"""Fused MoE kernel with FP8 weight using Ampere."""

import vllm
import torch
from vllm import _custom_ops as ops
import cupy
from typing import Dict, Any, Optional
import triton.language as tl
import vllm.model_executor.layers.fused_moe as default_fused_moe

# <todo:wenxh> Kernels performance needs to be optimized
#   such as one thread deals with multiple elements to reduce memory transaction.

convert_fp8e4m3_to_half = cupy.RawKernel(
    r"""
#include "cuda_fp8.h"
#include "cuda_fp16.h"
extern "C" __global__
void convert_fp8e4m3_to_half(const __nv_fp8_storage_t* x, float scale, half* y, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
        y[tid] = __nv_cvt_fp8_to_halfraw(x[tid], __NV_E4M3) * scale;
}
""",
    "convert_fp8e4m3_to_half",
)

convert_fp8e4m3_to_bfloat16 = cupy.RawKernel(
    r"""
#include "cuda_fp8.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
extern "C" __global__
void convert_fp8e4m3_to_bfloat16(const __nv_fp8_storage_t* x, float scale, __nv_bfloat16* y, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
        y[tid] = __float2bfloat16(__nv_cvt_fp8_to_halfraw(x[tid], __NV_E4M3) * scale);
}
""",
    "convert_fp8e4m3_to_bfloat16",
)

def dequantize_fp8(t_fp8, scale, dtype=torch.float16):
    s = torch.empty_like(t_fp8, dtype=dtype)
    scale = cupy.float32(scale.item())
    convert = convert_fp8e4m3_to_half if dtype == torch.float16 else convert_fp8e4m3_to_bfloat16
    convert(
        ((t_fp8.numel() + 1024 - 1) // 1024,),
        (1024,),
        (t_fp8.data_ptr(), scale, s.data_ptr(), t_fp8.numel()),
    )
    return s


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    training: bool = False,
    sparse_mixer: bool = False,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    This layer works the same as fused_moe, but it is used for the Ampere arch, which does not support fp8.
    By default, to be more comparable to Hopper, we reuse E4M3 configuration.
    <todo:wenxh> Use FP8E4b16 to reduce overhead:
        https://github.com/triton-lang/triton/blob/d7c8b3d7890125f5fc1b9f046e3189baa2665be4/python/triton/language/extra/cuda/utils.py#L34

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    w1_f = dequantize_fp8(w1, w1_scale, hidden_states.dtype)
    w2_f = dequantize_fp8(w2, w2_scale, hidden_states.dtype)

    out = default_fused_moe.fused_moe(
        hidden_states,
        w1_f,
        w2_f,
        gating_output,
        topk,
        renormalize,
        training,
        sparse_mixer,
        inplace,
        override_config,
        use_fp8=False,
    )

    del w1_f
    del w2_f

    return out

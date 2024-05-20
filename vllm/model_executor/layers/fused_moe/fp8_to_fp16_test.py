import torch
import triton
import triton.language as tl
import vllm
from vllm import _custom_ops as ops
import ampere_fp8_fused_moe

@triton.jit
def cast_kernel(x_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    the_output = tl.extra.cuda.convert_fp8e4b15_as_fp8e4m3_to_float16(x)
    tl.store(output_ptr + offsets, the_output, mask=mask)


torch.manual_seed(0)
size = 128
x = torch.rand(size, device='cuda') - 0.5
xs, s = ops.scaled_fp8_quant(x)
xs = triton.reinterpret(xs, tl.float8e4b15)
output = torch.empty_like(x, dtype=torch.float16, device='cuda')
n_elements = output.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
cast_kernel[grid](xs, output, n_elements, BLOCK_SIZE=1024)

print(x, s)
print(output*s)

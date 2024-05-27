import os, sys
import torch
os.environ["CUBLAS_LIBNAME"] = os.path.join(os.path.dirname(sys.executable), "..", "lib/libcublas.so.12")

from vllm import _custom_ops as ops

#!pip install git+https://github.com/wenxcs/pycublas.git
import pycublas


def moe_perf(
    tokens=1024,
    experts=8,
    topk=2,
    intermediate_size=14336,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)
    hidden_state = torch.randn(tokens*topk, hidden_size).cuda().bfloat16()
    w1 = torch.randn(intermediate_size * 2, hidden_size).cuda().bfloat16()
    w2 = torch.randn(hidden_size, intermediate_size * 2).cuda().bfloat16()


    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        intermediate_cache2 = torch.empty((tokens*topk, intermediate_size*2),
                                      device=hidden_state.device,
                                      dtype=hidden_state.dtype)
        a1 = pycublas.function.matmul.matmul_nt_bf16_fp32.apply(hidden_state, w1).bfloat16() # (tokens*topk, intermediate_size*2)
        ops.silu_and_mul(intermediate_cache2, a1)
        a2 = pycublas.function.matmul.matmul_nt_bf16_fp32.apply(intermediate_cache2, w2).bfloat16()

        end.record()
        torch.cuda.synchronize()
        if j >= 10:
            all_time += start.elapsed_time(end)
    
    return all_time/times

searchspace = [1] + list(range(0, 256, 32))[1:] + list(range(256, 4097, 256))
#searchspace = [1, 4096]
intermediate_size = 6400
expert_num = 16

for tk in searchspace:
    print(
        tk,
        ",",
        moe_perf(tokens=tk, experts=expert_num, intermediate_size=intermediate_size, use_fp8=False),
    )

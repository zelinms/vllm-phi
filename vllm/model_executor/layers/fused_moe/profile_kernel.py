import vllm
import ampere_fp8_fused_moe
import torch
from vllm import _custom_ops as ops
import argparse
import json
import sys
import os
import shutil


def moe_perf(
    tokens=1024,
    experts=8,
    topk=2,
    intermediate_size=14336,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8=True,
):
    torch.manual_seed(0)

    hidden_state = torch.randn(tokens, hidden_size).cuda().half()

    if use_fp8:
        w1, ws_scale = ops.scaled_fp8_quant(
            torch.randn(experts, intermediate_size * topk, hidden_size).cuda().half()
        )
        w2, w2s_scale = ops.scaled_fp8_quant(
            torch.randn(experts, hidden_size, intermediate_size).cuda().half()
        )
        _, h_scale = ops.scaled_fp8_quant(hidden_state)
        ws_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device)
        w2s_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device)
    else:
        w1 = torch.randn(experts, intermediate_size * topk, hidden_size).cuda().half()
        w2 = torch.randn(experts, hidden_size, intermediate_size).cuda().half()
        h_scale = None
        ws_scale = None
        w2s_scale = None

    gatew = torch.randn(hidden_size, experts).cuda().half()
    gating_output = torch.matmul(hidden_state, gatew).float()

    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        ampere_fp8_fused_moe.fused_moe(
            hidden_states=hidden_state,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
            topk=topk,
            override_config=config,
            renormalize=True,
            inplace=True,
            use_fp8=use_fp8,
            w1_scale=ws_scale,
            w2_scale=w2s_scale,
            a1_scale=h_scale,
            a2_scale=h_scale,
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)

        if j >= 10:
            all_time += elapsed_time_ms

    return all_time / times


if __name__ == "__main__":

    input_str = sys.stdin.read()

    try:
        data = json.loads(input_str)
        print(
            moe_perf(
                tokens=data["tokens"],
                experts=data["expert_num"],
                intermediate_size=data["intermediate_size"],
                config=data["config"],
                times=100,
            )
        )

    except json.JSONDecodeError:
        print("Invalid JSON data.")

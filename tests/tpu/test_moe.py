# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.models.llama4 import Llama4MoE

NUM_EXPERTS = [16]
TOP_KS = [1]

torch.manual_seed(12345)


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
    w2 = torch.randn((e, k, n), dtype=dtype) / 10
    score = torch.randn((m, e), dtype=dtype)

    apply_router_weight_on_input = True
    custom_routing_function = Llama4MoE.custom_routing_function
    # custom_routing_function = None
    # torch_output = torch_moe(a, w1, w2, score, topk,
    #                          apply_router_weight_on_input=apply_router_weight_on_input,
    #                          custom_routing_function=custom_routing_function)
    iterative_output = iterative_moe(
        a,
        w1,
        w2,
        score,
        topk,
        global_num_experts=e,
        apply_router_weight_on_input=apply_router_weight_on_input,
        custom_routing_function=custom_routing_function)

    a = a.to('xla')
    w1 = w1.to('xla')
    w2 = w2.to('xla')
    score = score.to('xla')
    xla_output = iterative_moe(
        a,
        w1,
        w2,
        score,
        topk,
        global_num_experts=e,
        apply_router_weight_on_input=apply_router_weight_on_input,
        custom_routing_function=custom_routing_function)

    xla_output = xla_output.cpu()

    print(xla_output)
    print(iterative_output)
    print(torch.max(xla_output - iterative_output))

    torch.testing.assert_close(iterative_output, xla_output, atol=5e-2, rtol=0)

# SPDX-License-Identifier: Apache-2.0
import functools
import tempfile

import pytest
import torch
import torch.utils._pytree as pytree
import torchax
from torchax.interop import extract_all_buffers, jax_jit

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)

torchax.enable_globally()

num_tokens = 32
num_heads = 12
head_size = 128
num_kv_heads = 2
num_blocks = 1024
max_num_reqs = 8
max_num_blocks_per_req = 8

torch.set_default_dtype(torch.bfloat16)

block_size = 16


def _setup_environment(model):
    engine_args = EngineArgs(model=model, )
    vllm_config = engine_args.create_engine_config()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        # Under single worker mode, full model is init first and then
        # partitioned using GSPMD.
        ensure_model_parallel_initialized(1, 1)
    return vllm_config


class M(torch.nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=0.08838834764831845,
            num_kv_heads=num_kv_heads,
            cache_config=CacheConfig(
                block_size=block_size,
                swap_space=4,
                cache_dtype='auto',
            ),
        )

    def forward(self, q, k, v):
        return self.attn(q, k, v)


def wrapped_module(m, vllm_config, forward_context):

    @functools.partial(jax_jit, kwargs_for_jax_jit={"static_argnums": (4, )})
    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            for layer_name, cache in kv_caches.items():
                forward_context[layer_name].kv_cache = [cache]
            res = torch.func.functional_call(m, weights, inputs)
            new_kv_caches = dict()
            for layer_name, cache in kv_caches.items():
                new_kv_caches[layer_name] = forward_context[
                    layer_name].kv_cache
            return res, new_kv_caches

    return func


slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
max_num_reqs = 8
max_num_blocks_per_req = 8
block_tables = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                           dtype=torch.int32)
context_lens = torch.ones((max_num_reqs, ), dtype=torch.int32)
query_lens = [1] * max_num_reqs
query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                            dtype=torch.int32),
                               dim=0,
                               dtype=torch.int32)
num_seqs = torch.tensor([max_num_reqs], dtype=torch.int32)

num_blocks = 1024


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen2-1.5B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
    ])
@torch.no_grad()
def test_tpu_model_loader(model):
    torchax.enable_globally()
    vllm_config = _setup_environment(model)
    with set_current_vllm_config(vllm_config):
        m = M()
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)

        block_tables = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                                   dtype=torch.int32)
        context_lens = torch.ones((max_num_reqs, ), dtype=torch.int32)
        query_lens = [1] * max_num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32)
        num_seqs = torch.tensor([max_num_reqs], dtype=torch.int32)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping.to('jax'),
            block_tables=block_tables.to('jax'),
            context_lens=context_lens.to('jax'),
            query_start_loc=query_start_loc.to('jax'),
            num_seqs=num_seqs.to('jax'),
        )
        kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)
        # simulate bind
        m = m.to("jax")
        print(f"check kv_cache_shape {kv_cache_shape}")
        kv_cache = torch.rand(kv_cache_shape).to('jax')
        # m.attn.kv_cache = [torch.rand(kv_cache_shape).to('jax')]
        q = torch.rand(num_tokens, num_heads * head_size).to('jax')
        k = torch.rand(num_tokens, num_kv_heads * head_size).to('jax')
        v = torch.rand(num_tokens, num_kv_heads * head_size).to('jax')
        fwd_func = jax_jit(m.forward)
        params, buffers = extract_all_buffers(m)
        params = pytree.tree_map(lambda x: x.to('jax'), params)
        buffers = pytree.tree_map(lambda x: x.to('jax'), buffers)
        params_and_buffers = params
        params_and_buffers.update(buffers)
        print("params", params)
        print("buffers", buffers)

        # Wrap func
        forward_context = {'attn': m.attn}
        wrapped_func = wrapped_module(m, vllm_config, forward_context)

        # with set_forward_context(attn_metadata,
        #                          vllm_config,
        #                          num_tokens=num_tokens):
        # JAX lowering
        # args = (q, k, v)

        # jax_args = env.t2j_iso(args)
        # lowered_ir = jax.jit(jax_view(m.forward)).lower(*jax_args).compiler_ir()
        # print(lowered_ir)

        # Model fwd
        # out = fwd_func(q, k, v)
        # print(out)

        # Wrap the model

        kv_caches = {'attn': kv_cache}
        # kv_caches = [kv_cache]
        out = wrapped_func(params_and_buffers, (q, k, v), kv_caches,
                           attn_metadata, num_tokens)
        print(out)

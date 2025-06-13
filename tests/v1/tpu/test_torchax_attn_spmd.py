# SPDX-License-Identifier: Apache-2.0
import functools
import tempfile

import jax
import pytest
import torch
import torch.utils._pytree as pytree
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torchax.interop import extract_all_buffers, jax_jit

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.distributed.tpu_distributed_utils import shard_model
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)

torchax.enable_globally()

num_tokens = 32
num_heads = 32
head_size = 128
num_kv_heads = 8
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
        # "Qwen/Qwen2-1.5B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
    ])
@torch.no_grad()
def test_attn(model):
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('x', ))
    torchax.enable_globally()
    vllm_config = _setup_environment(model)
    hf_config = vllm_config.model_config.hf_config
    print(f"HF config: {hf_config}")
    with set_current_vllm_config(vllm_config):
        from vllm.model_executor.models.llama import LlamaDecoderLayer
        decoder_model = LlamaDecoderLayer(
            hf_config,
            prefix="decoder.layers.0",
        )
        print("check decoder model", decoder_model)
        shard_model(decoder_model, mesh)
        decoder_model = decoder_model.to('jax')
        print("check decoder model", decoder_model)
        breakpoint()
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

        # Sharding
        duplicated_sharding = NamedSharding(mesh, P())
        kv_cache_sharding = NamedSharding(mesh, P(None, None, 'x', None))
        qkv_sharding = NamedSharding(mesh, P(None, 'x'))

        # Replicate metadata
        slot_mapping = slot_mapping.to('jax').apply_jax(
            jax.device_put, duplicated_sharding)
        block_tables = block_tables.to('jax').apply_jax(
            jax.device_put, duplicated_sharding)
        context_lens = context_lens.to('jax').apply_jax(
            jax.device_put, duplicated_sharding)
        query_start_loc = query_start_loc.to('jax').apply_jax(
            jax.device_put, duplicated_sharding)
        num_seqs = num_seqs.to('jax').apply_jax(jax.device_put,
                                                duplicated_sharding)

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
        )
        kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)
        # simulate bind
        m = m.to("jax")
        print(f"check kv_cache_shape {kv_cache_shape}")

        # Shard kv cache
        kv_cache = torch.rand(kv_cache_shape).to('jax').apply_jax(
            jax.device_put, kv_cache_sharding)
        # m.attn.kv_cache = [torch.rand(kv_cache_shape).to('jax')]

        q = torch.rand(num_tokens, num_heads * head_size).to('jax').apply_jax(
            jax.device_put, qkv_sharding)
        k = torch.rand(num_tokens,
                       num_kv_heads * head_size).to('jax').apply_jax(
                           jax.device_put, qkv_sharding)
        v = torch.rand(num_tokens,
                       num_kv_heads * head_size).to('jax').apply_jax(
                           jax.device_put, qkv_sharding)

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

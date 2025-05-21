# SPDX-License-Identifier: Apache-2.0
import functools
import tempfile

import pytest
import torch
import torchax
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers, jax_jit

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.tpu import TPUModelLoader
from vllm.v1.attention.backends.pallas import PallasMetadata


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


def wrapped_module(m, vllm_config, static_forward_context):

    @functools.partial(jax_jit, kwargs_for_jax_jit={"static_argnums": (4, )})
    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            # m.attn.kv_cache = kv_cache
            for layer_name, cache in static_forward_context.items():
                static_forward_context[layer_name].kv_cache = [
                    kv_caches[layer_name]
                ]
            # TODO: some buffers are tied, investigate how it works.
            res = torch.func.functional_call(m,
                                             weights,
                                             kwargs={
                                                 "input_ids": inputs[0],
                                                 "positions": inputs[1],
                                             },
                                             tie_weights=False)
            # new_kv_cache = m.attn.kv_cache
            new_kv_cache = dict()
            for layer_name, cache in static_forward_context.items():
                new_kv_cache[layer_name] = static_forward_context[
                    layer_name].kv_cache
            return res, new_kv_cache

    return func


def _create_pallas_metadata_from_dict(dict):
    dict = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'), dict)

    def _convert_one(d):
        pallas_metadata = PallasMetadata(slot_mapping=d['slot_mapping'],
                                         block_tables=d['block_tables'],
                                         context_lens=d['context_lens'],
                                         query_start_loc=d['query_start_loc'],
                                         num_seqs=d['num_seqs'])
        return pallas_metadata

    res = {}
    for key, value in dict.items():
        res[key] = _convert_one(value)
    return res


def _load_dump(path):
    with torchax.default_env():
        dump_dict = torch.load(path)
        pallas_metadata = _create_pallas_metadata_from_dict(
            dump_dict["attn_metadata"])
        input_ids = dump_dict['input_ids'].to('jax')
        position_ids = dump_dict['position_ids'].to('jax')
        return pallas_metadata, input_ids, position_ids


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen2-1.5B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
    ])
def test_tpu_model_loader(model):
    vllm_config = _setup_environment(model)
    loader = TPUModelLoader(load_config=vllm_config.load_config)
    model = loader.load_model(None, vllm_config)

    torchax.enable_globally()
    # env = torchax.default_env()
    # with env:
    model = model.to('jax')
    print(model)
    # for name, param in model.named_parameters():
    #     print(f"name {name}: param {param}")
    attn_metadata, input_ids, position_ids = \
        _load_dump('/home/lsiyuan/torchax_dump/attn_metadata.pt')

    kv_caches = torch.load('/home/lsiyuan/torchax_dump/kv_caches.pt')
    for key, value in kv_caches.items():
        kv_caches[key] = value[:1024]
    kv_caches = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
                                     kv_caches)

    # Simulate bind kv cache
    static_forward_context = \
        vllm_config.compilation_config.static_forward_context

    # breakpoint()
    wrapped_func = wrapped_module(model, vllm_config, static_forward_context)
    params, buffers = extract_all_buffers(model)
    params_and_buffers = {**params, **buffers}
    params_and_buffers = pytree.tree_map_only(torch.Tensor,
                                              lambda x: x.to('jax'),
                                              params_and_buffers)
    input_args = (input_ids, position_ids)
    num_tokens = 9  # Not used?
    out = wrapped_func(params_and_buffers, input_args, kv_caches,
                       attn_metadata, num_tokens)

    print(out)
    # breakpoint()

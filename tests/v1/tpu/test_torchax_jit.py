# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import os
import tempfile

import jax
import pytest
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torchax
from jax.sharding import Mesh
from torch.nn.utils import stateless as torch_stateless
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers, jax_jit

from vllm.attention.layer import Attention
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.distributed.tpu_distributed_utils import (
    create_torchax_tensor_with_partition_spec)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.tpu import TPUModelLoader
from vllm.v1.attention.backends.pallas import PallasMetadata

MESH = None


def _get_spmd_mesh():
    global MESH
    if MESH is None:
        if os.environ.get('VLLM_TORCHAX_ENABLED', '0') == '1':
            devices = jax.devices()
            MESH = Mesh(devices, axis_names=('x', ))
        else:
            xr.use_spmd()
            num_devices = xr.global_runtime_device_count()
            mesh_shape = (num_devices, 1)
            device_ids = np.array(range(num_devices))
            MESH = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))
    return MESH


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

    @functools.partial(
        jax_jit,
        kwargs_for_jax_jit={
            "static_argnums": (4, ),
            "donate_argnums": (2, )
        },
    )
    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            # Bind kv cache
            for layer_name, attn in static_forward_context.items():
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
            for layer_name, attn in static_forward_context.items():
                new_kv_cache[layer_name] = static_forward_context[
                    layer_name].kv_cache
            return res, new_kv_cache

    return func


def _create_pallas_metadata_from_dict(dict):
    with torchax.default_env():
        # dict = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'), dict)
        dict = pytree.tree_map_only(
            torch.Tensor,
            lambda x: create_torchax_tensor_with_partition_spec(x, MESH,
                                                                ()), dict)

        def _convert_one(d):
            pallas_metadata = PallasMetadata(
                slot_mapping=d['slot_mapping'],
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
    dump_dict = torch.load(path)
    with torchax.default_env():
        pallas_metadata = _create_pallas_metadata_from_dict(
            dump_dict["attn_metadata"])
        input_ids = create_torchax_tensor_with_partition_spec(
            dump_dict['input_ids'], MESH, ())
        position_ids = create_torchax_tensor_with_partition_spec(
            dump_dict['position_ids'], MESH, ())
        return pallas_metadata, input_ids, position_ids


def functional_call(model, method_name, params, buffers, *args, **kwargs):
    kwargs = kwargs or {}
    params_copy = copy.copy(params)
    params_copy.update(buffers)
    # # reinflate the state dict so there are not any missing keys
    # for k, v in self._extra_dumped_weights.items():
    #     for new_key in v:
    #         params_copy[new_key] = params_copy[k]
    with torch_stateless._reparametrize_module(model, params_copy):
        res = getattr(model, method_name)(*args, **kwargs)
    return res


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
    mesh = _get_spmd_mesh()
    model = loader.load_model(vllm_config, vllm_config.model_config, mesh)

    attn_metadata, input_ids, position_ids = \
        _load_dump('/home/lsiyuan/torchax_dump/attn_metadata.pt')

    torchax.enable_globally()
    model = model.to('jax')
    print(model)

    assert isinstance(model.model.layers[0].self_attn.attn, Attention)

    n_blocks = 1024
    block_size = 16
    num_kv_heads = model.model.layers[0].self_attn.attn.num_kv_heads
    head_size = model.model.layers[0].self_attn.attn.head_size
    kv_dtype = torch.bfloat16

    kv_caches = dict()
    kv_cache_shape = (n_blocks, block_size, num_kv_heads * 2, head_size)
    for i in range(len(model.model.layers)):
        key = f"model.layers.{i}.self_attn.attn"
        kv_caches[key] = torch.zeros(kv_cache_shape, dtype=kv_dtype)

    # kv_caches = torch.load('/home/lsiyuan/torchax_dump/kv_caches.pt')
    # for key, value in kv_caches.items():
    #     kv_caches[key] = value[:1024]
    #     print(f"key {key}: value {kv_caches[key].shape}")

    # This is for copy
    # kv_caches = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
    #                                  kv_caches)

    # This is for sharding
    # kv_caches = pytree.tree_map_only(torch.Tensor, lambda x:
    #                 create_torchax_tensor_with_partition_spec(x, mesh, (None, None, 'x', None)),
    #                                   kv_caches)
    # Duplicate since the dump is for qwen, and qwen has 4 heads
    kv_caches = pytree.tree_map_only(
        torch.Tensor,
        lambda x: create_torchax_tensor_with_partition_spec(x, mesh,
                                                            ()), kv_caches)

    # Simulate bind kv cache
    static_forward_context = \
        vllm_config.compilation_config.static_forward_context

    wrapped_func = wrapped_module(model, vllm_config, static_forward_context)
    params, buffers = extract_all_buffers(model)
    # It's already placed on device by tpu loader.
    # params, buffers = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
    #                                        (params, buffers))
    params_and_buffers = {**params, **buffers}
    input_args = (input_ids, position_ids)
    num_tokens = 9  # Not used?
    for name, tensor in params_and_buffers.items():
        if not isinstance(tensor, torchax.tensor.Tensor):
            # print("name: {}, tensor: {}".format(name, tensor))
            params_and_buffers[name] = \
                create_torchax_tensor_with_partition_spec(tensor, mesh, ())
    # breakpoint()
    hidden_states, new_kv_caches = wrapped_func(params_and_buffers, input_args,
                                                kv_caches, attn_metadata,
                                                num_tokens)
    print(hidden_states)
    for new_kv_cache in new_kv_caches.values():
        # Ensure kv cache is updated.
        assert torch.count_nonzero(new_kv_cache[0]) > 0

    @jax_jit
    def wrapped_compute_logits(params, buffers, hidden_states, sampling_param):
        return functional_call(model, "compute_logits", params, buffers,
                               hidden_states, sampling_param)

    logits = wrapped_compute_logits(params, buffers, hidden_states, None)
    # logits = model.compute_logits(hidden_states, None)
    print(logits)

# SPDX-License-Identifier: Apache-2.0
import functools

import torch
from torch.nn.utils import stateless as torch_stateless
from torchax.interop import jax_jit

from vllm.forward_context import set_forward_context


def wrap_model(m, vllm_config, static_forward_context):

    @functools.partial(
        jax_jit,
        kwargs_for_jax_jit={
            "static_argnums": (4, ),
            "donate_argnums": (2, )  # KV cache buffer donation.
        },
    )
    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            for layer_name, attn in static_forward_context.items():
                attn.kv_cache = [kv_caches[layer_name]]
            # TODO: some buffers are tied, investigate how it works.
            res = torch.func.functional_call(m,
                                             weights,
                                             kwargs={
                                                 "input_ids": inputs[0],
                                                 "positions": inputs[1],
                                             },
                                             tie_weights=False)
            new_kv_cache = dict()
            for layer_name, attn in static_forward_context.items():
                new_kv_cache[layer_name] = attn.kv_cache
            return res, new_kv_cache

    return func


def wrap_model_func(model, method_name):

    @jax_jit
    def func(params_and_buffers, *args, **kwargs):
        with torch_stateless._reparametrize_module(model, params_and_buffers):
            res = getattr(model, method_name)(*args, **kwargs)
        return res

    return func

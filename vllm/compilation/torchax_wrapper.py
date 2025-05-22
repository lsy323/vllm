# SPDX-License-Identifier: Apache-2.0
import copy
import functools

import torch
from torch.nn.utils import stateless as torch_stateless
from torchax.interop import jax_jit

from vllm.forward_context import set_forward_context


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
            new_kv_cache = dict()
            for layer_name, attn in static_forward_context.items():
                new_kv_cache[layer_name] = static_forward_context[
                    layer_name].kv_cache
            return res, new_kv_cache

    return func


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


@jax_jit
def wrapped_compute_logits(params, buffers, hidden_states, sampling_param):
    return functional_call(model, "compute_logits", params, buffers,
                           hidden_states, sampling_param)

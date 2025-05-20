# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torchax
from torch.utils import _pytree as pytree
from torchax.interop import jax_jit

_forward_context = None


@contextmanager
def set_forward_context(attn_metadata, num_tokens):
    global _forward_context
    _forward_context = {
        # A dataclass containing torch tensors (Maybe other primitive types)
        "attn_metadata": attn_metadata,
        # An integer (not used for now)
        "num_tokens": num_tokens,
    }
    yield


def get_forward_context():
    assert _forward_context is not None
    return _forward_context


@dataclass
class AttentionMetadata:
    a: torch.Tensor
    b: torch.Tensor


class M(torch.nn.Module):

    def __init__(self):
        super(M, self).__init__()

    def forward(self, x):
        fwd_context = get_forward_context()
        attn_metadata = fwd_context["attn_metadata"]
        a = attn_metadata.a
        b = attn_metadata.b
        return a * (x + self.kv_cache) + b

    def set_kv_cache(self, kv_cache):
        self.kv_cache = kv_cache


a = torch.tensor([2], dtype=torch.int64)
b = torch.tensor([3], dtype=torch.int64)
attn_metadata = AttentionMetadata(a, b)

kv_cache = torch.tensor([4], dtype=torch.int64)
m = M()
m.set_kv_cache(kv_cache)
x = torch.tensor([5], dtype=torch.int64)
args = (x, )

with set_forward_context(attn_metadata, 8):
    res = m(*args)
    print(res)

# Run with torchax
torchax.enable_globally()
attn_metadata = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
                                     attn_metadata)
args = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'), args)


def wrap_func(m):

    def wrapped_func(kv_cache, metadata_a, metadata_b, num_tokens, inputs):
        m.set_kv_cache(kv_cache)
        metadata = AttentionMetadata(metadata_a, metadata_b)
        with set_forward_context(metadata, num_tokens):
            res = torch.func.functional_call(m, {}, inputs)
        new_kv_cache = m.kv_cache
        return res, new_kv_cache

    return wrapped_func


wrapped_func = wrap_func(m)
jax_jitted_func = jax_jit(wrapped_func)
jax_args = pytree.tree_map_only(
    torch.Tensor, lambda x: x.to('jax'),
    (kv_cache, attn_metadata.a, attn_metadata.b, 8, x))
jax_res, new_kv_cache = jax_jitted_func(*jax_args)
print(jax_res)

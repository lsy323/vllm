# SPDX-License-Identifier: Apache-2.0
import jax
import torch
import torchax
from torchax.interop import jax_jit, jax_view

torchax.enable_globally()
env = torchax.default_env()

# def update_cache(cache, indices, k):
#     k = k + 1
#     return k


def update_cache(cache, indices, k):
    cache.index_copy_(0, indices, k)
    return new_cache


cache = torch.zeros((64, 8 * 128)).to('jax')
indices = torch.tensor([5, 6, 7], dtype=torch.int64).to('jax')
k = torch.rand(3, 8 * 128).to('jax')

jitted_func = jax_jit(update_cache)

args = (cache, indices, k)
# update_cache(cache, indices, k)
new_cache = jitted_func(*args)
print(cache[4:9])
print(new_cache[4:9])

jax_args = env.t2j_iso(args)
# breakpoint()
lowered_ir = jax.jit(jax_view(update_cache)).lower(*jax_args).compiler_ir()

# Print the IR as text
print(lowered_ir)

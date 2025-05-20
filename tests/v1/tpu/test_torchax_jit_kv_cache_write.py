# SPDX-License-Identifier: Apache-2.0
import torch
import torchax
from torchax.interop import jax_jit

from vllm.v1.attention.backends.pallas import write_to_kv_cache

torchax.enable_globally()

# breakpoint()

num_tokens = 3
num_kv_heads = 2
head_size = 3
num_blocks = 64
block_size = 16

key = torch.rand(num_tokens, num_kv_heads * head_size).to('jax')
value = torch.rand(num_tokens, num_kv_heads * head_size).to('jax')
kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads * 2,
                       head_size).to('jax')
indices = torch.tensor([3, 4, 5], dtype=torch.int64).to('jax')


def update_cache(kv_cache, indices, key, value):
    new_cache = write_to_kv_cache(key, value, kv_cache, indices)
    return new_cache


jitted_func = jax_jit(update_cache)
args = (kv_cache, indices, key, value)

new_cache = jitted_func(*args)
print(kv_cache[3:6])
print(new_cache[3:6])

# SPDX-License-Identifier: Apache-2.0
import torch
import torchax
from torchax.interop import jax_jit

from vllm.v1.attention.backends.pallas import PallasMetadata

torchax.enable_globally()

metadata = PallasMetadata(
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'))

metadata2 = PallasMetadata(
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'),
    torch.rand(5).to('jax'))

metadata_dict = {"metadata_1": metadata, "metadata_2": metadata2}


@jax_jit
def func(metadata: PallasMetadata):
    return metadata.block_tables + metadata.slot_mapping + \
        metadata.context_lens + metadata.query_start_loc + metadata.num_seqs


out = func(metadata)
print(out)


@jax_jit
def func(metadata_dict):
    metadata_1 = metadata_dict["metadata_1"]
    metadata_2 = metadata_dict["metadata_2"]
    metadata_sum_1 = metadata_1.block_tables + metadata_1.slot_mapping + \
        metadata_1.context_lens + metadata_1.query_start_loc \
            + metadata_1.num_seqs
    metadata_sum_2 = metadata_2.block_tables + metadata_2.slot_mapping + \
        metadata_2.context_lens + metadata_2.query_start_loc \
            + metadata_2.num_seqs
    return metadata_sum_1 + metadata_sum_2


out = func(metadata_dict)
print(out)

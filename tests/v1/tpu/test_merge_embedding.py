# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla.core.xla_model as xm

from vllm.model_executor.models.utils import merge_multimodal_embeddings

# placeholder_token_id = 32000
# input_ids = torch.zeros(1024, dtype=torch.int32)
# input_ids[:576] = placeholder_token_id
# input_ids = input_ids.to('xla')
# multimodal_embeddings = torch.rand(1, 576, 4096).to(torch.bfloat16).to('xla')
# inputs_embeds = torch.rand(1024, 4096).to(torch.bfloat16).to('xla')

dump_dict = torch.load("/home/lsiyuan/mm_dump/mm_embedding_dict.pt")
placeholder_token_id = dump_dict["image_token_index"]
input_ids = dump_dict["input_ids"].to('xla')
multimodal_embeddings = dump_dict["multimodal_embeddings"]
for i in range(len(multimodal_embeddings)):
    multimodal_embeddings[i] = multimodal_embeddings[i].to('xla')
inputs_embeds = dump_dict["inputs_embeds"].to('xla')

embed = merge_multimodal_embeddings(
    input_ids,
    inputs_embeds,
    multimodal_embeddings,
    placeholder_token_id,
)
xm.mark_step()
xm.wait_device_ops()

embed = embed.cpu()
print(embed)
print(embed.shape)
# SPDX-License-Identifier: Apache-2.0

import torch

# def test_ragged_paged_attention():
#     # We verify that the kernel inputs such as sliding_window, etc. are passed
#     # in from the model correctly.
#     # The correctness of the paged attention kernel is tested in the kernel
#     # library.
#     num_heads = 4
#     head_size = 128
#     scale = 1.0
#     num_kv_heads = 4
#     sliding_window = 128
#     logits_soft_cap = 50.0
#     attn_impl = PallasAttentionBackendImpl(
#         num_heads=num_heads,
#         head_size=head_size,
#         scale=scale,
#         num_kv_heads=num_kv_heads,
#         alibi_slopes=None,
#         sliding_window=sliding_window,
#         kv_cache_dtype="auto",
#         logits_soft_cap=logits_soft_cap,
#         attn_type=AttentionType.DECODER,
#     )

#     class FakeAttentionLayer:
#         _k_scale_float: float
#         _v_scale_float: float

#     layer = FakeAttentionLayer()
#     layer._k_scale_float = 1.0
#     layer._v_scale_float = 1.0

#     num_tokens = 16
#     num_blocks = 1024
#     block_size = 16
#     query = torch.zeros(num_tokens, num_heads * head_size)
#     key = torch.zeros(num_tokens, num_kv_heads * head_size)
#     value = torch.zeros(num_tokens, num_kv_heads * head_size)
#     kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads * 2, head_size)
#     slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
#     max_num_reqs = 8
#     max_num_blocks_per_req = 8
#     block_tables = torch.zeros((max_num_reqs, max_num_blocks_per_req),
#                                dtype=torch.int32)
#     context_lens = torch.ones((max_num_reqs, ), dtype=torch.int32)
#     query_lens = [1] * max_num_reqs
#     query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
#                                                 dtype=torch.int32),
#                                    dim=0,
#                                    dtype=torch.int32)
#     num_seqs = torch.tensor([max_num_reqs], dtype=torch.int32)
#     attn_metadata = PallasMetadata(
#         slot_mapping=slot_mapping,
#         block_tables=block_tables,
#         context_lens=context_lens,
#         query_start_loc=query_start_loc,
#         num_seqs=num_seqs,
#     )

#     with patch("torch.ops.xla.ragged_paged_attention"
#                ) as mock_ragged_paged_attention:
#         attn_impl.forward(
#             layer=layer,
#             query=query,
#             key=key,
#             value=value,
#             kv_cache=kv_cache,
#             attn_metadata=attn_metadata,
#         )

#         mock_ragged_paged_attention.assert_called_once_with(
#             ANY,  # query
#             ANY,  # kv_cache
#             ANY,  # context_lens
#             ANY,  # block_tables
#             ANY,  # query_start_loc
#             ANY,  # num_seqs
#             num_kv_pages_per_block=None,
#             num_queries_per_block=None,
#             vmem_limit_bytes=None,
#             use_kernel=True,
#             sm_scale=scale,
#             sliding_window=sliding_window,
#             soft_cap=logits_soft_cap,
#         )

# def test_ragged_paged_attention():
#     query = torch.zeros(16, 32, 128).to(torch.bfloat16).to('xla')
#     kv_cache = torch.zeros(277, 16, 64, 128).to(torch.bfloat16).to('xla')
#     context_lens = torch.ones(8, dtype=torch.int32).to('xla')
#     block_tables = torch.zeros(8, 64, dtype=torch.int32).to('xla')
#     query_start_loc = torch.arange(9).to(torch.int32).to('xla')
#     num_seqs = torch.tensor([8]).to(torch.int32).to('xla')

#     out_placeholder = torch.zeros_like(query)
#     out = torch.ops.xla.ragged_paged_attention(
#         query,
#         kv_cache,
#         context_lens,
#         block_tables,
#         query_start_loc,
#         num_seqs,
#         num_kv_pages_per_block=None,
#         num_queries_per_block=None,
#         vmem_limit_bytes=None,
#         use_kernel=False,
#         sm_scale=1.0,
#         sliding_window=None,
#         soft_cap=None,
#     )
#     out_placeholder[:out.shape[0]] = out[:]
#     out = out.cpu()
#     out_placeholder = out_placeholder.cpu()
#     print(out.shape)
#     print(out_placeholder.shape)


def test_ragged_paged_attention():
    query = torch.rand(1024, 32 * 128).to(torch.bfloat16).to('xla')
    kv_cache = torch.rand(277, 16, 64, 128).to(torch.bfloat16).to('xla')
    context_lens = torch.tensor([581, 0, 0, 0, 0, 0, 0, 0],
                                dtype=torch.int32).to('xla')
    block_tables = torch.zeros(8, 64, dtype=torch.int32)
    block_idx = torch.arange(1, 38, dtype=torch.int32)
    block_tables[0, :len(block_idx)] = block_idx
    # print(block_tables)
    block_tables = block_tables.to('xla')
    query_start_loc = torch.tensor([0, 576, 1, 1, 1, 1, 1, 1,
                                    1]).to(torch.int32).to('xla')
    num_seqs = torch.tensor([1]).to(torch.int32).to('xla')

    query = query.view(-1, 32, 128)
    out = torch.ops.xla.ragged_paged_attention(
        query,
        kv_cache,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        num_kv_pages_per_block=None,
        num_queries_per_block=None,
        vmem_limit_bytes=None,
        use_kernel=True,
        sm_scale=1 / 1.4,
        sliding_window=None,
        soft_cap=None,
    )
    out = out.view(-1, 32 * 128)
    out = out.cpu()
    print(out.shape)

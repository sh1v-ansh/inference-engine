#pragma once
#include <cuda_runtime.h>

void launch_paged_attention(
    void* out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int* block_tables,
    const int* context_lens,
    float scale,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    cudaStream_t stream
);

#pragma once
#include <cuda_runtime.h>

void launch_reshape_and_cache(
    const void* key,
    const void* value,
    void* key_cache,
    void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
);

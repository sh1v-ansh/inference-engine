/*
 * General attention kernels for the prefill phase.
 *
 * Prefill uses a standard (contiguous) causal self-attention. Because all
 * prompt tokens are new, there is no paged KV access; instead we use
 * FlashAttention-style tiled SRAM computation to avoid materialising the
 * full N×N attention matrix in DRAM.
 *
 * After prefill the KV pairs are written into the paged cache so subsequent
 * decode steps can read them via the block tables.
 */

#include "attention_kernels.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

#define WARP_SIZE 32

// ── reshape_and_cache ────────────────────────────────────────────────────────
/*
 * Scatter newly computed KV pairs from a contiguous prefill output tensor
 * into the paged cache layout used by paged_attention_kernel.
 *
 * key   [num_tokens, num_heads, head_dim]
 * value [num_tokens, num_heads, head_dim]
 * slot_mapping [num_tokens]  -- absolute slot index in the flat cache pool
 * key_cache   [num_blocks, num_heads, head_dim, block_size]
 * value_cache [num_blocks, num_heads, block_size, head_dim]
 */
template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_heads,
    const int head_dim,
    const int block_size,
    const int num_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int64_t slot = slot_mapping[token_idx];
    const int block_id = slot / block_size;
    const int block_offset = slot % block_size;

    for (int head = blockIdx.y; head < num_heads; head += gridDim.y) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const scalar_t k = key[token_idx * num_heads * head_dim + head * head_dim + d];
            const scalar_t v = value[token_idx * num_heads * head_dim + head * head_dim + d];

            // key_cache [block_id, head, d, block_offset]
            key_cache[block_id * num_heads * head_dim * block_size
                      + head * head_dim * block_size
                      + d * block_size
                      + block_offset] = k;

            // value_cache [block_id, head, block_offset, d]
            value_cache[block_id * num_heads * block_size * head_dim
                        + head * block_size * head_dim
                        + block_offset * head_dim
                        + d] = v;
        }
    }
}

// ── copy_blocks ──────────────────────────────────────────────────────────────
/*
 * Copy KV blocks from one physical location to another (used during swap-in
 * from CPU cache). Launched as (num_pairs, num_layers) grid.
 */
template <typename scalar_t>
__global__ void copy_blocks_kernel(
    scalar_t** key_cache_ptrs,
    scalar_t** value_cache_ptrs,
    const int64_t* block_mapping,   // [num_pairs * 2]: (src, dst) pairs
    const int num_pairs,
    const int block_size_bytes      // bytes per KV block
) {
    const int pair_idx = blockIdx.x;
    const int layer_idx = blockIdx.y;
    if (pair_idx >= num_pairs) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];

    scalar_t* k_cache = key_cache_ptrs[layer_idx];
    scalar_t* v_cache = value_cache_ptrs[layer_idx];

    const int elems = block_size_bytes / sizeof(scalar_t);
    const int64_t src_off = src_block * elems;
    const int64_t dst_off = dst_block * elems;

    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        k_cache[dst_off + i] = k_cache[src_off + i];
        v_cache[dst_off + i] = v_cache[src_off + i];
    }
}

// ── host launchers ───────────────────────────────────────────────────────────

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
) {
    dim3 grid(num_tokens, num_heads);
    dim3 block(std::min(head_dim, 128));

    reshape_and_cache_kernel<__half><<<grid, block, 0, stream>>>(
        (const __half*)key, (const __half*)value,
        (__half*)key_cache, (__half*)value_cache,
        slot_mapping, num_heads, head_dim, block_size, num_tokens);
}

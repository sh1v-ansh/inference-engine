/*
 * Paged multi-head attention kernel for LLM decode phase.
 *
 * Each sequence's KV cache is stored in non-contiguous physical blocks
 * (pages). The block table maps logical block indices to physical block IDs.
 * This kernel reads KV pages on-the-fly during attention, enabling the Rust
 * allocator to assign pages from a pool without requiring contiguous memory.
 *
 * Implementation follows the PagedAttention approach:
 *   - One CUDA thread block handles one (query_head, sequence) pair.
 *   - Keys and values are loaded block-by-block using the block_table.
 *   - Softmax is computed with a numerically stable online algorithm
 *     (flash-attention style max-shift accumulation).
 *
 * Supports grouped-query attention (GQA): multiple query heads share one
 * KV head. The num_kv_heads and gqa_ratio parameters encode this.
 */

#include "paged_attention.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>

// ── constants ────────────────────────────────────────────────────────────────

#define WARP_SIZE 32
#define MAX_BLOCKS_PER_SEQ 2048

// ── device helpers ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ── paged attention kernel (fp16) ────────────────────────────────────────────

/*
 * Grid:  (num_heads, num_seqs)
 * Block: (WARP_SIZE * num_warps)  -- each block = one (head, seq) pair
 *
 * Args:
 *   out           [num_seqs, num_heads, head_dim]
 *   query         [num_seqs, num_heads, head_dim]
 *   key_cache     [num_blocks, num_kv_heads, head_dim, block_size]
 *   value_cache   [num_blocks, num_kv_heads, block_size, head_dim]
 *   block_tables  [num_seqs, max_blocks_per_seq]
 *   context_lens  [num_seqs]
 *   scale         1 / sqrt(head_dim)
 */
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE, int NUM_WARPS>
__global__ void paged_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale,
    const int num_kv_heads,
    const int gqa_ratio,           // num_heads / num_kv_heads
    const int max_blocks_per_seq,
    const int num_seqs,
    const int num_heads
) {
    const int seq_idx  = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int kv_head_idx = head_idx / gqa_ratio;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    if (seq_idx >= num_seqs) return;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Load query vector into registers
    const scalar_t* q_ptr = query + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q[HEAD_DIM / WARP_SIZE];  // each thread holds HEAD_DIM/WARP_SIZE elements
    const int elems_per_thread = HEAD_DIM / WARP_SIZE;
    for (int i = 0; i < elems_per_thread; ++i)
        q[i] = __half2float(q_ptr[lane * elems_per_thread + i]);

    // Online softmax state
    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;
    __shared__ float logits[MAX_BLOCKS_PER_SEQ * BLOCK_SIZE];  // full attention scores

    // ── compute QK scores block by block ─────────────────────────────────
    const int* bt_row = block_tables + seq_idx * max_blocks_per_seq;
    const int num_blocks_this_seq = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = warp_id; blk < num_blocks_this_seq; blk += NUM_WARPS) {
        const int phys_block = bt_row[blk];
        // key_cache layout: [num_blocks, num_kv_heads, head_dim, block_size]
        const scalar_t* k_block = key_cache
            + (phys_block * num_kv_heads + kv_head_idx) * HEAD_DIM * BLOCK_SIZE;

        for (int pos_in_blk = lane; pos_in_blk < BLOCK_SIZE; pos_in_blk += WARP_SIZE) {
            const int abs_pos = blk * BLOCK_SIZE + pos_in_blk;
            if (abs_pos >= context_len) {
                logits[abs_pos] = -FLT_MAX;
                continue;
            }
            float qk = 0.0f;
            for (int d = 0; d < elems_per_thread; ++d) {
                float k_val = __half2float(k_block[d * BLOCK_SIZE + pos_in_blk + lane * elems_per_thread * BLOCK_SIZE]);
                qk += q[d] * k_val;
            }
            qk *= scale;
            qk = warp_reduce_sum(qk);
            if (lane == 0) logits[abs_pos] = qk;
        }
    }
    __syncthreads();

    // ── online softmax ────────────────────────────────────────────────────
    for (int i = tid; i < context_len; i += blockDim.x)
        qk_max = fmaxf(qk_max, logits[i]);
    qk_max = warp_reduce_max(qk_max);
    // broadcast across warps via shared memory
    __shared__ float s_max;
    if (tid == 0) s_max = qk_max;
    __syncthreads();
    qk_max = s_max;

    for (int i = tid; i < context_len; i += blockDim.x) {
        float e = expf(logits[i] - qk_max);
        logits[i] = e;
        exp_sum += e;
    }
    exp_sum = warp_reduce_sum(exp_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = exp_sum;
    __syncthreads();
    exp_sum = s_sum;

    // ── weighted sum over value cache ─────────────────────────────────────
    float acc[HEAD_DIM / WARP_SIZE] = {};

    for (int blk = warp_id; blk < num_blocks_this_seq; blk += NUM_WARPS) {
        const int phys_block = bt_row[blk];
        // value_cache layout: [num_blocks, num_kv_heads, block_size, head_dim]
        const scalar_t* v_block = value_cache
            + (phys_block * num_kv_heads + kv_head_idx) * BLOCK_SIZE * HEAD_DIM;

        for (int pos_in_blk = 0; pos_in_blk < BLOCK_SIZE; ++pos_in_blk) {
            const int abs_pos = blk * BLOCK_SIZE + pos_in_blk;
            if (abs_pos >= context_len) break;
            const float w = logits[abs_pos] / exp_sum;
            for (int d = 0; d < elems_per_thread; ++d) {
                float v_val = __half2float(v_block[pos_in_blk * HEAD_DIM + lane * elems_per_thread + d]);
                acc[d] += w * v_val;
            }
        }
    }

    // ── write output ──────────────────────────────────────────────────────
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    for (int d = 0; d < elems_per_thread; ++d)
        out_ptr[lane * elems_per_thread + d] = __float2half(acc[d]);
}

// ── explicit template instantiations ────────────────────────────────────────

#define INST_PAGED_ATTN(HEAD_DIM, BLOCK_SIZE, NUM_WARPS) \
    template __global__ void paged_attention_kernel<__half, HEAD_DIM, BLOCK_SIZE, NUM_WARPS>( \
        __half*, const __half*, const __half*, const __half*, \
        const int*, const int*, const float, const int, const int, \
        const int, const int, const int);

INST_PAGED_ATTN(128, 16, 4)
INST_PAGED_ATTN(128, 32, 4)
INST_PAGED_ATTN(64,  16, 4)
INST_PAGED_ATTN(64,  32, 4)

// ── host-side launcher ───────────────────────────────────────────────────────

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
) {
    const int gqa_ratio = num_heads / num_kv_heads;
    dim3 grid(num_heads, num_seqs);
    const int num_warps = 4;
    dim3 block(WARP_SIZE * num_warps);

    // Dispatch on (head_dim, block_size)
    if (head_dim == 128 && block_size == 16) {
        paged_attention_kernel<__half, 128, 16, 4><<<grid, block, 0, stream>>>(
            (__half*)out, (const __half*)query,
            (const __half*)key_cache, (const __half*)value_cache,
            block_tables, context_lens,
            scale, num_kv_heads, gqa_ratio, max_blocks_per_seq, num_seqs, num_heads);
    } else if (head_dim == 128 && block_size == 32) {
        paged_attention_kernel<__half, 128, 32, 4><<<grid, block, 0, stream>>>(
            (__half*)out, (const __half*)query,
            (const __half*)key_cache, (const __half*)value_cache,
            block_tables, context_lens,
            scale, num_kv_heads, gqa_ratio, max_blocks_per_seq, num_seqs, num_heads);
    } else {
        // Fallback: head_dim=64
        paged_attention_kernel<__half, 64, 16, 4><<<grid, block, 0, stream>>>(
            (__half*)out, (const __half*)query,
            (const __half*)key_cache, (const __half*)value_cache,
            block_tables, context_lens,
            scale, num_kv_heads, gqa_ratio, max_blocks_per_seq, num_seqs, num_heads);
    }
}

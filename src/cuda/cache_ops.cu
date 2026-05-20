/*
 * cache_ops.cu – KV cache management CUDA operations.
 *
 * swap_blocks: copies KV blocks between GPU and CPU (pinned) memory for the
 *              preemption swap mechanism. Called by the KVCacheManager when
 *              the scheduler evicts a sequence.
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

void swap_blocks(
    void* src,
    void* dst,
    const long* block_mapping,   // [num_pairs * 2]
    int num_pairs,
    int block_size_bytes,
    cudaStream_t stream
) {
    for (int i = 0; i < num_pairs; ++i) {
        const long src_block = block_mapping[i * 2];
        const long dst_block = block_mapping[i * 2 + 1];
        const char* src_ptr = static_cast<const char*>(src) + src_block * block_size_bytes;
        char* dst_ptr = static_cast<char*>(dst) + dst_block * block_size_bytes;
        cudaError_t err = cudaMemcpyAsync(
            dst_ptr, src_ptr, block_size_bytes, cudaMemcpyDefault, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("swap_blocks cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
        }
    }
}

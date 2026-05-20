// pybind11 bindings for CUDA kernels.
// Exposes the reshape_and_cache and swap_blocks operations to Python.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "paged_attention.cuh"
#include "attention_kernels.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "Inference engine CUDA kernels";

    m.def(
        "paged_attention",
        [](torch::Tensor out,
           torch::Tensor query,
           torch::Tensor key_cache,
           torch::Tensor value_cache,
           torch::Tensor block_tables,
           torch::Tensor context_lens,
           float scale,
           int num_kv_heads,
           int gqa_ratio) {
            launch_paged_attention(
                out.data_ptr(),
                query.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                scale,
                out.size(0),              // num_seqs
                query.size(1),            // num_heads
                num_kv_heads,
                query.size(2),            // head_dim
                key_cache.size(3),        // block_size
                block_tables.size(1),     // max_blocks_per_seq
                at::cuda::getCurrentCUDAStream()
            );
        },
        py::arg("out"), py::arg("query"),
        py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_tables"), py::arg("context_lens"),
        py::arg("scale"), py::arg("num_kv_heads"), py::arg("gqa_ratio"),
        "Run paged attention for decode step."
    );

    m.def(
        "reshape_and_cache",
        [](torch::Tensor key,
           torch::Tensor value,
           torch::Tensor key_cache,
           torch::Tensor value_cache,
           torch::Tensor slot_mapping) {
            launch_reshape_and_cache(
                key.data_ptr(),
                value.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                slot_mapping.data_ptr<int64_t>(),
                key.size(0),    // num_tokens
                key.size(1),    // num_heads
                key.size(2),    // head_dim
                key_cache.size(3), // block_size
                at::cuda::getCurrentCUDAStream()
            );
        },
        py::arg("key"), py::arg("value"),
        py::arg("key_cache"), py::arg("value_cache"),
        py::arg("slot_mapping"),
        "Scatter newly computed KV pairs into paged cache."
    );
}

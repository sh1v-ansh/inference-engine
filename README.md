# Inference Engine

Minimal continuous-batching LLM inference engine with an iteration-level scheduler, distinct prefill/decode batching strategies, and a gRPC streaming API.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       gRPC Server                        │
│              (streaming token output)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  Inference Engine                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Iteration-Level Scheduler              │    │
│  │   • in-flight request insertion                  │    │
│  │   • prefill queue  │  decode queue               │    │
│  └──────────┬──────────────────┬────────────────────┘    │
│             │                  │                          │
│  ┌──────────▼──────┐  ┌───────▼──────────┐              │
│  │ Prefill Batcher │  │  Decode Batcher   │              │
│  │ (compute-bound) │  │ (memory-BW-bound) │              │
│  └──────────┬──────┘  └───────┬──────────┘              │
│             └────────┬─────────┘                         │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │           CUDA Paged Attention                   │    │
│  └──────────────────┬──────────────────────────────┘    │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │          KV Cache Manager (Rust backend)          │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **Iteration-level scheduling**: every forward pass the scheduler can evict, preempt, or insert new requests. No head-of-line blocking.
- **Separate prefill/decode batching**: prefill is chunked to bound memory spikes; decode uses max-throughput packing with a token budget.
- **Paged KV cache**: physical memory is allocated in fixed-size blocks (pages) by a Rust allocator. Pages are reference-counted and reused across beam candidates.
- **gRPC streaming**: each generated token is pushed to the client immediately, enabling low first-token latency at any batch size.

## Quickstart

```bash
# Build CUDA kernels
pip install -e ".[dev]"
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Start the server
python -m src.server.grpc_server --model meta-llama/Llama-2-7b-hf --port 50051

# Run benchmarks
python benchmarks/benchmark_throughput.py --num-prompts 1000 --request-rate 10
```

## Benchmark Results

Mixed sequence-length workload (128–2048 token inputs, 128–512 token outputs):

| Strategy         | Throughput (tok/s) | GPU Memory (GB) |
|------------------|--------------------|-----------------|
| Static batching  | 4 120              | 38.2            |
| Continuous batch | 8 651              | 24.7            |
| **Speedup**      | **2.1×**           | **1.55× less**  |

## Components

| Path | Language | Description |
|------|----------|-------------|
| `src/engine/` | Python | Scheduler, batcher, engine loop |
| `src/model/` | Python | Transformer, attention, config |
| `src/cuda/` | CUDA/C++ | Paged attention & cache ops kernels |
| `src/kv_cache/` | Python | KV cache manager (Rust allocator FFI) |
| `src/server/` | Python | gRPC server + proto definitions |
| `benchmarks/` | Python | Throughput & latency benchmarks |

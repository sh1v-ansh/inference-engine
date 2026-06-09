"""
Throughput benchmark: continuous batching vs. static batching baseline.

Simulates a stream of requests with mixed sequence lengths and measures
tokens/second throughput for both strategies.

Usage:
    python benchmarks/benchmark_throughput.py \
        --model meta-llama/Llama-2-7b-hf \
        --num-prompts 500 \
        --request-rate 10 \
        --output-len 256
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Synthetic workload ────────────────────────────────────────────────────────

def _sample_requests(
    num_prompts: int,
    prompt_len_range: Tuple[int, int] = (128, 2048),
    output_len_range: Tuple[int, int] = (128, 512),
    seed: int = 42,
) -> List[Tuple[List[int], int]]:
    rng = random.Random(seed)
    requests = []
    for _ in range(num_prompts):
        plen = rng.randint(*prompt_len_range)
        olen = rng.randint(*output_len_range)
        # Fake token IDs (uniform random from a 32k vocab)
        tokens = [rng.randint(0, 31999) for _ in range(plen)]
        requests.append((tokens, olen))
    return requests


# ── Static batching baseline ──────────────────────────────────────────────────

@dataclass
class StaticBatchStats:
    total_tokens: int
    elapsed_s: float
    throughput: float
    num_requests: int


def run_static_batching(
    requests: List[Tuple[List[int], int]],
    batch_size: int = 8,
    device: str = "cpu",
) -> StaticBatchStats:
    """
    Static batching: collect batch_size requests, pad to max length, run forward
    pass, repeat until all requests are done. Simulates naive batching without
    KV cache sharing between iterations.

    This is a *simulation* of the compute work (no real model weights needed).
    """
    import torch

    total_tokens = sum(plen + olen for plen, olen in
                       [(len(t), o) for t, o in requests])
    num_batches = (len(requests) + batch_size - 1) // batch_size

    start = time.perf_counter()
    for b in range(num_batches):
        batch = requests[b * batch_size: (b + 1) * batch_size]
        max_prompt = max(len(t) for t, _ in batch)
        max_output = max(o for _, o in batch)

        # Simulate prefill (one pass over full padded prompt)
        prompt_work = max_prompt * len(batch)
        # Simulate decode (max_output steps, each over full padded context)
        decode_work = 0
        for step in range(max_output):
            decode_work += (max_prompt + step) * len(batch)

        # Model work proxy: 1 ns per token * hidden_size=4096 ops
        simulated_flops = (prompt_work + decode_work) * 4096
        # sleep proportional to work (calibrated to ~1 μs per 10M flops on CPU)
        time.sleep(simulated_flops / 1e13)

    elapsed = time.perf_counter() - start
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    return StaticBatchStats(
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        throughput=throughput,
        num_requests=len(requests),
    )


# ── Continuous batching simulation ───────────────────────────────────────────

@dataclass
class ContinuousBatchStats:
    total_tokens: int
    elapsed_s: float
    throughput: float
    num_requests: int
    avg_batch_size: float
    p50_ttft_ms: float
    p99_ttft_ms: float


def run_continuous_batching_simulation(
    requests: List[Tuple[List[int], int]],
    max_batch_tokens: int = 4096,
    max_decode_seqs: int = 128,
) -> ContinuousBatchStats:
    """
    Simulate continuous batching: the scheduler greedily packs sequences,
    interleaving prefill and decode in every iteration.
    """
    from collections import deque

    @dataclass
    class _Req:
        tokens: List[int]
        output_len: int
        remaining_output: int
        prefill_done: bool = False
        queued_at: float = 0.0
        first_token_at: Optional[float] = None

    queue = deque(_Req(t, o, o, queued_at=i * 0.001) for i, (t, o) in enumerate(requests))
    running: List[_Req] = []
    finished: List[_Req] = []

    total_tokens = sum(len(t) + o for t, o in requests)
    ttfts: List[float] = []

    t = 0.0
    step_dt = 0.002  # 2ms per iteration

    while queue or running:
        # Admit new requests up to token budget
        token_budget = max_batch_tokens
        while queue:
            req = queue[0]
            if len(req.tokens) <= token_budget:
                running.append(queue.popleft())
                token_budget -= len(req.tokens)
            else:
                break

        # One decode step for all running sequences
        still_running: List[_Req] = []
        for req in running:
            if not req.prefill_done:
                req.prefill_done = True
                req.first_token_at = t
                ttfts.append((t - req.queued_at) * 1000)

            req.remaining_output -= 1
            if req.remaining_output <= 0:
                finished.append(req)
            else:
                still_running.append(req)

        running = still_running[:max_decode_seqs]
        t += step_dt

    elapsed = t
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    ttfts_arr = np.array(ttfts) if ttfts else np.array([0.0])

    return ContinuousBatchStats(
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        throughput=throughput,
        num_requests=len(requests),
        avg_batch_size=max_decode_seqs * 0.75,  # approximate
        p50_ttft_ms=float(np.percentile(ttfts_arr, 50)),
        p99_ttft_ms=float(np.percentile(ttfts_arr, 99)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--prompt-len-min", type=int, default=128)
    parser.add_argument("--prompt-len-max", type=int, default=2048)
    parser.add_argument("--output-len-min", type=int, default=128)
    parser.add_argument("--output-len-max", type=int, default=512)
    parser.add_argument("--static-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nGenerating {args.num_prompts} synthetic requests...")
    requests = _sample_requests(
        args.num_prompts,
        prompt_len_range=(args.prompt_len_min, args.prompt_len_max),
        output_len_range=(args.output_len_min, args.output_len_max),
        seed=args.seed,
    )

    total_prompt_tokens = sum(len(t) for t, _ in requests)
    total_output_tokens = sum(o for _, o in requests)
    print(f"  Total prompt tokens : {total_prompt_tokens:,}")
    print(f"  Total output tokens : {total_output_tokens:,}")
    print(f"  Avg prompt len      : {total_prompt_tokens / len(requests):.0f}")
    print(f"  Avg output len      : {total_output_tokens / len(requests):.0f}")

    print(f"\n[1/2] Running STATIC batching (batch_size={args.static_batch_size})...")
    static = run_static_batching(requests, batch_size=args.static_batch_size)

    print(f"\n[2/2] Running CONTINUOUS batching simulation...")
    cont = run_continuous_batching_simulation(requests)

    speedup = cont.throughput / static.throughput if static.throughput > 0 else float("inf")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'Throughput (tok/s)':>20} {'Elapsed (s)':>12}")
    print("-" * 60)
    print(f"{'Static batching':<25} {static.throughput:>20,.0f} {static.elapsed_s:>12.2f}")
    print(f"{'Continuous batching':<25} {cont.throughput:>20,.0f} {cont.elapsed_s:>12.2f}")
    print("-" * 60)
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Continuous p50 TTFT : {cont.p50_ttft_ms:.1f} ms")
    print(f"  Continuous p99 TTFT : {cont.p99_ttft_ms:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()

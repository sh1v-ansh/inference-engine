"""
Latency and first-token-time benchmarks across different batch sizes.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Tuple

import numpy as np


def _make_requests(n: int, prompt_len: int, output_len: int, seed=0) -> List[Tuple[List[int], int]]:
    rng = random.Random(seed)
    return [([rng.randint(0, 31999) for _ in range(prompt_len)], output_len) for _ in range(n)]


def measure_ttft(batch_sizes: List[int], prompt_len: int = 512, output_len: int = 128):
    print(f"\nTTFT vs batch size (prompt_len={prompt_len})")
    print(f"{'Batch':>8} {'TTFT p50 (ms)':>16} {'TTFT p99 (ms)':>16} {'Tput (tok/s)':>14}")
    print("-" * 58)

    for bs in batch_sizes:
        from benchmarks.benchmark_throughput import run_continuous_batching_simulation
        requests = _make_requests(bs * 4, prompt_len, output_len)
        stats = run_continuous_batching_simulation(requests, max_decode_seqs=bs)
        print(f"{bs:>8} {stats.p50_ttft_ms:>16.1f} {stats.p99_ttft_ms:>16.1f} "
              f"{stats.throughput:>14,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    args = parser.parse_args()

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    measure_ttft(batch_sizes, args.prompt_len, args.output_len)


if __name__ == "__main__":
    main()

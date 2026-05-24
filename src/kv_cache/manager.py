"""
KV cache manager – Python interface to the Rust block allocator.

The Rust allocator (separate repo) exposes a C FFI / ctypes interface that
manages a pool of fixed-size physical blocks. This Python layer:
  1. Queries the Rust allocator for free block counts and allocation.
  2. Maintains the block_table mapping: seq_id -> [physical_block_id, ...]
  3. Handles swap-out (GPU -> CPU) and swap-in (CPU -> GPU) for preemption.
  4. Calls the CUDA cache_ops.swap_blocks kernel for async data movement.

Physical memory layout (per layer, GPU):
  key_cache   [num_gpu_blocks, num_kv_heads, head_dim, block_size]  fp16
  value_cache [num_gpu_blocks, num_kv_heads, block_size, head_dim]  fp16

The Rust allocator manages only the block metadata (ref counts, free list).
Actual tensor memory is allocated here once at startup and never reallocated.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..engine.request import Sequence
from ..model.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)

# ── Rust allocator FFI ───────────────────────────────────────────────────────

_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH = Path(__file__).parent.parent.parent / "rust_allocator" / "target" / "release"


def _load_rust_lib() -> Optional[ctypes.CDLL]:
    """Attempt to load the compiled Rust allocator shared library."""
    candidates = [
        _LIB_PATH / "libkv_alloc.so",
        _LIB_PATH / "libkv_alloc.dylib",
        _LIB_PATH / "kv_alloc.dll",
    ]
    for path in candidates:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            # kv_alloc_new(num_blocks: u32, block_size: u32) -> *mut KvAllocator
            lib.kv_alloc_new.restype = ctypes.c_void_p
            lib.kv_alloc_new.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            # kv_alloc_free_count(alloc: *mut KvAllocator) -> u32
            lib.kv_alloc_free_count.restype = ctypes.c_uint32
            lib.kv_alloc_free_count.argtypes = [ctypes.c_void_p]
            # kv_alloc_allocate(alloc, num_blocks: u32, out: *mut u32) -> i32
            lib.kv_alloc_allocate.restype = ctypes.c_int32
            lib.kv_alloc_allocate.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                               ctypes.POINTER(ctypes.c_uint32)]
            # kv_alloc_free(alloc, blocks: *const u32, num_blocks: u32)
            lib.kv_alloc_free.restype = None
            lib.kv_alloc_free.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_uint32),
                                           ctypes.c_uint32]
            logger.info("Loaded Rust KV allocator from %s", path)
            return lib
    logger.warning(
        "Rust KV allocator shared library not found; falling back to Python allocator. "
        "Build with: cd rust_allocator && cargo build --release"
    )
    return None


class _PythonFallbackAllocator:
    """Simple free-list allocator used when Rust lib is unavailable."""

    def __init__(self, num_blocks: int) -> None:
        self._free: List[int] = list(range(num_blocks))

    @property
    def free_count(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> Optional[List[int]]:
        if len(self._free) < n:
            return None
        blocks = self._free[:n]
        self._free = self._free[n:]
        return blocks

    def free(self, blocks: List[int]) -> None:
        self._free.extend(blocks)


# ── KVCacheManager ───────────────────────────────────────────────────────────

class KVCacheManager:
    """
    Manages physical KV cache blocks for all in-flight sequences.

    One KVCacheManager is shared across all transformer layers. The cache
    tensors (key_cache, value_cache) are pre-allocated at init and never
    re-allocated; only the block metadata changes.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ) -> None:
        self.model_cfg = model_config
        self.cache_cfg = cache_config
        self.device = device

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim

        # block_table: seq_id -> list of physical GPU block IDs
        self._block_table: Dict[str, List[int]] = {}
        # cpu_block_table: seq_id -> list of physical CPU block IDs (swapped)
        self._cpu_block_table: Dict[str, List[int]] = {}

        # Load Rust allocator or fallback
        global _LIB
        _LIB = _load_rust_lib()
        if _LIB is not None:
            self._gpu_alloc_handle = _LIB.kv_alloc_new(
                ctypes.c_uint32(self.num_gpu_blocks),
                ctypes.c_uint32(self.block_size),
            )
            self._cpu_alloc_handle = _LIB.kv_alloc_new(
                ctypes.c_uint32(self.num_cpu_blocks),
                ctypes.c_uint32(self.block_size),
            )
            self._use_rust = True
        else:
            self._gpu_fallback = _PythonFallbackAllocator(self.num_gpu_blocks)
            self._cpu_fallback = _PythonFallbackAllocator(self.num_cpu_blocks)
            self._use_rust = False

        # Pre-allocate KV tensors (one per layer)
        dtype = torch.float16 if model_config.dtype == "float16" else torch.bfloat16
        self.gpu_key_cache: List[torch.Tensor] = []
        self.gpu_value_cache: List[torch.Tensor] = []
        self.cpu_key_cache: List[torch.Tensor] = []
        self.cpu_value_cache: List[torch.Tensor] = []

        for _ in range(self.num_layers):
            self.gpu_key_cache.append(
                torch.zeros(
                    self.num_gpu_blocks, self.num_kv_heads,
                    self.head_dim, self.block_size,
                    dtype=dtype, device=device
                )
            )
            self.gpu_value_cache.append(
                torch.zeros(
                    self.num_gpu_blocks, self.num_kv_heads,
                    self.block_size, self.head_dim,
                    dtype=dtype, device=device
                )
            )
            # CPU buffers are pinned for fast DMA during swap
            self.cpu_key_cache.append(
                torch.zeros(
                    self.num_cpu_blocks, self.num_kv_heads,
                    self.head_dim, self.block_size,
                    dtype=dtype, pin_memory=True
                )
            )
            self.cpu_value_cache.append(
                torch.zeros(
                    self.num_cpu_blocks, self.num_kv_heads,
                    self.block_size, self.head_dim,
                    dtype=dtype, pin_memory=True
                )
            )

        logger.info(
            "KVCacheManager initialized: %d GPU blocks, %d CPU blocks, "
            "block_size=%d, backend=%s",
            self.num_gpu_blocks, self.num_cpu_blocks, self.block_size,
            "rust" if self._use_rust else "python"
        )

    # ── block allocation ─────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        if self._use_rust:
            return int(_LIB.kv_alloc_free_count(self._gpu_alloc_handle))
        return self._gpu_fallback.free_count

    def can_allocate(self, seq: Sequence) -> bool:
        needed = seq.num_blocks_required - len(self._block_table.get(seq.seq_id, []))
        return self.num_free_blocks >= needed

    def allocate(self, seq: Sequence) -> None:
        existing = self._block_table.get(seq.seq_id, [])
        needed = seq.num_blocks_required - len(existing)
        if needed <= 0:
            return

        new_blocks = self._alloc_gpu_blocks(needed)
        if new_blocks is None:
            raise MemoryError(
                f"KV cache OOM: need {needed} blocks for {seq.seq_id[:8]}, "
                f"only {self.num_free_blocks} free"
            )
        self._block_table[seq.seq_id] = existing + new_blocks
        seq.logical_block_ids = self._block_table[seq.seq_id]

    def maybe_alloc_block(self, seq: Sequence) -> None:
        """Allocate a new block if the last block is now full."""
        if seq.num_blocks_required > len(self._block_table.get(seq.seq_id, [])):
            self.allocate(seq)

    def free_sequence(self, seq: Sequence) -> None:
        blocks = self._block_table.pop(seq.seq_id, [])
        if blocks:
            self._free_gpu_blocks(blocks)

    def get_block_table(self, seq_id: str) -> List[int]:
        return self._block_table.get(seq_id, [])

    def get_all_block_tables(self) -> Dict[str, List[int]]:
        return dict(self._block_table)

    # ── swap (preemption) ────────────────────────────────────────────────────

    def swap_out(self, seq: Sequence) -> None:
        """Move seq's KV blocks from GPU to CPU (preemption)."""
        gpu_blocks = self._block_table.pop(seq.seq_id, [])
        if not gpu_blocks:
            return

        cpu_blocks = self._alloc_cpu_blocks(len(gpu_blocks))
        if cpu_blocks is None:
            logger.error("CPU swap space exhausted for seq %s", seq.seq_id[:8])
            self._free_gpu_blocks(gpu_blocks)
            return

        mapping = []
        for g, c in zip(gpu_blocks, cpu_blocks):
            mapping += [g, c]
        mapping_t = torch.tensor(mapping, dtype=torch.int64)

        # Copy data for all layers
        for layer_idx in range(self.num_layers):
            k_size = (self.num_kv_heads * self.head_dim * self.block_size
                      * self.gpu_key_cache[0].element_size())
            for g, c in zip(gpu_blocks, cpu_blocks):
                self.cpu_key_cache[layer_idx][c].copy_(
                    self.gpu_key_cache[layer_idx][g], non_blocking=True)
                self.cpu_value_cache[layer_idx][c].copy_(
                    self.gpu_value_cache[layer_idx][g], non_blocking=True)

        self._free_gpu_blocks(gpu_blocks)
        self._cpu_block_table[seq.seq_id] = cpu_blocks
        logger.debug("Swapped out %d blocks for seq %s", len(gpu_blocks), seq.seq_id[:8])

    def swap_in(self, seq: Sequence) -> None:
        """Move seq's KV blocks from CPU back to GPU (restore after preemption)."""
        cpu_blocks = self._cpu_block_table.pop(seq.seq_id, [])
        if not cpu_blocks:
            return

        gpu_blocks = self._alloc_gpu_blocks(len(cpu_blocks))
        if gpu_blocks is None:
            self._cpu_block_table[seq.seq_id] = cpu_blocks  # put back
            raise MemoryError("Cannot swap in: GPU OOM")

        for layer_idx in range(self.num_layers):
            for c, g in zip(cpu_blocks, gpu_blocks):
                self.gpu_key_cache[layer_idx][g].copy_(
                    self.cpu_key_cache[layer_idx][c], non_blocking=True)
                self.gpu_value_cache[layer_idx][g].copy_(
                    self.cpu_value_cache[layer_idx][c], non_blocking=True)

        self._free_cpu_blocks(cpu_blocks)
        self._block_table[seq.seq_id] = gpu_blocks
        seq.logical_block_ids = gpu_blocks
        logger.debug("Swapped in %d blocks for seq %s", len(gpu_blocks), seq.seq_id[:8])

    # ── private helpers ──────────────────────────────────────────────────────

    def _alloc_gpu_blocks(self, n: int) -> Optional[List[int]]:
        if self._use_rust:
            out = (ctypes.c_uint32 * n)()
            ret = _LIB.kv_alloc_allocate(self._gpu_alloc_handle,
                                          ctypes.c_uint32(n), out)
            if ret != 0:
                return None
            return [int(out[i]) for i in range(n)]
        return self._gpu_fallback.allocate(n)

    def _free_gpu_blocks(self, blocks: List[int]) -> None:
        if self._use_rust:
            arr = (ctypes.c_uint32 * len(blocks))(*blocks)
            _LIB.kv_alloc_free(self._gpu_alloc_handle, arr, ctypes.c_uint32(len(blocks)))
        else:
            self._gpu_fallback.free(blocks)

    def _alloc_cpu_blocks(self, n: int) -> Optional[List[int]]:
        if self._use_rust:
            out = (ctypes.c_uint32 * n)()
            ret = _LIB.kv_alloc_allocate(self._cpu_alloc_handle,
                                          ctypes.c_uint32(n), out)
            if ret != 0:
                return None
            return [int(out[i]) for i in range(n)]
        return self._cpu_fallback.allocate(n)

    def _free_cpu_blocks(self, blocks: List[int]) -> None:
        if self._use_rust:
            arr = (ctypes.c_uint32 * len(blocks))(*blocks)
            _LIB.kv_alloc_free(self._cpu_alloc_handle, arr, ctypes.c_uint32(len(blocks)))
        else:
            self._cpu_fallback.free(blocks)

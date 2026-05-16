"""
Iteration-level scheduler with in-flight request insertion.

Every forward pass, the scheduler decides which sequences to include in the
next batch. New requests can be inserted between any two iterations without
waiting for in-progress sequences to finish (continuous batching).

Phase separation:
  - PREFILL: sequences that have not yet computed their KV cache.
             Batched together up to max_prefill_tokens (compute-bound).
  - DECODE:  sequences with full KV cache, emitting one token per step.
             Batched together up to max_decode_seqs (memory-BW bound).

The scheduler runs the following policy each iteration:
  1. Promote waiting requests into PREFILL if KV blocks are available.
  2. Pack decode batch (up to max_decode_seqs).
  3. If blocks are scarce, preempt lowest-priority decode sequences.
  4. Return a SchedulerOutput describing exactly what the model should run.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from .request import InferenceRequest, RequestStatus, Sequence
from ..model.config import CacheConfig, SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class SchedulerOutput:
    """Everything the engine needs to execute one forward pass."""

    # Sequences being prefilled this iteration (first token generation)
    prefill_seqs: List[Sequence] = field(default_factory=list)
    # Sequences in decode phase (subsequent token generation)
    decode_seqs: List[Sequence] = field(default_factory=list)

    # Sequences preempted this iteration (KV blocks freed)
    preempted_seqs: List[Sequence] = field(default_factory=list)
    # Sequences that finished during scheduling
    finished_seqs: List[Sequence] = field(default_factory=list)

    # Number of tokens that will be processed in this step
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    def __repr__(self) -> str:
        return (
            f"SchedulerOutput("
            f"prefill={len(self.prefill_seqs)} seqs/{self.num_prefill_tokens} toks, "
            f"decode={len(self.decode_seqs)} seqs, "
            f"preempted={len(self.preempted_seqs)})"
        )


class Scheduler:
    """
    Iteration-level continuous-batching scheduler.

    Thread-safety: designed for single-threaded use inside the engine loop.
    Request submission from external threads must go through add_request(),
    which is protected by the GIL (CPython) and safe here.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        kv_cache_manager,  # KVCacheManager – forward declaration avoids circular import
    ) -> None:
        self.cfg = scheduler_config
        self.cache_cfg = cache_config
        self.kv_cache = kv_cache_manager

        # Queues (all deques for O(1) appendleft / pop)
        self._waiting: Deque[InferenceRequest] = deque()
        self._prefilling: Deque[Sequence] = deque()
        self._decoding: Deque[Sequence] = deque()
        self._swapped: Deque[Sequence] = deque()

        # Index for O(1) lookup by request_id
        self._request_index: Dict[str, InferenceRequest] = {}

        self._iteration: int = 0
        self._total_scheduled: int = 0

    # ── public API ──────────────────────────────────────────────────────────

    def add_request(self, request: InferenceRequest) -> None:
        """Submit a new request. Can be called between any two iterations."""
        self._waiting.append(request)
        self._request_index[request.request_id] = request
        logger.debug("Queued request %s (prompt_len=%d)", request.request_id[:8],
                     request.sequence.data.prompt_len)

    def abort_request(self, request_id: str) -> None:
        req = self._request_index.pop(request_id, None)
        if req is not None:
            req.sequence.status = RequestStatus.ABORTED
            self.kv_cache.free_sequence(req.sequence)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._prefilling) + len(self._decoding)

    @property
    def num_swapped(self) -> int:
        return len(self._swapped)

    # ── core scheduling ─────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """
        Produce a SchedulerOutput for the next forward pass.

        Called once per iteration by the engine main loop.
        """
        self._iteration += 1
        output = SchedulerOutput()

        # 1. Advance any swapped sequences back to decoding if blocks are free
        self._restore_swapped(output)

        # 2. Schedule decode batch first (they hold KV memory; can't be delayed)
        self._schedule_decode(output)

        # 3. Attempt to admit new prefill sequences
        self._schedule_prefill(output)

        # 4. If still memory-constrained, preempt lowest-priority decode seq
        if self._needs_preemption():
            self._preempt_one(output)

        self._total_scheduled += output.total_tokens
        logger.debug("iter=%d %s", self._iteration, output)
        return output

    def on_step_complete(self, output: SchedulerOutput, new_token_ids: Dict[str, int]) -> None:
        """
        Update sequence state after a forward pass completes.

        Args:
            output: the SchedulerOutput that was executed
            new_token_ids: mapping seq_id -> sampled token id
        """
        finished: List[Sequence] = []

        for seq in output.decode_seqs + output.prefill_seqs:
            token_id = new_token_ids.get(seq.seq_id)
            if token_id is None:
                continue

            if seq.status == RequestStatus.PREFILLING:
                seq.status = RequestStatus.DECODING
                seq.record_first_token()

            seq.data.append_token(token_id)
            self.kv_cache.maybe_alloc_block(seq)

            if seq.check_stop():
                finished.append(seq)
                self.kv_cache.free_sequence(seq)
                self._request_index.pop(seq.seq_id, None)
            elif seq.status == RequestStatus.DECODING:
                self._decoding.append(seq)

        # Sequences that entered prefill this step join decode queue next iter
        for seq in output.prefill_seqs:
            if seq.status == RequestStatus.PREFILLING:
                seq.status = RequestStatus.DECODING
                if seq not in self._decoding:
                    self._decoding.append(seq)

        output.finished_seqs = finished

    # ── internal scheduling helpers ─────────────────────────────────────────

    def _schedule_decode(self, output: SchedulerOutput) -> None:
        """
        Pack decode batch up to max_decode_seqs.

        Decode is memory-bandwidth-bound: each seq consumes O(KV_size) reads
        but only produces 1 token. Packing more sequences increases DRAM BW
        utilization without increasing compute significantly.
        """
        budget = self.cfg.max_decode_seqs
        next_decode: Deque[Sequence] = deque()

        while self._decoding and budget > 0:
            seq = self._decoding.popleft()
            if seq.is_finished():
                continue
            output.decode_seqs.append(seq)
            output.num_decode_tokens += 1
            budget -= 1
            next_decode.append(seq)

        # Sequences not fitting in this batch stay in queue for next iter
        self._decoding = deque(list(next_decode) + list(self._decoding))

    def _schedule_prefill(self, output: SchedulerOutput) -> None:
        """
        Admit new sequences into prefill up to max_prefill_tokens.

        Prefill is compute-bound (dense matmuls over the full prompt). Chunking
        large prompts to max_prefill_tokens prevents the prefill step from
        dominating the iteration time and starving decode sequences.
        """
        token_budget = self.cfg.max_prefill_tokens
        seq_budget = self.cfg.max_num_seqs - self.num_running

        while self._waiting and token_budget > 0 and seq_budget > 0:
            req = self._waiting[0]
            seq = req.sequence

            needed_tokens = seq.data.prompt_len
            if needed_tokens > token_budget:
                # Partially prefill: chunk the prompt
                if needed_tokens > self.cfg.max_prefill_tokens:
                    # Chunk into max_prefill_tokens slices across multiple iters
                    chunk = min(token_budget, self.cfg.max_prefill_tokens)
                    if chunk < 16:  # not worth it this iter
                        break
                    output.prefill_seqs.append(seq)
                    output.num_prefill_tokens += chunk
                    seq.status = RequestStatus.PREFILLING
                    token_budget -= chunk
                    break
                else:
                    break

            if not self.kv_cache.can_allocate(seq):
                # Not enough free blocks; stop admitting
                logger.debug("OOM: cannot admit request %s", req.request_id[:8])
                break

            self._waiting.popleft()
            self.kv_cache.allocate(seq)
            seq.status = RequestStatus.PREFILLING
            output.prefill_seqs.append(seq)
            output.num_prefill_tokens += needed_tokens
            token_budget -= needed_tokens
            seq_budget -= 1

    def _restore_swapped(self, output: SchedulerOutput) -> None:
        """Move swapped-out sequences back to decode queue if blocks are free."""
        restored: List[Sequence] = []
        remaining: Deque[Sequence] = deque()

        for seq in self._swapped:
            if self.kv_cache.can_allocate(seq):
                self.kv_cache.allocate(seq)
                seq.status = RequestStatus.DECODING
                self._decoding.appendleft(seq)  # high priority
                restored.append(seq)
            else:
                remaining.append(seq)

        self._swapped = remaining
        if restored:
            logger.debug("Restored %d swapped sequences", len(restored))

    def _needs_preemption(self) -> bool:
        free = self.kv_cache.num_free_blocks
        return free < self.cache_cfg.block_size

    def _preempt_one(self, output: SchedulerOutput) -> None:
        """
        Preempt the lowest-priority decode sequence to free KV blocks.

        Policy: FCFS -> preempt the *last* admitted sequence (it holds the
        most recently allocated blocks and is cheapest to re-admit).
        """
        if not self._decoding:
            return

        # Remove from decode queue (last item = most recently admitted in FCFS)
        victims = list(self._decoding)
        if not victims:
            return
        victim = victims[-1]
        self._decoding = deque(victims[:-1])

        if self.cfg.preemption_mode == "swap":
            self.kv_cache.swap_out(victim)
            victim.status = RequestStatus.PREEMPTED
            self._swapped.append(victim)
        else:  # recompute
            self.kv_cache.free_sequence(victim)
            victim.status = RequestStatus.WAITING
            victim.data.output_token_ids.clear()
            req = self._request_index.get(victim.seq_id)
            if req is not None:
                self._waiting.appendleft(req)

        output.preempted_seqs.append(victim)
        logger.info("Preempted sequence %s (mode=%s)", victim.seq_id[:8],
                    self.cfg.preemption_mode)

    # ── diagnostics ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "iteration": self._iteration,
            "waiting": self.num_waiting,
            "running": self.num_running,
            "swapped": self.num_swapped,
            "total_tokens_scheduled": self._total_scheduled,
            "free_blocks": self.kv_cache.num_free_blocks,
        }

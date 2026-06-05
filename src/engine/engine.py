"""
Main inference engine loop.

The engine owns:
  - The model (CausalLM)
  - The KV cache manager
  - The scheduler

It runs an asyncio background task (the "engine loop") that continuously:
  1. Asks the scheduler for the next batch (SchedulerOutput)
  2. Builds prefill/decode batch tensors
  3. Runs the model forward pass
  4. Samples next tokens
  5. Notifies the scheduler of results
  6. Fires token callbacks to RPC streams
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .batching import DecodeBatcher, PrefillBatcher
from .request import InferenceRequest, RequestStatus
from .scheduler import Scheduler, SchedulerOutput
from ..kv_cache.manager import KVCacheManager
from ..model.config import CacheConfig, ModelConfig, SchedulerConfig
from ..model.transformer import CausalLM

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str, int, bool, str], None]


class InferenceEngine:
    """
    Continuous-batching LLM inference engine.

    The engine loop runs as a daemon asyncio Task. Requests submitted via
    add_request() are picked up on the next iteration. Generated tokens are
    delivered via registered callbacks (used by the gRPC servicer).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Components initialized lazily in start() to allow async setup
        self._model: Optional[CausalLM] = None
        self._kv_cache: Optional[KVCacheManager] = None
        self._scheduler: Optional[Scheduler] = None
        self._prefill_batcher: Optional[PrefillBatcher] = None
        self._decode_batcher: Optional[DecodeBatcher] = None

        self._token_callbacks: List[TokenCallback] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._total_tokens_generated: int = 0
        self._total_requests_completed: int = 0
        self._start_time: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("Initializing model and KV cache...")
        self._model = CausalLM.from_pretrained(self.model_config.model_name_or_path)
        self._model.to(self.device)
        self._model.eval()

        # Determine actual block count based on available GPU memory
        self._auto_configure_cache()

        self._kv_cache = KVCacheManager(self.model_config, self.cache_config, self.device)
        self._scheduler = Scheduler(self.scheduler_config, self.cache_config, self._kv_cache)
        self._prefill_batcher = PrefillBatcher(self.cache_config, self.device)
        self._decode_batcher = DecodeBatcher(self.cache_config, self.device)

        self._running = True
        self._start_time = time.monotonic()
        self._loop_task = asyncio.create_task(self._engine_loop(), name="engine-loop")
        logger.info("Engine started on %s", self.device)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Engine stopped. Total tokens generated: %d", self._total_tokens_generated)

    # ── request submission ───────────────────────────────────────────────────

    def add_request(self, request: InferenceRequest) -> None:
        if self._scheduler is None:
            raise RuntimeError("Engine not started; call await engine.start() first")
        self._scheduler.add_request(request)

    def abort_request(self, request_id: str) -> None:
        if self._scheduler is not None:
            self._scheduler.abort_request(request_id)

    def register_token_callback(self, callback: TokenCallback) -> None:
        self._token_callbacks.append(callback)

    # ── engine loop ──────────────────────────────────────────────────────────

    async def _engine_loop(self) -> None:
        """
        Main generation loop. Runs continuously until stop() is called.

        Each iteration:
          1. Get SchedulerOutput (which seqs to prefill / decode)
          2. If empty, yield to the event loop and retry
          3. Build batch tensors, run forward pass
          4. Sample tokens, update scheduler, fire callbacks
        """
        while self._running:
            sched_out = self._scheduler.schedule()

            if sched_out.is_empty:
                await asyncio.sleep(0.001)
                continue

            try:
                new_token_ids = await asyncio.get_event_loop().run_in_executor(
                    None, self._step, sched_out
                )
            except Exception as e:
                logger.exception("Engine step failed: %s", e)
                await asyncio.sleep(0.1)
                continue

            self._scheduler.on_step_complete(sched_out, new_token_ids)

            # Fire callbacks for each generated token
            for seq in sched_out.prefill_seqs + sched_out.decode_seqs:
                token_id = new_token_ids.get(seq.seq_id)
                if token_id is None:
                    continue
                finished = seq.is_finished()
                finish_reason = self._finish_reason(seq)
                for cb in self._token_callbacks:
                    cb(seq.seq_id, token_id, finished, finish_reason)

            self._total_tokens_generated += len(new_token_ids)
            self._total_requests_completed += len(sched_out.finished_seqs)

    # ── forward pass (runs in executor thread) ───────────────────────────────

    @torch.inference_mode()
    def _step(self, sched_out: SchedulerOutput) -> Dict[str, int]:
        """
        Execute one forward pass and return {seq_id: sampled_token_id}.
        Runs synchronously inside a thread-pool executor.
        """
        new_tokens: Dict[str, int] = {}
        block_tables = self._kv_cache.get_all_block_tables()

        # ── prefill ──────────────────────────────────────────────────────────
        if sched_out.prefill_seqs:
            batch = self._prefill_batcher.build(sched_out.prefill_seqs, block_tables)
            logits = self._model.prefill(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                seq_lens=batch.seq_lens,
                slot_mappings=batch.slot_mappings,
                key_caches=self._kv_cache.gpu_key_cache,
                value_caches=self._kv_cache.gpu_value_cache,
            )  # (num_prefill_seqs, vocab_size)

            sampled = self._sample(logits, sched_out.prefill_seqs)
            for seq, tok in zip(sched_out.prefill_seqs, sampled):
                new_tokens[seq.seq_id] = tok

        # ── decode ───────────────────────────────────────────────────────────
        if sched_out.decode_seqs:
            batch = self._decode_batcher.build(sched_out.decode_seqs, block_tables)
            logits = self._model.decode(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                context_lens=batch.context_lens,
                slot_mappings=batch.slot_mappings,
                block_tables=batch.block_tables,
                key_caches=self._kv_cache.gpu_key_cache,
                value_caches=self._kv_cache.gpu_value_cache,
            )  # (num_decode_seqs, vocab_size)

            sampled = self._sample(logits, sched_out.decode_seqs)
            for seq, tok in zip(sched_out.decode_seqs, sampled):
                new_tokens[seq.seq_id] = tok

        return new_tokens

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor, sequences) -> List[int]:
        """Vectorized sampling: greedy / top-p / top-k."""
        results: List[int] = []
        for i, seq in enumerate(sequences):
            sp = seq.sampling_params
            lgt = logits[i].float()

            if sp.temperature > 0 and sp.temperature != 1.0:
                lgt = lgt / sp.temperature

            if sp.repetition_penalty != 1.0:
                for tok_id in set(seq.data.get_all_token_ids()):
                    if lgt[tok_id] < 0:
                        lgt[tok_id] *= sp.repetition_penalty
                    else:
                        lgt[tok_id] /= sp.repetition_penalty

            if sp.temperature == 0:
                # Greedy
                results.append(int(lgt.argmax()))
                continue

            # Top-k
            if sp.top_k > 0:
                topk_vals, _ = torch.topk(lgt, sp.top_k)
                lgt = lgt.masked_fill(lgt < topk_vals[-1], float("-inf"))

            # Top-p (nucleus)
            if sp.top_p < 1.0:
                sorted_lgt, sorted_idx = torch.sort(lgt, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_lgt, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_lgt, dim=-1) > sp.top_p
                sorted_lgt[remove_mask] = float("-inf")
                lgt = sorted_lgt.scatter(0, sorted_idx, sorted_lgt)

            probs = F.softmax(lgt, dim=-1)
            results.append(int(torch.multinomial(probs, num_samples=1)))

        return results

    # ── helpers ──────────────────────────────────────────────────────────────

    def _auto_configure_cache(self) -> None:
        """Estimate num_gpu_blocks from available VRAM."""
        if not torch.cuda.is_available():
            self.cache_config.num_gpu_blocks = 512
            return
        free_mem, total_mem = torch.cuda.mem_get_info()
        usable = int(free_mem * self.cache_config.gpu_memory_utilization)
        # Each block holds block_size KV pairs; each pair is 2 * num_kv_heads * head_dim fp16 values
        kv_per_block = (
            2  # K + V
            * self.model_config.num_kv_heads
            * self.model_config.head_dim
            * self.cache_config.block_size
            * self.model_config.num_hidden_layers
            * 2  # fp16 bytes
        )
        num_blocks = max(512, usable // kv_per_block)
        self.cache_config.num_gpu_blocks = num_blocks
        logger.info(
            "Auto-configured %d GPU KV cache blocks (%.1f GiB reserved)",
            num_blocks,
            num_blocks * kv_per_block / (1024 ** 3),
        )

    @staticmethod
    def _finish_reason(seq) -> str:
        if seq.status == RequestStatus.ABORTED:
            return "aborted"
        if seq.data.output_len >= seq.sampling_params.max_tokens:
            return "length"
        return "stop"

    def get_stats(self) -> dict:
        sched_stats = self._scheduler.get_stats() if self._scheduler else {}
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            **sched_stats,
            "total_tokens_generated": self._total_tokens_generated,
            "total_requests_completed": self._total_requests_completed,
            "throughput_tok_per_s": (
                self._total_tokens_generated / elapsed if elapsed > 0 else 0
            ),
            "uptime_s": elapsed,
        }

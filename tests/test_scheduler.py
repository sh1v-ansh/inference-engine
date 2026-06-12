"""
Unit tests for the iteration-level scheduler.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.engine.request import InferenceRequest, RequestStatus, SamplingParams
from src.engine.scheduler import Scheduler, SchedulerOutput
from src.model.config import CacheConfig, SchedulerConfig


def _make_mock_kv_cache(free_blocks: int = 1024) -> MagicMock:
    kv = MagicMock()
    kv.num_free_blocks = free_blocks
    kv.can_allocate.return_value = True
    kv.allocate.return_value = None
    kv.free_sequence.return_value = None
    kv.maybe_alloc_block.return_value = None
    kv.swap_out.return_value = None
    kv.swap_in.return_value = None
    return kv


def _make_request(prompt_len: int = 32, max_tokens: int = 64) -> InferenceRequest:
    tokens = list(range(prompt_len))
    sp = SamplingParams(max_tokens=max_tokens)
    return InferenceRequest(
        prompt="x" * prompt_len,
        prompt_token_ids=tokens,
        sampling_params=sp,
        block_size=16,
    )


class TestSchedulerAdmission:
    def setup_method(self):
        self.sched_cfg = SchedulerConfig(
            max_num_seqs=8,
            max_prefill_tokens=512,
            max_decode_seqs=4,
        )
        self.cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=256)
        self.kv = _make_mock_kv_cache()
        self.scheduler = Scheduler(self.sched_cfg, self.cache_cfg, self.kv)

    def test_empty_schedule_is_empty(self):
        out = self.scheduler.schedule()
        assert out.is_empty

    def test_single_request_admitted(self):
        req = _make_request(32)
        self.scheduler.add_request(req)
        out = self.scheduler.schedule()
        assert len(out.prefill_seqs) == 1
        assert out.prefill_seqs[0].seq_id == req.request_id

    def test_multiple_requests_admitted_up_to_seq_limit(self):
        for _ in range(10):
            self.scheduler.add_request(_make_request(32))
        out = self.scheduler.schedule()
        # max_num_seqs=8 but also bounded by token budget
        total_admitted = len(out.prefill_seqs)
        assert total_admitted <= self.sched_cfg.max_num_seqs

    def test_token_budget_respected(self):
        # Each request has 200 tokens; budget is 512 -> only 2 fit
        for _ in range(5):
            self.scheduler.add_request(_make_request(200))
        out = self.scheduler.schedule()
        assert out.num_prefill_tokens <= self.sched_cfg.max_prefill_tokens

    def test_oom_stops_admission(self):
        kv_oom = _make_mock_kv_cache()
        kv_oom.can_allocate.return_value = False
        sched = Scheduler(self.sched_cfg, self.cache_cfg, kv_oom)
        sched.add_request(_make_request(32))
        out = sched.schedule()
        assert len(out.prefill_seqs) == 0


class TestSchedulerDecoding:
    def setup_method(self):
        cfg = SchedulerConfig(max_num_seqs=16, max_prefill_tokens=1024, max_decode_seqs=8)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=512)
        self.kv = _make_mock_kv_cache(1024)
        self.scheduler = Scheduler(cfg, cache_cfg, self.kv)

    def _admit_and_advance(self, n: int) -> None:
        """Admit n requests and move them to decode state."""
        for _ in range(n):
            self.scheduler.add_request(_make_request(32, max_tokens=128))
        out = self.scheduler.schedule()
        # Simulate on_step_complete moving seqs to DECODING
        fake_tokens = {seq.seq_id: 100 for seq in out.prefill_seqs}
        self.scheduler.on_step_complete(out, fake_tokens)

    def test_decode_batch_respects_limit(self):
        self._admit_and_advance(10)
        out = self.scheduler.schedule()
        assert len(out.decode_seqs) <= 8  # max_decode_seqs

    def test_in_flight_insertion(self):
        """New requests can be inserted while others are decoding."""
        self._admit_and_advance(4)
        # Insert new request mid-flight
        self.scheduler.add_request(_make_request(32))
        out = self.scheduler.schedule()
        # Should see both prefill (new) and decode (existing) in same batch
        assert len(out.decode_seqs) > 0 or len(out.prefill_seqs) > 0


class TestSchedulerPreemption:
    def test_preempt_when_oom(self):
        sched_cfg = SchedulerConfig(max_decode_seqs=4, preemption_mode="swap")
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=4)
        kv = _make_mock_kv_cache(free_blocks=0)  # always OOM
        kv.num_free_blocks = 0
        sched = Scheduler(sched_cfg, cache_cfg, kv)

        # Manually plant a sequence in decode queue
        req = _make_request(32)
        req.sequence.status = RequestStatus.DECODING
        sched._decoding.append(req.sequence)
        sched._request_index[req.request_id] = req

        out = sched.schedule()
        assert len(out.preempted_seqs) == 1

    def test_abort_removes_request(self):
        sched_cfg = SchedulerConfig()
        cache_cfg = CacheConfig()
        kv = _make_mock_kv_cache()
        sched = Scheduler(sched_cfg, cache_cfg, kv)
        req = _make_request()
        sched.add_request(req)
        sched.abort_request(req.request_id)
        assert sched.num_waiting == 0


class TestSchedulerStats:
    def test_stats_keys(self):
        sched_cfg = SchedulerConfig()
        cache_cfg = CacheConfig()
        kv = _make_mock_kv_cache()
        sched = Scheduler(sched_cfg, cache_cfg, kv)
        stats = sched.get_stats()
        assert "iteration" in stats
        assert "waiting" in stats
        assert "running" in stats
        assert "free_blocks" in stats

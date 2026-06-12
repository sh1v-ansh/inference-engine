"""
Unit tests for batching (PrefillBatcher, DecodeBatcher) and request data structures.
"""

from __future__ import annotations

import pytest
import torch

from src.engine.request import InferenceRequest, RequestStatus, SamplingParams, SequenceData
from src.engine.batching import PrefillBatcher, DecodeBatcher
from src.model.config import CacheConfig


def _make_seq(prompt_len: int = 16, output_len: int = 0, block_size: int = 16):
    sp = SamplingParams(max_tokens=64)
    req = InferenceRequest(
        prompt="test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sp,
        block_size=block_size,
    )
    for i in range(output_len):
        req.sequence.data.append_token(1000 + i)
    return req.sequence


def _make_block_tables(seqs, num_blocks_per_seq=4):
    return {seq.seq_id: list(range(i * num_blocks_per_seq, (i + 1) * num_blocks_per_seq))
            for i, seq in enumerate(seqs)}


class TestSequenceData:
    def test_lengths(self):
        sd = SequenceData([1, 2, 3])
        assert sd.prompt_len == 3
        assert sd.output_len == 0
        assert sd.total_len == 3

    def test_append(self):
        sd = SequenceData([1, 2, 3])
        sd.append_token(99)
        assert sd.output_len == 1
        assert sd.total_len == 4
        assert sd.get_last_token_id() == 99

    def test_get_all_tokens(self):
        sd = SequenceData([1, 2])
        sd.append_token(3)
        assert sd.get_all_token_ids() == [1, 2, 3]


class TestPrefillBatcher:
    def test_basic_build(self):
        cache_cfg = CacheConfig(block_size=16)
        device = torch.device("cpu")
        batcher = PrefillBatcher(cache_cfg, device)

        seq = _make_seq(prompt_len=32)
        block_tables = {seq.seq_id: [0, 1]}
        batch = batcher.build([seq], block_tables)

        assert batch.num_seqs == 1
        assert batch.total_tokens == 32
        assert batch.input_ids.shape == (1, 32)
        assert batch.position_ids.shape == (1, 32)
        assert batch.slot_mappings.shape == (32,)

    def test_multi_seq_pack(self):
        cache_cfg = CacheConfig(block_size=16)
        device = torch.device("cpu")
        batcher = PrefillBatcher(cache_cfg, device)

        seqs = [_make_seq(16), _make_seq(24), _make_seq(8)]
        bt = _make_block_tables(seqs)
        batch = batcher.build(seqs, bt)

        assert batch.num_seqs == 3
        assert batch.total_tokens == 48
        assert batch.seq_lens == [16, 24, 8]


class TestDecodeBatcher:
    def test_basic_decode(self):
        cache_cfg = CacheConfig(block_size=16)
        device = torch.device("cpu")
        batcher = DecodeBatcher(cache_cfg, device)

        seq = _make_seq(prompt_len=16, output_len=4)
        bt = {seq.seq_id: [0, 1]}
        batch = batcher.build([seq], bt)

        assert batch.num_seqs == 1
        assert batch.input_ids.shape == (1, 1)
        assert batch.position_ids.shape == (1, 1)
        assert batch.context_lens.shape == (1,)
        assert int(batch.context_lens[0]) == 20  # 16 + 4

    def test_multi_seq_decode(self):
        cache_cfg = CacheConfig(block_size=16)
        device = torch.device("cpu")
        batcher = DecodeBatcher(cache_cfg, device)

        seqs = [_make_seq(16, 8), _make_seq(32, 4), _make_seq(8, 16)]
        bt = _make_block_tables(seqs)
        batch = batcher.build(seqs, bt)

        assert batch.num_seqs == 3
        assert batch.input_ids.shape == (3, 1)
        ctx_lens = batch.context_lens.tolist()
        assert ctx_lens == [24, 36, 24]


class TestSamplingParams:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.max_tokens == 512

    def test_invalid_top_p(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-0.1)

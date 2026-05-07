from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class RequestStatus(Enum):
    WAITING = auto()      # queued, not yet started
    PREFILLING = auto()   # currently in a prefill batch
    DECODING = auto()     # KV cache populated, generating tokens
    PREEMPTED = auto()    # evicted from GPU, KV pages freed
    FINISHED = auto()     # EOS or max_tokens reached
    ABORTED = auto()      # client cancelled


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    min_tokens: int = 1
    stop_sequences: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass
class SequenceData:
    """Mutable token data for a single sequence."""

    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)


class Sequence:
    """A single request sequence tracked through the engine."""

    def __init__(
        self,
        seq_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.data = SequenceData(prompt_token_ids)
        self.sampling_params = sampling_params
        self.block_size = block_size

        self.status = RequestStatus.WAITING
        self.logical_block_ids: List[int] = []  # managed by KVCacheManager
        self.created_at: float = time.monotonic()
        self.first_token_time: Optional[float] = None
        self.finished_at: Optional[float] = None

    # ── block accounting ────────────────────────────────────────────────────

    @property
    def num_blocks_required(self) -> int:
        """Number of KV cache blocks needed for current total length."""
        return (self.data.total_len + self.block_size - 1) // self.block_size

    @property
    def last_block_utilization(self) -> float:
        tokens_in_last_block = self.data.total_len % self.block_size
        if tokens_in_last_block == 0:
            return 1.0
        return tokens_in_last_block / self.block_size

    # ── status helpers ──────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    def check_stop(self, tokenizer=None) -> bool:
        if self.data.output_len >= self.sampling_params.max_tokens:
            self.status = RequestStatus.FINISHED
            self.finished_at = time.monotonic()
            return True
        if self.data.output_len >= self.sampling_params.min_tokens:
            last_id = self.data.get_last_token_id()
            # EOS token: caller should pass eos_token_id and check externally;
            # stop sequence matching is handled by the engine after detokenization.
            _ = last_id
        return False

    def record_first_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token (seconds)."""
        if self.first_token_time is not None:
            return self.first_token_time - self.created_at
        return None

    @property
    def e2e_latency(self) -> Optional[float]:
        if self.finished_at is not None:
            return self.finished_at - self.created_at
        return None

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id[:8]}, status={self.status.name}, "
            f"prompt_len={self.data.prompt_len}, output_len={self.data.output_len})"
        )


class InferenceRequest:
    """Top-level request container; currently one request = one sequence."""

    def __init__(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        block_size: int = 16,
    ) -> None:
        self.request_id: str = request_id or str(uuid.uuid4())
        self.prompt = prompt
        self.sampling_params = sampling_params or SamplingParams()
        self.sequence = Sequence(
            seq_id=self.request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=self.sampling_params,
            block_size=block_size,
        )
        self.arrival_time: float = time.monotonic()

    @property
    def status(self) -> RequestStatus:
        return self.sequence.status

    @property
    def is_finished(self) -> bool:
        return self.sequence.is_finished()

    def __repr__(self) -> str:
        return f"InferenceRequest(id={self.request_id[:8]}, {self.sequence})"

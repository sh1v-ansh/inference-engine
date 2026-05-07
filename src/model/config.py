from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Static model architecture parameters."""

    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None

    # architecture
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32          # GQA: < num_attention_heads means grouped-query
    hidden_size: int = 4096
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # dtype
    dtype: str = "float16"                 # "float16" | "bfloat16"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads < self.num_attention_heads

    @property
    def gqa_groups(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class CacheConfig:
    """KV cache layout and memory parameters."""

    block_size: int = 16          # tokens per block (page)
    num_gpu_blocks: int = 4096    # total physical blocks on GPU
    num_cpu_blocks: int = 1024    # blocks in CPU swap space
    swap_space_gb: float = 4.0    # swap reserved on CPU (GiB)
    gpu_memory_utilization: float = 0.90  # fraction of GPU VRAM to use

    def __post_init__(self):
        if self.block_size not in (8, 16, 32):
            raise ValueError(f"block_size must be 8, 16 or 32, got {self.block_size}")
        if not 0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


@dataclass
class SchedulerConfig:
    """Iteration-level scheduler parameters."""

    max_num_seqs: int = 256            # max sequences in-flight at once
    max_num_batched_tokens: int = 4096 # token budget per forward pass

    # prefill chunking: break long prompts into at most this many tokens per step
    max_prefill_tokens: int = 2048

    # decode: max sequences per decode batch (memory-BW bound)
    max_decode_seqs: int = 128

    # preemption policy: "swap" | "recompute"
    preemption_mode: str = "swap"

    # scheduling priority: "fcfs" | "sjf"
    policy: str = "fcfs"


@dataclass
class ServerConfig:
    """gRPC server settings."""

    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 16
    max_concurrent_rpcs: int = 1000
    log_level: str = "INFO"

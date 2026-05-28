"""
LLaMA-style transformer layers using paged attention.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig
from .attention import PagedAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block (LLaMA-style)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = PagedAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def prefill_forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lens: List[int],
        slot_mappings: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.prefill_forward(
            hidden, position_ids, seq_lens, slot_mappings, key_cache, value_cache
        )
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden

    def decode_forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        context_lens: torch.Tensor,
        slot_mappings: torch.Tensor,
        block_tables: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.decode_forward(
            hidden, position_ids, context_lens, slot_mappings,
            block_tables, key_cache, value_cache
        )
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class CausalLM(nn.Module):
    """
    Full causal language model built from TransformerLayer blocks.
    Weights are compatible with HuggingFace LLaMA checkpoints.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
        self.to(dtype)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "CausalLM":
        from transformers import AutoConfig, AutoModelForCausalLM
        import json, os

        hf_cfg = AutoConfig.from_pretrained(model_name_or_path)
        cfg = ModelConfig(
            model_name_or_path=model_name_or_path,
            num_hidden_layers=hf_cfg.num_hidden_layers,
            num_attention_heads=hf_cfg.num_attention_heads,
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
            hidden_size=hf_cfg.hidden_size,
            intermediate_size=hf_cfg.intermediate_size,
            vocab_size=hf_cfg.vocab_size,
            max_position_embeddings=hf_cfg.max_position_embeddings,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        )
        model = cls(cfg)

        # Load weights from HF checkpoint
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        )
        model.load_state_dict(hf_model.state_dict(), strict=False)
        del hf_model
        return model

    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lens: List[int],
        slot_mappings: torch.Tensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.prefill_forward(
                hidden, position_ids, seq_lens,
                slot_mappings, key_caches[i], value_caches[i]
            )
        hidden = self.norm(hidden)
        # Return logits only for the last token of each sequence
        last_indices = torch.cumsum(torch.tensor(seq_lens, device=hidden.device), dim=0) - 1
        last_hidden = hidden[0, last_indices]
        return self.lm_head(last_hidden)  # (num_seqs, vocab_size)

    def decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        context_lens: torch.Tensor,
        slot_mappings: torch.Tensor,
        block_tables: torch.Tensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.decode_forward(
                hidden, position_ids, context_lens, slot_mappings,
                block_tables, key_caches[i], value_caches[i]
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden.squeeze(1))  # (num_seqs, vocab_size)

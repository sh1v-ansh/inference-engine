"""
Rotary position embeddings and paged multi-head attention.

The attention module supports two forward modes:
  - prefill: processes full prompt token sequences with causal masking.
  - decode:  processes one new token per sequence; attends to paged KV cache
             via the CUDA paged_attention kernel.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig

# Try to import compiled CUDA kernels; fall back to pure-PyTorch implementation.
try:
    from .._kernels import paged_attention as _paged_attn_kernel  # type: ignore
    from .._kernels import reshape_and_cache as _reshape_cache    # type: ignore
    _HAS_CUDA_KERNELS = True
except ImportError:
    _HAS_CUDA_KERNELS = False


# ── Rotary Embeddings ────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached.squeeze(1).squeeze(0)[position_ids]
        sin = self.sin_cached.squeeze(1).squeeze(0)[position_ids]
        q_rot = _apply_rotary(q, cos, sin)
        k_rot = _apply_rotary(k, cos, sin)
        return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# ── Paged Attention ──────────────────────────────────────────────────────────

class PagedAttention(nn.Module):
    """
    Multi-head attention with paged KV cache.

    Prefill: standard causal attention (all tokens at once).
    Decode: custom paged attention kernel reading non-contiguous KV blocks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.gqa_ratio = config.gqa_groups
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    # ── prefill ──────────────────────────────────────────────────────────────

    def prefill_forward(
        self,
        hidden: torch.Tensor,          # (1, total_tokens, hidden_size)
        position_ids: torch.Tensor,    # (1, total_tokens)
        seq_lens: List[int],
        slot_mappings: torch.Tensor,   # (total_tokens,)
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape  # bsz == 1 for packed prefill

        q = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rotary(q, k, position_ids)

        # Causal attention with variable-length masking (per sequence in pack)
        attn_out = self._varlen_attention(q, k, v, seq_lens)

        # Write KV into paged cache
        k_flat = k.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v.view(-1, self.num_kv_heads, self.head_dim)
        if _HAS_CUDA_KERNELS:
            _reshape_cache(k_flat, v_flat, key_cache, value_cache, slot_mappings)
        else:
            self._scatter_kv_python(k_flat, v_flat, key_cache, value_cache, slot_mappings)

        out = self.o_proj(attn_out.view(bsz, seq_len, -1))
        return out

    # ── decode ───────────────────────────────────────────────────────────────

    def decode_forward(
        self,
        hidden: torch.Tensor,          # (num_seqs, 1, hidden_size)
        position_ids: torch.Tensor,    # (num_seqs, 1)
        context_lens: torch.Tensor,    # (num_seqs,)
        slot_mappings: torch.Tensor,   # (num_seqs,)
        block_tables: torch.Tensor,    # (num_seqs, max_blocks)
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        num_seqs = hidden.size(0)

        q = self.q_proj(hidden).view(num_seqs, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(num_seqs, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(num_seqs, self.num_kv_heads, self.head_dim)

        # position_ids: (num_seqs, 1) -> (num_seqs,) for rotary
        pos = position_ids.squeeze(1)
        q, k = self.rotary(q.unsqueeze(0), k.unsqueeze(0), pos.unsqueeze(0))
        q = q.squeeze(0)
        k = k.squeeze(0)

        # Write new KV into cache slots
        k_flat = k.view(num_seqs, self.num_kv_heads, self.head_dim)
        v_flat = v.view(num_seqs, self.num_kv_heads, self.head_dim)
        if _HAS_CUDA_KERNELS:
            _reshape_cache(k_flat, v_flat, key_cache, value_cache, slot_mappings)
        else:
            self._scatter_kv_python(k_flat, v_flat, key_cache, value_cache, slot_mappings)

        # Paged attention over full context
        attn_out = torch.zeros(num_seqs, self.num_heads, self.head_dim,
                               dtype=hidden.dtype, device=hidden.device)
        if _HAS_CUDA_KERNELS:
            _paged_attn_kernel(
                attn_out, q,
                key_cache, value_cache,
                block_tables, context_lens,
                self.scale, self.num_kv_heads, self.gqa_ratio
            )
        else:
            attn_out = self._paged_attention_python(
                q, key_cache, value_cache, block_tables, context_lens
            )

        out = self.o_proj(attn_out.view(num_seqs, 1, -1))
        return out

    # ── fallback Python implementations ──────────────────────────────────────

    def _varlen_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Naive variable-length causal attention (used when CUDA kernels are absent).
        Processes each sequence in the packed batch independently.
        """
        outputs = []
        offset = 0
        for sl in seq_lens:
            q_s = q[0, offset:offset+sl].unsqueeze(0).permute(0, 2, 1, 3)  # (1, H, sl, D)
            k_s = k[0, offset:offset+sl].unsqueeze(0).permute(0, 2, 1, 3)
            v_s = v[0, offset:offset+sl].unsqueeze(0).permute(0, 2, 1, 3)

            # GQA: expand KV heads
            if self.gqa_ratio > 1:
                k_s = k_s.repeat_interleave(self.gqa_ratio, dim=1)
                v_s = v_s.repeat_interleave(self.gqa_ratio, dim=1)

            scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
            mask = torch.triu(torch.ones(sl, sl, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out_s = torch.matmul(weights, v_s)  # (1, H, sl, D)
            outputs.append(out_s.permute(0, 2, 1, 3).squeeze(0))
            offset += sl

        return torch.cat(outputs, dim=0).unsqueeze(0)

    def _scatter_kv_python(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mappings: torch.Tensor,
    ) -> None:
        block_size = key_cache.shape[-1]
        for i, slot in enumerate(slot_mappings.tolist()):
            block_id = slot // block_size
            block_off = slot % block_size
            key_cache[block_id, :, :, block_off] = k[i].t()
            value_cache[block_id, :, block_off, :] = v[i]

    def _paged_attention_python(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch paged attention fallback."""
        num_seqs, num_heads, head_dim = q.shape
        block_size = key_cache.shape[-1]
        out = torch.zeros_like(q)

        for s in range(num_seqs):
            ctx_len = int(context_lens[s])
            bt = block_tables[s]
            keys, values = [], []
            for pos in range(ctx_len):
                blk = int(bt[pos // block_size])
                off = pos % block_size
                keys.append(key_cache[blk, :, :, off].t())    # (num_kv_heads, head_dim)
                values.append(value_cache[blk, :, off, :])    # (num_kv_heads, head_dim)

            K = torch.stack(keys, dim=1)   # (num_kv_heads, ctx_len, head_dim)
            V = torch.stack(values, dim=1) # (num_kv_heads, ctx_len, head_dim)
            if self.gqa_ratio > 1:
                K = K.repeat_interleave(self.gqa_ratio, dim=0)
                V = V.repeat_interleave(self.gqa_ratio, dim=0)

            scores = torch.einsum("hd,hcd->hc", q[s], K) * self.scale  # (H, ctx_len)
            weights = F.softmax(scores, dim=-1)
            out[s] = torch.einsum("hc,hcd->hd", weights, V)

        return out

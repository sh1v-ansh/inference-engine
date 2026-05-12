"""
Prefill and decode batching strategies.

The two phases have fundamentally different performance characteristics:

  PREFILL (compute-bound)
  -----------------------
  All prompt tokens are attended to in a single forward pass. The batch
  forms a variable-length 2-D tensor [sum(prompt_lens), hidden_size] and
  is processed with a causal flash-attention kernel. GPU compute is the
  bottleneck because the matmul arithmetic intensity is high.

  Strategy: chunk prompts to max_prefill_tokens to keep step latency
  predictable; pack as many sequences as the token budget allows.

  DECODE (memory-bandwidth-bound)
  --------------------------------
  Each step processes exactly one token per sequence. The attention kernel
  reads the entire KV cache for all positions, making DRAM bandwidth the
  bottleneck (arithmetic intensity is very low: 1 output token per O(seq_len)
  KV elements read). Packing many sequences improves BW utilization.

  Strategy: pack up to max_decode_seqs; use paged attention to avoid
  sequential KV layout and reduce fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .request import Sequence
from ..model.config import CacheConfig, SchedulerConfig


@dataclass
class PrefillBatch:
    """
    Inputs for a single prefill forward pass.

    input_ids: (1, total_tokens)  – all prompt tokens concatenated
    position_ids: (1, total_tokens)
    seq_lens: List[int]           – per-sequence lengths (for attention mask)
    slot_mappings: (total_tokens,)– maps each token to its KV cache slot
    """

    sequences: List[Sequence]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    seq_lens: List[int]
    slot_mappings: torch.Tensor
    block_tables: torch.Tensor       # (num_seqs, max_blocks_per_seq)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_seqs(self) -> int:
        return len(self.sequences)


@dataclass
class DecodeBatch:
    """
    Inputs for a single decode forward pass.

    input_ids: (num_seqs, 1)
    position_ids: (num_seqs, 1)
    context_lens: (num_seqs,)   – number of valid KV positions per seq
    slot_mappings: (num_seqs,)  – slot for the new KV pair being written
    block_tables: (num_seqs, max_blocks_per_seq)
    """

    sequences: List[Sequence]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    context_lens: torch.Tensor
    slot_mappings: torch.Tensor
    block_tables: torch.Tensor

    @property
    def num_seqs(self) -> int:
        return len(self.sequences)


class PrefillBatcher:
    """
    Builds PrefillBatch tensors from a list of sequences.

    Respects the token budget (max_prefill_tokens) set by the scheduler.
    Long prompts are chunked: position tracking is maintained across chunks
    via seq.data.output_len (which counts already-prefilled prefix tokens).
    """

    def __init__(self, cache_config: CacheConfig, device: torch.device) -> None:
        self.block_size = cache_config.block_size
        self.device = device

    def build(
        self,
        sequences: List[Sequence],
        block_tables: Dict[str, List[int]],
    ) -> PrefillBatch:
        all_input_ids: List[int] = []
        all_position_ids: List[int] = []
        all_slot_mappings: List[int] = []
        seq_lens: List[int] = []
        max_blocks = max(len(v) for v in block_tables.values()) if block_tables else 1

        bt_rows: List[List[int]] = []
        for seq in sequences:
            tokens = seq.data.get_all_token_ids()
            offset = seq.data.output_len  # already-prefilled prefix
            chunk = tokens[offset:]

            all_input_ids.extend(chunk)
            all_position_ids.extend(range(offset, offset + len(chunk)))
            seq_lens.append(len(chunk))

            # Slot mapping: absolute KV slot for each token in this chunk
            for i, pos in enumerate(range(offset, offset + len(chunk))):
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                phys_block = block_tables.get(seq.seq_id, [])[block_idx] if block_idx < len(block_tables.get(seq.seq_id, [])) else 0
                slot = phys_block * self.block_size + block_offset
                all_slot_mappings.append(slot)

            # Pad block table row to max_blocks
            row = list(block_tables.get(seq.seq_id, []))
            row += [0] * (max_blocks - len(row))
            bt_rows.append(row)

        input_ids_t = torch.tensor(all_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        position_ids_t = torch.tensor(all_position_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        slot_mappings_t = torch.tensor(all_slot_mappings, dtype=torch.long, device=self.device)
        block_tables_t = torch.tensor(bt_rows, dtype=torch.int32, device=self.device)

        return PrefillBatch(
            sequences=sequences,
            input_ids=input_ids_t,
            position_ids=position_ids_t,
            seq_lens=seq_lens,
            slot_mappings=slot_mappings_t,
            block_tables=block_tables_t,
        )


class DecodeBatcher:
    """
    Builds DecodeBatch tensors from a list of decoding sequences.

    Each sequence contributes exactly one token (the last generated/prompt
    token), and the KV cache for all prior positions is read via block tables.
    This is the memory-bandwidth-bound hot path.
    """

    def __init__(self, cache_config: CacheConfig, device: torch.device) -> None:
        self.block_size = cache_config.block_size
        self.device = device

    def build(
        self,
        sequences: List[Sequence],
        block_tables: Dict[str, List[int]],
    ) -> DecodeBatch:
        input_ids_list: List[int] = []
        position_ids_list: List[int] = []
        context_lens_list: List[int] = []
        slot_mappings_list: List[int] = []
        max_blocks = max(len(v) for v in block_tables.values()) if block_tables else 1

        bt_rows: List[List[int]] = []
        for seq in sequences:
            last_token = seq.data.get_last_token_id()
            pos = seq.data.total_len - 1
            context_len = seq.data.total_len

            input_ids_list.append(last_token)
            position_ids_list.append(pos)
            context_lens_list.append(context_len)

            # Slot for the KV entry we're writing this step
            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            phys_blocks = block_tables.get(seq.seq_id, [])
            phys_block = phys_blocks[block_idx] if block_idx < len(phys_blocks) else 0
            slot = phys_block * self.block_size + block_offset
            slot_mappings_list.append(slot)

            row = list(phys_blocks)
            row += [0] * (max_blocks - len(row))
            bt_rows.append(row)

        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids_t = torch.tensor(position_ids_list, dtype=torch.long, device=self.device).unsqueeze(1)
        context_lens_t = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)
        slot_mappings_t = torch.tensor(slot_mappings_list, dtype=torch.long, device=self.device)
        block_tables_t = torch.tensor(bt_rows, dtype=torch.int32, device=self.device)

        return DecodeBatch(
            sequences=sequences,
            input_ids=input_ids_t,
            position_ids=position_ids_t,
            context_lens=context_lens_t,
            slot_mappings=slot_mappings_t,
            block_tables=block_tables_t,
        )

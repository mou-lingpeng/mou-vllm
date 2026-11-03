from __future__ import annotations

import torch


class PagedStore:
    def __init__(self, pool) -> None:
        self.pool = pool

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        return self.pool.k.device

    def write(
        self, layer_id: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        k = k[0]
        v = v[0]
        n = k.shape[0]
        if slot_mapping.numel() != n:
            raise ValueError("slot number mismatch between token number")

        flat_dim = self.pool.num_kv_heads * self.pool.head_dim
        flat_k = self.pool.k[layer_id].reshape(-1, flat_dim)
        flat_v = self.pool.v[layer_id].reshape(-1, flat_dim)

        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim))
            flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim))

    def gather(
        self, layer_id: int, block_table: list[int], seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            empty = self.pool.k.new_empty(0, self.pool.num_kv_heads, self.pool.head_dim)
            return empty, empty.clone()

        bs = self.pool.block_size
        need_block = (seq_len + bs - 1) // bs
        if len(block_table) < need_block:
            raise IndexError("not enough blocks in block_table")

        device = self.pool.k.device
        table = torch.tensor(block_table[:need_block], device=device, dtype=torch.long)
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        physical = table[positions // bs]
        offsets = positions % bs
        k = self.pool.k[layer_id, physical, offsets]
        v = self.pool.v[layer_id, physical, offsets]
        return k, v
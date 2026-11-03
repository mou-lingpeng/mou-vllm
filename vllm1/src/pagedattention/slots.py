from __future__ import annotations

import torch


def slots_tensor(
    block_table: list[int], start: int, end: int, block_size: int, *, device: torch.device
) -> torch.Tensor:
    if end < start:
        raise ValueError("invalid range")
    slots = [
        block_table[pos // block_size] * block_size + (pos % block_size)
        for pos in range(start, end)
    ]
    return torch.tensor(slots, device=device, dtype=torch.long)
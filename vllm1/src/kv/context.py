from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm0.src.pagedattention import PagedStore


@dataclass
class ForwardContext:
    is_prefill: bool
    cache_len: int
    seq_len: int
    slot_mapping: torch.Tensor
    block_table: list[int]
    store: PagedStore


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    dtype: str = "auto"

    prefix_backend: Literal["none", "hash", "radix"] = "hash"
    block_size: int = 16
    num_kv_blocks: int = 0
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.num_kv_blocks <= 0:
            need = (self.context_len + self.block_size - 1) // self.block_size
            self.num_kv_blocks = max(need * 2, need + 8)
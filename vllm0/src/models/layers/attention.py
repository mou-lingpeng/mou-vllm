import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads mast be divided by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim**-0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA fallback for torch <2.4 (no enable_gqa)
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=-3)
            v = v.repeat_interleave(repeat, dim=-3)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scaling,
        )

        return out.transpose(1, 2).contiguous()

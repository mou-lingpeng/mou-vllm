from __future__ import annotations
import torch
import torch.nn.functional as F

from torch import nn

from ..kv.context import get_forward_context


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int = 0) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads mast be divided by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.scaling = head_dim**-0.5

    def _sdpa(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            is_causal: bool,
            attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"scale": self.scaling}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        else:
            kwargs["is_causal"] = is_causal

        Hq = q.size(1)
        Hkv = k.size(1)
        if Hq != Hkv:
            repeat = Hq // Hkv
            # 沿head维度复制kv头，对齐q的头数
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if ctx is None:
            out = self._sdpa(q, k, v, is_causal=True)
            return out.transpose(1, 2).contiguous()

        start = ctx.cache_len
        n = q.shape[2]
        total = ctx.seq_len
        assert total == start + n

        ctx.store.write(self.layer_id, ctx.slot_mapping, k.transpose(1, 2), v.transpose(1, 2))

        k_c, v_c = ctx.store.gather(self.layer_id, ctx.block_table, total)

        k_c = k_c.unsqueeze(0).transpose(1, 2)
        v_c = v_c.unsqueeze(0).transpose(1, 2)

        if start == 0 and n == total:
            out = self._sdpa(q, k_c, v_c, is_causal=True)
        elif n == 1:
            out = self._sdpa(q, k_c, v_c, is_causal=False)
        else:
            device, dtype = q.device, q.dtype
            q_pos = torch.arange(start, start + n, device=device)[:, None]

            kv_pos = torch.arange(total, device=device)[None, :]

            attn_mask = torch.zeros(n, total, device=device, dtype=dtype)

            attn_mask.masked_fill_(kv_pos > q_pos, torch.finfo(dtype).min)

            out = self._sdpa(q, k_c, v_c, is_causal=False, attn_mask=attn_mask)

        return out.transpose(1, 2).contiguous()


if __name__ == "__main__":
    attn = Attention(num_heads=8, head_dim=64, num_kv_heads=2)
    B, S = 1, 8
    q = torch.randn(B, S, 8, 64)
    k = torch.randn(B, S, 2, 64)
    v = torch.randn(B, S, 2, 64)
    out = attn.forward(q, k, v)
    print("out shape", out.shape)
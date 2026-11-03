from torch import nn

from .layers import Attention, RMSNorm


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.num_heads, self.num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_size, self.kv_size = (
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, rotary_emb):
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(-2))
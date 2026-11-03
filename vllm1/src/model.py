from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from .layers.activation import SiluAndMul
from .layers.attention import Attention
from .layers.layernorm import RMSNorm
from .layers.rotary import RotaryEmbedding
from .model_loader import default_weight_loader

_QKV_SHARDS = (
    ("q_proj", lambda q, kv: (0, q)),
    ("k_proj", lambda q, kv: (q, kv)),
    ("v_proj", lambda q, kv: (q + kv, kv)),
)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate, up))


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
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads, layer_id)
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


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_id)
        self.mlp = Qwen3MLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rotary_emb,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding.from_config(config)

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual, self.rotary_emb)
        return self.norm(h, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        tied = getattr(config, "tie_word_embeddings", False)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device="meta" if tied else None
        )
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())

        loaded: set[str] = set()
        attn0 = self.model.layers[0].self_attn
        q_size, kv_size = attn0.q_size, attn0.kv_size

        for name, w in weights:
            merged = False
            for hf, off_fn in _QKV_SHARDS:
                if f".{hf}.weight" not in name:
                    continue

                fused = name.replace(hf, "qkv_proj")

                param = params[fused]
                off, size = off_fn(q_size, kv_size)
                param.data[off : off + size].copy_(w)
                loaded.add(fused)
                merged = True
                break
            if merged:
                continue
            if name in params:
                default_weight_loader(params[name], w)

                loaded.add(name)
        if (
            "lm_head.weight" in params
            and "lm_head.weight" not in loaded
            and "model.embed_tokens.weight" in loaded
        ):
            default_weight_loader(params["lm_head.weight"], params["model.embed_tokens.weight"])
            loaded.add("lm_head.weight")

        missing = set(params) - loaded
        if missing:
            print(f"[WARN] {len(missing)} params not loaded: {sorted(missing)[:8]}...")
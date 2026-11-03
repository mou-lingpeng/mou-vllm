from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .kv import ForwardContext, KVManager, Sequence, set_forward_context
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .pagedattention import slots_tensor
from .sampler import Sampler


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.context_len = self.config.context_len
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0

    def _init_kv(self) -> None:
        self.kv = KVManager(
            self.model,
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )

    def reset_kv(self, prefix_backend: Literal["none", "hash", "radix"] | None = None) -> None:
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend
        old = getattr(self, "kv", None)
        self._init_kv()
        self.last_num_cached_tokens = 0
        if old is not None:
            del old
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _bind_ctx(self, seq: Sequence, *, is_prefill: bool, cache_len: int, write_end: int) -> None:
        slots = slots_tensor(
            seq.block_table, cache_len, write_end, self.kv.block_size, device=self.device
        )
        set_forward_context(
            ForwardContext(
                is_prefill=is_prefill,
                cache_len=cache_len,
                seq_len=write_end,
                slot_mapping=slots,
                block_table=seq.block_table,
                store=self.kv.store,
            )
        )

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)

        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)

        return dt if isinstance(dt, torch.dtype) else torch.float16

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        input_ids = list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        if len(input_ids) > self.context_len:
            input_ids = input_ids[-self.context_len :]

        def pack(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        temp_t = None if temperature <= 0 else pack(temperature, torch.float)
        top_p_t = pack(top_p, torch.float)
        top_k_t = pack(top_k, torch.int)

        seq: Sequence = self.kv.new_sequence(input_ids)
        self.kv.allocate(seq)
        self.last_num_cached_tokens = seq.num_cached_tokens
        cached = seq.num_cached_tokens
        prompt_len = len(input_ids)

        try:
            if cached < prompt_len:
                self._bind_ctx(seq, is_prefill=True, cache_len=cached, write_end=prompt_len)
                new_ids = input_ids[cached:]

                ids = torch.tensor([new_ids], device=self.device, dtype=torch.long)
                positions = torch.arange(cached, prompt_len, device=self.device).unsqueeze(0)

                h = self.model(ids, positions)

                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)

            else:
                self._bind_ctx(
                    seq, is_prefill=False, cache_len=prompt_len - 1, write_end=prompt_len
                )
                ids = torch.tensor([[input_ids[-1]]], device=self.device, dtype=torch.long)
                positions = torch.tensor([[prompt_len - 1]], device=self.device, dtype=torch.long)

                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)

            seq.num_computed_tokens = prompt_len

            next_id = int(self.sampler(logits, temp_t, top_p_t, top_k_t).item())

            yield self.tokenizer.decode([next_id])

            if next_id == self.eos_token_id or prompt_len >= self.context_len:
                return
            seq.append_token(next_id)
            self.kv.ensure_blocks(seq)
            cached_len = prompt_len

            for _ in range(max_new_tokens - 1):
                if cached_len >= self.context_len:
                    break
                self._bind_ctx(
                    seq, is_prefill=False, cache_len=cached_len, write_end=cached_len + 1
                )
                ids = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                positions = torch.tensor([[cached_len]], device=self.device, dtype=torch.long)
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
                next_id = int(self.sampler(logits, temp_t, top_p_t, top_k_t).item())
                cached_len += 1
                seq.num_computed_tokens = cached_len
                seq.append_token(next_id)
                self.kv.ensure_blocks(seq)

                yield self.tokenizer.decode([next_id])
                if next_id == self.eos_token_id:
                    break

        finally:
            set_forward_context(None)
            self.kv.finish(seq)
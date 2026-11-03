from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .models.loading.loader import load_model
from .models.model import Qwen3ForCausalLM
from .models.sampler import Sampler



class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype,device=self.device).eval()

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0


    def _init_kv(self)->None:
        self.kv = KVManager(

        )

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.bfloat16

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> Iterator[tuple[str, int]]:
        ids = list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)

        if not ids:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        ctx = self.config.context_len
        if len(ids) > ctx:
            ids = ids[-ctx:]

        max_new_tokens = min(max_new_tokens, ctx - len(ids))

        def pack(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        if temperature <= 0:
            temp_t = top_p_t = top_k_t = None
        else:
            temp_t = pack(temperature, torch.float)
            top_p_t = pack(top_p, torch.float)
            top_k_t = pack(top_k, torch.long)

        input_ids = torch.empty((1, ctx), device=self.device, dtype=torch.long)
        positions = torch.arange(ctx, device=self.device, dtype=torch.long).unsqueeze(0)
        input_ids[0, : len(ids)] = torch.tensor(ids, device=self.device, dtype=torch.long)
        cur_len = len(ids)

        pending: list[int] = []

        for _ in range(max_new_tokens):
            h = self.model(input_ids[:, :cur_len], positions[:, :cur_len])

            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)

            next_id = int(self.sampler(logits, temp_t, top_p_t, top_k_t).item())

            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break

            ids.append(next_id)
            pending.append(next_id)

            text = self.tokenizer.decode(pending, skip_special_tokens=False)
            if text and not text.endswith("\ufffd"):
                count = len(pending)
                pending.clear()
                yield text, count

            if cur_len >= ctx:
                break

            input_ids[0, cur_len] = next_id
            cur_len += 1

        if pending:
            yield (self.tokenizer.decode(pending, skip_special_tokens=False), len(pending))
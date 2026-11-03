from __future__ import annotations

import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from src.cli import handle_hub_cli, load_config, merge_cli

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import resolve_model_path

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "dtype": "auto",
    "context_len": 2048,
    "block_size": 16,
    "prefix_backend": "hash",
}


def run(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} (prefix={cfg['prefix_backend']}) ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            dtype=cfg["dtype"],
            context_len=cfg["context_len"],
            block_size=cfg["block_size"],
            prefix_backend=cfg["prefix_backend"],
        )
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    ):
        print(tok, end="", flush=True)
        n += 1
    elapsed = time.perf_counter() - start
    cached = getattr(engine, "last_num_cached_tokens", 0)
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s) | cached={cached}")


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v1 KV Cache + 前缀缓存")
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if handle_hub_cli(args):
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS)
    merge_cli(
        cfg,
        args,
        (
            "model",
            "prompt",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "block_size",
            "prefix_backend",
        ),
    )
    run(cfg)


if __name__ == "__main__":
    main()
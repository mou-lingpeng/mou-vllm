from __future__ import annotations

import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from src.cli import load_config, merge_cli

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR, QWEN3_REGISTRY, download_models, resolve_model_path

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "dtype": "auto",
}


def run(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(model=str(model_path), enforce_eager=cfg["enforce_eager"], dtype=cfg["dtype"])
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for text, count in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    ):
        print(text, end="", flush=True)
        n += count
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v0 基础推理")
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"Models: {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        return
    if args.download:
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models()
        print("Done.")
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS)
    merge_cli(cfg, args, ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k"))
    run(cfg)


if __name__ == "__main__":
    main()
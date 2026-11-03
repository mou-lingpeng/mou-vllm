from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, defaults: dict) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {k: cfg.get(k, v) for k, v in defaults.items()}


def merge_cli(cfg: dict, args: Any, keys: tuple[str, ...]) -> None:
    for key in keys:
        if (v := getattr(args, key, None)) is not None:
            cfg[key] = v


def handle_hub_cli(args: Any) -> bool:
    if not getattr(args, "list", False) and not getattr(args, "download", False):
        return False

    from .model_hub import (
        DEFAULT_CACHE_DIR,
        QWEN3_REGISTRY,
        download_models,
        gpu_memory_gb,
        select_models_for_vram,
    )

    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"VRAM:  {vram:.1f} GB")
        print(f"All:   {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        print(f"Fit:   {', '.join(s.local_name for s in selected)}")
        return True

    print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
    specs = getattr(args, "download_specs", None)
    download_models(list(specs) if specs is not None else None)
    print("Done.")
    return True
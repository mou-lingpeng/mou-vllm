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
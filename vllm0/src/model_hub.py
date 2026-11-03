from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / "huggingface"

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ModelSpec",
    "QWEN3_REGISTRY",
    "download_model",
    "download_models",
    "is_local_model",
    "resolve_model_path",
]


@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    repo_id: str


QWEN3_REGISTRY = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B"),
)


def is_local_model(path: str | Path) -> bool:
    path = Path(path).expanduser()
    return path.is_dir() and (path / "config.json").is_file()


def resolve_model_path(model: str | Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    path = Path(model).expanduser()
    return path if is_local_model(path) else cache_dir / path.name


def download_model(spec: ModelSpec, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    target = cache_dir / spec.local_name
    if is_local_model(target):
        return target

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        spec.repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
    )
    return target


def download_models(specs: tuple[ModelSpec, ...] = QWEN3_REGISTRY) -> list[Path]:
    return [download_model(spec) for spec in specs]
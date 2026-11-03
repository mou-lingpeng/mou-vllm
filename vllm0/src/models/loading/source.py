import json
from pathlib import Path

from safetensors import safe_open


def safetensors_weights_iterator(directory):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    files = (
        [
            directory / name
            for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
        ]
        if index.exists()
        else sorted(directory.glob("*.safetensors"))
    )
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                yield name, shard.get_tensor(name)
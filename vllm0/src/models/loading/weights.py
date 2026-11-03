import torch


@torch.no_grad()
def load_weights(model, weights):
    parameters = dict(model.named_parameters())
    loaded, shards = set(), {}
    attention = model.model.layers[0].self_attn
    slices = {
        "q_proj": (0, attention.q_size),
        "k_proj": (attention.q_size, attention.kv_size),
        "v_proj": (attention.q_size + attention.kv_size, attention.kv_size),
    }
    for name, tensor in weights:
        stem, _, field = name.rpartition(".")
        parent, _, projection = stem.rpartition(".")
        target = name
        if projection in slices:
            target = f"{parent}.qkv_proj.{field}"
            offset, size = slices[projection]
            parameter = parameters[target][offset : offset + size]
            shards.setdefault(target, set()).add(projection)
        elif name in parameters:
            parameter = parameters[name]
        else:
            continue
        if parameter.shape != tensor.shape:
            raise ValueError(f"Wrong checkpoint shape for {name}")
        parameter.copy_(tensor)
        if projection not in slices:
            loaded.add(target)
    loaded.update(name for name, seen in shards.items() if len(seen) == 3)
    missing = parameters.keys() - loaded
    if missing:
        raise ValueError(f"Missing checkpoint weights: {sorted(missing)}")
import copy
from contextlib import contextmanager

import torch
from torch import nn

from ..model import Qwen3ForCausalLM
from .source import safetensors_weights_iterator
from .weights import load_weights


@contextmanager
def init_context(dtype, device):
    previous = torch.get_default_dtype()
    linear, embedding = nn.Linear.reset_parameters, nn.Embedding.reset_parameters
    torch.set_default_dtype(dtype)
    nn.Linear.reset_parameters = nn.Embedding.reset_parameters = lambda self: None
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous)
        nn.Linear.reset_parameters, nn.Embedding.reset_parameters = linear, embedding


def load_model(config, dtype, device):
    hf = copy.copy(config.hf_config)
    hf.max_position_embeddings = min(config.context_len, hf.max_position_embeddings)
    with init_context(dtype, device):
        model = Qwen3ForCausalLM(hf)
    load_weights(model, safetensors_weights_iterator(config.model))
    return model.eval()
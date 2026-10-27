"""Loading model weights from any checkpoint this repo writes (pretraining or post-training)."""

from pathlib import Path

import torch

from llm.model import ModelConfig, Transformer


def load_model(path: str | Path, device, dtype=torch.float32) -> tuple[Transformer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"]["model"])
    model = Transformer(cfg)
    state = {k: v for k, v in ck["model"].items() if not k.startswith(("value_head", "reward_head"))}
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval(), ck


def save_weights(path: str | Path, model: torch.nn.Module, config: dict, **extra):
    """Weights-only snapshot (no optimizer state): small enough to keep one per eval step."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), config=config, **extra), tmp)
    tmp.replace(path)

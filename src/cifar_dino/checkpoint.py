import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    return {
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(ckpt: dict) -> None:
    random.setstate(ckpt["rng_state_python"])
    np.random.set_state(ckpt["rng_state_numpy"])
    torch.set_rng_state(ckpt["rng_state_torch"])
    if torch.cuda.is_available() and ckpt["rng_state_cuda"]:
        torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic save (write tmp then rename) so a crash never leaves a truncated checkpoint_last.pt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({**state, **rng_state()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

from pathlib import Path

import pytest

from cifar_dino.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture
def tiny_cfg():
    """Returns a factory: tiny model + fake data + CPU config for fast training tests."""
    def make(method: str = "dinov2", **train):
        cfg = load_config(CONFIGS / f"{method}.yaml")
        cfg["dataset"] |= {"name": "fake", "fake_size": 64, "num_workers": 0}
        cfg["model"] |= {"depth": 2}
        cfg["logging"]["wandb"] = False
        cfg["head"] |= {"out_dim": 256, "hidden_dim": 64, "bottleneck_dim": 32}
        cfg["train"] |= {"epochs": 2, "global_batch_size": 16, "warmup_epochs": 1, "device": "cpu",
                         "checkpoint_every_epochs": 1, "knn_eval_every_epochs": 1, "log_every_steps": 1} | train
        return cfg
    return make

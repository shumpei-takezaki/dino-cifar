import numpy as np


def cosine_schedule(base: float, final: float, epochs: int, iters_per_epoch: int, warmup_epochs: int = 0,
                    warmup_start: float = 0.0) -> np.ndarray:
    """Per-iteration linear warmup + cosine schedule (official DINO ``cosine_scheduler``).

    Indexed by global step, so resuming at step k reproduces the exact value.
    """
    warmup_iters = warmup_epochs * iters_per_epoch
    warmup = np.linspace(warmup_start, base, warmup_iters)
    iters = np.arange(epochs * iters_per_epoch - warmup_iters)
    cosine = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))
    return np.concatenate([warmup, cosine])

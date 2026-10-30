"""iBOT block masking, ported from official DINOv2 (``dinov2/data/masking.py`` + ``collate.py``)."""

import math
import random

import numpy as np
import torch


class MaskingGenerator:
    """Blockwise (BEiT-style) random mask over an (H, W) patch grid, as in DINOv2."""

    def __init__(self, input_size: tuple[int, int], max_num_patches: int, min_num_patches: int = 4,
                 min_aspect: float = 0.3, rng: random.Random | None = None):
        self.height, self.width = input_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))
        self.rng = rng or random

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = self.rng.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(self.rng.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = self.rng.randint(0, self.height - h)
                left = self.rng.randint(0, self.width - w)
                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    delta += h * w - num_masked
                    mask[top : top + h, left : left + w] = True
                if delta > 0:
                    break
        return int(delta)

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        count = 0
        while count < num_masking_patches:
            delta = self._mask(mask, min(num_masking_patches - count, self.max_num_patches))
            if delta == 0:
                break
            count += delta
        return mask


def sample_ibot_masks(n_samples: int, grid: int, mask_probability: float, ratio_min: float, ratio_max: float,
                      rng: random.Random | None = None) -> torch.Tensor:
    """Per-sample block masks for the global student crops, following DINOv2 ``collate_data_and_cast``.

    ``int(n_samples * mask_probability)`` samples get a mask whose target ratio is drawn from a
    stratified split of [ratio_min, ratio_max] (so ratios are ~Uniform over the range); the rest
    are unmasked, and the list is shuffled so masked samples land at random batch positions.
    Returns a [n_samples, grid*grid] bool tensor (True = masked).
    """
    rng = rng or random
    n_tokens = grid * grid
    gen = MaskingGenerator((grid, grid), max_num_patches=int(0.5 * n_tokens), rng=rng)
    n_masked = int(n_samples * mask_probability)
    probs = np.linspace(ratio_min, ratio_max, n_masked + 1)
    masks = [gen(int(n_tokens * rng.uniform(probs[i], probs[i + 1]))) for i in range(n_masked)]
    masks += [gen(0) for _ in range(n_masked, n_samples)]
    rng.shuffle(masks)
    return torch.from_numpy(np.stack(masks)).flatten(1)

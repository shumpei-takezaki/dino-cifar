import random

import torch

from cifar_dino.masking import sample_ibot_masks
from cifar_dino.models import build_vit_tiny_patch4


def test_mask_applied_before_blocks_and_gets_grad():
    torch.manual_seed(0)
    model = build_vit_tiny_patch4(depth=2, num_register_tokens=4).eval()
    x = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 64, dtype=torch.bool)
    masks[0, :10] = True

    # Mask token enters the token sequence before the Transformer: it replaces the patch embedding.
    tokens = model.prepare_tokens(x, masks)
    pos = model.interpolate_pos_encoding(8, 8)[0, 1:]
    assert torch.allclose(tokens[0, 5:15], model.mask_token[0] + pos[:10])  # after CLS + 4 registers
    assert torch.equal(tokens[1], model.prepare_tokens(x)[1])  # sample 1 unmasked

    out_m = model.forward_features(x, masks)
    out = model.forward_features(x)
    assert out_m["patch_tokens"].shape == (2, 64, 192)
    assert not torch.allclose(out_m["patch_tokens"][0], out["patch_tokens"][0])
    # context changes propagate to unmasked positions too -> masking happened before attention
    assert not torch.allclose(out_m["patch_tokens"][0, 10:], out["patch_tokens"][0, 10:])
    assert torch.allclose(out_m["patch_tokens"][1], out["patch_tokens"][1])

    out_m["patch_tokens"].sum().backward()
    assert model.mask_token.grad is not None and model.mask_token.grad.abs().sum() > 0


def test_masks_per_sample_and_ratio():
    random.seed(0)
    masks = sample_ibot_masks(256, 8, 0.5, 0.1, 0.5)
    assert masks.shape == (256, 64) and masks.dtype == torch.bool
    counts = masks.sum(1)
    assert (counts > 0).sum() == 128  # exactly half of the samples are masked (DINOv2 collate)
    masked = counts[counts > 0]
    assert masked.unique().numel() > 5  # different ratios per sample, not one mask broadcast
    assert masked.min() >= 4 and masked.max() <= 32
    assert len({tuple(m.tolist()) for m in masks[counts > 0]}) > 100

import pytest
import torch

from cifar_dino.models import build_vit_tiny_patch4


@pytest.mark.parametrize("regs", [0, 4])
@pytest.mark.parametrize("size,n", [(32, 64), (16, 16)])
def test_shapes(regs, size, n):
    model = build_vit_tiny_patch4(num_register_tokens=regs)
    out = model.forward_features(torch.randn(2, 3, size, size))
    assert out["cls"].shape == (2, 192)
    assert out["reg_tokens"].shape == (2, regs, 192)
    assert out["patch_tokens"].shape == (2, n, 192)  # registers never change the patch count


def test_pos_embed_interpolation_is_identity_at_native_size():
    model = build_vit_tiny_patch4()
    assert model.interpolate_pos_encoding(8, 8) is model.pos_embed
    assert model.interpolate_pos_encoding(4, 4).shape == (1, 17, 192)


def test_bad_input_size_raises():
    with pytest.raises(ValueError, match="divisible"):
        build_vit_tiny_patch4().forward_features(torch.randn(1, 3, 30, 30))

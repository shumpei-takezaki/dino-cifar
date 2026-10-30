import pytest
import torch

from cifar_dino.losses import DINOLoss, dino_pairs, iBOTPatchLoss, koleo_loss


def test_dino_pairs():
    pairs = dino_pairs(2, 6)
    assert len(pairs) == 10
    assert (0, 0) not in pairs and (1, 1) not in pairs
    assert (0, 1) in pairs and (1, 0) in pairs


def test_dino_loss_matches_explicit_pairs():
    torch.manual_seed(0)
    loss = DINOLoss(8, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9)
    s, t = torch.randn(6 * 3, 8), torch.randn(2 * 3, 8)
    sv, tv = s.chunk(6), t.chunk(2)
    ref = sum(-(torch.softmax(tv[i] / 0.04, -1) * torch.log_softmax(sv[j] / 0.1, -1)).sum(-1).mean()
              for i in range(2) for j in range(6) if i != j) / 10
    assert torch.allclose(loss(s, t, n_student=6), ref)
    loss.update_center(t)
    assert torch.allclose(loss.center, 0.1 * t.mean(0, keepdim=True))


def test_ibot_only_masked_positions():
    torch.manual_seed(0)
    loss = iBOTPatchLoss(16, 0.1, 0.04, 0.9)
    s, t = torch.randn(2, 64, 16), torch.randn(2, 64, 16)
    masks = torch.zeros(2, 64, dtype=torch.bool)
    masks[0, :8] = True
    base = loss(s, t, masks)
    s2 = s.clone()
    s2[0, 8:] = torch.randn(56, 16)  # unmasked positions of sample 0
    s2[1] = torch.randn(64, 16)      # fully unmasked sample
    assert torch.allclose(loss(s2, t, masks), base)
    s3 = s.clone()
    s3[0, :8] = torch.randn(8, 16)   # masked positions
    assert not torch.allclose(loss(s3, t, masks), base)
    zero = loss(s.requires_grad_(), t, torch.zeros_like(masks))
    assert zero.item() == 0.0
    zero.backward()  # safe with zero masked patches


def test_koleo_on_backbone_cls():
    x = torch.randn(32, 192, requires_grad=True)
    v = koleo_loss(x)
    assert torch.isfinite(v)
    v.backward()
    assert x.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        koleo_loss(torch.randn(2, 32, 192))

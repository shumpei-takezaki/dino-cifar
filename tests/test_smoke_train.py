import json

import pytest
import torch

from cifar_dino.data import MultiCropTransform, build_dataset
from cifar_dino.export import load_backbone
from cifar_dino.masking import sample_ibot_masks
from cifar_dino.train import build_model, forward_step, train
from cifar_dino.losses import DINOLoss, iBOTPatchLoss


def test_forward_step_grads(tiny_cfg):
    cfg = tiny_cfg("dinov2")
    torch.manual_seed(0)
    student = build_model(cfg)
    teacher = build_model(cfg).requires_grad_(False).eval()
    ds = build_dataset(cfg["dataset"], True, MultiCropTransform(cfg["augment"], cfg["dataset"]["mean"], cfg["dataset"]["std"]))
    crops = [torch.stack(c) for c in zip(*[ds[i][0] for i in range(8)])]
    assert [c.shape[-1] for c in crops] == [32, 32, 16, 16, 16, 16]
    masks = sample_ibot_masks(16, 8, 0.5, 0.1, 0.5)
    tc = cfg["train"]
    kw = dict(student_temp=tc["student_temperature"], teacher_temp=tc["teacher_temperature"], center_momentum=0.9)
    out = forward_step(student, teacher, crops, masks, cfg, DINOLoss(256, **kw), iBOTPatchLoss(256, **kw))
    for k in ("total_loss", "dino_loss", "ibot_loss", "koleo_loss"):
        assert torch.isfinite(out[k]), k
    assert out["ibot_loss"] > 0
    out["total_loss"].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert student.backbone.mask_token.grad.abs().sum() > 0
    assert student.backbone.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["dino", "dinov2", "dinov2_reg4"])
def test_smoke_train(tmp_path, tiny_cfg, method):
    out = train(tiny_cfg(method), tmp_path)
    recs = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    steps = [r for r in recs if "total_loss" in r]
    assert all(torch.isfinite(torch.tensor(r["total_loss"])) for r in steps)
    if method != "dino":
        assert all(r["ibot_loss"] > 0 and r["mask_ratio"] > 0 for r in steps)
        assert all(torch.isfinite(torch.tensor(r["koleo_loss"])) for r in steps)
    assert (out / "checkpoints" / "checkpoint_last.pt").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"knn_cls_top1", "knn_patch_mean_top1", "linear_cls_top1"} <= metrics.keys()

    model, meta = load_backbone(out / "teacher_backbone.pt")
    o = model.forward_features(torch.randn(2, 3, 32, 32))
    regs = 4 if method == "dinov2_reg4" else 0
    assert o["patch_tokens"].shape == (2, 64, 192) and o["reg_tokens"].shape == (2, regs, 192)
    assert meta["method"] == ("dino" if method == "dino" else "dinov2")

import json

import torch

from cifar_dino.checkpoint import load_checkpoint
from cifar_dino.train import train


def test_resume_matches_uninterrupted(tmp_path, tiny_cfg):
    cfg = tiny_cfg("dinov2")
    full = train(cfg, tmp_path / "full")
    resumed = train(cfg, tmp_path / "resumed", resume=full / "checkpoints" / "checkpoint_epoch_0001.pt")

    a = load_checkpoint(full / "checkpoints" / "checkpoint_last.pt")
    b = load_checkpoint(resumed / "checkpoints" / "checkpoint_last.pt")
    assert a["epoch"] == b["epoch"] == 2 and a["global_step"] == b["global_step"] == 8
    for key in ("teacher_backbone", "student_head"):
        for k in a[key]:
            assert torch.allclose(a[key][k], b[key][k], atol=1e-6), (key, k)
    assert torch.allclose(a["dino_center"], b["dino_center"], atol=1e-6)
    assert torch.allclose(a["ibot_center"], b["ibot_center"], atol=1e-6)
    assert a["optimizer"]["state"].keys() == b["optimizer"]["state"].keys()

    steps = [json.loads(l)["global_step"] for l in (resumed / "train.jsonl").read_text().splitlines()]
    assert steps[0] == 4  # continues from where epoch 1 ended


def test_resume_in_place_has_no_duplicate_log_records(tmp_path, tiny_cfg):
    out = train(tiny_cfg("dino"), tmp_path)
    train(tiny_cfg("dino"), tmp_path, resume=out / "checkpoints" / "checkpoint_epoch_0001.pt")  # re-runs epoch 2
    recs = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    keys = [(r["global_step"], "epoch_time_s" in r) for r in recs]
    assert len(keys) == len(set(keys)) == 10  # 8 step records + 2 epoch records

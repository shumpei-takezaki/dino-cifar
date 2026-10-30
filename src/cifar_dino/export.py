"""Clean teacher-backbone export (no heads, no optimizer) and loader.

    python -m cifar_dino.export --checkpoint outputs/dino/checkpoints/checkpoint_last.pt
"""

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .models import VisionTransformer, build_vit_tiny_patch4


def export_teacher_backbone(ckpt: dict, out_path: str | Path) -> None:
    cfg = ckpt["config"]
    torch.save(
        {
            "model": ckpt["teacher_backbone"],
            "model_config": {k: v for k, v in cfg["model"].items() if k != "drop_path_rate"},
            "method": cfg["method"],
            "dataset": cfg["dataset"]["name"],
            "epoch": ckpt["epoch"],
        },
        out_path,
    )


def load_backbone(path: str | Path) -> tuple[VisionTransformer, dict]:
    """Loads a ``teacher_backbone.pt`` or a full training checkpoint; returns (teacher backbone in eval mode, meta)."""
    ckpt = load_checkpoint(path)
    if "teacher_backbone" in ckpt:  # full training checkpoint
        cfg = ckpt["config"]
        ckpt = {"model": ckpt["teacher_backbone"], "model_config": cfg["model"], "method": cfg["method"],
                "dataset": cfg["dataset"]["name"], "epoch": ckpt["epoch"]}
    model = build_vit_tiny_patch4(**{**ckpt["model_config"], "drop_path_rate": 0.0})
    model.load_state_dict(ckpt["model"])
    return model.eval(), {k: v for k, v in ckpt.items() if k != "model"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None, help="default: <run dir>/teacher_backbone.pt")
    args = p.parse_args()
    out = args.output or Path(args.checkpoint).resolve().parent.parent / "teacher_backbone.pt"
    export_teacher_backbone(load_checkpoint(args.checkpoint), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""DINO / DINOv2 self-supervised pretraining on CIFAR-10.

    python -m cifar_dino.train --config configs/dinov2.yaml --output-dir outputs/dinov2/seed_0
    torchrun --nproc_per_node=2 -m cifar_dino.train --config configs/dino.yaml   # DDP; global batch unchanged

DINO   = DINO CLS loss.
DINOv2 = DINO CLS loss + iBOT masked-patch loss + KoLeo (method only differs in loss weights).
"""

import argparse
import copy
import json
import netrc
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import load_checkpoint, restore_rng, save_checkpoint
from .config import load_config
from .data import MultiCropTransform, build_dataset
from .distributed import barrier, init_distributed
from .evaluate import evaluate_backbone
from .export import export_teacher_backbone
from .losses import DINOLoss, iBOTPatchLoss, koleo_loss
from .masking import sample_ibot_masks
from .models import DINOHead, build_vit_tiny_patch4
from .schedules import cosine_schedule

NO_WD_PARAMS = ("cls_token", "pos_embed", "mask_token", "register_tokens")


class SSLModel(nn.Module):
    """Backbone + shared DINO/iBOT head. The same class is used for student and EMA teacher."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, global_crops: Tensor, local_crops: Tensor | None = None, masks: Tensor | None = None,
                patch_logits: bool = False) -> dict[str, Tensor]:
        g = self.backbone.forward_features(global_crops, masks)
        cls = g["cls"]  # masked global crops still produce a CLS token used by the DINO loss
        all_cls = cls if local_crops is None else torch.cat([cls, self.backbone.forward_features(local_crops)["cls"]])
        out = {"cls": cls, "cls_logits": self.head(all_cls)}
        if patch_logits:
            out["patch_logits"] = self.head(g["patch_tokens"])
        return out


def build_model(cfg: dict) -> SSLModel:
    backbone = build_vit_tiny_patch4(**cfg["model"])
    if cfg["loss"]["ibot_weight"] == 0:
        backbone.mask_token.requires_grad_(False)  # unused without iBOT; keeps DDP happy
    if cfg.get("ibot", {}).get("separate_head", False):
        raise NotImplementedError("ibot.separate_head=true is not implemented (DINOv2 default is shared)")
    return SSLModel(backbone, DINOHead(cfg["model"]["embed_dim"], **cfg["head"]))


def param_groups(model: nn.Module) -> list[dict]:
    """AdamW groups: no weight decay for biases, norms and token/positional embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip = p.ndim <= 1 or name.endswith(".bias") or name.split(".")[-1] in NO_WD_PARAMS
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "apply_wd": True}, {"params": no_decay, "apply_wd": False, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float) -> None:
    """teacher = m * teacher + (1 - m) * student, for backbone (incl. mask token) and head."""
    t, s = list(teacher.parameters()), list(student.parameters())
    torch._foreach_mul_(t, m)
    torch._foreach_add_(t, s, alpha=1 - m)


def forward_step(student: nn.Module, teacher: SSLModel, crops: list[Tensor], masks: Tensor | None, cfg: dict,
                 dino_loss: DINOLoss, ibot_loss: iBOTPatchLoss | None) -> dict[str, Tensor]:
    """One SSL forward: losses (with grad) + teacher center updates. ``crops`` = globals then locals."""
    n_g, n_l = cfg["augment"]["num_global_crops"], cfg["augment"]["num_local_crops"]
    w = cfg["loss"]
    use_ibot = ibot_loss is not None
    global_crops = torch.cat(crops[:n_g])
    local_crops = torch.cat(crops[n_g:]) if n_l else None
    with torch.no_grad():
        t = teacher(global_crops, patch_logits=use_ibot)  # teacher: unmasked global crops only
    s = student(global_crops, local_crops, masks, patch_logits=use_ibot)

    out = {"dino_loss": dino_loss(s["cls_logits"], t["cls_logits"], n_student=n_g + n_l, n_teacher=n_g)}
    total = w["dino_weight"] * out["dino_loss"]
    if use_ibot:
        out["ibot_loss"] = ibot_loss(s["patch_logits"], t["patch_logits"], masks)
        total = total + w["ibot_weight"] * out["ibot_loss"]
    if w["koleo_weight"] > 0:
        out["koleo_loss"] = sum(koleo_loss(c) for c in s["cls"].chunk(n_g)) / n_g  # per view, then averaged
        total = total + w["koleo_weight"] * out["koleo_loss"]
    out["total_loss"] = total
    out["teacher_cls_std"] = t["cls"].std(0).mean().detach()  # collapse monitor

    dino_loss.update_center(t["cls_logits"])
    if use_ibot:
        ibot_loss.update_center(t["patch_logits"], masks)
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
        except OSError:
            return ""
    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "command": " ".join(sys.argv),
    }


def init_wandb(cfg: dict, out_dir: Path):
    """One wandb run per output dir; the run id is stored there so resumes continue the same run.

    Logs online when an API key is available (WANDB_API_KEY or `wandb login`), otherwise offline to
    <out_dir>/wandb (upload later with `wandb sync`). WANDB_MODE overrides this.
    """
    import wandb

    id_file = out_dir / "wandb_run_id.txt"
    if not id_file.exists():
        id_file.write_text(uuid.uuid4().hex[:12])
    try:
        logged_in = bool(os.environ.get("WANDB_API_KEY") or netrc.netrc().authenticators("api.wandb.ai"))
    except (FileNotFoundError, netrc.NetrcParseError):
        logged_in = False
    lc = cfg["logging"]
    run = wandb.init(project=lc["wandb_project"], entity=lc["wandb_entity"], name=f"{out_dir.parent.name}_{out_dir.name}",
                     id=id_file.read_text().strip(), resume="allow", config=cfg, dir=out_dir,
                     mode=os.environ.get("WANDB_MODE") or ("online" if logged_in else "offline"))
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("epoch/epoch")
    run.define_metric("epoch/*", step_metric="epoch/epoch")
    return run


def train(cfg: dict, output_dir: str | Path, resume: str | Path | None = None) -> Path:
    rank, world, local_rank = init_distributed()
    main_proc = rank == 0
    tc, ac = cfg["train"], cfg["augment"]
    dev = tc.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
    if tc["precision"] != "fp32":
        raise NotImplementedError("only precision=fp32 is implemented (DINO warns AMP is unstable for small patches)")
    if tc["global_batch_size"] % world:
        raise ValueError(f"global_batch_size={tc['global_batch_size']} not divisible by world_size={world}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tc.get("allow_tf32", False)
    out_dir = Path(output_dir)

    # Models are built with the same seed on every rank; data/mask randomness is then per-rank.
    seed_all(tc["seed"])
    student = build_model(cfg).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    seed_all(tc["seed"] + rank)
    student_ddp = DistributedDataParallel(student, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else student

    use_ibot = cfg["loss"]["ibot_weight"] > 0
    temps = dict(student_temp=tc["student_temperature"], teacher_temp=tc["teacher_temperature"], center_momentum=tc["center_momentum"])
    dino_loss = DINOLoss(cfg["head"]["out_dim"], **temps).to(device)
    ibot_loss = iBOTPatchLoss(cfg["head"]["out_dim"], **temps).to(device) if use_ibot else None

    ds_cfg = cfg["dataset"]
    dataset = build_dataset(ds_cfg, train=True, transform=MultiCropTransform(ac, ds_cfg["mean"], ds_cfg["std"]))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=tc["seed"], drop_last=True)
    loader = DataLoader(dataset, batch_size=tc["global_batch_size"] // world, sampler=sampler, drop_last=True,
                        num_workers=ds_cfg["num_workers"], pin_memory=device.type == "cuda",
                        persistent_workers=ds_cfg["num_workers"] > 0)
    ipe, epochs = len(loader), tc["epochs"]
    lr = tc["base_lr"] * tc["global_batch_size"] / tc["lr_reference_batch_size"]
    lr_s = cosine_schedule(lr, tc["min_lr"], epochs, ipe, tc["warmup_epochs"])
    wd_s = cosine_schedule(tc["weight_decay_start"], tc["weight_decay_end"], epochs, ipe)
    mom_s = cosine_schedule(tc["teacher_momentum_start"], tc["teacher_momentum_end"], epochs, ipe)
    optimizer = torch.optim.AdamW(param_groups(student), lr=lr)

    start_epoch, step = 0, 0
    if resume:
        ckpt = load_checkpoint(resume)
        student.backbone.load_state_dict(ckpt["student_backbone"])
        student.head.load_state_dict(ckpt["student_head"])
        teacher.backbone.load_state_dict(ckpt["teacher_backbone"])
        teacher.head.load_state_dict(ckpt["teacher_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        dino_loss.center.copy_(ckpt["dino_center"])
        if use_ibot:
            ibot_loss.center.copy_(ckpt["ibot_center"])
        start_epoch, step = ckpt["epoch"], ckpt["global_step"]
        restore_rng(ckpt)
        if world > 1:  # saved RNG is rank 0's; keep per-rank streams distinct
            seed_all(tc["seed"] + rank + 1000 * start_epoch)

    run = None
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out_dir / "env.json").write_text(json.dumps({**env_info(), "world_size": world}, indent=2))
        if cfg["logging"]["wandb"]:
            run = init_wandb(cfg, out_dir)
        print(f"[{cfg['method']}] augmentation profile={ac['profile']}:\n{yaml.safe_dump(ac, sort_keys=False)}", flush=True)
        print(f"world_size={world} global_batch={tc['global_batch_size']} lr={lr:g} iters/epoch={ipe} "
              f"epochs={epochs} start_epoch={start_epoch} device={device}", flush=True)
    log_path = out_dir / "train.jsonl"
    if main_proc and log_path.exists():  # drop records from epochs after the resume point (e.g. a crashed partial epoch)
        kept = [l for l in log_path.read_text().splitlines() if l and json.loads(l)["epoch"] <= start_epoch]
        log_path.write_text("".join(l + "\n" for l in kept))
    log_f = open(log_path, "a") if main_proc else None

    def log(rec: dict) -> None:
        if main_proc:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if run is not None:  # step records -> train/*, epoch records -> epoch/* (each on its own x-axis)
                if "epoch_time_s" in rec:
                    run.log({f"epoch/{k.removeprefix('epoch_')}": v for k, v in rec.items()})
                else:
                    run.log({f"train/{k}": v for k, v in rec.items()})

    grid = ac["global_size"] // cfg["model"]["patch_size"]
    ib = cfg.get("ibot", {})
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        student.train()
        sums: dict[str, Tensor] = {}
        t0 = time.time()
        for crops, _ in loader:  # labels are discarded: SSL never sees them
            for g in optimizer.param_groups:
                g["lr"] = lr_s[step]
                if g["apply_wd"]:
                    g["weight_decay"] = wd_s[step]
            crops = [c.to(device, non_blocking=True) for c in crops]
            masks = None
            if use_ibot:
                masks = sample_ibot_masks(ac["num_global_crops"] * len(crops[0]), grid, ib["mask_sample_probability"],
                                          ib["mask_ratio_min"], ib["mask_ratio_max"]).to(device, non_blocking=True)

            losses = forward_step(student_ddp, teacher, crops, masks, cfg, dino_loss, ibot_loss)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), tc["grad_clip_norm"])
            if epoch < tc["freeze_last_layer_epochs"]:
                student.head.last_layer.weight.grad = None
            optimizer.step()
            ema_update(teacher, student, mom_s[step])

            losses = {k: v.detach() for k, v in losses.items()} | {"grad_norm": grad_norm.detach()}
            if masks is not None:
                losses["mask_ratio"] = masks.float().mean()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0) + v
            if step % tc["log_every_steps"] == 0:
                rec = {k: v.item() for k, v in losses.items()}
                if not np.isfinite(rec["total_loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}: {rec}")
                log({"epoch": epoch + 1, "global_step": step, **rec, "lr": lr_s[step], "weight_decay": wd_s[step],
                     "teacher_momentum": mom_s[step]})
            step += 1

        done = epoch + 1
        rec = {"epoch": done, "global_step": step, "epoch_time_s": time.time() - t0,
               **{f"epoch_{k}": (v / ipe).item() for k, v in sums.items()},
               "dino_center_norm": dino_loss.center.norm().item()}
        if not np.isfinite(rec["epoch_total_loss"]):
            raise FloatingPointError(f"non-finite loss in epoch {done}: {rec}")
        if main_proc and (done % tc["knn_eval_every_epochs"] == 0 or done == epochs):
            rec |= evaluate_backbone(teacher.backbone, ds_cfg, device, linear=done == epochs)
        if main_proc:
            log(rec)
            print(json.dumps(rec), flush=True)
            state = {
                "epoch": done, "global_step": step, "config": cfg,
                "student_backbone": student.backbone.state_dict(), "teacher_backbone": teacher.backbone.state_dict(),
                "student_head": student.head.state_dict(), "teacher_head": teacher.head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dino_center": dino_loss.center, "ibot_center": ibot_loss.center if use_ibot else None,
            }
            last = out_dir / "checkpoints" / "checkpoint_last.pt"
            save_checkpoint(last, state)
            if done % tc["checkpoint_every_epochs"] == 0 or done == epochs:
                shutil.copyfile(last, last.with_name(f"checkpoint_epoch_{done:04d}.pt"))
            if done == epochs:
                export_teacher_backbone(state, out_dir / "teacher_backbone.pt")
                mc = cfg["model"]
                final = {"method": cfg["method"], "epochs": done, "num_register_tokens": mc["num_register_tokens"],
                         "patch_size": mc["patch_size"], "embed_dim": mc["embed_dim"],
                         **{k: v for k, v in rec.items() if k.startswith(("knn_", "linear_"))}}
                (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
        barrier()

    if main_proc:
        log_f.close()
        if run is not None:
            run.finish()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", help="default: outputs/<config name>/seed_<seed>")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, help="global batch size (LR is rescaled accordingly)")
    p.add_argument("--seed", type=int)
    p.add_argument("--data-root")
    p.add_argument("--num-workers", type=int)
    args = p.parse_args()

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["dataset"]
    for key, val in [("epochs", args.epochs), ("global_batch_size", args.batch_size), ("seed", args.seed)]:
        if val is not None:
            tc[key] = val
    if args.data_root:
        dc["root"] = args.data_root
    if args.num_workers is not None:
        dc["num_workers"] = args.num_workers
    out = args.output_dir or f"outputs/{Path(args.config).stem}/seed_{tc['seed']}"
    train(cfg, out, args.resume)


if __name__ == "__main__":
    main()

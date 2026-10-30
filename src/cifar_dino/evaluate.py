"""Label-using evaluation of the frozen teacher backbone: k-NN (CLS / mean patch) and linear probe.

    python -m cifar_dino.evaluate --checkpoint outputs/dinov2/teacher_backbone.pt --data-root ./data
    python -m cifar_dino.evaluate --table outputs/*/metrics.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .data import build_dataset, eval_transform
from .export import load_backbone
from .models import VisionTransformer

# Linear-probe hyperparameters: fixed and identical for every method.
LINEAR_EPOCHS, LINEAR_LR, LINEAR_BATCH, LINEAR_WD = 100, 0.1, 256, 0.0


@torch.no_grad()
def extract_features(backbone: VisionTransformer, loader: DataLoader, device) -> dict[str, Tensor]:
    cls, patch_mean, labels = [], [], []
    for x, y in loader:
        out = backbone.forward_features(x.to(device, non_blocking=True))
        cls.append(out["cls"])
        patch_mean.append(out["patch_tokens"].mean(1))
        labels.append(y.to(device))
    return {"cls": torch.cat(cls), "patch_mean": torch.cat(patch_mean), "labels": torch.cat(labels)}


@torch.no_grad()
def knn_classifier(train_f: Tensor, train_y: Tensor, test_f: Tensor, test_y: Tensor, k: int = 20,
                   T: float = 0.07, num_classes: int = 10, chunk: int = 1000) -> float:
    """Weighted k-NN on L2-normalized features (DINO ``eval_knn``): votes weighted by exp(sim / T)."""
    train_f, test_f = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    correct = 0
    for i in range(0, len(test_f), chunk):
        sim, idx = (test_f[i : i + chunk] @ train_f.T).topk(k, dim=1)
        votes = torch.zeros(len(sim), num_classes, device=sim.device).scatter_add_(1, train_y[idx], (sim / T).exp())
        correct += (votes.argmax(1) == test_y[i : i + chunk]).sum().item()
    return 100.0 * correct / len(test_f)


def linear_probe(train_f: Tensor, train_y: Tensor, test_f: Tensor, test_y: Tensor, num_classes: int = 10,
                 seed: int = 0) -> float:
    """Linear classifier on frozen, standardized features (SGD + cosine LR, fixed hyperparameters)."""
    mean, std = train_f.mean(0), train_f.std(0) + 1e-6
    train_f, test_f = (train_f - mean) / std, (test_f - mean) / std
    g = torch.Generator(device="cpu").manual_seed(seed)
    clf = torch.nn.Linear(train_f.shape[1], num_classes).to(train_f.device)
    opt = torch.optim.SGD(clf.parameters(), lr=LINEAR_LR, momentum=0.9, weight_decay=LINEAR_WD)
    steps_per_epoch = max(1, len(train_f) // LINEAR_BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, LINEAR_EPOCHS * steps_per_epoch)
    for _ in range(LINEAR_EPOCHS):
        perm = torch.randperm(len(train_f), generator=g).to(train_f.device)
        for i in range(steps_per_epoch):
            b = perm[i * LINEAR_BATCH : (i + 1) * LINEAR_BATCH]
            loss = F.cross_entropy(clf(train_f[b]), train_y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return 100.0 * (clf(test_f).argmax(1) == test_y).float().mean().item()


def evaluate_backbone(backbone: VisionTransformer, ds_cfg: dict, device, linear: bool = False,
                      batch_size: int = 512) -> dict[str, float]:
    t = eval_transform(ds_cfg["mean"], ds_cfg["std"])
    kw = dict(batch_size=batch_size, shuffle=False, num_workers=ds_cfg["num_workers"], pin_memory=True)
    was_training = backbone.training
    backbone.eval()
    tr = extract_features(backbone, DataLoader(build_dataset(ds_cfg, True, t), **kw), device)
    te = extract_features(backbone, DataLoader(build_dataset(ds_cfg, False, t), **kw), device)
    backbone.train(was_training)
    metrics = {f"knn_{f}_top1": knn_classifier(tr[f], tr["labels"], te[f], te["labels"]) for f in ("cls", "patch_mean")}
    if linear:
        with torch.enable_grad():
            metrics |= {f"linear_{f}_top1": linear_probe(tr[f], tr["labels"], te[f], te["labels"]) for f in ("cls", "patch_mean")}
    return metrics


def make_table(metric_files: list[str]) -> str:
    rows = ["| Method | Registers | Epochs | Backbone | Patch | Dim | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for f in metric_files:
        m = json.loads(Path(f).read_text())
        r = m["num_register_tokens"]
        name = {"dino": "DINO", "dinov2": "DINOv2"}[m["method"]] + (f"-Reg{r}" if r else "")
        fmt = lambda k: f"{m[k]:.2f}" if k in m else "-"
        rows.append(f"| {name} | {r} | {m['epochs']} | ViT-Tiny | {m['patch_size']} | {m['embed_dim']} | "
                    f"{fmt('knn_cls_top1')} | {fmt('knn_patch_mean_top1')} | {fmt('linear_cls_top1')} | {fmt('linear_patch_mean_top1')} |")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="teacher_backbone.pt or a training checkpoint")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-linear", action="store_true")
    p.add_argument("--output", help="metrics json path (default: metrics.json next to the checkpoint)")
    p.add_argument("--table", nargs="+", help="metrics.json files -> print Markdown table and write results.json/.md")
    args = p.parse_args()

    if args.table:
        table = make_table(args.table)
        Path("results.md").write_text(table + "\n")
        Path("results.json").write_text(json.dumps([json.loads(Path(f).read_text()) for f in args.table], indent=2))
        print(table)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, meta = load_backbone(args.checkpoint)
    backbone.to(device)
    ds_cfg = {"name": args.dataset, "root": args.data_root, "num_workers": args.num_workers,
              "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}
    mc = meta["model_config"]
    metrics = {"method": meta["method"], "epochs": meta.get("epoch"), "num_register_tokens": mc["num_register_tokens"],
               "patch_size": mc["patch_size"], "embed_dim": mc["embed_dim"],
               **evaluate_backbone(backbone, ds_cfg, device, linear=not args.no_linear)}
    out = Path(args.output) if args.output else Path(args.checkpoint).with_name("metrics.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

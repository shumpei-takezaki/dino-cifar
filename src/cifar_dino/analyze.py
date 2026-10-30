"""Qualitative analysis of trained teacher backbones on the CIFAR-10 test set.

- ``tsne.png``: t-SNE of the L2-normalized CLS and mean-patch features, coloured by class.
- ``patch_pca.png``: DINOv2-style PCA of patch tokens. The first 3 PCs of all patches -> RGB, and the same
  restricted to the foreground (PC1 > 0), with the background in black. The per-position mean patch token
  (over ``--n-pos-mean`` test images) is subtracted first: otherwise ~12% of the patch variance is position
  alone and PC1 just splits the top from the bottom of the image.

    python -m cifar_dino.analyze outputs/{dino,dinov2,dinov2_reg4}/seed_0/teacher_backbone.pt --out-dir outputs/analysis
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .data import build_dataset, eval_transform
from .evaluate import extract_features
from .export import load_backbone

MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _minmax(a: np.ndarray) -> np.ndarray:
    return (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-8)


def patch_pca_rgb(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """patches [N, h*w, D] -> (rgb_all [N,h,w,3], rgb_fg [N,h,w,3], fg mask [N,h,w]).

    PCA is fit jointly over all N images so colours are comparable across images. PC1's sign is chosen so the
    central patches score higher than the rest (CIFAR objects are roughly centred); foreground = PC1 > 0.
    """
    N, P, D = patches.shape
    s = int(round(P**0.5))
    x = patches.reshape(-1, D)
    pc = PCA(3).fit_transform(x)
    pc1 = pc[:, 0].reshape(N, s, s)
    c = slice(s // 4, s - s // 4)
    if pc1[:, c, c].mean() < 0:
        pc1 = -pc1
    fg = (pc1 > 0).ravel()
    rgb_fg = np.zeros((N * P, 3))
    rgb_fg[fg] = _minmax(PCA(3).fit_transform(x[fg]))
    return _minmax(pc).reshape(N, s, s, 3), rgb_fg.reshape(N, s, s, 3), fg.reshape(N, s, s)


def method_name(meta: dict) -> str:
    r = meta["model_config"]["num_register_tokens"]
    return {"dino": "DINO", "dinov2": "DINOv2"}[meta["method"]] + (f"-Reg{r}" if r else "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="teacher_backbone.pt files (one column per model)")
    p.add_argument("--out-dir", default="outputs/analysis")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake"])
    p.add_argument("--n-tsne", type=int, default=5000, help="test images used for t-SNE")
    p.add_argument("--n-pca-per-class", type=int, default=2, help="images per class shown in the patch PCA figure")
    p.add_argument("--n-pos-mean", type=int, default=2000, help="test images used for the per-position mean patch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_dataset({"name": args.dataset, "root": args.data_root}, False, eval_transform(MEAN, STD))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
    rng = np.random.default_rng(args.seed)

    names, tsne, pca, labels, imgs = [], [], [], None, None
    for ckpt in args.checkpoints:
        backbone, meta = load_backbone(ckpt)
        backbone.to(device)
        names.append(method_name(meta))
        feats = extract_features(backbone, loader, device)
        if labels is None:  # same images for every model
            labels = feats["labels"].cpu().numpy()
            tsne_idx = rng.choice(len(labels), min(args.n_tsne, len(labels)), replace=False)
            pca_idx = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[: args.n_pca_per_class] for c in range(10)])
            imgs = torch.stack([ds[i][0] for i in pca_idx])
        print(f"[{names[-1]}] t-SNE on {len(tsne_idx)} images", flush=True)
        tsne.append({f: TSNE(2, init="pca", random_state=args.seed).fit_transform(
            F.normalize(feats[f][tsne_idx], dim=1).cpu().numpy()) for f in ("cls", "patch_mean")})
        with torch.no_grad():
            pos_imgs = torch.stack([ds[i][0] for i in range(min(args.n_pos_mean, len(ds)))]).to(device)
            pos_mean = backbone.forward_features(pos_imgs)["patch_tokens"].mean(0)
            pca.append(patch_pca_rgb((backbone.forward_features(imgs.to(device))["patch_tokens"] - pos_mean).cpu().numpy()))

    # t-SNE: rows = feature, cols = model
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 10), squeeze=False)
    y = labels[tsne_idx]
    for j, (name, t) in enumerate(zip(names, tsne)):
        for i, (f, label) in enumerate([("cls", "CLS"), ("patch_mean", "PatchMean")]):
            ax = axes[i, j]
            sc = ax.scatter(*t[f].T, c=y, cmap="tab10", vmin=-0.5, vmax=9.5, s=2)
            ax.set_title(f"{name} / {label}")
            ax.set_xticks([]), ax.set_yticks([])
    fig.legend(sc.legend_elements()[0], CLASSES, loc="lower center", ncol=10, markerscale=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "tsne.png", dpi=150)

    # Patch PCA: row 0 = images, then per model (PCs of all patches, foreground-only PCs)
    n = len(imgs)
    rows = [("image", (imgs.permute(0, 2, 3, 1) * torch.tensor(STD) + torch.tensor(MEAN)).clamp(0, 1).numpy())]
    for name, (rgb_all, rgb_fg, _) in zip(names, pca):
        rows += [(f"{name}\nPCA", rgb_all), (f"{name}\nPCA fg", rgb_fg)]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.9, len(rows) * 0.95), squeeze=False)
    for i, (label, ims) in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(ims[j], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=7)
            if i == 0:
                ax.set_title(CLASSES[labels[pca_idx[j]]], fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_dir / "patch_pca.png", dpi=150)
    print(f"wrote {out_dir / 'tsne.png'} and {out_dir / 'patch_pca.png'}")


if __name__ == "__main__":
    main()

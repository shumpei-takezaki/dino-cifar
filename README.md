# dino-cifar

Controlled DINO vs DINOv2 self-supervised pretraining on CIFAR-10 with ViT-Tiny/4 (192-d, 8×8 = 64 patch tokens).

| config | method | registers | role |
|---|---|---:|---|
| `configs/dino.yaml` | DINO (CLS loss) | 0 | strict comparison |
| `configs/dinov2.yaml` | DINO + iBOT + KoLeo | 0 | strict comparison |
| `configs/dinov2_reg4.yaml` | DINO + iBOT + KoLeo | 4 | RAE-aligned variant, reported separately |

All three share `configs/base.yaml` (via `_base_`): the same data, `mrnghia90` augmentation, backbone, head, optimizer and schedules.
`configs/dinov2_vits.yaml` is the same DINOv2 recipe with a ViT-S/4 backbone (384-d, 6 heads; trained with TF32), used to
compare encoder sizes in [rae-cifar](https://github.com/shumpei-takezaki/rae-cifar).

## Results (seed 0, 1000 epochs, CIFAR-10 test top-1 %)

| Model | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|
| DINO, ViT-Tiny/4 | 82.04 | 78.27 | 79.86 | 77.74 |
| DINOv2, ViT-Tiny/4 | 85.49 | 86.05 | 81.25 | 86.59 |
| DINOv2-Reg4, ViT-Tiny/4 | 84.59 | 84.04 | 79.84 | 85.00 |
| DINOv2, ViT-S/4 | 86.96 | 87.04 | 76.59 | 89.54 |

The linear-probe hyperparameters are fixed and untuned, so the CLS linear numbers are provisional. Details, training curves
and qualitative analysis (in Japanese): [docs/results.md](docs/results.md).

The DINOv2 ViT-Tiny/4 and ViT-S/4 teacher backbones and the DINO ViT-Tiny/4 backbone are released with rae-cifar on
Hugging Face ([shumpei777/rae-cifar](https://huggingface.co/shumpei777/rae-cifar): `encoder_dinov2_vit_tiny4.pt`,
`encoder_dinov2_vit_s4.pt`, `disc_dino_vit_tiny4.pt`); load them with `load_backbone` below.

## Setup

```bash
uv sync
echo 'WANDB_API_KEY=...' > .env   # optional; scripts load it. Without a key, wandb logs offline (`wandb sync outputs/*/wandb/offline-run-*`)
# CIFAR-10 is downloaded to ./data automatically (or symlink an existing data/cifar-10-batches-py)
uv run pytest -q
```

## Training

Run long jobs in a detached tmux session. The scripts resume automatically from `checkpoint_last.pt`.

```bash
# smoke (2 epochs)
uv run python -m cifar_dino.train --config configs/dino.yaml   --epochs 2 --batch-size 32 --output-dir outputs/smoke_dino
uv run python -m cifar_dino.train --config configs/dinov2.yaml --epochs 2 --batch-size 32 --output-dir outputs/smoke_dinov2

# primary 1000-epoch runs (seed 0; another seed: append --seed N --output-dir outputs/<config>_seedN)
tmux new-session -d -s dino        'scripts/train_dino.sh        > outputs/dino.log 2>&1'
tmux new-session -d -s dinov2      'scripts/train_dinov2.sh      > outputs/dinov2.log 2>&1'
tmux new-session -d -s dinov2_reg4 'scripts/train_dinov2_reg4.sh > outputs/dinov2_reg4.log 2>&1'

# explicit resume
uv run python -m cifar_dino.train --config configs/dinov2.yaml --output-dir outputs/dinov2 \
  --resume outputs/dinov2/checkpoints/checkpoint_last.pt

# DDP (the global batch stays 256; the per-GPU batch is 256 / world_size)
uv run torchrun --nproc_per_node=2 -m cifar_dino.train --config configs/dino.yaml --output-dir outputs/dino
```

Measured on one A100-40GB, bs 256, fp32: about 55–75 s/epoch, so each 1000-epoch run takes about 16–20 h.
The GPU stays at ~98% utilization, and TF32 (`train.allow_tf32: true`) only saves about 6%.

Each run writes to `outputs/<config>/`:
- `config.yaml`: the resolved config, including all augmentation parameters
- `env.json`: git commit, library versions and GPU
- `train.jsonl`: step records every 50 steps and epoch averages, with each loss logged separately, plus kNN every 50 epochs. This is the canonical log.
- `wandb/`, `wandb_run_id.txt`: wandb run (project `dino-cifar`, run `<config>`). Step records go under `train/*` and epoch records under `epoch/*`, and a resumed run continues the same wandb run.
- `checkpoints/checkpoint_last.pt` and `checkpoints/checkpoint_epoch_XXXX.pt`
- `teacher_backbone.pt`
- `metrics.json`: final kNN and linear probe results

## Evaluation

```bash
uv run python -m cifar_dino.evaluate --checkpoint outputs/dinov2/teacher_backbone.pt --data-root ./data
uv run python -m cifar_dino.evaluate --table outputs/{dino,dinov2,dinov2_reg4}/metrics.json  # -> results.md / results.json
```

- **kNN:** k=20, T=0.07, run on the L2-normalized CLS token and on the mean patch token.
- **Linear probe:** trains on frozen, standardized features with SGD for 100 epochs (lr 0.1, cosine). The hyperparameters are fixed and identical for all methods.

### Qualitative analysis

```bash
uv run python -m cifar_dino.analyze outputs/{dino,dinov2,dinov2_reg4}/teacher_backbone.pt --out-dir outputs/analysis
```

Writes `tsne.png` (t-SNE of the L2-normalized CLS / mean-patch features on 5000 test images) and `patch_pca.png`
(DINOv2-style PCA of patch tokens on 2 test images per class). The per-position mean patch token is subtracted before
the PCA; without this, PC1 mostly separates the top of the image from the bottom. Takes about 1.5 min on one A100.

## Using the teacher backbone

```python
from cifar_dino.export import load_backbone

model, meta = load_backbone("outputs/dinov2/teacher_backbone.pt")  # no projection head needed
with torch.no_grad():
    out = model.forward_features(images)        # images: CIFAR-normalized [B,3,32,32]
cls, regs, patches = out["cls"], out["reg_tokens"], out["patch_tokens"]  # [B,192], [B,R,192], [B,64,192]
patch_map = patches.transpose(1, 2).reshape(-1, 192, 8, 8)
```

All outputs are taken after the final LayerNorm. Register tokens are never included in `patch_tokens`.

## Implementation notes

- **iBOT masking** follows DINOv2 `prepare_tokens_with_masks`. The mask token replaces patch embeddings before the positional embedding and the Transformer. Registers are inserted after CLS and get no positional embedding.
- **Mask sampling** follows the official DINOv2 `MaskingGenerator` and `collate_data_and_cast`. Exactly `int(0.5·2B)` global student samples are masked, with ratios stratified over [0.1, 0.5], and the masks are shuffled across the batch. The per-sample marginal is Bernoulli(0.5) × Uniform(0.1, 0.5), but the number of masked samples per batch is fixed.
- **Centering** uses DINO-style EMA centering for both losses, with separate centers for CLS and iBOT. The iBOT center is updated from masked positions only. Official DINOv2 defaults to Sinkhorn-Knopp centering; this repo does not.
- **KoLeo** is computed per global view on the student backbone CLS token and then averaged over views. Official DINOv2 sums over views.
- **DINO head:** the last layer is `F.linear(x, normalize(W))`. This is exactly weight norm with a frozen unit gain. The last layer is frozen for epoch 1.
- **Not implemented** (all raise an error): AMP, a separate iBOT head, and augmentation profiles other than `mrnghia90`.

## Acknowledgements

- [DINO](https://github.com/facebookresearch/dino) ([Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294))
  and [DINOv2](https://github.com/facebookresearch/dinov2) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)):
  the algorithms and the reference implementations (losses, centering, heads, iBOT masking, KoLeo).
- [iBOT](https://arxiv.org/abs/2111.07832) (masked patch objective) and [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (register tokens).
- [mrnghia90/dino-from-scratch](https://huggingface.co/mrnghia90/dino-from-scratch): the CIFAR-10 ViT-Tiny/4 setup and the `mrnghia90` augmentation recipe.
- [DOV4CL](https://github.com/xieyc99/DOV4CL) and [solo-learn](https://github.com/vturrisi/solo-learn): CIFAR DINO implementations used as references.

## License

MIT ([LICENSE](LICENSE)).

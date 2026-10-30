# 03. Encoder size: DINOv2 ViT-Tiny/4 vs. ViT-S/4

- **Status:** done (seed 0, 1000 epochs). ViT-B/4 coming soon.

## Goal

Train DINOv2 with a larger backbone to compare encoder sizes in rae-cifar
([experiment 06 there](https://github.com/shumpei-takezaki/rae-cifar/blob/main/docs/experiments/06_encoder_capacity.md)).

## Setup

`configs/dinov2_vits.yaml`: the DINOv2 recipe of [01](01_dino_vs_dinov2.md) with a ViT-S/4 backbone (width 384, 6 heads;
the size of the official ViT-S), no registers. Trained with TF32 (`train.allow_tf32: true`); strict fp32 is about 2.4× slower
for ViT-S. About 18 h on one A100-40GB.

## Results

Final top-1 accuracy on CIFAR-10 test (%):

| Encoder | Width | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|---:|
| ViT-Tiny/4 | 192 | 85.49 | 86.05 | **81.25** | 86.59 |
| ViT-S/4 | 384 | **86.96** | **87.04** | 76.59 | **89.54** |

kNN during training (CLS / PatchMean):

| Epoch | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| ViT-Tiny/4 | 81.5 / 78.1 | 83.5 / 81.6 | 84.7 / 83.3 | 85.1 / 85.1 | 85.5 / 86.1 |
| ViT-S/4 | 82.3 / 80.7 | 84.5 / 84.1 | 85.2 / 84.5 | 86.2 / 86.2 | 87.0 / 87.0 |

- **ViT-S is better than ViT-Tiny** (kNN +1.0–1.5 pt, PatchMean linear +3.0 pt), at every point of training, and is still
  improving at epoch 1000.
- **Only the CLS linear probe is lower for ViT-S** (76.59). As noted in [01](01_dino_vs_dinov2.md), the CLS linear numbers
  are provisional because the linear-probe hyperparameters are fixed and untuned.
- **Different numerical precision:** ViT-Tiny was trained in strict fp32, ViT-S with TF32.

## Reproduce

```bash
scripts/train_dinov2_vits.sh   # outputs/dinov2_vits/ (metrics.json is written at the end of training)
```

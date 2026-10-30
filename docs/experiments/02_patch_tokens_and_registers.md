# 02. Patch tokens as a latent, and whether registers are needed

- **Status:** done (seed 0, 1000 epochs)

## Goal

[rae-cifar](https://github.com/shumpei-takezaki/rae-cifar) uses the teacher patch tokens as the latent of a representation
autoencoder (RAE). Before choosing the encoder, we looked at the geometry of the patch tokens and checked for the high-norm
artifact tokens that motivate registers in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).

## Setup

Teacher patch tokens of 5000 test images × 64 patches, for the three runs of [01](01_dino_vs_dinov2.md). Computed with a
one-off script (not included in the repository).

## Results

| | DINO | DINOv2 | DINOv2-Reg4 |
|---|---:|---:|---:|
| Norm after the final LayerNorm | 41.2 ± 0.2 | 40.9 ± 0.2 | 40.4 ± 0.2 |
| Per-channel std after the final LayerNorm | 2.1–6.6 | 1.9–6.4 | 1.9–4.3 |
| Variance of PC1 / dims for 90% of the variance / effective rank | 4.6% / 129 / 126 | 4.6% / 116 / 118 | 4.3% / 111 / 115 |
| Variance explained by position alone | 13.6% | 12.1% | 11.7% |
| Cosine: neighbouring patch / same image / same position in another image | 0.61 / 0.06 / 0.15 | 0.74 / 0.15 / 0.16 | 0.74 / 0.14 / 0.17 |
| Patch norm before the final LayerNorm: median / max | 1.3 / 2.4 | 1.9 / 3.4 | 1.9 / 3.2 |

- The variance is spread over almost all 192 dimensions; the tokens do not live in a low-dimensional subspace.
- For DINO, a patch is more similar to the patch at the same position in another image than to other patches of its own image:
  position dominates. DINOv2 is spatially smoother and more coherent within an image, matching the patch PCA in [01](01_dino_vs_dinov2.md).
- **Registers are not needed here.** Even without registers, the largest pre-LayerNorm norm is at most 1.8× the median, so
  there are no artifact tokens with extreme norms. The Reg4 register norms (about 1.3) are similar to that of CLS.
  rae-cifar therefore uses DINOv2 without registers.

# Experiments

Self-supervised pretraining of DINO and DINOv2 from scratch on CIFAR-10. One file per experiment in `experiments/`, figures in
`figures/<experiment>/`, final metrics of all runs in [results.json](results.json).

| # | Experiment | Main result |
|---|---|---|
| 01 | [DINO vs. DINOv2 (ViT-Tiny/4)](experiments/01_dino_vs_dinov2.md) | DINOv2 is better on every metric, most of all for the mean patch token (kNN 86.05 vs. 78.27) |
| 02 | [Patch tokens as a latent, and registers](experiments/02_patch_tokens_and_registers.md) | No high-norm artifact tokens without registers, so registers are not needed |
| 03 | [Encoder size: ViT-Tiny/4 vs. ViT-S/4](experiments/03_encoder_size.md) | ViT-S is better (PatchMean kNN 87.04, linear 89.54) |

**Coming soon:** DINOv2 ViT-B/4; more seeds; a linear-probe protocol with hyperparameters chosen on a validation split.

#!/usr/bin/env bash
# Usage: scripts/train_dinov2_reg4.sh [extra train args...]   (resumes automatically if a checkpoint exists)
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
OUT=outputs/dinov2_reg4
RESUME=()
[[ -f $OUT/checkpoints/checkpoint_last.pt ]] && RESUME=(--resume $OUT/checkpoints/checkpoint_last.pt)
uv run python -m cifar_dino.train --config configs/dinov2_reg4.yaml --output-dir $OUT "${RESUME[@]}" "$@"

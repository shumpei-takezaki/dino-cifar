#!/usr/bin/env bash
# Usage: scripts/train_dinov2.sh [seed] [extra train args...]   (resumes automatically if a checkpoint exists)
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
SEED=${1:-0}; shift || true
OUT=outputs/dinov2/seed_$SEED
RESUME=()
[[ -f $OUT/checkpoints/checkpoint_last.pt ]] && RESUME=(--resume $OUT/checkpoints/checkpoint_last.pt)
uv run python -m cifar_dino.train --config configs/dinov2.yaml --seed $SEED --output-dir $OUT "${RESUME[@]}" "$@"

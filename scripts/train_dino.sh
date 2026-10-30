#!/usr/bin/env bash
# Usage: scripts/train_dino.sh [extra train args...]   (resumes automatically if a checkpoint exists)
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
OUT=outputs/dino
RESUME=()
[[ -f $OUT/checkpoints/checkpoint_last.pt ]] && RESUME=(--resume $OUT/checkpoints/checkpoint_last.pt)
uv run python -m cifar_dino.train --config configs/dino.yaml --output-dir $OUT "${RESUME[@]}" "$@"

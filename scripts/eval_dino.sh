#!/usr/bin/env bash
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
SEED=${1:-0}
uv run python -m cifar_dino.evaluate --checkpoint outputs/dino/seed_$SEED/teacher_backbone.pt --data-root ./data

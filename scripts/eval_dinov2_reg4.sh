#!/usr/bin/env bash
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
uv run python -m cifar_dino.evaluate --checkpoint outputs/dinov2_reg4/teacher_backbone.pt --data-root ./data

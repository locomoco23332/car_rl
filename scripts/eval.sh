#!/usr/bin/env bash
set -e
source .venv/bin/activate

python -m src.eval \
  --env-id CarRacing-v3 \
  --model checkpoints/ppo_carracing_v3.zip \
  --episodes 5 \
  --render human \
  --record out_videos  # 제거하면 기록 안 함


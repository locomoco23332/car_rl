#!/usr/bin/env bash
set -e
source .venv/bin/activate

python -m src.train \
  --env-id CarRacing-v3 \
  --n-envs 4 \
  --total-steps 300000 \
  --logdir runs/carracing_ppo \
  --save-path checkpoints/ppo_carracing_v3.zip


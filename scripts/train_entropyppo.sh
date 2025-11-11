#!/usr/bin/env bash
set -e
source .venv/bin/activate

mkdir -p checkpoints

python -m src.train_entropyppo \
  --env-id CarRacing-v2 \
  --n-envs 4 \
  --total-steps 300000 \
  --rollout-steps 1024 \
  --frame-stack 4 \
  --resize 84 84 \
  --gray \
  --clip 0.2 \
  --ppo-epoch 4 \
  --mini-batch 8 \
  --lr 3e-4 \
  --vf-coef 0.5 \
  --ent-coef 0.01 \
  # --adaptive-entropy  # ← 자동 엔트로피 튜닝 쓰려면 주석 해제


#!/usr/bin/bash

# ------------------------------------------------------------------------------
# SAC State
# ------------------------------------------------------------------------------

# PickCube-v1
python -m scripts.sac.train ms3-rgb --env.env-id PickCube-v1 --seed 1 \
  --env.max-episode-steps 50 \
  --env.ignore-terminations \
  \
  --num-eval-steps 50 \
  --eval-env.num-envs 16 \
  --eval-env.ignore-terminations \
  \
  --env.num-envs 256 \
  --batch_size 1024 \
  --buffer_size 1_000_000 \
  --learning_starts $((256 * 32)) \
  \
  --total_timesteps 10_000_000 \
  --steps_per_env_per_iteration 1 \
  --grad_steps_per_iteration 20 \
  --sac.policy-frequency 1 \
  --sac.target-network-frequency 2 \
  \
  --sac.gamma 0.8 \
  --sac.policy-lr 3e-4 \
  --sac.q-lr 3e-4 \
  --sac.tau 0.005 \
  \
  --sac.alpha 1.0 \
  --sac.autotune \
  --sac.alpha-lr 3e-4 \
  \
  --eval-freq 50 \
  --logger_freq 2 \
  \
  --compile --cudagraphs \
  \
  --logger.exp-name "sac-PickCube-v1-rgb-1" \
  --logger.wandb \
  --logger.wandb-entity arth-shukla

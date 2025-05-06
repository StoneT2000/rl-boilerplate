#!/usr/bin/bash

# # ------------------------------------------------------------------------------
# # SAC State PickCube-v1
# # ------------------------------------------------------------------------------

# python -m scripts.sac.train ms3-state --env.env-id PickCube-v1 --seed 1 \
#   --env.max-episode-steps 50 \
#   --env.ignore-terminations \
#   \
#   --num-eval-steps 50 \
#   --eval-env.num-envs 16 \
#   --eval-env.ignore-terminations \
#   \
#   --logger.exp-name "sac-PickCube-v1-state-1" \
#   --logger.wandb \
#   --logger.wandb-entity arth-shukla \
#   \
#   --env.num-envs 1024 \
#   --batch_size 4096 \
#   --buffer_size 1_000_000 \
#   --learning_starts $((1024 * 32)) \
#   \
#   --total_timesteps 3_000_000 \
#   --steps_per_env_per_iteration 1 \
#   --grad_steps_per_iteration 20 \
#   --sac.policy-frequency 1 \
#   --sac.target-network-frequency 2 \
#   \
#   --sac.gamma 0.8 \
#   --sac.policy-lr 3e-4 \
#   --sac.q-lr 3e-4 \
#   --sac.tau 0.005 \
#   \
#   --sac.alpha 1.0 \
#   --sac.autotune \
#   --sac.alpha-lr 3e-4 \
#   \
#   --eval-freq 50 \
#   --logger_freq 2 \
#   \
#   --compile --cudagraphs
# python -m scripts.sac.train ms3-state --env.env-id PickCube-v1 --seed 1 \
#   --env.max-episode-steps 50 \
#   --env.ignore-terminations \
#   \
#   --num-eval-steps 50 \
#   --eval-env.num-envs 16 \
#   --eval-env.ignore-terminations \
#   \
#   --logger.exp-name "redq-PickCube-v1-state-1" \
#   --logger.wandb \
#   --logger.wandb-entity arth-shukla \
#   \
#   --env.num-envs 1024 \
#   --batch_size 4096 \
#   --buffer_size 1_000_000 \
#   --learning_starts $((1024 * 32)) \
#   \
#   --total_timesteps 3_000_000 \
#   --steps_per_env_per_iteration 1 \
#   --grad_steps_per_iteration 20 \
#   --sac.policy-frequency 1 \
#   --sac.target-network-frequency 2 \
#   \
#   --sac.gamma 0.8 \
#   --sac.policy-lr 3e-4 \
#   --sac.q-lr 3e-4 \
#   --sac.tau 0.005 \
#   \
#   --sac.alpha 1.0 \
#   --sac.autotune \
#   --sac.alpha-lr 3e-4 \
#   \
#   --sac.num-q 10 \
#   --sac.min-q 2 \
#   --sac.ensemble-reduction mean \
#   \
#   --eval-freq 50 \
#   --logger_freq 2 \
#   \
#   --compile --cudagraphs

# # ------------------------------------------------------------------------------


# # ------------------------------------------------------------------------------
# # SAC State PegInsertionSide-v1
# # ------------------------------------------------------------------------------

# python -m scripts.sac.train ms3-state --env.env-id PegInsertionSide-v1 --seed 1 \
#   --env.max-episode-steps 100 \
#   --env.ignore-terminations \
#   \
#   --num-eval-steps 100 \
#   --eval-env.num-envs 16 \
#   --eval-env.ignore-terminations \
#   \
#   --logger.exp-name "sac-PegInsertionSide-v1-state-1" \
#   --logger.wandb \
#   --logger.wandb-entity arth-shukla \
#   \
#   --env.num-envs 1024 \
#   --batch_size 4096 \
#   --buffer_size 1_000_000 \
#   --learning_starts $((1024 * 32)) \
#   \
#   --total_timesteps 30_000_000 \
#   --steps_per_env_per_iteration 1 \
#   --grad_steps_per_iteration 20 \
#   --sac.policy-frequency 5 \
#   --sac.target-network-frequency 2 \
#   \
#   --sac.gamma 0.99 \
#   --sac.policy-lr 3e-4 \
#   --sac.q-lr 3e-4 \
#   --sac.tau 0.005 \
#   \
#   --sac.alpha 1.0 \
#   --sac.autotune \
#   --sac.alpha-lr 3e-4 \
#   \
#   --eval-freq 400 \
#   --logger_freq 10 \
#   \
#   --compile --cudagraphs
# python -m scripts.sac.train ms3-state --env.env-id PegInsertionSide-v1 --seed 1 \
#  --env.max-episode-steps 100 \
#  --env.ignore-terminations \
#  \
#  --num-eval-steps 100 \
#  --eval-env.num-envs 16 \
#  --eval-env.ignore-terminations \
#  \
#  --logger.exp-name "redq-PegInsertionSide-v1-state-1" \
#  --logger.wandb \
#  --logger.wandb-entity arth-shukla \
#  \
#  --env.num-envs 1024 \
#  --batch_size 4096 \
#  --buffer_size 1_000_000 \
#  --learning_starts $((1024 * 32)) \
#  \
#  --total_timesteps 30_000_000 \
#  --steps_per_env_per_iteration 1 \
#  --grad_steps_per_iteration 20 \
#  --sac.policy-frequency 5 \
#  --sac.target-network-frequency 2 \
#  \
#  --sac.gamma 0.99 \
#  --sac.policy-lr 3e-4 \
#  --sac.q-lr 3e-4 \
#  --sac.tau 0.005 \
#  \
#  --sac.alpha 1.0 \
#  --sac.autotune \
#  --sac.alpha-lr 3e-4 \
#  \
#  --sac.num-q 10 \
#  --sac.min-q 2 \
#  --sac.ensemble-reduction mean \
#  \
#  --eval-freq 400 \
#  --logger_freq 10 \
#  \
#  --compile --cudagraphs

# # ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# SAC State UnitreeG1TransportBox-v1
# ------------------------------------------------------------------------------

python -m scripts.sac.train ms3-state --env.env-id UnitreeG1TransportBox-v1 --seed 1 \
  --env.max-episode-steps 100 \
  --env.ignore-terminations \
  \
  --num-eval-steps 100 \
  --eval-env.num-envs 16 \
  --eval-env.ignore-terminations \
  \
  --logger.exp-name "sac-UnitreeG1TransportBox-v1-state-1" \
  --logger.wandb \
  --logger.wandb-entity arth-shukla \
  \
  --env.num-envs 4096 \
  --batch_size 2048 \
  --buffer_size 1_000_000 \
  --learning_starts $((1024 * 32)) \
  \
  --total_timesteps 50_000_000 \
  --steps_per_env_per_iteration 1 \
  --grad_steps_per_iteration 20 \
  --sac.policy-frequency 1 \
  --sac.target-network-frequency 2 \
  \
  --sac.gamma 0.99 \
  --sac.policy-lr 3e-4 \
  --sac.q-lr 3e-4 \
  --sac.tau 0.005 \
  \
  --sac.alpha 1.0 \
  --sac.autotune \
  --sac.alpha-lr 3e-4 \
  \
  --eval-freq 400 \
  --logger_freq 10 \
  \
  --compile --cudagraphs
python -m scripts.sac.train ms3-state --env.env-id UnitreeG1TransportBox-v1 --seed 1 \
  --env.max-episode-steps 100 \
  --env.ignore-terminations \
  \
  --num-eval-steps 100 \
  --eval-env.num-envs 16 \
  --eval-env.ignore-terminations \
  \
  --logger.exp-name "redq-UnitreeG1TransportBox-v1-state-1" \
  --logger.wandb \
  --logger.wandb-entity arth-shukla \
  \
  --env.num-envs 4096 \
  --batch_size 2024 \
  --buffer_size 1_000_000 \
  --learning_starts $((1024 * 32)) \
  \
  --total_timesteps 50_000_000 \
  --steps_per_env_per_iteration 1 \
  --grad_steps_per_iteration 20 \
  --sac.policy-frequency 1 \
  --sac.target-network-frequency 2 \
  \
  --sac.gamma 0.99 \
  --sac.policy-lr 3e-4 \
  --sac.q-lr 3e-4 \
  --sac.tau 0.005 \
  \
  --sac.alpha 1.0 \
  --sac.autotune \
  --sac.alpha-lr 3e-4 \
  \
  --sac.num-q 10 \
  --sac.min-q 2 \
  --sac.ensemble-reduction mean \
  \
  --eval-freq 400 \
  --logger_freq 10 \
  \
  --compile --cudagraphs

# ------------------------------------------------------------------------------

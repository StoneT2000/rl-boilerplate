#!/usr/bin/bash

PROJECT_NAME="rl-boilerplate-rgbd"
ENTITY="stonet2000"

NUM_ENVS=256
args=(
  "--env.num-envs=$NUM_ENVS"
  "--env.ignore-terminations"
  "--eval-env.num-envs=16"
  "--eval-env.ignore-terminations"
  #
  "--buffer-size=100_000"
  "--batch-size=1024"
  "--learning_starts=$((NUM_ENVS * 32))"
  #
  "--steps_per_env_per_iteration=1"
  "--grad_steps_per_iteration=10"
  "--sac.policy-frequency=1"
  "--sac.target-network-frequency=2"
  #
  "--sac.policy-lr=3e-4"
  "--sac.q-lr=3e-4"
  "--sac.tau=0.005"
  #
  "--sac.alpha=1.0"
  "--sac.autotune"
  "--sac.alpha-lr=3e-4"
  #
  "--eval-freq=50"
  "--logger_freq=2"
  "--compile"
  "--cudagraphs"
  #
  "--logger.wandb" \
  "--logger.wandb-entity=$ENTITY" \
  "--logger.wandb-project=$PROJECT_NAME" \
)

seeds=(9351 4796 1788)

for seed in "${seeds[@]}"
do

  env_id=PushCube-v1
  max_episode_steps=50
  python -m scripts.sac.train ms3-rgb \
    "${args[@]}" \
    --sac.gamma=0.8 \
    --total_timesteps=2_000_000 \
    --env.max-episode-steps=$max_episode_steps \
    --num-eval-steps=$max_episode_steps \
    --env.env-id=$env_id \
    --seed=$seed \
    --logger.exp-name="sac-$env_id-rgb-$seed-untuned"

  env_id=PickCube-v1
  max_episode_steps=50
  python -m scripts.sac.train ms3-rgb \
    "${args[@]}" \
    --sac.gamma=0.8 \
    --total_timesteps=2_000_000 \
    --num-eval-steps=$max_episode_steps \
    --env.env-id=$env_id \
    --seed=$seed \
    --logger.exp-name="sac-$env_id-rgb-$seed-untuned"

  env_id=PushT-v1
  max_episode_steps=100
  python -m scripts.sac.train ms3-rgb \
    "${args[@]}" \
    --sac.gamma=0.99 \
    --total_timesteps=12_000_000 \
    --env.max-episode-steps=$max_episode_steps \
    --num-eval-steps=$max_episode_steps \
    --env.env-id=$env_id \
    --seed=$seed \
    --logger.exp-name="sac-$env_id-rgb-$seed-untuned"

  env_id=AnymalC-Reach-v1
  max_episode_steps=200
  python -m scripts.sac.train ms3-rgb \
    "${args[@]}" \
    --sac.gamma=0.99 \
    --total_timesteps=12_000_000 \
    --env.max-episode-steps=$max_episode_steps \
    --num-eval-steps=$max_episode_steps \
    --env.env-id=$env_id \
    --seed=$seed \
    --logger.exp-name="sac-$env_id-rgb-$seed-untuned"

  env_id=PickSingleYCB-v1
  max_episode_steps=50
  python -m scripts.sac.train ms3-rgb \
    "${args[@]}" \
    --sac.gamma=0.8 \
    --total_timesteps=12_000_000 \
    --env.max-episode-steps=$max_episode_steps \
    --num-eval-steps=$max_episode_steps \
    --env.env-id=$env_id \
    --seed=$seed \
    --logger.exp-name="sac-$env_id-rgb-$seed-untuned"

done
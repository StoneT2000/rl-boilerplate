python scripts/sac/train.py ms3-state --env.env-id PegInsertionSide-v1 --seed 1 --num_eval_steps 100 \
  --total_timesteps 30_000_000 \
  --buffer_size 1_000_000 \
  --learning_starts 32768 \
  --batch_size 4096 \
  --grad_steps_per_iteration 20 \
  --sac.gamma 0.99 --sac.tau 5e-3 --sac.policy_frequency 5\
  --sac.ensemble_reduction "mean" \
  --env.ignore_terminations \
  --logger.exp-name "sac-PegInsertionSide-v1-state-1-ensemble_reduction=mean" --logger.wandb --logger.wandb_project "rl-boilerplate-ms3" \
  --compile --cudagraphs

python scripts/sac/train.py ms3-state --env.env-id PickCube-v1 --seed 1 \
  --num_eval_steps 50 \
  --total_timesteps 5_000_000 --buffer_size 1_000_000 --batch_size 4096 \
  --learning_starts 32768 --grad_steps_per_iteration 20 \
  --sac.gamma 0.8 \
  --logger.exp-name "sac-PickCube-v1-state-1-walltime-efficient" --logger.wandb_project "rl-boilerplate-ms3" \
  --compile --cudagraphs



# SCRATCH #
python scripts/sac/train.py ms3-rgb --env.env-id PickCube-v1 --seed 1   --logger.exp-name "sac-PickCube-v1-rgb-1" --logger.clear_out \
  --compile --cudagraphs

python scripts/sac/train.py ms3-rgb --env.env-id PickCube-v1 --seed 1   --logger.exp-name "sac-PickCube-v1-rgb-1-slow5" --logger.clear_out \
  --env.num_envs 32 --steps_per_env_per_iteration 2 --grad_steps_per_iteration 16 --batch_size 512 --sac.target_network_frequency 1 --learning_starts 4000 --sac.alpha 0.2 \
  --compile --cudagraphs

python scripts/sac/train.py ms3-rgb --env.env-id PickCube-v1 --seed 1   --logger.exp-name "sac-PickCube-v1-rgb-1-slow5" --logger.clear_out \
  --env.num_envs 64 --steps_per_env_per_iteration 1 --grad_steps_per_iteration 16 --batch_size 512 --sac.target_network_frequency 1 --learning_starts 4000 --sac.alpha 0.2 \
  --compile --cudagraphs --sac.buffer-size 200_000
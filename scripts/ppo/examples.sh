seeds=(1 2 3 4 5)
# MS3 RGB
for seed in ${seeds[@]}
do
  python scripts/ppo/train.py ms3-rgb --env.env-id PickCube-v1 \
    --seed ${seed} \
    --cudagraphs \
    --logger.exp-name="ppo-PickCube-v1-rgb-${seed}" --logger.clear-out \
    --logger.wandb_entity stonet2000 --logger.wandb \
    --logger.wandb_project "rl-boilerplate" --logger.wandb_group "PPO-baseline"
done

# MS3 State
for seed in ${seeds[@]}
do
  python scripts/ppo/train.py ms3-state --env.env-id PickCube-v1 \
    --seed ${seed} \
    --cudagraphs \
    --logger.exp-name="ppo-PickCube-v1-state-${seed}" --logger.clear-out \
    --logger.wandb_entity stonet2000 --logger.wandb \
    --logger.wandb_project "rl-boilerplate" --logger.wandb_group "PPO-baseline"
done
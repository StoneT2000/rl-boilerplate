seed=1
python scripts/ppo/train.py ms3-state --env.env-id PickCube-v1 \
  --seed ${seed} \
  --logger.no-wandb --logger.clear-out \
  --cudagraphs \
  --logger.exp-name="ppo-PickCube-v1-state-${seed}" 
seeds=(1 2 3 4 5)
for seed in ${seeds[@]}
do
  python scripts/ppo/train.py ms3-rgb --env.env-id PickCube-v1 \
    --seed ${seed} \
    --logger.clear-out \
    --cudagraphs \
    --logger.exp-name="ppo-PickCube-v1-rgb-${seed}" \
    --logger.wandb_entity stonet2000 --logger.wandb \
    --logger.wandb_project "PPO" --logger.wandb_group "PPO-baseline"
done

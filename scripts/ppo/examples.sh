seed=1
python scripts/ppo/train.py ms3-state --env.env-id PickCube-v1 \
  --seed ${seed} \
  --logger.no-wandb --logger.clear-out \
  --cudagraphs \
  --logger.exp-name="ppo-PickCube-v1-state-${seed}" 

python scripts/ppo/train.py ms3-rgb --env.env-id PickCube-v1 \
  --seed ${seed} \
  --logger.no-wandb --logger.clear-out \
  --cudagraphs \
  --logger.exp-name="ppo-PickCube-v1-rgb-${seed}" 
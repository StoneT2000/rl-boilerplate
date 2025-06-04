# rl-boilerplate

A repo that I copy for any project that uses some RL, with some focus on robotics/continuous control.

It is fairly minimal (perhaps a mid-point between CleanRL and StableBaselines). It is also nearly completely typed, runs quite fast, and is torch based.

What's included:
- [Tyro](https://github.com/brentyi/tyro) based CLI control with dataclass based python configuration (no yamls!)
- PPO and SAC algorithms (with torch.compile/cudagraphs support)
- Basic models (CNNs, MLPs etc.) configurable with python dataclasses
- Loggers for tensorboard and wandb
- A general purpose `make_env_from_config` function to replace `gym.make` that handles all dependency and wrapper madness for a bunch of continuous control environments.

General practices:
- All code is typed when possible
- All data is batched whenever possible, even if its just some sample input to figure out shapes. Batched is the default
- Dictionary of torch tensors are moved to tensordict when possible

While some repos like to make things as modular as possible, this repo does not do that. The only objects that are intended to be imported from the library in `rl` are environment configurations and a function to create environments from configs, neural network models and a function to build models from configs, and replay buffer designs. Things that are generally standardizable or often are never changed in RL experiments (e.g. neural net architectures). 

Algorithm specific code is in the `scripts/<algo>` folder and is organized on a per-algorithm basis. Typically they each have at least a `scripts/<algo>/config.py` file with some default configs and configs for the algorithm itself. There is also a `scripts/<algo>/README.md` file which contains
- High-level overview of algorithm
- Problems with algorithm
- Tricks/modifications used in the code base (for better performance / faster training etc. )
- Citation bibtex

## To Use

Copy the contents of this repo somewhere into your project.

Merge your setup.py file / update your dependencies to include the dependencies in this repo's setup.py file.
<!-- 
Replace / update the following files

`environment.yml` - change the name and add/remove pkgs

`pkgname` - rename folder to the actual project name

Then `mamba create env` or `conda create env` -->

Or to test this repo directly clone it and run

```bash
mamba create -n "rl" "python==3.11"
mamba activate rl
pip install -e . torch # pick your torch version
```

Example train script

```bash
python scripts/ppo/train.py --help
python scripts/ppo/train.py ms3-state --env.env-id PickCube-v1 --seed 1 --logger.exp-name "ppo-PickCube-v1-state-1"

python scripts/sac/train.py --help
python scripts/sac/train.py ms3-state --env.env-id PickCube-v1 --seed 1 --logger.exp-name "sac-PickCube-v1-state-1"

python scripts/sac/train.py ms3-state --env.env-id PegInsertionSide-v1 --seed 1 --logger.exp-name "sac-PegInsertionSide-v1-state-1" --num_eval_steps 100 \
  --total_timesteps 30_000_000 \
  --buffer_size 1_000_000 \
  --learning_starts 32768 \
  --batch_size 4096 \
  --grad_steps_per_iteration 20 \
  --sac.gamma 0.99 \
  --sac.tau 5e-3 \
  --sac.policy_frequency 5 \
  --env.ignore_terminations

python scripts/sac/train.py ms3-rgb --env.env-id PickCube-v1 --seed 1 \
  --logger.exp-name "sac-PickCube-v1-rgb-1"
```

The way it works with tyro CLI configuration is you can first specify a default config (train.py and config.py defines ms3-state and ms3-rgb for now), and then override all the other things as needed.

If you want to make your own changes/algorithm etc. recommend you read how ppo/train.py and ppo/config.py is written for a fairly clean example of fully typed configs and easy experimentation:
- Define a TrainConfig dataclass that contains all other dataclass configs (e.g. PPO hyperparemters, env configs, network configs, logger configs etc.)
- save a pickle file of the python config (which can include non json serializable objects such as gym wrapper classes)
- save a JSON readable version of the config
- save evaluation videos of the agent
- log things to wandb

## Organization

`rl/` the directory for all commonly re-used code. Code that is standardizable (e.g. replay buffers) because there is an obvious goal (e.g. max memory efficiency / read/write speeds for replay buffers) and many algos use it, or code that is often re-used and is rarely ever heavily experimented with outside some common choices (e.g. neural net models), are candidates for being placed here.

`rl/logger` - code for logging utilities

`rl/models` - contains all neural network models

`rl/envs` - folder for any custom environments and utilities to create environments from standardized environment configs

`scripts/<algo>` - all files related for running an algorithm

`scripts/<algo>/config.py` - all relevant configs as python files

## Citation

If you use this codebase no need to cite it (you are welcome to cite the github repo), just make sure to cite the algorithm(s) you are using.
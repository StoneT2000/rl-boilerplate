# rl-boilerplate

A repo that I copy for any project that uses some RL, with some focus on robotics/continuous control.

The point of this repo is that it **should be extendible**, with instructions on how to add new algorithms, modify config definitions etc.

It is as minimal as it gets while still being updated to run somewhat fast and is torch based.

If you want to copy this repo / use it you can just copy the rl folder and place it somewhere in your own project and hack/modify whatever you need.

What's included in the main branch
- [Dacite](https://github.com/konradhalas/dacite) + dataclass based yaml configuration system
- [Tyro](https://github.com/brentyi/tyro) based CLI control
- PPO and SAC algorithms (with torch.compile/cudagraphs support)
- Basic models (CNNs, MLPs etc.) configurable with yaml files
- Loggers for tensorboard and wandb
- A general purpose `make_env` function to replace `gym.make` that handles all dependency and wrapper madness for a bunch of continuous control environments


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

## Organization


`configs` holds all config files stored as yml files
`configs/<name>` holds all config files (and should hold a default of some sort) for a particular script (e.g PPO training script) or a group of experiments (e.g. BC on CartPole env)

`rl` the directory for all code

`rl/models` contains all models (probably torch)

`rl/envs` - folder for any custom environments

`rl/cfg` configuration tooling. Usually doesn't need to be modified unless you want to change the config system.

`scripts/` contains various scripts that use modules in the package to do things, e.g. sample random environment data, train with PPO etc.
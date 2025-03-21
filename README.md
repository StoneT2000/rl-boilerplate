# rl-boilerplate

A repo that I copy for any project that uses some RL, with some focus on robotics/continuous control.

The point of this repo is that it **should be extendible**, with instructions on how to add new algorithms, modify config definitions etc.

It is fairly minimal (perhaps a mid-point between CleanRL and StableBaselines). It is also completely typed, runs somewhat fast, and is torch based.

If you want to copy this repo / use it you can just copy the rl folder and place it somewhere in your own project and hack/modify whatever you need.

What's included in the main branch
- [Dacite](https://github.com/konradhalas/dacite) + dataclass based yaml configuration system
- [Tyro](https://github.com/brentyi/tyro) based CLI control
- PPO and SAC algorithms (with torch.compile/cudagraphs support)
- Basic models (CNNs, MLPs etc.) configurable with yaml files
- Loggers for tensorboard and wandb
- A general purpose `make_env` function to replace `gym.make` that handles all dependency and wrapper madness for a bunch of continuous control environments

General practices
- All code is typed when possible
- All data is batched whenever possible, even if its just some sample input to figure out shapes. Batched is the default


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

`rl/` the directory for all code

`rl/agent` - folder for RL algorithm classes. `rl/agent/<algo_name>` should have most of its logic fairly self-contained. The only things that are outside are just model definitions.

`rl/models` contains all models (probably torch)

`rl/envs` - folder for any custom environments

`rl/cfg` configuration tooling. Usually doesn't need to be modified unless you want to change the config system.

`scripts/` contains various scripts that use modules in the package to do things, e.g. sample random environment data, train with PPO etc.

`scripts/<algo>` all files related for running an algorithm

`scripts/<algo>/config.py` all relevant configs as python files
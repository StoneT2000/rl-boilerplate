# rl-boilerplate
a repo that I copy for new rl projects

## To Use

clone this repo and make a new repo.

Replace / update the following files

`environment.yml` - change the name and add/remove pkgs

`pkgname` - rename folder to the actual project name

Then `mamba create env` or `conda create env`

## Organization


`cfgs` holds all config files stored as yml files
`cfgs/<name>` holds all config files (and should hold a default of some sort) for a particular script (e.g PPO training script) or a group of experiments (e.g. BC on CartPole env)

`pkgname` the directory for all code

`pkgname/models` contains all models (probably torch)

`pkgname/envs` contains various custom envs / wrapped envs

`pkgname/cfg` configuration tooling

`scripts` contains various scripts (e.g. PPO training, data generation what not)
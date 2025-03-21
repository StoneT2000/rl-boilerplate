import copy
import os
import tyro
from config import PPOTrainConfig, default_config
from rl.envs.make_env.make_env import make_env_from_config
from rl.logger.logger import Logger
from rl.models.builder import build_network_from_cfg
from rl.models.mlp import layer_init
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from dataclasses import asdict

class Agent(nn.Module):
    def __init__(self, sample_obs, sample_act, device=None):
        super().__init__()
        self.critic_feature_net = build_network_from_cfg(sample_obs, config.network.critic)
        self.critic_head = layer_init(nn.Linear(256, 1, device=device))
        self.actor_feature_net = build_network_from_cfg(sample_obs, config.network.actor)
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(n_obs, 256, device=device)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256, device=device)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256, device=device)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        # )
        self.actor_head = layer_init(nn.Linear(256, sample_act.shape[1], device=device))
        self.actor_logstd = nn.Parameter(torch.zeros(1, sample_act.shape[1], device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

def main(config: PPOTrainConfig):
    # background setup and seeding
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    ### Initialize logger ###
    orig_config = copy.deepcopy(config)
    logger = Logger(config.logger)
    # make modifications to config for this training script
    config.eval_env.env_id = config.env.env_id
    config.eval_env.record_episode_kwargs["max_steps_per_video"] = config.train.num_eval_steps

    
    model_path = os.path.join(logger.workspace, logger.exp_name, "models")
    video_path = None if config.eval_env.record_video_path is None else os.path.join(logger.workspace, logger.exp_name, config.eval_env.record_video_path)
    if video_path is not None:
        config.eval_env.record_video_path = video_path

    ### Create Environments ###
    env, env_meta = make_env_from_config(config.env)
    eval_env, eval_env_meta = make_env_from_config(config.eval_env)

    logger.init(asdict(config), orig_config)


    ### Create Agent ###
    agent = Agent(env_meta.sample_obs, env_meta.sample_acts)

    

    import ipdb; ipdb.set_trace()

    
if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_config)
    main(config)


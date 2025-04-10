from collections import defaultdict
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from config import SACNetworkConfig, SACTrainConfig, ms3_configs
import tyro

from rl.envs.make_env.make_env import make_env_from_config
from rl.logger.logger import Logger
from rl.models.builder import build_network_from_cfg
from rl.models.mlp import layer_init
from torchrl.data import LazyTensorStorage, ReplayBuffer
from tensordict import TensorDict

class Agent(nn.Module):
    def __init__(self, config: SACNetworkConfig, sample_obs: torch.Tensor, sample_acts: torch.Tensor, device=None):
        super().__init__()
        if config.shared_backbone is not None:
            self.shared_feature_net, sample_obs = build_network_from_cfg(config.shared_backbone, sample_obs, device=device)
        else:
            self.shared_feature_net = None
        self.critic_feature_net, critic_sample_obs = build_network_from_cfg(config.critic, sample_obs, device=device)
        self.actor_feature_net, actor_sample_obs = build_network_from_cfg(config.actor, sample_obs, device=device)
        
        # self.critic_head = layer_init(nn.Linear(critic_sample_obs.shape[1], 1, device=device))            
        # self.actor_head = layer_init(nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device), std=0.01*np.sqrt(2))
        # self.actor_logstd = nn.Parameter(torch.ones(1, sample_act.shape[1], device=device) * config.init_logstd)

def main(config: SACTrainConfig):
    # background setup and seeding
    config.setup()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    ### Initialize logger ###
    orig_config = copy.deepcopy(config)
    logger = Logger(config.logger)
    
    model_path = os.path.join(logger.workspace, logger.exp_name, "models")
    video_path = None if config.eval_env.record_video_path is None else os.path.join(logger.workspace, logger.exp_name, config.eval_env.record_video_path)
    if video_path is not None:
        config.eval_env.record_video_path = video_path

    ### Create Environments ###
    envs, env_meta = make_env_from_config(config.env)
    eval_envs, eval_env_meta = make_env_from_config(config.eval_env)

    # backfill the max episode steps of the env
    config.env.max_episode_steps = env_meta.max_episode_steps
    config.eval_env.max_episode_steps = eval_env_meta.max_episode_steps

    logger.init(config, orig_config)
    print("Config: ", config)

    ### Load Checkpoint ###
    checkpoint = None
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location=device)

    ### Create Agent ###
    agent = Agent(config.network, env_meta.sample_obs, env_meta.sample_acts, device=device)
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])

    # lazy tensor storage is nice, determines the structure of the data based on the first added data point
    rb = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size, device=device))


    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=config.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=config.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    pbar = tqdm.tqdm(range(config.total_timesteps))
    cumulative_times = defaultdict(float)
    print("=== Starting SAC Training ===")
    for iteration in range(1, config.num_iterations + 1):
        if config.eval_freq > 0 and iteration % config.eval_freq == 1:
            # evaluate
            agent.eval()
            # stime = time.perf_counter()
            # eval_obs, _ = eval_envs.reset()
            # eval_metrics = defaultdict(list)
            # num_episodes = 0
        rollout_time = time.perf_counter()
        for _ in range(config.steps_per_env_per_iteration):
            global_step += 1 * config.env.num_envs

            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = agent.get_action(obs)
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()

            # always bootstrap strategy
            need_final_obs = truncations | terminations # always need final obs when episode ends
            stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                # else: # bootstrap at truncated
                #     need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                #     stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            
            transition = TensorDict({
                "observation": obs,
                "next_observation": real_next_obs,
                "actions": actions,
                "rewards": rewards,
                "stop_bootstrap": stop_bootstrap,
            }, batch_size=1024)
            rb.extend(transition)

            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(config.steps_per_iteration)

        if global_step < config.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(config.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(config.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(ms3_configs)
    main(config)


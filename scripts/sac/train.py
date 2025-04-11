from collections import defaultdict
import copy
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tqdm
from config import SACNetworkConfig, SACTrainConfig, ms3_configs
import tyro
import torch.optim as optim
import torch.nn.functional as F
from rl.envs.make_env.make_env import make_env_from_config
from rl.logger.logger import Logger
from rl.models.builder import build_network_from_cfg
from rl.models.mlp import layer_init
from torchrl.data import LazyTensorStorage, ReplayBuffer
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule


class SoftQNetwork(nn.Module):
    def __init__(self, config: SACNetworkConfig, encoder: nn.Module, x: torch.Tensor, device=None):
        super().__init__()
        self.encoder = encoder
        self.net, x = build_network_from_cfg(config.critic, x, device=device)
        self.head = layer_init(nn.Linear(x.shape[1], 1, device=device))

    def forward(self, obs, a):
        if self.encoder is not None:
            obs = self.encoder(obs)
        x = torch.cat([obs, a], 1)
        return self.head(self.net(x))

class Actor(nn.Module):
    def __init__(self, config: SACNetworkConfig, encoder: nn.Module, sample_obs: torch.Tensor, sample_act: torch.Tensor, single_action_space: gym.spaces.Box, device=None):
        super().__init__()
        self.encoder = encoder
        self.actor_feature_net, actor_sample_obs = build_network_from_cfg(config.actor, sample_obs, device=device)
        self.actor_head = layer_init(nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device))
        self.actor_logstd = layer_init(nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device))

        h, l = single_action_space.high, single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32, device=device))
    
    def forward(self, obs):
        if self.encoder is None:
            actor_features = obs
        else:
            actor_features = self.encoder(obs)
        if self.actor_feature_net is not None:
            actor_features = self.actor_feature_net(actor_features)
        mean = self.actor_head(actor_features)
        log_std = self.actor_logstd(actor_features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def get_eval_action(self, obs):
        if self.encoder is None:
            actor_features = obs
        else:
            actor_features = self.encoder(obs)
        if self.actor_feature_net is not None:
            actor_features = self.actor_feature_net(actor_features)
        mean = self.actor_head(actor_features)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action
    
    def get_action_and_value(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Agent(nn.Module):
    def __init__(self, config: SACNetworkConfig, sample_obs: torch.Tensor, sample_act: torch.Tensor, single_action_space: gym.spaces.Box, device=None):
        super().__init__()
        if config.shared_backbone is not None:
            self.shared_encoder, sample_obs = build_network_from_cfg(config.shared_backbone, sample_obs, device=device)
        else:
            self.shared_encoder = None

        ### set up q networks ###
        
        # for optimization we don't need to save qf1 and qf2, we just save the parameters of the networks
        qf1 = SoftQNetwork(config, self.shared_encoder, torch.cat([sample_obs, sample_act], dim=1), device=device)
        qf2 = SoftQNetwork(config, self.shared_encoder, torch.cat([sample_obs, sample_act], dim=1), device=device)
        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target = self.qnet_params.data.clone()

        # discard params of net
        # use the "meta" torch device to create abstraction, but the actual parameters are held in self.qnet_params (q1, q2) and self.qnet_target
        # this is similar to how jax would work, with separate function (qnet) and parameters (qnet_params, qnet_target)
        self.qnet = SoftQNetwork(config, self.shared_encoder, torch.cat([sample_obs, sample_act], dim=1), device=device)
        self.qnet = self.qnet.to("meta")
        self.qnet_params.to_module(self.qnet)
        
        ### set up actor networks ###
        self.actor = Actor(config, self.shared_encoder, sample_obs, sample_act, single_action_space, device=device)
    

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
    agent = Agent(config.network, env_meta.sample_obs, env_meta.sample_acts, envs.single_action_space, device=device)

    q_optimizer = optim.Adam(agent.qnet.parameters(), lr=config.sac.q_lr, capturable=config.cudagraphs and not config.compile)
    actor_optimizer = optim.Adam(list(agent.actor.parameters()), lr=config.sac.policy_lr, capturable=config.cudagraphs and not config.compile)

    # we don't store actor_detach in the Agent class since we don't need to save a duplicate actor set of weights
    actor_detach = Actor(config.network, agent.shared_encoder, env_meta.sample_obs, env_meta.sample_acts, envs.single_action_space, device=device)
    from_module(agent.actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action_and_value, in_keys=["obs"], out_keys=["action"])
    eval_policy = TensorDictModule(actor_detach.get_eval_action, in_keys=["obs"], out_keys=["action"])

    
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])
        actor_detach.load_state_dict(checkpoint["actor_detach"])
        q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

    # Automatic entropy tuning
    if config.sac.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=config.sac.alpha_lr, capturable=config.cudagraphs and not config.compile)
    else:
        alpha = torch.as_tensor(config.sac.alpha, device=device)

    

    # lazy tensor storage is nice, determines the structure of the data based on the first added data point
    rb = ReplayBuffer(storage=LazyTensorStorage(config.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(agent.qnet):
            vals = agent.qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data):
        # optimize the model
        # NOTE (from arth): we update shared encoder only during critic updates, not actor updates, we detach encoder
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = agent.actor.get_action_and_value(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                agent.qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                ~data["dones"].flatten()
            ).float() * config.sac.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            agent.qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())
    def update_pol(data):
        actor_optimizer.zero_grad()
        # TODO (stao): detach encoder!
        pi, log_pi, _ = agent.actor.get_action_and_value(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(agent.qnet_params.data, data["observations"], pi)
        if config.sac.ensemble_reduction == "min":
            qf = qf_pi.min(0).values
        elif config.sac.ensemble_reduction == "mean":
            qf = qf_pi.mean(0)
        else:
            raise ValueError(f"Invalid ensembling reduction: {config.sac.ensemble_reduction}")
        actor_loss = ((alpha * log_pi) - qf).mean()

        actor_loss.backward()
        actor_optimizer.step()

        if config.sac.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = agent.actor.get_action_and_value(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach())

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(config.batch_size)

    is_extend_compiled = False
    if config.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if config.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)


    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=config.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=config.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    pbar = tqdm.tqdm(range(config.total_timesteps))
    cumulative_times = defaultdict(float)
    start_time = time.perf_counter()
    print("=== Starting SAC Training ===")
    for iteration in range(1, config.num_iterations + 1):
        if config.eval_freq > 0 and iteration % config.eval_freq == 1:
            agent.eval()
            stime = time.perf_counter()
            
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(config.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_policy(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            agent.train()
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
        rollout_time = time.perf_counter()
        for _ in range(config.steps_per_env_per_iteration):
            global_step += 1 * config.env.num_envs

            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions = policy(obs)
                # actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()

            # always bootstrap strategy
            need_final_obs = truncations | terminations # always need final obs when episode ends
            dones = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
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
                "observations": obs,
                "next_observations": real_next_obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
            }, batch_size=config.env.num_envs)
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
            data = rb.sample(batch_size=config.batch_size)
            metrics = update_main(data)

            if global_update % config.sac.policy_frequency == 0:
                metrics.update(update_pol(data))
                alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % config.sac.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                agent.qnet_target.lerp_(agent.qnet_params.data, config.sac.tau)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        if iteration % config.logger_freq == 1:
            for k, v in metrics.items():
                logger.add_scalar(f"losses/{k}", v, global_step)
            logger.add_scalar("time/SPS", int(global_step / (time.perf_counter() - start_time)), global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", config.steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(ms3_configs)
    main(config)


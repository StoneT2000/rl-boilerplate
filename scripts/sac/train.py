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
from .config import SACNetworkConfig, SACTrainConfig, ms3_configs
import tyro
import torch.optim as optim
import torch.nn.functional as F
from rl.envs.make_env.make_env import make_env_from_config
from rl.logger.logger import Logger
from rl.models.builder import build_network_from_cfg
from torchrl.data import LazyTensorStorage, ReplayBuffer
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class SoftQNetwork(nn.Module):
    def __init__(self, config: SACNetworkConfig, x: torch.Tensor, sample_act: torch.Tensor, device=None):
        super().__init__()
        self.net, x = build_network_from_cfg(config.critic, torch.cat([x, sample_act], dim=1), device=device)
        self.head = nn.Linear(x.shape[1], 1, device=device)
        
        self.apply(weight_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.head(self.net(x))

class Actor(nn.Module):
    def __init__(self, config: SACNetworkConfig, log_std_range: tuple[float, float], sample_q_obs: torch.Tensor, sample_act: torch.Tensor, single_action_space: gym.spaces.Box, device=None):
        super().__init__()
        self.actor_feature_net, actor_sample_obs = build_network_from_cfg(config.actor, sample_q_obs, device=device)
        self.actor_head = nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device)
        self.actor_logstd = nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device)

        h, l = single_action_space.high, single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32, device=device))
        self._logstd_min = log_std_range[0]
        self._logstd_max = log_std_range[1]

        self.apply(weight_init)
    
    def forward(self, actor_features):
        actor_features = self.actor_feature_net(actor_features)
        mean = self.actor_head(actor_features)
        log_std = self.actor_logstd(actor_features)
        log_std = torch.tanh(log_std)
        log_std = self._logstd_min + 0.5 * (self._logstd_max - self._logstd_min) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def get_eval_action(self, actor_features):
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

class Agent(nn.Module):
    def __init__(self, config: SACTrainConfig, sample_obs: torch.Tensor, sample_act: torch.Tensor, single_action_space: gym.spaces.Box, device=None):
        super().__init__()
        if config.network.shared_backbone is not None:
            self.shared_encoder, _ = build_network_from_cfg(config.network.shared_backbone, sample_obs, device=device)
            self.shared_encoder.apply(weight_init)
            self.target_encoder, _ = build_network_from_cfg(config.network.shared_backbone, sample_obs, device=device)
            self.target_encoder.load_state_dict(self.shared_encoder.state_dict())
        else:
            self.shared_encoder = nn.Identity()
            self.target_encoder = nn.Identity()
        with torch.no_grad():
            sample_q_obs = self.shared_encoder(sample_obs)

            ### set up q networks ###
            
            # for optimization we don't need to save qfs, we just save the parameters of the networks
            qfs = [SoftQNetwork(config.network, sample_q_obs, sample_act, device=device) for _ in range(config.sac.num_q)]
            self.qnet_params = from_modules(*qfs, as_module=True)
            self.qnet_target = self.qnet_params.data.clone()

            # discard params of net
            # use the "meta" torch device to create abstraction, but the actual parameters are held in self.qnet_params (q1, q2) and self.qnet_target
            # this is similar to how jax would work, with separate function (qnet) and parameters (qnet_params, qnet_target)
            self.qnet = SoftQNetwork(config.network, sample_q_obs, sample_act, device=device)
            self.qnet = self.qnet.to("meta")
            self.qnet_params.to_module(self.qnet)
            
            ### set up actor networks ###
            self.actor = Actor(config.network, (config.sac.log_std_min, config.sac.log_std_max), sample_q_obs, sample_act, single_action_space, device=device)
        

def main(config: SACTrainConfig):
    # background setup and seeding
    config.setup()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    buffer_device = torch.device("cuda" if torch.cuda.is_available() and config.buffer_cuda else "cpu")

    ### Initialize logger ###
    orig_config = copy.deepcopy(config)
    logger = Logger(config.logger)
    
    model_path = os.path.join(logger.workspace, logger.exp_name, "models")
    video_path = None if config.eval_env.record_video_path is None else os.path.join(logger.workspace, logger.exp_name, config.eval_env.record_video_path)
    if video_path is not None:
        config.eval_env.record_video_path = video_path

    ### Create Environments ###
    envs, env_meta = make_env_from_config(config.env)
    if config.eval_freq > 0:
        eval_envs, eval_env_meta = make_env_from_config(config.eval_env)

    # backfill the max episode steps of the env
    config.env.max_episode_steps = env_meta.max_episode_steps
    if config.eval_freq > 0:
        config.eval_env.max_episode_steps = eval_env_meta.max_episode_steps

    logger.init(config, orig_config)
    print("Config: ", config)

    ### Load Checkpoint ###
    checkpoint = None
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location=device)

    ### Create Agent ###
    agent = Agent(config, env_meta.sample_obs, env_meta.sample_acts, envs.single_action_space, device=device)
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])

    q_optimizer = optim.Adam(list(agent.qnet.parameters()) + list(agent.shared_encoder.parameters()), lr=config.sac.q_lr, capturable=config.cudagraphs and not config.compile)
    actor_optimizer = optim.Adam(list(agent.actor.parameters()), lr=config.sac.policy_lr, capturable=config.cudagraphs and not config.compile)

    # we don't store actor_detach in the Agent class since we don't need to save a duplicate actor set of weights
    actor_detach = Actor(config.network, (config.sac.log_std_min, config.sac.log_std_max), agent.shared_encoder(env_meta.sample_obs).detach(), env_meta.sample_acts, envs.single_action_space, device=device)
    from_module(agent.actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action_and_value, in_keys=["obs"], out_keys=["action"])
    eval_policy = TensorDictModule(actor_detach.get_eval_action, in_keys=["obs"], out_keys=["action"])

    # compile encoders
    if config.network.shared_backbone is not None:
        shared_encoder_detach, _ = build_network_from_cfg(config.network.shared_backbone, env_meta.sample_obs, device=device)
        from_module(agent.shared_encoder).data.to_module(shared_encoder_detach)
        shared_encoder = TensorDictModule(shared_encoder_detach.forward, in_keys=["obs"], out_keys=["obs"])
    else:
        shared_encoder = nn.Identity()
    
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
    # NOTE (arth): when rb is on cpu, rb.sample() and cpu->gpu are both big bottlenecks. pin_memory makes 
    #       cpu->gpu negligible, prefetch loads batches async to make rb.sample() faster. downside of prefetch
    #       is it will data which is 1 iter stale, but doesn't seem to affect training noticeably negatively
    rb_cpu_to_gpu = not config.buffer_cuda and config.cuda
    rb = ReplayBuffer(
        storage=LazyTensorStorage(config.buffer_size, device=buffer_device),
        batch_size=config.batch_size,
        prefetch=min(4, config.grad_steps_per_iteration) if rb_cpu_to_gpu else None,
        pin_memory=rb_cpu_to_gpu,
    )

    # NOTE (arth): removed encoder from batched_qf, instead encode once and reuse for each qf
    #       typically batchnorm, dropout, etc is not used for these encoders
    def batched_qf(params, encoded_obs, action, next_q_value=None):
        with params.to_module(agent.qnet):
            vals = agent.qnet(encoded_obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data):
        # optimize the model
        # NOTE (from arth): we update shared encoder only during critic updates, not actor updates, we detach encoder
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = agent.actor.get_action_and_value(agent.shared_encoder(data["next_observations"]))

            # randomly select min_q number of qfs
            selected_indices = torch.randperm(config.sac.num_q)[:config.sac.min_q]
            subset_qnet_target = agent.qnet_target[selected_indices]
            qf_next_target = torch.vmap(batched_qf, (0, None, None, None))(
                subset_qnet_target, agent.target_encoder(data["next_observations"]), next_state_actions, None
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                ~data["dones"].flatten()
            ).float() * config.sac.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            agent.qnet_params, agent.shared_encoder(data["observations"]), data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())
    def update_pol(data):
        actor_optimizer.zero_grad()
        with torch.no_grad():
            encoded_obs = agent.shared_encoder(data["observations"])
        pi, log_pi, _ = agent.actor.get_action_and_value(encoded_obs)
        qf_pi = torch.vmap(batched_qf, (0, None, None, None))(agent.qnet_params.data, encoded_obs, pi, None)
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
                _, log_pi, _ = agent.actor.get_action_and_value(encoded_obs)
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach())

    is_extend_compiled = False
    if config.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)
        # eval_policy = torch.compile(eval_policy, mode=mode)
        shared_encoder = torch.compile(shared_encoder, mode=mode)

    if config.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)
        # shared_encoder = CudaGraphModule(shared_encoder)


    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=config.seed) # in Gymnasium, seed is given to reset() instead of seed()
    if config.eval_freq > 0:
        eval_obs, _ = eval_envs.reset(seed=config.seed)

    init_transition = TensorDict({
        "next_observations": obs,
        "actions": torch.zeros((env_meta.num_envs, *env_meta.sample_acts.shape[1:]), dtype=torch.float32),
        "rewards": torch.zeros(env_meta.num_envs, dtype=torch.float32),
        "dones": torch.zeros(env_meta.num_envs, dtype=torch.bool),
    }, batch_size=config.env.num_envs)
    rb.extend(init_transition.to(buffer_device))

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
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_policy(shared_encoder(eval_obs)))
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
            if config.save_model:
                torch.save(
                    dict(
                        agent=agent.state_dict(),
                    ),
                    os.path.join(model_path, f"sac_model_{global_step}.pth")
                )
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
                if config.network.shared_backbone is None:
                    actions = policy(obs)
                else:
                    actions = policy(shared_encoder(obs=obs)) # for some reason obs=obs is needed here, but not later

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            if isinstance(next_obs, dict):
                next_obs = TensorDict(next_obs, batch_size=config.env.num_envs)
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
                if isinstance(infos["final_observation"], dict):
                    infos["final_observation"] = TensorDict(infos["final_observation"], batch_size=config.env.num_envs)
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            
            transition = TensorDict({
                "next_observations": real_next_obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
            }, batch_size=config.env.num_envs)
            
            rb.extend(transition.to(buffer_device))

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
            data, info = rb.sample(batch_size=config.batch_size, return_info=True)
            # fetch observations o_t given the index of o_t+1. This works because the replay buffer stores data in shape (N, ...) for N samples, and appends
            # B environment frames of data directly each time we add to buffer, so the previous B samples correspond with the previous frame.
            data["observations"] = rb[info["index"] - env_meta.num_envs]["next_observations"]
            metrics = update_main(data.to(device))

            if global_update % config.sac.policy_frequency == 0:
                metrics.update(update_pol(data))
                alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_update % config.sac.target_network_frequency == 0:
                with torch.no_grad():
                    for target_param, online_param in zip(agent.target_encoder.parameters(), agent.shared_encoder.parameters()):
                        target_param.lerp_(online_param, config.sac.tau)

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
            if config.max_time_s is not None:
                if sum(cumulative_times.values()) > config.max_time_s:
                    print(f"=== Exceeded {config.max_time_s} seconds, ending training early ===")
                    break
            

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(ms3_configs)
    main(config)


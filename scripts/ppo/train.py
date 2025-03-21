import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

from collections import defaultdict
import gymnasium as gym
import copy
import random
import time
import tqdm
import tyro

import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim

from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.normal import Normal


from config import PPOTrainConfig, default_config, PPONetworkConfig
from rl.envs.make_env.make_env import make_env_from_config
from rl.logger.logger import Logger
from rl.models.builder import build_network_from_cfg
from rl.models.mlp import layer_init


class Agent(nn.Module):
    def __init__(self, config: PPONetworkConfig, sample_obs: torch.Tensor, sample_act: torch.Tensor, device=None):
        super().__init__()
        if config.shared_backbone is not None:
            self.shared_feature_net, sample_obs = build_network_from_cfg(sample_obs, config.shared_backbone, device=device)
        else:
            self.shared_feature_net = None
        self.critic_feature_net, critic_sample_obs = build_network_from_cfg(sample_obs, config.critic, device=device)
        self.actor_feature_net, actor_sample_obs = build_network_from_cfg(sample_obs, config.actor, device=device)
        
        self.critic_head = layer_init(nn.Linear(critic_sample_obs.shape[1], 1, device=device))            
        self.actor_head = layer_init(nn.Linear(actor_sample_obs.shape[1], sample_act.shape[1], device=device), std=0.01*np.sqrt(2))
        self.actor_logstd = nn.Parameter(torch.zeros(1, sample_act.shape[1], device=device))
        import ipdb; ipdb.set_trace()
    def get_value(self, x):
        if self.shared_feature_net is None:
            critic_features = x
        else:
            critic_features = self.shared_feature_net(x)
        if self.critic_feature_net is not None:
            critic_features = self.critic_feature_net(critic_features)
        return self.critic_head(critic_features)
    
    def get_eval_action(self, obs: torch.Tensor):
        if self.shared_feature_net is None:
            actor_features = obs
        else:
            actor_features = self.shared_feature_net(obs)
        if self.actor_feature_net is not None:
            actor_features = self.actor_feature_net(actor_features)
        return self.actor_head(actor_features)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        if self.shared_feature_net is None:
            actor_features = obs
            critic_features = obs
        else:
            actor_features = self.shared_feature_net(obs)
            critic_features = actor_features
        if self.actor_feature_net is not None:
            actor_features = self.actor_feature_net(actor_features)
        if self.critic_feature_net is not None:
            critic_features = self.critic_feature_net(critic_features)

        action_mean = self.actor_head(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic_head(critic_features)

def main(config: PPOTrainConfig):
    # background setup and seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    ### Initialize logger ###
    orig_config = copy.deepcopy(config)
    logger = Logger(config.logger)

    # make modifications to config for this training script
    batch_size = int(config.env.num_envs * config.num_steps)
    config.minibatch_size = batch_size // config.ppo.num_minibatches
    config.batch_size = config.ppo.num_minibatches * config.minibatch_size
    config.num_iterations = config.total_timesteps // config.batch_size
    config.env.seed = config.seed
    config.eval_env.seed = config.seed
    
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

    # Make a version of agent with detached params
    agent_inference = Agent(config.network, env_meta.sample_obs, env_meta.sample_acts, device=device)
    agent_inference_p = from_module(agent).data # type: ignore
    agent_inference_p.to_module(agent_inference) # type: ignore

    ### Optimizer ###
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(config.ppo.learning_rate, device=device),
        eps=1e-5,
        capturable=config.cudagraphs and not config.compile,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    ### Executables ###
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # define gae, update, and rollout functions
    def gae(next_obs, next_done, container, final_values):
        # bootstrap value if not done
        next_value = get_value(next_obs).reshape(-1)
        lastgaelam = 0
        nextnonterminals = (~container["dones"]).float().unbind(0)
        vals = container["vals"]
        vals_unbind = vals.unbind(0)
        rewards = container["rewards"].unbind(0)

        advantages = []
        nextnonterminal = (~next_done).float()
        nextvalues = next_value
        for t in range(config.num_steps - 1, -1, -1):
            cur_val = vals_unbind[t]
            # real_next_values = nextvalues * nextnonterminal
            real_next_values = nextnonterminal * nextvalues + final_values[t] # t instead of t+1
            delta = rewards[t] + config.ppo.gamma * real_next_values - cur_val
            advantages.append(delta + config.ppo.gamma * config.ppo.gae_lambda * nextnonterminal * lastgaelam)
            lastgaelam = advantages[-1]

            nextnonterminal = nextnonterminals[t]
            nextvalues = cur_val

        advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
        container["returns"] = advantages + vals
        return container

    def step_func(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info
    def rollout(obs, done):
        ts = []
        final_values = torch.zeros((config.num_steps, config.env.num_envs), device=device)
        for step in range(config.num_steps):
            # ALGO LOGIC: action logic
            action, logprob, _, value = policy(obs=obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, infos = step_func(action)

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(config.env.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)

            ts.append(
                tensordict.TensorDict._new_unsafe(
                    obs=obs,
                    # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                    dones=done,
                    vals=value.flatten(),
                    actions=action,
                    logprobs=logprob,
                    rewards=reward,
                    batch_size=(config.env.num_envs,),
                )
            )
            # NOTE (stao): change here for gpu env
            obs = next_obs = next_obs
            done = next_done
        # NOTE (stao): need to do .to(device) i think? otherwise container.device is None, not sure if this affects anything
        container = torch.stack(ts, 0).to(device)
        return next_obs, done, container, final_values


    def update(obs, actions, logprobs, advantages, returns, vals):
        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config.ppo.clip_coef).float().mean()

        if config.ppo.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - config.ppo.clip_coef, 1 + config.ppo.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if config.ppo.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = vals + torch.clamp(
                newvalue - vals,
                -config.ppo.clip_coef,
                config.ppo.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config.ppo.ent_coef * entropy_loss + v_loss * config.ppo.vf_coef

        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), config.ppo.max_grad_norm)
        optimizer.step()

        return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


    update = tensordict.nn.TensorDictModule(
        update,
        in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
        out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
    )

    ### Compile ###
    if config.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if config.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)
    
    global_step = 0
    start_time = time.time()
    container_local = None
    next_obs = envs.reset()[0]
    next_done = torch.zeros(config.env.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, config.num_iterations + 1))
    cumulative_times = defaultdict(float)

    print("=== Starting PPO Training ===")
    for iteration in pbar:
        agent.eval()
        if iteration % config.eval_freq == 1:
            stime = time.perf_counter()
            
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(config.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_eval_action(eval_obs))
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
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            # agent.train()

        if config.save_model and iteration % config.eval_freq == 1:
            torch.save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict()),  os.path.join(model_path, f"ckpt_{iteration}.pt"))
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if config.ppo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.ppo.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        update_time = time.perf_counter()
        container = gae(next_obs, next_done, container, final_values)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(config.ppo.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(config.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                clipfracs.append(out["clipfrac"])
                if config.ppo.target_kl is not None and out["approx_kl"] > config.ppo.target_kl:
                    break
            else:
                continue
            break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # log all the things
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
        logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
        logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
        logger.add_scalar("losses/old_approx_kl", out["old_approx_kl"].item(), global_step)
        logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
        logger.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().cpu().item(), global_step)
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", config.env.num_envs * config.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if config.save_model:
        torch.save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict()), os.path.join(model_path, "final_ckpt.pt"))
        print(f"model saved to {model_path}")
    logger.close()
    envs.close()
    eval_envs.close()
    
if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_config)
    main(config)


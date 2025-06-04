from rl.envs.make_env.make_env import EnvConfig
from rl.logger.logger import LoggerConfig
from rl.models.types import NetworkConfig
from dataclasses import dataclass, field

@dataclass
class PPONetworkConfig:
    shared_backbone: NetworkConfig | None = None
    """the shared backbone network. Data is first processed through this network before being passed to the actor and critic networks."""
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    """the actor network"""
    critic: NetworkConfig = field(default_factory=NetworkConfig)
    """the critic network"""
    init_logstd: float = 0
    """the initial log standard deviation for the actor network"""
    
from dataclasses import dataclass

@dataclass
class PPOHyperparametersConfig:
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches per update epoch"""
    update_epochs: int = 8
    """the number of epochs to update the policy after a rollout"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False
@dataclass
class PPOTrainConfig:
    seed: int
    """seed for the experiment"""
    env: EnvConfig
    """environment configurations"""
    eval_env: EnvConfig
    """evaluation environment configurations. You should only modify at most num_eval_envs, ignore_terminations, record_video_path, record_episode_kwargs. The rest are copied from env.env_kwargs"""
    network: PPONetworkConfig = field(default_factory=PPONetworkConfig)
    """actor critic neural net configurations"""
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """logger configurations"""
    ppo: PPOHyperparametersConfig = field(default_factory=PPOHyperparametersConfig)
    """ppo hyperparameters"""

    total_timesteps: int = 100_000_000
    """total timesteps to train for"""
    eval_freq: int = 25
    """evaluation frequency in terms of training iterations"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint: str | None = None
    """path to a pretrained checkpoint file to start training from"""



    cudagraphs: bool = False
    """whether to use cudagraphs"""
    compile: bool = False
    """whether to use torch.compile"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def __post_init__(self):
        self.eval_env.env_id = self.env.env_id

        # self.eval_env.env_kwargs = dict(**self.env.env_kwargs)
        # any kwargs in env.env_kwargs that are not in eval_env.env_kwargs will be added to eval_env.env_kwargs
        for k, v in self.env.env_kwargs.items():
            if k not in self.eval_env.env_kwargs:
                print(f"Adding {k} to eval_env.env_kwargs from env.env_kwargs")
                self.eval_env.env_kwargs[k] = v
        
        self.eval_env.record_episode_kwargs["max_steps_per_video"] = self.num_eval_steps
        batch_size = int(self.env.num_envs * self.num_steps)
        self.minibatch_size = batch_size // self.ppo.num_minibatches
        self.batch_size = self.ppo.num_minibatches * self.minibatch_size
        self.num_iterations = self.total_timesteps // self.batch_size
        self.env.seed = self.seed
        self.eval_env.seed = self.seed

    def setup(self):
        """run any code before any of the PPO training script runs"""
        pass

try:
    import mani_skill.envs
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    ms3_configs = {
        "ms3-state": (
            "ManiSkill3 State based PPO training",
            PPOTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=4096,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="state",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
                        # not sure how to permit other changes to this in CLI, waiting on https://github.com/brentyi/tyro/issues/277
                    )
                ),
                eval_env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=16,
                    vectorization_method="gpu",
                    ignore_terminations=True,
                    env_kwargs=dict(
                        obs_mode="state",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=1,
                        render_mode="rgb_array",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    )
                ),
                network=PPONetworkConfig(
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="tanh",
                            output_activation="tanh",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="tanh",
                            output_activation="tanh",
                        ),
                    )
                ),
                num_steps=4,
                total_timesteps=10_000_000,
            )
        ),
        "ms3-rgb": (
            "ManiSkill3 RGB based PPO training",
            PPOTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=1024,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
                        # not sure how to permit other changes to this in CLI, waiting on https://github.com/brentyi/tyro/issues/277
                    ),
                    wrappers=[FlattenRGBDObservationWrapper]
                ),
                eval_env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=16,
                    vectorization_method="gpu",
                    ignore_terminations=True,
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=1,
                        render_mode="all",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    ),
                    wrappers=[FlattenRGBDObservationWrapper]
                ),
                network=PPONetworkConfig(
                    init_logstd=-0.5,
                    shared_backbone=NetworkConfig(
                        type="nature_cnn",
                        arch_cfg=dict(
                            state_features=[256],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[512],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[512],
                            activation="relu",
                            output_activation="relu",
                        ),
                    )
                ),
                num_steps=16,
                total_timesteps=20_000_000,
                ppo=PPOHyperparametersConfig(
                    target_kl=0.2,
                )
            )
        )
    }
except:
    pass
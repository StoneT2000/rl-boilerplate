from dataclasses import dataclass, field
from rl.envs.make_env.make_env import EnvConfig
from rl.logger.logger import LoggerConfig
from rl.models.types import NetworkConfig


@dataclass
class SACNetworkConfig:
    shared_backbone: NetworkConfig | None = None
    """the shared backbone network. Data is first processed through this network before being passed to the actor and critic networks."""
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    """the actor network"""
    critic: NetworkConfig = field(default_factory=NetworkConfig)
    """the critic network"""

@dataclass
class SACHyperparametersConfig:
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2
    """the frequency of updates for the target nerworks"""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    alpha_lr: float = 3e-4
    """the learning rate of the entropy coefficient optimizer"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    ensemble_reduction: str = "min"
    """the reduction to use for ensembling the Q-values when updating the actor. min is the original SAC implementation, mean function is used by REDQ"""
    log_std_max: float = 2.0
    """the maximum value of the log std"""
    log_std_min: float = -5.0
    """the minimum value of the log std"""

@dataclass
class SACTrainConfig:
    seed: int
    """seed for the experiment"""
    env: EnvConfig
    """environment configurations"""
    eval_env: EnvConfig
    """evaluation environment configurations. You should only modify at most num_eval_envs, ignore_terminations, record_video_path, record_episode_kwargs. The rest are copied from env.env_kwargs"""
    network: SACNetworkConfig = field(default_factory=SACNetworkConfig)
    """actor critic neural net configurations"""
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """logger configurations"""
    sac: SACHyperparametersConfig = field(default_factory=SACHyperparametersConfig)
    """sac hyperparameters"""

    total_timesteps: int = 10_000_000
    """total timesteps to train for"""
    steps_per_env_per_iteration: int = 1
    """number of steps per environment per training iteration (also known as number of parallel env steps)"""
    grad_steps_per_iteration: int = 10
    """number of gradient steps per training iteration"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 1024 * 128
    """timestep to start learning"""
    
    # to be filled in runtime
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    steps_per_iteration: int = 0
    """number of environment samples collected per training iteration (computed in runtime)"""
    utd: float = 0.0
    """the update to data ratio (computed in runtime)"""
    per_env_buffer_size: int = 0
    """the replay memory buffer size per environment (computed in runtime)"""

    logger_freq: int = 2
    """logger frequency in terms of training iterations. Does not affect logging of training episode metrics"""
    eval_freq: int = 1000
    """evaluation frequency in terms of training iterations"""

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
    buffer_cuda: bool = True
    """if toggled, the replay buffer will be stored on GPU"""


    def __post_init__(self):
        self.eval_env.env_id = self.env.env_id

        # self.eval_env.env_kwargs = dict(**self.env.env_kwargs)
        # any kwargs in env.env_kwargs that are not in eval_env.env_kwargs will be added to eval_env.env_kwargs
        for k, v in self.env.env_kwargs.items():
            if k not in self.eval_env.env_kwargs:
                self.eval_env.env_kwargs[k] = v
        
        self.eval_env.record_episode_kwargs["max_steps_per_video"] = self.num_eval_steps

        self.steps_per_iteration = self.steps_per_env_per_iteration * self.env.num_envs
        self.utd = self.grad_steps_per_iteration / self.steps_per_iteration
        self.per_env_buffer_size = self.buffer_size // self.env.num_envs
        self.num_iterations = self.total_timesteps // self.steps_per_iteration

    def setup(self):
        """run any code before any of the SAC training script runs"""
        pass


try:
    import mani_skill.envs
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    ms3_configs = {
        "ms3-state": (
            "ManiSkill3 State based SAC training",
            SACTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=1024,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="state",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
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
                network=SACNetworkConfig(
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                            use_layer_norm=True,
                        ),
                    )
                ),
                total_timesteps=10_000_000,
                learning_starts=1024 * 128,
                buffer_size=100_000,
                batch_size=4096,
                steps_per_env_per_iteration=1,
                grad_steps_per_iteration=10,
            )
        ),
        "ms3-rgb-ddpg": (
            "ManiSkill3 RGB based SAC training",
            SACTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=256,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
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
                        render_mode="rgb_array",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    wrappers=[FlattenRGBDObservationWrapper],
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    )
                ),
                network=SACNetworkConfig(
                    shared_backbone=NetworkConfig(
                        type="ddpg_cnn",
                        arch_cfg=dict(activation="relu"),
                    ),
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                            use_layer_norm=True,
                        ),
                    )
                ),
                total_timesteps=20_000_000,
                learning_starts=1024 * 32,
                buffer_size=100_000,
                batch_size=1024,
                steps_per_env_per_iteration=1,
                grad_steps_per_iteration=10,
                buffer_cuda=False,
            )
        ),
        "ms3-rgb-nature": (
            "ManiSkill3 RGB based SAC training",
            SACTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=256,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
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
                        render_mode="rgb_array",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    wrappers=[FlattenRGBDObservationWrapper],
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    )
                ),
                network=SACNetworkConfig(
                    shared_backbone=NetworkConfig(
                        type="nature_cnn_proj",
                        arch_cfg=dict(activation="relu"),
                    ),
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="relu",
                            output_activation="relu",
                            use_layer_norm=True,
                        ),
                    )
                ),
                total_timesteps=20_000_000,
                learning_starts=1024 * 32,
                buffer_size=100_000,
                batch_size=1024,
                steps_per_env_per_iteration=1,
                grad_steps_per_iteration=10,
                buffer_cuda=False,
            )
        )
    }
except ImportError:
    ms3_configs = {}

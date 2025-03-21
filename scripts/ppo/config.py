from rl.agents.ppo.config import PPOConfig
from rl.envs.make_env.make_env import EnvConfig
from rl.logger.logger import LoggerConfig
from rl.models.types import NetworkConfig
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    total_timesteps: int
    """total timesteps to train for"""
    eval_freq: int = 25
    """evaluation frequency in terms of training iterations"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""


    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint: str | None = None
    """path to a pretrained checkpoint file to start training from"""

@dataclass
class PPONetworkConfig:
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    critic: NetworkConfig = field(default_factory=NetworkConfig)

@dataclass
class PPOTrainConfig:
    seed: int
    """seed for the experiment"""
    env: EnvConfig
    """environment configurations"""
    eval_env: EnvConfig
    """evaluation environment configurations. You should only modify at most num_eval_envs, ignore_terminations, record_video_path, record_episode_kwargs. The rest are copied from env.env_kwargs"""
    train: TrainConfig
    """training configurations"""
    network: PPONetworkConfig = field(default_factory=PPONetworkConfig)
    """actor critic neural net configurations"""
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """logger configurations"""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    """ppo hyperparameters"""
    cudagraphs: bool = False
    """whether to use cudagraphs"""
    compile: bool = False
    """whether to use torch.compile"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    def __post_init__(self):
        self.eval_env.env_id = self.env.env_id

        # self.eval_env.env_kwargs = dict(**self.env.env_kwargs)
        # any kwargs in env.env_kwargs that are not in eval_env.env_kwargs will be added to eval_env.env_kwargs
        for k, v in self.env.env_kwargs.items():
            if k not in self.eval_env.env_kwargs:
                self.eval_env.env_kwargs[k] = v

default_config = {
    "ms3-state": (
        "ManiSkill3 State based PPO training",
        PPOTrainConfig(
            seed=0,
            env=EnvConfig(
                env_id="PickCube-v1",
                num_envs=16,
                vectorization_method="gpu",
                ignore_terminations=False, # partial resets
                env_kwargs=dict(
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
                    sim_backend="physx_cuda",
                    reconfiguration_freq=1,
                    render_mode="rgb_array",
                    human_render_camera_configs=dict(shader_pack="default")
                ),
                record_video_path="videos",
            ),
            train=TrainConfig(
                total_timesteps=10_000_000,
            ),
            network=PPONetworkConfig(
                actor=NetworkConfig(
                    type="mlp",
                    arch_cfg=dict(
                        features=[256, 256],
                        activation="tanh",
                        output_activation="tanh",
                    ),
                ),
                critic=NetworkConfig(
                    type="mlp",
                    arch_cfg=dict(
                        features=[256, 256],
                        activation="tanh",
                        output_activation="tanh",
                    ),
                )
            ),
            
        )
    )
}
from dataclasses import dataclass, field

import yaml

from rl.agents.ppo.config import PPOConfig
from rl.envs.make_env.make_env import EnvConfig
from rl.logger.logger import LoggerConfig
from rl.models.types import NetworkConfig
import tyro

@dataclass
class TrainConfig:
    total_timesteps: int
@dataclass
class PPONetworkConfig:
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    critic: NetworkConfig = field(default_factory=NetworkConfig)

@dataclass
class Args:
    seed: int
    env: EnvConfig
    train: TrainConfig
    network: PPONetworkConfig = field(default_factory=PPONetworkConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    algo: str = "ppo"

def main(args: Args):
    pass

if __name__ == "__main__":
    default_config = yaml.safe_load(open("configs/ppo/default.yml", "r"))
    import ipdb; ipdb.set_trace()
    args = tyro.cli(Args, default=default_config)
    main(args)

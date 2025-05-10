from rl.models.mlp import MLP
from rl.models.types import NetworkConfig
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class DDPGCNNArchConfig:
    activation: nn.Module | str | None = "relu"
    """internal activation"""
    output_activation: nn.Module | str | None = None
    """activation after final layer, default is None meaning no activation"""

@dataclass
class DDPGCNNConfig(NetworkConfig):
    type = "ddpg_cnn"
    arch_cfg: DDPGCNNArchConfig


class DDPGCNN(nn.Module):
    def __init__(self, config: DDPGCNNConfig, sample_obs: torch.Tensor, device: torch.device | None = None):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding="valid",
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding="valid",
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding="valid",
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding="valid",
                device=device,
            ),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2)).shape[1]
            extractors["rgb"] = nn.Sequential(
                cnn,
                nn.Linear(n_flatten, feature_size, device=device),
                nn.LayerNorm(feature_size, device=device),
                nn.Tanh(),
            )

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Sequential(
                nn.Linear(state_size, feature_size, device=device),
                nn.LayerNorm(feature_size, device=device),
                nn.Tanh(),
            )

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)
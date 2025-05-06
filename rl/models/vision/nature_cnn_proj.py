from rl.models.mlp import MLP
from rl.models.types import NetworkConfig
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class NatureCNNProjArchConfig:
    activation: nn.Module | str | None = "relu"
    """internal activation"""

@dataclass
class NatureCNNProjConfig(NetworkConfig):
    type = "nature_cnn_proj"
    arch_cfg: NatureCNNProjArchConfig


class NatureCNNProj(nn.Module):
    def __init__(self, config: NatureCNNProjConfig, sample_obs: torch.Tensor, device: torch.device | None = None):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = sample_obs["rgb"].shape[1:3]
        assert image_size == (128, 128), "nature_cnn type currently only supports 128x128 images"
        # here we use a NatureCNNProj architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
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
from rl.cfg import parse_cfg
import os.path as osp
import numpy as np
import torch

from rl.envs.make_env.make_env import make_env
if __name__ == "__main__":
    parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "../configs/template.yml")) # auto parses CLI as well
    env, env_meta = make_env(
        env_id="PickCube-v1",
        vectorization_method="gpu",
        num_envs=1,
        seed=0,
        env_kwargs={"render_mode": "rgb_array"},
        record_video_path="videos",
    )
    for _ in range(100):
        env.step(env_meta.sample_acts)
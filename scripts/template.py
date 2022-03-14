from pkgname.cfg import parse_cfg
import os.path as osp
import numpy as np
import torch
if __name__ == "__main__":
    parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "../cfgs/template.yml")) # auto parses CLI as well
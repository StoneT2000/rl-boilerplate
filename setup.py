from setuptools import setup, find_packages

setup(
    name="rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dacite",
        "tyro",
        "gymnasium==0.29.1",
        "wandb",
        "tensorboard",
        "torch",
        "torchrl",
        "tensordict",
        "tqdm"
    ],
)
from rl.agents.base import BasePolicy


class PPO(BasePolicy):
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.config = config

    def train(self):
        pass

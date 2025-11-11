from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

class EveryNStepsCallback(BaseCallback):
    def __init__(self, n_steps: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.n_steps = n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.n_steps == 0:
            if self.verbose > 0:
                ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean")
                if ep_rew is not None:
                    print(f"[{self.num_timesteps}] mean_ep_rew={ep_rew:.2f}")
        return True


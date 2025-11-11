import gymnasium as gym
from typing import Optional, Tuple, Callable
from .wrappers import apply_common_wrappers

def make_env(
    env_id: str = "CarRacing-v3",
    seed: int = 0,
    render_mode: Optional[str] = None,   # "human" | "rgb_array" | None
    resize: Optional[Tuple[int, int]] = (84, 84),
    gray: bool = True,
    reward_shaping: Optional[dict] = None,  # ← 트랙이탈 페널티 설정 dict
) -> Callable[[], gym.Env]:
    def _thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = apply_common_wrappers(env, resize=resize, gray=gray, reward_shaping=reward_shaping)
        env.reset(seed=seed)
        return env
    return _thunk

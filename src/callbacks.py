# src/callbacks.py
import os, time
from pathlib import Path
from typing import Optional
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder

from .make_env import make_env


class LiveRenderCallback(BaseCallback):
    """
    학습 중 일정 간격으로 별도 env에서 human 렌더로 짧게 플레이를 보여줍니다.
    - 주의: GUI가 없는 서버에서는 사용 불가 (대신 VideoRecord 콜백 쓰세요)
    """
    def __init__(
        self,
        env_id: str = "CarRacing-v3",
        resize=(84,84),
        gray=True,
        frame_stack: int = 4,
        seed: int = 0,
        every_steps: int = 50000,
        max_steps: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.resize = resize
        self.gray = gray
        self.frame_stack = frame_stack
        self.seed = seed
        self.every_steps = every_steps
        self.max_steps = max_steps
        self._env = None

    def _init_env(self):
        env = DummyVecEnv([
            make_env(self.env_id, seed=self.seed, render_mode="human",
                     resize=self.resize, gray=self.gray)
        ])
        env = VecFrameStack(env, n_stack=self.frame_stack)
        self._env = env

    def _on_step(self) -> bool:
        if self.every_steps <= 0:
            return True
        if self.n_calls % self.every_steps != 0:
            return True

        if self._env is None:
            self._init_env()

        obs = self._env.reset()
        done = False
        steps = 0
        ep_rew = 0.0
        while not done and steps < self.max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self._env.step(action)
            ep_rew += float(reward)
            steps += 1
            # 약간의 sleep을 주면 렌더 프레임이 너무 빨라지지 않습니다.
            time.sleep(0.01)
        if self.verbose:
            print(f"[LiveRender] steps={steps} reward={ep_rew:.1f}")
        return True

    def _on_training_end(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class VideoEvalRecordCallback(BaseCallback):
    """
    학습 중 일정 간격으로 별도 env에서 rgb_array 렌더로 mp4를 저장합니다.
    - GUI 없는 서버에서도 동작
    """
    def __init__(
        self,
        output_dir: str = "out_videos",
        env_id: str = "CarRacing-v3",
        resize=(84,84),
        gray=True,
        frame_stack: int = 4,
        seed: int = 0,
        every_steps: int = 50000,
        video_length: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.env_id = env_id
        self.resize = resize
        self.gray = gray
        self.frame_stack = frame_stack
        self.seed = seed
        self.every_steps = every_steps
        self.video_length = video_length

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _record_once(self, idx: int):
        base_env = DummyVecEnv([
            make_env(self.env_id, seed=self.seed, render_mode="rgb_array",
                     resize=self.resize, gray=self.gray)
        ])
        env = VecFrameStack(base_env, n_stack=self.frame_stack)
        # 첫 스텝부터 녹화 시작, 길이는 video_length
        env = VecVideoRecorder(
            env,
            video_folder=str(self.output_dir),
            record_video_trigger=lambda step: step == 0,
            video_length=self.video_length,
            name_prefix=f"train_eval_{idx}",
        )
        obs = env.reset()
        done = False
        steps = 0
        ep_rew = 0.0
        while not done and steps < self.video_length:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward)
            steps += 1
        env.close()
        if self.verbose:
            print(f"[VideoRecord] saved {self.output_dir}/train_eval_{idx}.mp4, reward={ep_rew:.1f}")

    def _on_step(self) -> bool:
        if self.every_steps <= 0:
            return True
        if self.n_calls % self.every_steps != 0:
            return True
        idx = int(self.num_timesteps)
        self._record_once(idx)
        return True

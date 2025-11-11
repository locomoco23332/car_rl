import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from typing import Optional

# --- (1) 트랙 이탈 페널티 + 코너 보정 + 연속 이탈시 조기 종료 -------------------
class OffTrackPenaltyWrapper(gym.Wrapper):
    """
    화면 하단 중앙 ROI에서 '녹색(잔디) 픽셀 비율'을 off-road 지표로 추정.
    - 기본 페널티: reward -= weight * offroad_ratio
    - 코너 보정: |steer|가 steer_thresh보다 크면 추가 페널티
                 reward -= corner_weight * offroad_ratio * max(0, |steer|-steer_thresh)
    - 연속 이탈 조기 종료: offroad_ratio >= offroad_thresh 상태가 offroad_consec 프레임 지속되면 truncated=True,
                           reward에서 early_term_penalty만큼 추가 감점(양수값을 넣으면 자동으로 -부호 적용)
    * 색 정보를 쓰므로 Grayscale 이전에 적용할 것.
    """
    def __init__(
        self,
        env: gym.Env,
        # 기본 페널티
        weight: float = 0.5,
        # 코너(조향) 추가 페널티
        corner_weight: float = 0.5,
        steer_thresh: float = 0.30,  # |steer|>0.3부터 추가 페널티
        # 연속 off-road 조기 종료
        early_terminate: bool = True,
        offroad_thresh: float = 0.35,    # ROI 내 초록 비율이 이 값 이상이면 off-road로 간주
        offroad_consec: int = 20,        # 연속 프레임 수 임계
        early_term_penalty: float = 5.0, # 종료 시 추가 감점(보상에서 -적용)
        # ROI (이미지 비율)
        win_y0: float = 0.60, win_y1: float = 1.00,
        win_x0: float = 0.35, win_x1: float = 0.65,
        # 잔디 검출 파라미터
        green_thr: int = 120,
        green_margin: int = 30,
    ):
        super().__init__(env)
        self.weight = float(weight)
        self.corner_weight = float(corner_weight)
        self.steer_thresh = float(steer_thresh)

        self.early_terminate = bool(early_terminate)
        self.offroad_thresh = float(offroad_thresh)
        self.offroad_consec = int(offroad_consec)
        self.early_term_penalty = float(early_term_penalty)

        self.win_y0, self.win_y1 = float(win_y0), float(win_y1)
        self.win_x0, self.win_x1 = float(win_x0), float(win_x1)
        self.green_thr = int(green_thr)
        self.green_margin = int(green_margin)

        self._off_consec = 0  # 내부 연속 카운터

    def _to_hwc(self, obs):
        if obs.ndim == 3 and obs.shape[-1] in (1, 3):     # HWC
            return obs
        if obs.ndim == 3 and obs.shape[0] in (1, 3):      # CHW
            return np.transpose(obs, (1, 2, 0))
        return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = self._to_hwc(obs)
        info = dict(info or {})

        offroad_ratio = 0.0
        base_penalty = 0.0
        corner_penalty = 0.0

        if img is not None and img.shape[-1] >= 3:
            h, w, _ = img.shape
            y0, y1 = int(h * self.win_y0), int(h * self.win_y1)
            x0, x1 = int(w * self.win_x0), int(w * self.win_x1)
            roi = img[y0:y1, x0:x1]

            r = roi[..., 0].astype(np.int32)
            g = roi[..., 1].astype(np.int32)
            b = roi[..., 2].astype(np.int32)
            green_mask = (g >= self.green_thr) & (g >= r + self.green_margin) & (g >= b + self.green_margin)

            total = green_mask.size
            offroad_ratio = float(green_mask.sum()) / float(total + 1e-6)

            # 기본 페널티
            base_penalty = self.weight * offroad_ratio
            reward = reward - base_penalty

            # 코너(조향) 추가 페널티: steer ∈ [-1,1], gas/brake ∈ [0,1]
            # action shape이 (3,) 또는 (n_env,3)일 수 있음 → 첫 차원 처리
            steer_abs = 0.0
            try:
                steer = float(action[0]) if np.ndim(action) == 1 else float(action[0][0])
                steer_abs = abs(steer)
            except Exception:
                pass

            if steer_abs > self.steer_thresh:
                corner_penalty = self.corner_weight * offroad_ratio * (steer_abs - self.steer_thresh)
                reward = reward - corner_penalty

            # 연속 off-road 카운트 & 조기 종료
            if self.early_terminate:
                if offroad_ratio >= self.offroad_thresh:
                    self._off_consec += 1
                else:
                    self._off_consec = 0

                if self._off_consec >= self.offroad_consec and not (terminated or truncated):
                    # 보상 추가 감점 + 조기 종료
                    reward = reward - abs(self.early_term_penalty)
                    truncated = True
                    info["early_terminated"] = "offroad_streak"

        info["offroad_ratio"] = offroad_ratio
        info["offroad_penalty_base"] = base_penalty
        info["offroad_penalty_corner"] = corner_penalty
        info["offroad_streak"] = self._off_consec
        return obs, reward, terminated, truncated, info


# --- (2) 관측 전처리 (Resize/Gray/CHW) -----------------------------------------
try:
    from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
    _HAS_BUILTIN = True
except Exception:
    _HAS_BUILTIN = False

def _fallback_grayscale(env: gym.Env, keep_dim: bool = True) -> gym.Env:
    class _Gray(gym.ObservationWrapper):
        def __init__(self, env, keep_dim=True):
            super().__init__(env)
            self.keep_dim = keep_dim
            assert len(self.observation_space.shape) == 3, "Expected HWC image"
            h, w, _ = self.observation_space.shape
            low, high = 0, 255
            shape = (h, w, 1) if keep_dim else (h, w)
            self.observation_space = gym.spaces.Box(low, high, shape=shape, dtype=np.uint8)

        def observation(self, obs):
            gray = (0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]).astype("uint8")
            if self.keep_dim:
                gray = gray[..., None]
            return gray
    return _Gray(env, keep_dim=keep_dim)

class ChannelFirst(gym.ObservationWrapper):
    """(H,W,C) -> (C,H,W)로 변환 (SB3 CnnPolicy와 호환)."""
    def __init__(self, env):
        super().__init__(env)
        assert len(self.observation_space.shape) == 3, "Expected HWC image"
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        return obs.transpose(2, 0, 1)

def apply_common_wrappers(
    env: gym.Env,
    resize=(84, 84),
    gray=True,
    reward_shaping: Optional[dict] = None,
) -> gym.Env:
    """
    권장 순서:
    1) (옵션) Reward shaping: OffTrackPenaltyWrapper ← RGB 필요시 Grayscale 이전
    2) Resize
    3) Grayscale(keep_dim=True)
    4) ChannelFirst (CHW)
    5) Monitor
    """
    if isinstance(reward_shaping, dict):
        env = OffTrackPenaltyWrapper(env, **reward_shaping)

    if resize is not None:
        if _HAS_BUILTIN:
            env = ResizeObservation(env, resize)
        else:
            import cv2
            class _Resize(gym.ObservationWrapper):
                def __init__(self, env, shape):
                    super().__init__(env)
                    self.shape = tuple(shape)
                    h, w = self.shape
                    ch = env.observation_space.shape[2]
                    self.observation_space = gym.spaces.Box(0, 255, (h, w, ch), dtype=np.uint8)
                def observation(self, obs):
                    return cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
            env = _Resize(env, resize)

    if gray:
        if _HAS_BUILTIN:
            env = GrayscaleObservation(env, keep_dim=True)
        else:
            env = _fallback_grayscale(env, keep_dim=True)

    env = ChannelFirst(env)
    env = Monitor(env)
    return env

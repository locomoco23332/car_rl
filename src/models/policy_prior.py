# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhDiagGaussian:
    """Squashed Gaussian with tanh. log_prob(actions) 지원.
    actions: steer∈[-1,1], gas/brake∈[0,1]이더라도, 내부적으로 [-1,1] 공간(a_tanh)로 변환해 Jacobian 보정."""
    def __init__(self, mean, log_std, eps=1e-6):
        self.mean = mean
        self.log_std = torch.clamp(log_std, -5.0, 2.0)
        self.std = torch.exp(self.log_std)
        self.eps = eps

    @staticmethod
    def to_tanh_space(actions):
        # actions: (...,3) = [steer(-1~1), gas(0~1), brake(0~1)]
        steer = actions[..., :1].clamp(-1.0, 1.0)
        gasbrake = actions[..., 1:].clamp(0.0, 1.0)
        gb_tanh = gasbrake * 2.0 - 1.0
        a_tanh = torch.cat([steer, gb_tanh], dim=-1)
        return a_tanh

    def log_prob(self, actions):
        a_tanh = self.to_tanh_space(actions)
        # pre_tanh
        a_tanh = a_tanh.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = 0.5 * torch.log((1 + a_tanh) / (1 - a_tanh))  # atanh
        # Gaussian log prob in pre-tanh space
        var = self.std ** 2
        logp = -0.5 * (((pre_tanh - self.mean) ** 2) / (var + self.eps) + 2 * self.log_std + math.log(2 * math.pi))
        logp = logp.sum(dim=-1)
        # tanh jacobian
        log_det = torch.log(1 - a_tanh ** 2 + self.eps).sum(dim=-1)
        return logp - log_det  # (…,)
    

class PolicyPrior(nn.Module):
    """관측 -> (mean, log_std) -> squashed Gaussian. IL 데이터로 학습하며, RL 중 KL 정규화에 사용."""
    def __init__(self, obs_shape, action_dim=3):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
        )
        # conv 출력 크기 계산
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            o = self.conv(dummy)
            conv_out = o.view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, obs):
        # obs: uint8(CHW) 또는 float -> float32(0~1)
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs
        z = self.conv(x).view(x.size(0), -1)
        z = self.head(z)
        mean = self.mu(z)
        log_std = self.log_std(z)
        return mean, log_std

    def dist(self, obs):
        mean, log_std = self.forward(obs)
        return TanhDiagGaussian(mean, log_std)

    def log_prob(self, obs, actions):
        return self.dist(obs).log_prob(actions)


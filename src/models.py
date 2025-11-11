# src/models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class GaussianSquashDiag:
    """
    tanh-squashed Gaussian + 차원별 범위 보정
    - a_raw ~ N(mu, sigma^2)
    - a_tanh = tanh(a_raw) in [-1,1]
    - steering: [-1,1] 그대로
    - gas/brake: (a_tanh+1)/2 in [0,1]
    """
    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        self.std = self.log_std.exp()

    def rsample(self):
        eps = torch.randn_like(self.mean)
        pre_tanh = self.mean + self.std * eps
        a_tanh = torch.tanh(pre_tanh)
        a = a_tanh.clone()
        # gas/brake rescale to [0,1]
        if a.shape[-1] >= 3:
            a[..., 1:] = 0.5 * (a[..., 1:] + 1.0)
        return a, pre_tanh, a_tanh

    def log_prob(self, actions, pre_tanh):
        # 역변환: steering -> tanh^-1(act), gas/brake -> 2*act-1 후 atanh
        z = pre_tanh  # pre-activation used during rsample
        # log prob in Gaussian
        log_prob_gauss = -0.5 * (((z - self.mean) / self.std)**2 + 2*self.log_std + math.log(2*math.pi))
        log_prob_gauss = log_prob_gauss.sum(dim=-1)

        # tanh jacobian correction: sum log(1 - tanh(z)^2)
        log_det = torch.log(1 - torch.tanh(z)**2 + 1e-6).sum(dim=-1)

        # gas/brake scaling by 0.5 (constant factor); PPO ratio에는 상수항이 상쇄되므로 무시 가능
        return (log_prob_gauss - log_det)

    def entropy(self):
        # approximate entropy before tanh (fine for PPO baseline)
        return (0.5 + 0.5*math.log(2*math.pi) + self.log_std).sum(dim=-1)


class CNNActorCritic(nn.Module):
    def __init__(self, obs_shape=(4, 84, 84), action_dim=3):
        super().__init__()
        self.action_dim = action_dim
        c, h, w = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            enc_dim = self.encoder(dummy).shape[-1]

        self.policy_head = nn.Sequential(
            nn.Linear(enc_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 2*action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(enc_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        self.is_recurrent = False

    def forward(self, x):
        z = self.encoder(x / 255.0)
        logits = self.policy_head(z)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        value = self.value_head(z).squeeze(-1)
        return mean, log_std, value

    def act(self, obs, rnn_hxs, masks):
        mean, log_std, value = self.forward(obs)
        dist = GaussianSquashDiag(mean, log_std)
        action, pre_tanh, a_tanh = dist.rsample()
        logp = dist.log_prob(action, pre_tanh)
        return value, action, logp, rnn_hxs

    def evaluate_actions(self, obs, rnn_hxs, masks, actions):
        mean, log_std, value = self.forward(obs)
        # 역전파 위해 pre_tanh를 다시 계산
        eps = torch.zeros_like(mean)  # determ. to get pre_tanh via atanh of transformed actions
        # Forward path to get pre_tanh consistent: reconstruct pre_tanh from actions
        # steer: atanh(a)
        # gas/brake: atanh(2a-1)
        a = actions.clamp(0, 1)  # safe
        steer = actions[..., :1].clamp(-1, 1)
        gasbrake = actions[..., 1:].clamp(0, 1)
        a_tanh = torch.cat([steer, 2*gasbrake - 1], dim=-1)
        pre_tanh = torch.atanh(a_tanh.clamp(-0.999, 0.999))

        dist = GaussianSquashDiag(mean, log_std)
        logp = dist.log_prob(actions, pre_tanh)
        entropy = dist.entropy().mean()
        return value, logp, entropy, rnn_hxs


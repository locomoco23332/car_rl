# src/storage.py
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutStorage:
    def __init__(self, num_steps, num_envs, obs_shape, action_dim, device):
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape, device=device, dtype=torch.uint8)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, device=device)
        self.returns = torch.zeros(num_steps + 1, num_envs, device=device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.masks = torch.ones(num_steps + 1, num_envs, device=device)

        self.num_steps = num_steps
        self.step = 0
        self.num_envs = num_envs
        self.device = device

    def to(self, device):
        return self

    def insert(self, obs, rnn_hxs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs.cpu())
        self.recurrent_hidden_states[self.step + 1].copy_(rnn_hxs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    @torch.no_grad()
    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_envs = self.num_steps, self.num_envs
        batch_size = num_envs * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), batch_size // num_mini_batch, drop_last=False)

        obs = self.obs[:-1].reshape(num_steps * num_envs, *self.obs.size()[2:]).float().to(self.device)
        actions = self.actions.reshape(num_steps * num_envs, -1).to(self.device)
        value_preds = self.value_preds[:-1].reshape(num_steps * num_envs).to(self.device)
        returns = self.returns[:-1].reshape(num_steps * num_envs).to(self.device)
        masks = self.masks[:-1].reshape(num_steps * num_envs).to(self.device)
        old_logp = self.action_log_probs.reshape(num_steps * num_envs).to(self.device)
        rnn_hxs = self.recurrent_hidden_states[:-1].reshape(num_steps * num_envs, -1).to(self.device)
        adv = advantages.reshape(-1).to(self.device)

        for idx in sampler:
            yield (obs[idx], rnn_hxs[idx], actions[idx],
                   value_preds[idx], returns[idx], masks[idx],
                   old_logp[idx], adv[idx])


# src/algos/entropy_ppo.py
import math
import torch
import torch.nn as nn
import torch.optim as optim

class EntropyPPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 # ---- Entropy-PPO (MaxEnt) options ----
                 adaptive_entropy: bool = False,
                 target_entropy: float = None,
                 entropy_lr: float = None,
                 init_entropy_coef: float = None):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # ---------- Entropy auto-tuning (SAC-style) ----------
        self.adaptive_entropy = adaptive_entropy
        if self.adaptive_entropy:
            action_dim = getattr(actor_critic, "action_dim", None)
            if action_dim is None:
                action_dim = 1
            if target_entropy is None:
                target_entropy = -float(action_dim)
            self.target_entropy = float(target_entropy)

            device = next(actor_critic.parameters()).device
            init_alpha = init_entropy_coef if init_entropy_coef is not None else entropy_coef
            init_alpha = max(1e-8, float(init_alpha))
            self.log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=device)

            if entropy_lr is None:
                entropy_lr = lr if lr is not None else 3e-4
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=entropy_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

    @property
    def alpha(self):
        if self.adaptive_entropy:
            return self.log_alpha.exp()
        return torch.tensor(self.entropy_coef, device=next(self.actor_critic.parameters()).device)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        alpha_value_epoch = 0.0

        for _ in range(self.ppo_epoch):
            if getattr(self.actor_critic, "is_recurrent", False):
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (obs_batch, rnn_hxs_batch, actions_batch,
                 value_preds_batch, return_batch, masks_batch,
                 old_logp_batch, adv_targ) = sample

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, rnn_hxs_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_logp_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if self.adaptive_entropy:
                    alpha = self.alpha.detach()
                    # note: maximize entropy == minimize (alpha * (-H)) == (alpha * log_prob)
                    entropy_term = (alpha * action_log_probs).mean()
                    total_loss = value_loss * self.value_loss_coef + action_loss + entropy_term
                else:
                    total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.adaptive_entropy:
                    with torch.no_grad():
                        neg_logp = (-action_log_probs).detach()
                    alpha_loss = (self.alpha * (neg_logp - self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_value_epoch += float(self.alpha.item())

                value_loss_epoch += float(value_loss.item())
                action_loss_epoch += float(action_loss.item())
                dist_entropy_epoch += float(dist_entropy.item())

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if self.adaptive_entropy:
            alpha_value_epoch /= num_updates
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, alpha_value_epoch
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


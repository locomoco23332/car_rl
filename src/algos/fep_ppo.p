# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

from .entropy_ppo import EntropyPPO


class FEP_PPO(EntropyPPO):
    """
    EntropyPPO + Free-Energy 기반 policy-prior 정규화
    - 손실에 λ * KL( posterior || prior ) = E[ logπ(a|s) - log p_prior(a|s) ] 추가
    - (선택) prior를 전문가 데이터로 동시학습(NLL 최소화)
    참고: FEP 확장에서 행동을 감각 입력에 포함하고(policy prior 도입), IL/RL의 EFE에 prior 항이 들어감. 
    """
    def __init__(self,
                 actor_critic,
                 policy_prior,                 # PolicyPrior 모듈
                 prior_coef: float = 1.0,      # λ (EFE의 prior 항 가중치)
                 prior_lr: float = 3e-4,
                 demos_loader=None,            # (옵션) 전문가 데이터 DataLoader[(obs, act)]
                 demos_steps_per_update: int = 0,
                 *args, **kwargs):
        super().__init__(actor_critic, *args, **kwargs)
        self.policy_prior = policy_prior
        self.prior_coef = float(prior_coef)
        self.demos_loader = demos_loader
        self.demos_iter = iter(demos_loader) if demos_loader is not None else None
        self.demos_steps_per_update = int(demos_steps_per_update)
        self.prior_optim = optim.Adam(self.policy_prior.parameters(), lr=prior_lr, eps=1e-5)

    def _kl_post_prior(self, obs, actions, action_log_probs):
        """E[ logπ(a|s) - log p_prior(a|s) ]"""
        with torch.no_grad():
            # posterior logp는 인자로 전달된 action_log_probs 사용 (정확히 같은 a로 평가)
            logp_post = action_log_probs
        logp_prior = self.policy_prior.log_prob(obs, actions)
        return (logp_post - logp_prior).mean()

    def _update_policy_prior_with_demos(self, device):
        if self.demos_loader is None or self.demos_steps_per_update <= 0:
            return {}
        self.policy_prior.train()
        logs = {}
        for _ in range(self.demos_steps_per_update):
            try:
                obs, act = next(self.demos_iter)
            except StopIteration:
                self.demos_iter = iter(self.demos_loader)
                obs, act = next(self.demos_iter)
            obs = obs.to(device)
            act = act.to(device)
            nll = - self.policy_prior.log_prob(obs, act).mean()
            self.prior_optim.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(self.policy_prior.parameters(), 1.0)
            self.prior_optim.step()
            logs["prior_nll"] = float(nll.item())
        return logs

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_epoch = 0.0
        alpha_value_epoch = 0.0

        device = next(self.actor_critic.parameters()).device

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (obs_batch, recurrent_hidden_states_batch, actions_batch,
                 value_preds_batch, return_batch, masks_batch,
                 old_action_log_probs_batch, adv_targ) = sample

                # Posterior 평가 (기존 PPO)
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
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

                # -------- FEP: KL( posterior || prior ) 정규화 --------
                with torch.no_grad():
                    # obs_batch는 uint8(CHW)일 수 있으므로 prior에서 처리
                    pass
                kl = self._kl_post_prior(obs_batch, actions_batch, action_log_probs)

                # -------- 총 손실 --------
                if self.adaptive_entropy:
                    alpha = self.alpha.detach()
                    entropy_term = (alpha * action_log_probs).mean()  # == -alpha * H
                    total_loss = (value_loss * self.value_loss_coef
                                  + action_loss
                                  + entropy_term
                                  + self.prior_coef * kl)
                else:
                    total_loss = (value_loss * self.value_loss_coef
                                  + action_loss
                                  - dist_entropy * self.entropy_coef
                                  + self.prior_coef * kl)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # alpha 갱신 (선택)
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
                kl_epoch += float(kl.item())

        # (옵션) 전문가 데이터로 policy prior 업데이트
        prior_logs = self._update_policy_prior_with_demos(device)

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_epoch /= num_updates

        out = (value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_epoch)
        if self.adaptive_entropy:
            out = out + (alpha_value_epoch / num_updates,)
        return out | prior_logs  # python3.9 dict merge-safe 아님 -> 필요 시 수정


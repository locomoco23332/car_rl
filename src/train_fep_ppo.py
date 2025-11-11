# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from .make_env import make_env
from .models import CNNActorCritic
from .models.policy_prior import PolicyPrior
from .algos.fep_ppo import FEP_PPO
from .storage import RolloutStorage


def maybe_make_demos_loader(npz_path, batch_size=64, shuffle=True):
    if npz_path is None:
        return None
    data = np.load(npz_path)
    obs = torch.tensor(data["obs"])      # (N,C,H,W) uint8/float
    act = torch.tensor(data["actions"])  # (N,3) float
    ds = TensorDataset(obs, act)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CarRacing-v2")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84])
    p.add_argument("--gray", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)

    # PPO
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epoch", type=int, default=4)
    p.add_argument("--mini-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--adaptive-entropy", action="store_true", default=False)
    p.add_argument("--target-entropy", type=float, default=None)

    # FEP prior 정규화
    p.add_argument("--efe-kl-coef", type=float, default=0.5)
    p.add_argument("--prior-lr", type=float, default=3e-4)
    p.add_argument("--demos", type=str, default=None, help=".npz 파일 경로 (obs, actions)")
    p.add_argument("--demos-steps-per-update", type=int, default=0)

    # Reward-shaping / 렌더 등은 기존 스크립트 옵션을 재사용
    p.add_argument("--offtrack-penalty", type=float, default=0.0)
    p.add_argument("--offtrack-green-thr", type=int, default=120)
    p.add_argument("--offtrack-green-margin", type=int, default=30)
    p.add_argument("--offtrack-win", type=float, nargs=4, default=[0.60, 1.00, 0.35, 0.65])
    p.add_argument("--corner-weight", type=float, default=0.5)
    p.add_argument("--steer-thresh", type=float, default=0.30)
    p.add_argument("--early-terminate", action="store_true", default=False)
    p.add_argument("--thresh-on", type=float, default=0.55)
    p.add_argument("--thresh-off", type=float, default=0.40)
    p.add_argument("--consec-on", type=int,   default=60)
    p.add_argument("--debounce-off", type=int, default=8)
    p.add_argument("--grace-steps", type=int, default=400)
    p.add_argument("--cooldown-steps", type=int, default=300)
    p.add_argument("--early-term-penalty", type=float, default=5.0)

    p.add_argument("--render-human", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "device")

    # demos loader (옵션)
    demos_loader = maybe_make_demos_loader(args.demos, batch_size=64)

    # reward shaping 구성
    reward_shaping_cfg = None
    if args.offtrack_penalty > 0.0:
        y0, y1, x0, x1 = args.offtrack_win
        reward_shaping_cfg = dict(
            weight=args.offtrack_penalty,
            green_thr=args.offtrack_green_thr,
            green_margin=args.offtrack_green_margin,
            win_y0=y0, win_y1=y1, win_x0=x0, win_x1=x1,
            corner_weight=args.corner_weight,
            steer_thresh=args.steer_thresh,
            early_terminate=args.early_terminate,
            thresh_on=args.thresh_on, thresh_off=args.thresh_off,
            consec_on=args.consec_on, debounce_off=args.debounce_off,
            grace_steps=args.grace_steps, cooldown_steps=args.cooldown_steps,
            early_term_penalty=args.early_term_penalty,
        )

    # VecEnv
    render_mode = "human" if args.render_human else None
    vec_env = make_vec_env(
        make_env(env_id=args.env_id, seed=args.seed, render_mode=render_mode,
                 resize=tuple(args.resize), gray=args.gray,
                 reward_shaping=reward_shaping_cfg),
        n_envs=args.n_envs, seed=args.seed
    )
    vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

    # Spaces
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space
    c, h, w = obs_space.shape
    action_dim = act_space.shape[0]

    # 모델/알고리즘/스토리지
    ac = CNNActorCritic(obs_shape=(c, h, w), action_dim=action_dim).to(device)
    prior = PolicyPrior(obs_shape=(c, h, w), action_dim=action_dim).to(device)
    algo = FEP_PPO(
        actor_critic=ac,
        policy_prior=prior,
        prior_coef=args.efe_kl_coef,
        prior_lr=args.prior_lr,
        demos_loader=demos_loader,
        demos_steps_per_update=args.demos_steps_per_update,
        clip_param=args.clip,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.mini_batch,
        value_loss_coef=args.vf_coef,
        entropy_coef=args.ent_coef,
        lr=args.lr,
        eps=1e-5,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        adaptive_entropy=args.adaptive_entropy,
        target_entropy=args.target_entropy,
        entropy_lr=args.lr,
        init_entropy_coef=args.ent_coef
    )
    storage = RolloutStorage(num_steps=args.rollout_steps, num_envs=args.n_envs,
                             obs_shape=(c, h, w), action_dim=action_dim, device=device)

    obs = vec_env.reset()
    storage.obs[0].copy_(torch.as_tensor(obs, dtype=torch.uint8))

    num_updates = args.total_steps // (args.rollout_steps * args.n_envs)
    gamma, gae_lambda = 0.99, 0.95
    global_step = 0

    for update in range(1, num_updates + 1):
        for step in range(args.rollout_steps):
            with torch.no_grad():
                obs_f = torch.as_tensor(obs, dtype=torch.float32, device=device)
                values, actions, logp, _ = ac.act(
                    obs_f,
                    torch.zeros(args.n_envs, 1, device=device),
                    torch.ones(args.n_envs, device=device)
                )
            next_obs, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            masks = torch.as_tensor(1.0 - dones.astype(np.float32), device=device)
            storage.insert(
                torch.as_tensor(next_obs, dtype=torch.uint8),
                torch.zeros(args.n_envs, 1, device=device),
                actions, logp, values, rewards_t, masks
            )
            obs = next_obs
            global_step += args.n_envs

        with torch.no_grad():
            obs_f = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, _, next_value = ac.forward(obs_f)

        storage.compute_returns(next_value, gamma=gamma, gae_lambda=gae_lambda)

        out = algo.update(storage)
        storage.after_update()

        # 로그
        if len(out) == 5:
            v_loss, a_loss, ent, kl, alpha = out
            print(f"[{update}/{num_updates}] v={v_loss:.3f} a={a_loss:.3f} H={ent:.3f} KL={kl:.3f} α={alpha:.4f} step={global_step}")
        else:
            v_loss, a_loss, ent, kl = out
            print(f"[{update}/{num_updates}] v={v_loss:.3f} a={a_loss:.3f} H={ent:.3f} KL={kl:.3f} step={global_step}")

    vec_env.close()
    torch.save(ac.state_dict(), "checkpoints/fep_ppo_actorcritic.pth")
    torch.save(prior.state_dict(), "checkpoints/fep_ppo_policyprior.pth")
    print("✅ Saved checkpoints.")
    

if __name__ == "__main__":
    main()


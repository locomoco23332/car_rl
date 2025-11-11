import argparse
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from .make_env import make_env
from .models import CNNActorCritic
from .algos.entropy_ppo import EntropyPPO
from .storage import RolloutStorage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CarRacing-v3")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84])
    p.add_argument("--gray", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)

    # PPO/EntropyPPO
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epoch", type=int, default=4)
    p.add_argument("--mini-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--adaptive-entropy", action="store_true", default=False)
    p.add_argument("--target-entropy", type=float, default=None)

    # Off-track reward shaping
    p.add_argument("--offtrack-penalty", type=float, default=0.0)
    p.add_argument("--offtrack-green-thr", type=int, default=120)
    p.add_argument("--offtrack-green-margin", type=int, default=30)
    p.add_argument("--offtrack-win", type=float, nargs=4, default=[0.60, 1.00, 0.35, 0.65])

    # 추가 옵션: 코너 페널티/조기 종료 파라미터 노출
    p.add_argument("--corner-weight", type=float, default=0.5)
    p.add_argument("--steer-thresh", type=float, default=0.30)
    p.add_argument("--early-terminate", action="store_true", default=True)
    p.add_argument("--offroad-thresh", type=float, default=0.35)
    p.add_argument("--offroad-consec", type=int, default=20)
    p.add_argument("--early-term-penalty", type=float, default=5.0)

    # 렌더링
    p.add_argument("--render-human", action="store_true", default=False)

    # TensorBoard
    p.add_argument("--logdir", type=str, default="runs/entropyppo")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "device")

    writer = SummaryWriter(log_dir=args.logdir)

    # Reward shaping 설정
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
            offroad_thresh=args.offroad_thresh,
            offroad_consec=args.offroad_consec,
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
    action_dim = act_space.shape[0]  # 3

    # Model / Algo / Storage
    ac = CNNActorCritic(obs_shape=(c, h, w), action_dim=action_dim).to(device)
    algo = EntropyPPO(
        actor_critic=ac,
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

    # 초기 상태
    obs = vec_env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=device)
    storage.obs[0].copy_(obs_t.cpu())

    num_updates = args.total_steps // (args.rollout_steps * args.n_envs)
    gamma, gae_lambda = 0.99, 0.95
    global_step = 0

    for update in range(1, num_updates + 1):
        # ---- TensorBoard 집계 변수 (이번 rollout 동안) ----
        off_sum, off_cnt = 0.0, 0
        pen_base_sum, pen_corner_sum = 0.0, 0.0
        streak_max = 0

        # Rollout 수집
        for step in range(args.rollout_steps):
            with torch.no_grad():
                obs_f = torch.as_tensor(obs, dtype=torch.float32, device=device)
                values, actions, logp, _ = ac.act(
                    obs_f,
                    torch.zeros(args.n_envs, 1, device=device),
                    torch.ones(args.n_envs, device=device)
                )

            next_obs, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            # info에서 오프로드 통계 수집
            for i in range(args.n_envs):
                inf = infos[i] if isinstance(infos, (list, tuple)) else {}
                if "offroad_ratio" in inf:
                    off_sum += float(inf["offroad_ratio"])
                    off_cnt += 1
                if "offroad_penalty_base" in inf:
                    pen_base_sum += float(inf["offroad_penalty_base"])
                if "offroad_penalty_corner" in inf:
                    pen_corner_sum += float(inf["offroad_penalty_corner"])
                if "offroad_streak" in inf:
                    streak_max = max(streak_max, int(inf["offroad_streak"]))

            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            masks = torch.as_tensor(1.0 - dones.astype(np.float32), device=device)

            storage.insert(
                torch.as_tensor(next_obs, dtype=torch.uint8, device=device),
                torch.zeros(args.n_envs, 1, device=device),
                actions, logp, values, rewards_t, masks
            )
            obs = next_obs
            global_step += args.n_envs

        # bootstrap value
        with torch.no_grad():
            obs_f = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, _, next_value = ac.forward(obs_f)

        storage.compute_returns(next_value, gamma=gamma, gae_lambda=gae_lambda)

        # 업데이트
        out = algo.update(storage)
        if len(out) == 4:
            v_loss, a_loss, d_ent, alpha_val = out
            print(f"[{update}/{num_updates}] v={v_loss:.3f} a={a_loss:.3f} H={d_ent:.3f} alpha={alpha_val:.4f} step={global_step}")
        else:
            v_loss, a_loss, d_ent = out
            print(f"[{update}/{num_updates}] v={v_loss:.3f} a={a_loss:.3f} H={d_ent:.3f} step={global_step}")

        # ---- TensorBoard 로그 ----
        if off_cnt > 0:
            writer.add_scalar("offroad/ratio_mean", off_sum / off_cnt, global_step)
        writer.add_scalar("offroad/penalty_base_sum", pen_base_sum, global_step)
        writer.add_scalar("offroad/penalty_corner_sum", pen_corner_sum, global_step)
        writer.add_scalar("offroad/streak_max", streak_max, global_step)
        writer.add_scalar("loss/value", v_loss, global_step)
        writer.add_scalar("loss/action", a_loss, global_step)
        writer.add_scalar("stats/entropy", d_ent, global_step)
        if len(out) == 4:
            writer.add_scalar("entropy/alpha", alpha_val, global_step)

        storage.after_update()

    vec_env.close()
    writer.close()
    torch.save(ac.state_dict(), "checkpoints/entropyppo_carracing_ac.pth")
    print("✅ Saved: checkpoints/entropyppo_carracing_ac.pth")


if __name__ == "__main__":
    main()

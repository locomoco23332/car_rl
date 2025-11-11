# 변경된 부분만 주석으로 표시합니다.
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from .make_env import make_env
from .callbacks import LiveRenderCallback, VideoEvalRecordCallback  # ✅ 추가


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CarRacing-v3")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--logdir", type=str, default="runs/carracing_ppo")
    p.add_argument("--save-path", type=str, default="checkpoints/ppo_carracing_v3.zip")
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84])
    p.add_argument("--gray", action="store_true", default=True)

    # ✅ 라이브 렌더/비디오 기록 옵션
    p.add_argument("--live-render-every", type=int, default=0, help="N>0이면 N 스텝마다 창으로 재생")
    p.add_argument("--live-render-steps", type=int, default=800)

    p.add_argument("--video-eval-every", type=int, default=100_000, help="N>0이면 N 스텝마다 mp4 기록")
    p.add_argument("--video-length", type=int, default=1000)
    p.add_argument("--video-dir", type=str, default="out_videos")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            render_mode="human",  # 학습은 렌더 없이
            resize=tuple(args.resize),
            gray=args.gray,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=4,
        clip_range=0.2,
        tensorboard_log=args.logdir,
        verbose=1,
        seed=args.seed,
    )

    # ✅ 콜백 구성
    callbacks = []
    if args.live_render_every > 0:
        callbacks.append(LiveRenderCallback(
            env_id=args.env_id,
            resize=tuple(args.resize),
            gray=args.gray,
            frame_stack=args.frame_stack,
            seed=args.seed,
            every_steps=args.live_render_every,
            max_steps=args.live_render_steps,
            verbose=1,
        ))
    if args.video_eval_every > 0:
        callbacks.append(VideoEvalRecordCallback(
            output_dir=args.video_dir,
            env_id=args.env_id,
            resize=tuple(args.resize),
            gray=args.gray,
            frame_stack=args.frame_stack,
            seed=args.seed,
            every_steps=args.video_eval_every,
            video_length=args.video_length,
            verbose=1,
        ))

    model.learn(total_timesteps=args.total_steps, callback=callbacks if callbacks else None)
    model.save(args.save_path)
    vec_env.close()
    print(f"✅ Saved model to {args.save_path}")


if __name__ == "__main__":
    main()

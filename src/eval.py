import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder

from .make_env import make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CarRacing-v2")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84])
    p.add_argument("--gray", action="store_true", default=True)
    p.add_argument("--render", type=str, default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--record", type=str, default=None, help="directory to save videos")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render == "none" else args.render

    # 평가용 단일 환경
    env = DummyVecEnv([
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            render_mode=render_mode,
            resize=tuple(args.resize),
            gray=args.gray,
        )
    ])
    env = VecFrameStack(env, n_stack=args.frame_stack)

    if args.record:
        Path(args.record).mkdir(parents=True, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder=args.record,
            record_video_trigger=lambda step: step == 0,  # 첫 에피소드만 저장
            video_length=1200,
            name_prefix="eval",
        )

    model = PPO.load(args.model, print_system_info=True)
    print(f"Loaded model: {args.model}")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward)
        print(f"[Episode {ep+1}] Reward: {ep_rew:.2f}")

    env.close()


if __name__ == "__main__":
    main()

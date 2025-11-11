#!/usr/bin/env python3
import argparse, numpy as np, gymnasium as gym

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="CarRacing-v3")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--save", default="datasets/carracing_random.npz")
    args = ap.parse_args()

    env = gym.make(args.env_id, render_mode=None)
    obs_list, act_list = [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            # 랜덤 행동: steer∈[-1,1], gas/brake∈[0,1]
            a = np.array([np.random.uniform(-1,1),
                          np.random.uniform(0,1),
                          np.random.uniform(0,1)], dtype=np.float32)
            nobs, r, done, truncated, info = env.step(a)
            obs_list.append(obs)    # (96,96,3) HWC uint8
            act_list.append(a)
            obs = nobs
        print(f"episode {ep+1}/{args.episodes}")

    env.close()
    obs = np.asarray(obs_list, dtype=np.uint8).transpose(0,3,1,2)  # (N,3,96,96) CHW
    acts = np.asarray(act_list, dtype=np.float32)                   # (N,3)
    np.savez_compressed(args.save, obs=obs, actions=acts)
    print(f"✅ saved {args.save} | obs={obs.shape}, actions={acts.shape}")

if __name__ == "__main__":
    main()


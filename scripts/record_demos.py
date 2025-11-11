import gymnasium as gym
import numpy as np
import pygame

def record_demos(env_id="CarRacing-v3", episodes=5, save_path="carracing_demos.npz"):
    env = gym.make(env_id, render_mode="human")
    obs_list, act_list = [], []

    print("Arrow keys로 조작 (↑=가속, ↓=브레이크, ←=좌회전, →=우회전). q키 종료")
    for ep in range(episodes):
        obs, info = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            # --- 키보드 입력 처리 ---
            action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = -1.0
            if keys[pygame.K_RIGHT]:
                action[0] = +1.0
            if keys[pygame.K_UP]:
                action[1] = +1.0
            if keys[pygame.K_DOWN]:
                action[2] = +0.8
            if keys[pygame.K_q]:
                done = True
                break

            # --- step ---
            next_obs, reward, done, truncated, info = env.step(action)

            # 저장
            obs_list.append(obs)
            act_list.append(action)

            obs = next_obs

        print(f"Episode {ep+1}/{episodes} 종료")

    env.close()

    obs_arr = np.array(obs_list, dtype=np.uint8)   # (N,96,96,3)
    obs_arr = obs_arr.transpose(0,3,1,2)          # (N,3,96,96) → CHW
    act_arr = np.array(act_list, dtype=np.float32)

    np.savez_compressed(save_path, obs=obs_arr, actions=act_arr)
    print(f"✅ 저장 완료: {save_path}, obs={obs_arr.shape}, actions={act_arr.shape}")


if __name__ == "__main__":
    record_demos(episodes=5, save_path="carracing_demos.npz")


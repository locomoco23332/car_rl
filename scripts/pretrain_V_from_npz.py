#!/usr/bin/env python3
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_G import WorldModelG  # encoder + LatentDiffusionAdapter 포함

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="datasets/*.npz (obs, actions)")
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--latent-dvec-dim", type=int, default=896)
    ap.add_argument("--noise-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt", default="checkpoints/pre_V.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.npz)
    obs = torch.tensor(data["obs"])        # (N,C,H,W) uint8/float
    C, H, W = obs.shape[1], obs.shape[2], obs.shape[3]

    # WorldModelG 생성: encoder + (옵션) latent diffusion
    WM = WorldModelG(
        obs_shape=(C, H, W), action_dim=3,
        z_dim=args.z_dim, h_dim=128,
        use_latent_diffusion=True,
        latent_dvec_dim=args.latent_dvec_dim,
        diffusion_noise_steps=args.noise_steps,
        device=device
    ).to(device)

    # encoder + latent_diff만 학습
    params = list(WM.encoder.parameters())
    if WM.latent_diff is not None:
        params += list(WM.latent_diff.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    dl = DataLoader(TensorDataset(obs), batch_size=args.bs, shuffle=True, drop_last=True)
    WM.train()
    for ep in range(args.epochs):
        losses = []
        for (b_obs,) in dl:
            b_obs = b_obs.to(device)
            z = WM.encode(b_obs)                          # (B, z_dim)
            # 잠재 디퓨전 정규화 손실(없으면 0)
            loss = WM.diffusion_reg_loss(z, t=5)
            # 폭주 방지를 위한 작은 L2 정규화
            loss = loss + 1e-4 * z.pow(2).mean()

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))
        print(f"[V-pretrain] epoch {ep+1}/{args.epochs} loss={np.mean(losses):.4f}")

    # 전체 WM state_dict 저장(추후 M-pretrain/학습에서 encoder 로딩용)
    torch.save(WM.state_dict(), args.ckpt)
    print(f"✅ saved V checkpoint: {args.ckpt}")

if __name__ == "__main__":
    main()


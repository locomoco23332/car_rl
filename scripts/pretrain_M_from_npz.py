#!/usr/bin/env python3
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_G import WorldModelG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--V-ckpt", required=True)   # pre_V.pt 로드
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt", default="checkpoints/pre_WM.pt")
    ap.add_argument("--modes", type=int, default=16)  # FNO modes (WorldModelG 내부 기본 1,1)
    ap.add_argument("--K", type=int, default=5)       # MDN mixtures
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.npz)
    obs = torch.tensor(data["obs"])        # (N,C,H,W)
    acts = torch.tensor(data["actions"])   # (N,3)
    C, H, W = obs.shape[1], obs.shape[2], obs.shape[3]

    WM = WorldModelG(
        obs_shape=(C, H, W), action_dim=3,
        z_dim=args.z_dim, h_dim=128,
        use_latent_diffusion=True,  # 구조 일치
        device=device
    ).to(device)

    # V 로드(encoder/latent_diff)
    WM.load_state_dict(torch.load(args.V_ckpt, map_location=device), strict=False)
    WM.eval()

    # 전체 Z 시퀀스 생성
    with torch.no_grad():
        Z = []
        B = 512
        for i in range(0, len(obs), B):
            z = WM.encode(obs[i:i+B].to(device))  # (b, z_dim)
            Z.append(z.cpu())
        Z = torch.cat(Z, dim=0)  # (N, z_dim)

    # (z_t, a_t, z_{t+1})
    Zt  = Z[:-1]
    Zt1 = Z[1:]
    At  = acts[:-1].float()
    ds = TensorDataset(Zt, At, Zt1)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)

    # ODE(FNO) + MDN만 학습: encoder/latent_diff는 freeze
    for p in WM.encoder.parameters(): p.requires_grad = False
    if WM.latent_diff is not None:
        for p in WM.latent_diff.parameters(): p.requires_grad = False
    train_params = list(WM.f.parameters()) + list(WM.mdn.parameters())
    opt = torch.optim.Adam(train_params, lr=args.lr)

    WM.train()
    for ep in range(args.epochs):
        losses = []
        for zt, at, zt1 in dl:
            zt, at, zt1 = zt.to(device), at.to(device), zt1.to(device)
            h0 = WM.init_state(zt.size(0))
            nll, _ = WM.loss_step(zt, at, h0, zt1)
            opt.zero_grad(); nll.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            losses.append(float(nll.item()))
        print(f"[M-pretrain] epoch {ep+1}/{args.epochs} NLL={np.mean(losses):.4f}")

    # 전체 WM 저장(encoder+latent_diff+ODE+MDN 포함) → 학습에서 한 번에 로드 가능
    torch.save(WM.state_dict(), args.ckpt)
    print(f"✅ saved WorldModel checkpoint: {args.ckpt}")

if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
DCGAN 단일 스크립트: 학습 → 학습된 모델로 이미지 생성(샘플/보간)
- 데이터: MNIST (28x28, 1채널)
- 결과물: runs_gan/
- samples_epochXX.png (에폭별 샘플 그리드)
- best_g.pt / last_g.pt (Generator 가중치)
- samples_final.png (최종 생성 그리드)
- interp_final.png (잠재공간 보간)
- 빠른 실습을 위해 기본 epochs=2 로 설정
"""

import os, argparse, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# ------------------------------------------------
# 재현성 / 장치
# ------------------------------------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# 모델
# ------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, base_ch=64, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
    nn.ConvTranspose2d(z_dim, base_ch*4, 4, 1, 0, bias=False), # 4x4
    nn.BatchNorm2d(base_ch*4), nn.ReLU(True),
    nn.ConvTranspose2d(base_ch*4, base_ch*2, 3, 2, 1, bias=False), # 7x7
    nn.BatchNorm2d(base_ch*2), nn.ReLU(True),
    nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1, bias=False), #14x14
    nn.BatchNorm2d(base_ch), nn.ReLU(True),
    nn.ConvTranspose2d(base_ch, out_ch, 4, 2, 1, bias=False), #28x28
    nn.Tanh()   # [-1,1] 범위로 정규화
    )
    def forward(self, z): # z: [B, z_dim, 1, 1]
        return self.net(z)


# ------------------------------------------------
# Discriminator: 이미지 → 진위 판별 스칼라
# ------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, base_ch=64, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_ch, base_ch, 4, 2, 1, bias=False), # 14x14
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_ch, base_ch*2, 4, 2, 1, bias=False), # 7x7
        nn.BatchNorm2d(base_ch*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_ch*2, 1, 7, 1, 0, bias=False), # 1x1

    )
    def forward(self, x): # x: [B,1,28,28]
        return self.net(x).view(-1) # [B]

# ------------------------------------------------
# 가중치 초기화
# ------------------------------------------------
def weights_init(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:  # bias가 있을 때만 초기화
            nn.init.zeros_(m.bias)

# ------------------------------------------------
# 데이터
# ------------------------------------------------
def make_loader(batch=128, workers=2, subset=-1):
    tfm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5,), std=(0.5,)) # [-1,1]
    ])
    ds = tv.datasets.MNIST(root="data", train=True, transform=tfm, download=True)
    if subset > 0 and subset < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(subset)))
    return DataLoader(ds, batch_size=batch, shuffle=True,num_workers=workers, pin_memory=(DEVICE.type=="cuda"))

# ------------------------------------------------
# 유틸(생성/저장)
# ------------------------------------------------
@torch.no_grad()
def save_grid_from_fixed(G, z_fixed, path, nrow=8):
    G.eval()
    imgs = (G(z_fixed).cpu() + 1) / 2   # [-1,1] → [0,1]
    grid = make_grid(imgs, nrow=nrow)
    save_image(grid, path)  # 그리드 이미지 저장

@torch.no_grad()
def save_interpolation(G, z_dim, path, steps=12):
    G.eval()
    # 이미지 생성
    z1 = torch.randn(1, z_dim, 1, 1, device=DEVICE)
    z2 = torch.randn(1, z_dim, 1, 1, device=DEVICE)
    zs = [ (1-a)*z1 + a*z2 for a in torch.linspace(0, 1, steps, device=DEVICE) ]
    z = torch.cat(zs, dim=0)
    imgs = (G(z).cpu() + 1) / 2
    grid = make_grid(imgs, nrow=steps)
    save_image(grid, path)

# ------------------------------------------------
# 학습
# ------------------------------------------------
def train_dcgan(epochs, batch, zdim, base_ch, lr, workers, out_dir, subset):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    G = Generator(z_dim=zdim, base_ch=base_ch).to(DEVICE)
    D = Discriminator(base_ch=base_ch).to(DEVICE)
    G.apply(weights_init); D.apply(weights_init)

    loader = make_loader(batch=batch, workers=workers, subset=subset)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # 고정 z
    torch.manual_seed(SEED)
    z_fixed = torch.randn(64, zdim, 1, 1, device=DEVICE)

    best_score = -1e9
    best_path = out_dir / "best_g.pt"

    for ep in range(1, epochs+1):
        G.train(); D.train()
        g_loss_sum, d_loss_sum = 0.0, 0.0
        for x, _ in loader:
            x = x.to(DEVICE)

            # --- Discriminator ---
            z = torch.randn(x.size(0), zdim, 1, 1, device=DEVICE)
            fake = G(z).detach()
            real_logits = D(x)
            fake_logits = D(fake)
            # 작은 라벨 스무딩(실습용): 진짜=0.9
            loss_d_real = bce(real_logits, torch.full_like(real_logits, 0.9))
            loss_d_fake = bce(fake_logits, torch.zeros_like(fake_logits))
            loss_d = 0.5*(loss_d_real + loss_d_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # --- Generator ---
            z = torch.randn(x.size(0), zdim, 1, 1, device=DEVICE)
            fake = G(z)
            fake_logits = D(fake)
            loss_g = bce(fake_logits, torch.ones_like(fake_logits))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_loss_sum += loss_d.item()
            g_loss_sum += loss_g.item()
            n = len(loader)
            print(f"[EPOCH {ep:02d}] G={g_loss_sum/max(1,n):.3f} D={d_loss_sum/max(1,n):.3f}")

            # 중간 샘플 저장
            save_grid_from_fixed(G, z_fixed, out_dir / f"samples_epoch{ep:02d}.png")

            # 간단한 점수(실습용): -G loss 기준
            score = -(g_loss_sum/max(1,n))
            if score > best_score:
                best_score = score
                torch.save(G.state_dict(), best_path)
                print(f"[INFO] Saved best generator → {best_path}")

    # 마지막 가중치도 저장
    torch.save(G.state_dict(), out_dir / "last_g.pt")
    return best_path

# ------------------------------------------------
# 메인: 학습 후 즉시 생성
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50, help="짧게 1~3 권장(수업용)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--zdim", type=int, default=100)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="runs_gan")
    ap.add_argument("--subset", type=int, default=10000, help="학습에 사용할 MNIST 샘플 수(빠른 실습용). 전체= -1")
    ap.add_argument("--gen_n", type=int, default=64, help="최종 샘플 그리드 이미지에 사용할 샘플 수")
    args, _ = ap.parse_known_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={DEVICE}")

    # 1) 학습(베스트 G 저장)
    best_g_path = train_dcgan(epochs=args.epochs, batch=args.batch, zdim=args.zdim,base_ch=args.base_ch,lr=args.lr, workers=args.workers, out_dir=out_dir, subset=args.subset)

    # 2) 학습된 모델 로드해서 최종 생성(샘플/보간)
    G = Generator(z_dim=args.zdim, base_ch=args.base_ch).to(DEVICE)
    G.load_state_dict(torch.load(best_g_path, map_location=DEVICE))
    G.eval()

    # 샘플 생성
    with torch.no_grad():
        z = torch.randn(args.gen_n, args.zdim, 1, 1, device=DEVICE)
        imgs = (G(z).cpu() + 1) / 2
        grid = make_grid(imgs, nrow=int(args.gen_n**0.5))
        save_image(grid, out_dir / "samples_final.png")

    # 보간 이미지
    save_interpolation(G, args.zdim, out_dir / "interp_final.png", steps=12)

    print(f"[DONE] Check images in: {out_dir}")
    print(f" - best_g.pt / last_g.pt")
    print(f" - samples_epoch*.png / samples_final.png / interp_final.png")

if __name__ == "__main__":
    main()
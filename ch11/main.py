# -*- coding: utf-8 -*-

import os, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, utils
from PIL import Image
import matplotlib.pyplot as plt


# 유틸
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def log_write(fpath, text):
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

def pil_open_rgb(path):
    img = Image.open(path).convert("RGB")
    return img

def make_transform(target_long_side):
# 짧은 변 기준으로 리사이즈(긴 변도 비율 유지), 센터크롭은 하지 않음
    return transforms.Compose([transforms.Lambda(lambda im: resize_by_short_side(im, target_long_side)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),])

def resize_by_short_side(img_pil: Image.Image, short_side: int):
    w, h = img_pil.size
    if min(w, h) == short_side:
        return img_pil
    if w < h:
        new_w = short_side
        new_h = int(h * (short_side / w))
    else:
        new_h = short_side
        new_w = int(w * (short_side / h))
    return img_pil.resize((new_w, new_h), Image.LANCZOS)

def tensor_to_pil(img_t):
    # 역정규화 후 클램프
    mean = torch.tensor([0.485,0.456,0.406], device=img_t.device).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=img_t.device).view(1,3,1,1)
    x = img_t * std + mean
    x = torch.clamp(x, 0, 1)
    x = x.detach().cpu()
    grid = x.squeeze(0)
    grid = transforms.ToPILImage()(grid)
    return grid

def save_tensor_as_png(img_t, path):
    pil = tensor_to_pil(img_t)
    pil.save(path)

def make_compare_grid(content_t, style_t, gen_t, out_path):
    # 세 이미지를 같은 해상도로 맞춰 그리드로 저장 (기준: 생성 이미지 크기)
    _, _, H, W = gen_t.shape
    def resize_like(t):
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    c = resize_like(content_t)
    s = resize_like(style_t)
    g = gen_t
    mean = torch.tensor([0.485, 0.456, 0.406], device=g.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=g.device).view(1, 3, 1, 1)
    def denorm(x): return torch.clamp(x * std + mean, 0, 1)
    grid = utils.make_grid(torch.cat([denorm(c), denorm(s), denorm(g)], dim=0),nrow=3)
    utils.save_image(grid.cpu(), out_path)

def plot_losses(loss_log, out_path):
    # loss_log: list of dict(step, total, content, style, tv)
    steps = [d["step"] for d in loss_log]
    total = [d["total"] for d in loss_log]
    content = [d["content"] for d in loss_log]
    style = [d["style"] for d in loss_log]
    tv = [d["tv"] for d in loss_log]
    plt.figure()
    plt.plot(steps, total, label="total")
    plt.plot(steps, content, label="content")
    plt.plot(steps, style, label="style")
    plt.plot(steps, tv, label="tv")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# VGG 특성 추출
VGG_LAYER_MAP = {
    "conv1_1": 0, "relu1_1": 1,
    "conv1_2": 2, "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5, "relu2_1": 6,
    "conv2_2": 7, "relu2_2": 8,
    "pool2": 9,
    "conv3_1":10, "relu3_1":11,
    "conv3_2":12, "relu3_2":13,
    "conv3_3":14, "relu3_3":15,
    "conv3_4":16, "relu3_4":17,
    "pool3":18,
    "conv4_1":19, "relu4_1":20,
    "conv4_2":21, "relu4_2":22, # 콘텐츠 주로 사용
    "conv4_3":23, "relu4_3":24,
    "conv4_4":25, "relu4_4":26,
    "pool4":27,
    "conv5_1":28, "relu5_1":29,
    "conv5_2":30, "relu5_2":31,
    "conv5_3":32, "relu5_3":33,
    "conv5_4":34, "relu5_4":35,
    "pool5":36,
}

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers_to_get):
        super().__init__()
        # 최신 torchvision 가중치 호출
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layers_to_get = [VGG_LAYER_MAP[n] for n in layers_to_get]

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers_to_get:
                # 인덱스 → 이름 매핑
                name = [k for k, v in VGG_LAYER_MAP.items() if v == i][0]
                feats[name] = x
                if len(feats) == len(self.layers_to_get):
                    break
        return feats

# 손실 함수
def gram_matrix(f):
    # f: [B,C,H,W]
    B, C, H, W = f.shape
    F_ = f.view(B, C, H * W)
    G = torch.bmm(F_, F_.transpose(1, 2)) / (C * H * W)
    return G

def total_variation(x, tv_type="l2"):
    # x: [B,3,H,W]
    dh = x[:, :, :, 1:] - x[:, :, :, :-1]
    dw = x[:, :, 1:, :] - x[:, :, :-1, :]
    if tv_type.lower() == "l1":
        return dh.abs().mean() + dw.abs().mean()
    else:
        return (dh ** 2).mean() + (dw ** 2).mean()

# 메인 루틴
def run_train(args):
    # 안전 프로파일 적용
    if args.safe:
        if args.size > 256:
            args.size = 256
        if args.steps > 300:
            args.steps = 300
        args.optimizer = "lbfgs"  # TDR 회피에 유리
        # amp는 lbfgs와 잘 맞지 않으므로 비활성 권장
        args.amp = False

    # 디바이스/AMP 설정
    raw_device = "cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    device = torch.device(raw_device)
    use_amp = bool(args.amp and (device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = ensure_dir(Path(args.out_dir))
    wlog = out_dir / "run_log.txt"
    if wlog.exists(): wlog.unlink()

    log_write(str(wlog), f"[ENV] torch={torch.__version__} cuda = {torch.cuda.is_available()} device = {device} amp = {use_amp}")
    log_write(str(wlog), f"[ARGS] size={args.size} steps={args.steps} alpha={args.alpha} beta = {args.beta} gamma = {args.gamma} opt = {args.optimizer}")

    # 이미지 로드
    assert os.path.exists(args.content), f"content not found: {args.content}"
    assert os.path.exists(args.style), f"style not found: {args.style}"
    Timg = make_transform(args.size)

    content_pil = pil_open_rgb(args.content)
    style_pil = pil_open_rgb(args.style)

    content = Timg(content_pil).unsqueeze(0).to(device)
    style = Timg(style_pil).unsqueeze(0).to(device)

    # 생성 이미지 초기화 = 콘텐츠 복사본
    gen = content.clone().requires_grad_(True)

    # VGG 특징 추출 구성
    content_layer = args.content_layer
    style_layers = args.style_layers
    layers_to_get = sorted(set([content_layer] + style_layers), key=lambda n: VGG_LAYER_MAP[n])
    feat_net = VGGFeatureExtractor(layers_to_get).to(device)

    # 고정된 타깃 특징 계산
    with torch.no_grad():
        c_feats = feat_net(content)[content_layer]
        s_feats = feat_net(style)
        s_grams = {k: gram_matrix(v) for k, v in s_feats.items() if k in style_layers}

    # 옵티마이저
    if args.optimizer.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([gen], max_iter=1)  # 외부 루프로 steps 제어
        # LBFGS와 AMP는 혼용 비권장
        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        optimizer = torch.optim.Adam([gen], lr=args.lr)

    loss_log = []
    last_save_step = -1
    save_every = max(1, args.save_every)
    t0 = time.time()

    def one_step():
        """한 스텝(순전파/역전파/옵티마이저 스텝) 수행 후 (total, lc, ls, ltv) 반환"""

        def closure():
            optimizer.zero_grad(set_to_none=True)
            # AMP 구간
            if use_amp:
                with torch.cuda.amp.autocast():
                    g_feats = feat_net(gen)
                    lc = F.mse_loss(g_feats[content_layer], c_feats)
                    ls = 0.0
                    for ln in style_layers:
                        g_gram = gram_matrix(g_feats[ln])
                        ls = ls + F.mse_loss(g_gram, s_grams[ln])
                    ltv = total_variation(gen, tv_type=args.tv)
                    total = args.alpha * lc + args.beta * ls + args.gamma * ltv
                scaler.scale(total).backward()
                return total, lc, ls, ltv
            else:
                g_feats = feat_net(gen)
                lc = F.mse_loss(g_feats[content_layer], c_feats)
                ls = 0.0
                for ln in style_layers:
                    g_gram = gram_matrix(g_feats[ln])
                    ls = ls + F.mse_loss(g_gram, s_grams[ln])
                ltv = total_variation(gen, tv_type=args.tv)
                total = args.alpha * lc + args.beta * ls + args.gamma * ltv
                total.backward()
                return total, lc, ls, ltv

        if args.optimizer.lower() == "lbfgs":
            # LBFGS는 내부에서 closure 호출 → total만 반환되므로 기록용 재계산
            total, lc, ls, ltv = closure()
            loss_for_step = total
            # LBFGS step (AMP 미사용)
            optimizer.step(lambda: closure()[0])
        else:
            total, lc, ls, ltv = closure()
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            loss_for_step = total
        return loss_for_step, lc, ls, ltv

    # while 루프로 구성하여 폴백 시 현재 스텝을 재시도 가능
    step = 1
    while step <= args.steps:
        try:
            total, lc, ls, ltv = one_step()

            # 값 추출
            t_val = float(total.detach().item())
            c_val = float(lc.detach().item())
            s_val = float(ls.detach().item())
            tv_val = float(ltv.detach().item())
            loss_log.append({"step": step, "total": t_val, "content": c_val, "style": s_val,"tv": tv_val})

            # 정규화 복원 → 다시 정규화(드리프트 방지)
            with torch.no_grad():
                mean = torch.tensor([0.485, 0.456, 0.406],device=gen.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225],device=gen.device).view(1, 3, 1, 1)
                denorm = torch.clamp(gen * std + mean, 0, 1)
                gen.copy_((denorm - mean) / std)

            if (not args.quick) and (step % 10 == 0 or step == 1 or step == args.steps):
                log_write(str(wlog), f"[{step:04d}/{args.steps}] total={t_val:.4f} c = {c_val: .4f} s = {s_val: .4f} tv = {tv_val: .6f}")

            # 주기적 샘플 저장
            if step % save_every == 0 or step in (1, args.steps):
                sample_path = out_dir / f"samples_step{step:04d}.png"
                save_tensor_as_png(gen, sample_path)
                last_save_step = step

            step += 1  # 정상 완료 시에만 증가

        except RuntimeError as e:
            # 윈도우 TDR 등 CUDA 타임아웃 폴백
            msg = str(e)
            timeout_sign = ("the launch timed out and was terminated" in msg) or \
                           ("CUDA error" in msg and "timed out" in msg)
            if timeout_sign and device.type == "cuda":
                log_write(str(wlog), "[WARN] CUDA timeout 발생 → CPU로 폴백하여 이어서 진행합니다.")
                # 캐시 정리 및 CPU 전환
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                device = torch.device("cpu")
                use_amp = False

                # 텐서 이동
                content = content.cpu()
                style = style.cpu()
                with torch.no_grad():
                    gen_data = gen.detach().cpu()
                gen = gen_data.requires_grad_(True)

                # 모델 이동
                feat_net = feat_net.cpu()

                # 옵티마이저 재생성(동일 하이퍼파라미터)
                if args.optimizer.lower() == "lbfgs":
                    optimizer = torch.optim.LBFGS([gen], max_iter=1)
                else:
                    optimizer = torch.optim.Adam([gen], lr=args.lr)
            # 같은 step을 CPU에서 재시도하므로 step 증가하지 않음
                continue
            else:
                # 다른 오류는 그대로 raise
                raise

    # 최종 결과 저장
    result_path = out_dir / "result_final.png"
    save_tensor_as_png(gen, result_path)

    # 비교 그리드 저장
    compare_path = out_dir / "compare_grid.png"
    make_compare_grid(content, style, gen, compare_path)

    # 손실 곡선 저장
    plot_losses(loss_log, out_dir / "loss_curve.png")

    # 요약 로그
    dt = time.time() - t0
    log_write(str(wlog), f"[DONE] steps={args.steps} time={dt:.1f}s last_sample_step = {last_save_step}")
    log_write(str(wlog), f"saved: {result_path.name}, compare_grid.png, loss_curve.png")
    print("Finish.")

def run_gen_adain_stub(args):
    # 간단 안내: 본 스크립트는 이미지 최적화 방식만 지원
    print("현재 스크립트는 --mode gen (AdaIN 네트워크) 생성을 지원하지 않습니다.")
    print("이미지 최적화 방식( --mode train )을 사용하세요.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train","gen"], default="train")
    p.add_argument("--content", type=str, default="/home/choe/Downloads/picasso.jpg")
    p.add_argument("--style", type=str, default="/home/choe/Downloads/ref.jpg")
    p.add_argument("--size", type=int, default=512, help="짧은 변 기준 리사이즈")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=5000.0)
    p.add_argument("--gamma", type=float, default=1e-5)
    p.add_argument("--tv", type=str, default="l2", choices=["l1","l2"])
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam","lbfgs"])
    p.add_argument("--content_layer", type=str, default="relu4_2")
    p.add_argument("--style_layers", nargs="+", default=["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"])
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="runs_style")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--quick", action="store_true", help="로깅/시각화 최소화")
    # 추가 옵션
    p.add_argument("--amp", action="store_true", help="CUDA AMP(FP16) 사용")
    p.add_argument("--safe", action="store_true", help="안전 프로파일: size<=256, steps<=300, LBFGS, AMP off")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_gen_adain_stub(args)
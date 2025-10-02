import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

# ----------------------------
# 재현성 & 상수
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
SUPPORTED = ["cnn", "resnet18"]

# ----------------------------
# 전처리
# ----------------------------
def build_transforms(use_colorjitter=False):
    aug = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    if use_colorjitter:
        aug.append(transforms.ColorJitter(0.2, 0.2, 0.2))
    train_tf = transforms.Compose(aug + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return train_tf, eval_tf

def inv_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3,1,1)
    std = torch.tensor(STD).view(3,1,1)
    return (x * std + mean).clamp(0,1)

# ----------------------------
# 데이터
# ----------------------------
def prepare_dataloaders(data_root: Path, batch: int, num_workers: int, device: str,
                        val_size: int = 5000, use_colorjitter=False):
    train_tf, eval_tf = build_transforms(use_colorjitter)
    full_train = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=eval_tf)

    n = len(full_train)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    train_ds = Subset(full_train, tr_idx)

    val_raw = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=eval_tf)
    val_ds = Subset(val_raw, val_idx)

    use_cuda = (device == "cuda")
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    val_dl = DataLoader(val_ds, batch_size=batch*2, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    test_dl = DataLoader(test_ds, batch_size=batch*2, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    return train_dl, val_dl, test_dl

# ----------------------------
# 모델
# ----------------------------
class CNNBN(nn.Module):
    def __init__(self, p1=0.2, p2=0.3, num_classes=10):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )
        self.features = nn.Sequential(
            blk(3,64), blk(64,64), nn.MaxPool2d(2), nn.Dropout(p1),
            blk(64,128), blk(128,128), nn.MaxPool2d(2), nn.Dropout(p2),
            blk(128,256), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

def build_model(name="cnn", num_classes=10, pretrained=False, freeze=False):
    name = name.lower()
    if name == "cnn":
        return CNNBN(num_classes=num_classes)
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if freeze:
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"지원 모델: {SUPPORTED}")

# ----------------------------
# 학습/검증
# ----------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cpu", amp=False, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, y_true, y_pred = 0.0, [], []

    if amp and device == "cuda" and scaler is None:
        from torch import amp as _amp
        scaler = _amp.GradScaler('cuda')

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if train and amp and device == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        tot_loss += loss.item() * xb.size(0)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())

    avg_loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc

def plot_curves(history: Dict[str, List[float]], out_dir: Path):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir/"loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir/"acc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

def evaluate_and_cm(model, loader, device, out_dir: Path, split="test", classes=None):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            ys.extend(yb.numpy())
            ps.extend(logits.argmax(1).cpu().numpy())
    ys, ps = np.array(ys), np.array(ps)
    acc = accuracy_score(ys, ps)
    cm = confusion_matrix(ys, ps, labels=list(range(10)))
    plt.figure(figsize=(6,6))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes if classes else list(range(10)))
    disp.plot(cmap="Blues", colorbar=False, ax=plt.gca())
    plt.title(f"{split} Confusion Matrix (acc={acc:.3f})")
    plt.savefig(out_dir/f"{split}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {split} accuracy = {acc:.4f}")
    return acc

# ----------------------------
# 단일 이미지 예측
# ----------------------------
def load_and_preprocess_image(img_path: Path) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img)

def predict_single(model, img_path: Path, device, out_dir: Path, class_names: List[str]):
    model.eval()
    x = load_and_preprocess_image(img_path).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(prob.argmax())
    pred_name = class_names[pred_idx]
    print(f"[PREDICT] {img_path.name} → {pred_name} (p={prob[pred_idx]:.4f})")

    x_vis = inv_normalize(x[0].cpu()).permute(1,2,0).numpy()
    plt.figure()
    plt.imshow(x_vis)
    plt.axis("off")
    plt.title(f"{pred_name} (p={prob[pred_idx]:.2f})")
    plt.savefig(out_dir/"single_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    top5 = np.argsort(prob)[::-1][:5]
    lines = [f"{i}: {class_names[i]} - {prob[i]:.4f}" for i in top5]
    (out_dir/"predict_top5.txt").write_text("\n".join(lines), encoding="utf-8")

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="runs_cifar10")
    ap.add_argument("--model", type=str, default="cnn", choices=SUPPORTED)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--predict_path", type=str, default="")
    args, _ = ap.parse_known_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    train_dl, val_dl, test_dl = prepare_dataloaders(
        data_root=data_dir, batch=args.batch, num_workers=4, device=device,
        val_size=5000, use_colorjitter=args.jitter
    )
    class_names = datasets.CIFAR10(root=str(data_dir), train=False, download=True).classes

    model = build_model(args.model, num_classes=10, pretrained=args.pretrained, freeze=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device=="cuda"))

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val, patience, patience_lim = 0.0, 0, 10
    best_path = out_dir/"best_cifar10.pt"

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, device, amp=(args.amp and device=="cuda"), scaler=scaler)
        val_loss, val_acc = run_epoch(model, val_dl, criterion, optimizer=None, device=device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"[EPOCH {ep:02d}] train {tr_acc*100:.2f}%/{tr_loss:.3f} | val {val_acc*100:.2f}%/{val_loss:.3f} | lr {scheduler.get_last_lr()[0]:.5f}")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] ✅ Saved best: {best_path} (val_acc={best_val:.4f})")
        else:
            patience += 1
            if patience >= patience_lim:
                print("[INFO] Early stopping triggered.")
                break

    plot_curves(history, out_dir)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}")
        test_acc = evaluate_and_cm(model, test_dl, device, out_dir, split="test", classes=class_names)
        print(f"[RESULT] Test accuracy = {test_acc:.4f}")

    if args.predict_path:
        p = Path(args.predict_path)
        if p.exists():
            predict_single(model, p, device, out_dir, class_names)
            print(f"[INFO] Saved single_prediction.png / predict_top5.txt to {out_dir}")
        else:
            print(f"[WARN] --predict_path not found: {p}")
    else:
        test_raw = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=transforms.ToTensor())
        idxs = random.sample(range(len(test_raw)), 8)
        plt.figure(figsize=(12,6))
        for i, idx in enumerate(idxs):
            img, y = test_raw[idx]
            x = transforms.Normalize(MEAN, STD)(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x).argmax(1).item()
            plt.subplot(2,4,i+1)
            plt.imshow(img.permute(1,2,0))
            plt.axis("off")
            plt.title(f"GT:{class_names[y]}\nPred:{class_names[pred]}")
        plt.tight_layout()
        plt.savefig(out_dir/"sample_predictions.png", dpi=150)
        plt.close()
        print(f"[INFO] Saved: sample_predictions.png")

if __name__ == "__main__":
    main()

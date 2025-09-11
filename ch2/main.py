import os, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(batch_size:int=64, num_workers:int=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set, valid_set = torch.utils.data.random_split(
        train_set, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, valid_loader, test_loader

class SimpleCNN(nn.Module):
    def __init__(self, dropout:float=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def train_one_epoch(model, loader, optimizer, scaler, use_amp:bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return avg_loss, acc, all_preds, all_targets


def plot_curves(history, save_path:str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["valid_acc"], label="Valid Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path:str, labels=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_misclassified_images(y_true, y_pred, save_path:str, max_samples:int=25):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    inv_transform = transforms.Compose([
        transforms.Normalize((0.0,), (1/0.3081,)),
        transforms.Normalize((-0.1307,), (1.0,))
    ])

    mis_idx = np.where(y_true != y_pred)[0][:max_samples]
    if len(mis_idx) == 0:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No misclassified samples", horizontalalignment="center", verticalalignment="center")
        plt.axis('off')
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    cols = 5
    rows = int(np.ceil(len(mis_idx) / cols))
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, idx in enumerate(mis_idx):
        img, label = test_set[idx]
        img = inv_transform(img)
        img_np = img.squeeze().numpy()
        pred = y_pred[idx]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_np, cmap="gray")
        plt.title(f"GT:{label} / Pred:{pred}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"✅ Device: {device} | AMP: {'ON' if (args.amp and device.type == 'cuda') else 'OFF'}")
    train_loader, valid_loader, test_loader = get_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = SimpleCNN(dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == 'cuda'))

    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
    best_acc, best_path = 0.0, os.path.join(args.out_dir, "best_mnist_cnn.pt")

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, use_amp=args.amp)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_loader)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_path)

        dt = time.time() - t0
        print(f"[Epoch {epoch:02d} / {args.epochs}]"
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} |"
              f"Valid: loss={valid_loss:.4f} acc={valid_acc:.4f} |"
              f"Time: {dt:.1f}s | BestAcc={best_acc:.4f}")

    plot_curves(history, os.path.join(args.out_dir, "train_curve.png"))

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader)
    print(f"\n[CHART] Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    labels = list(range(10))
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.out_dir, "confusion_matrix.png"), labels=labels)
    save_misclassified_images(y_true, y_pred, os.path.join(args.out_dir, "misclassified_samples.png"))

    print(f"\n[DIR] 결과 저장 위치: {os.path.abspath(args.out_dir)}")
    print(" - 모델 가중치: best_mnist_cnn.pt")
    print(" - 학습곡선: train_curve.png")
    print(" - 혼동행렬: confusion_matrix.png")
    print(" - 오분류샘플: misclassified_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST CNN Training (PyTorch)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="CUDA에서 자동 혼합 정밀도(AMP) 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./outputs")
    args = parser.parse_args()
    main(args)
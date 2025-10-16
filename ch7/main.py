# -*- coding: utf-8 -*-
"""
IMDB 감정 분석 올인원(토치텍스트 제거 + Windows 멀티워커 대응)
- 데이터: IMDB (train 25k / test 25k) [datasets 라이브러리 사용]
- 분할: train(90%) / val(10%) / test(25k)
- 모델: Embedding + (RNN/LSTM/GRU)
- 최적화: Adam, Early Stopping
- 결과: runs_imdb/ 에 곡선/혼동행렬/모델/예측 저장
"""

import os, re, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ----------------------------
# 재현성 & 상수
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_VOCAB = 20000
MIN_FREQ = 2
DEFAULT_MAX_LEN = 300
SUPPORTED = ["rnn", "lstm", "gru"]

# ----------------------------
# 유틸: 토큰화/정리
# ----------------------------
def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9'!?., ]+", " ", text)
    return text

def basic_tokenize(text: str) -> List[str]:
    return basic_clean(text).split()

# ----------------------------
# 어휘사전 빌드(순수 파이썬)
# ----------------------------
class Vocab:
    def __init__(self, counter: Counter, max_tokens: int, min_freq: int):
        self.itos = [PAD_TOKEN, UNK_TOKEN]
        words = [w for w, c in counter.items() if c >= min_freq and w not in {PAD_TOKEN, UNK_TOKEN}]
        words.sort(key=lambda w: (-counter[w], w))
        if max_tokens is not None:
            remain = max_tokens - len(self.itos)
            words = words[:max(0, remain)]
        self.itos.extend(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK_TOKEN])

    def get_stoi(self):
        return self.stoi

# ----------------------------
# 데이터셋 & 콜레이트
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, data: List[Tuple[int, List[int]]]):
        super().__init__()
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        label, ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor([float(label)], dtype=torch.float32)


def pad_truncate(seq: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


class PadCollate:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch):
        xs, ys = zip(*batch)

        def pad_truncate_local(tensor1d, max_len, pad_id):
            lst = tensor1d.tolist() if hasattr(tensor1d, "tolist") else list(tensor1d)
            if len(lst) >= max_len:
                return lst[:max_len]
            return lst + [pad_id] * (max_len - len(lst))

        xs = [pad_truncate_local(x, self.max_len, self.pad_id) for x in xs]
        return torch.tensor(xs, dtype=torch.long), torch.vstack(ys)

# ----------------------------
# 모델
# ----------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=100, hidden=128, rnn_type="lstm",
                 num_layers=1, bidirectional=False, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type.lower()
        self.bid = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"지원 모델: {SUPPORTED}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * self.num_dirs, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, h = self.rnn(emb)
        if self.rnn_type == "lstm":
            if self.bid:
                last_h = torch.cat([h[0][-2], h[0][-1]], dim=1)
            else:
                last_h = h[0][-1]
        else:
            if self.bid:
                last_h = torch.cat([h[-2], h[-1]], dim=1)
            else:
                last_h = h[-1]
        last_h = self.dropout(last_h)
        logit = self.fc(last_h)
        return logit

# ----------------------------
# 학습/검증 루프
# ----------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    tot_loss, y_true, y_pred = 0.0, [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        tot_loss += loss.item() * xb.size(0)
        prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        pred = (prob >= 0.5).astype(np.int64)
        y_true.extend(yb.cpu().numpy().ravel().astype(np.int64))
        y_pred.extend(pred)

    avg_loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc

def plot_curves(history: Dict[str, List[float]], out_dir: Path):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch");
    plt.ylabel("loss");
    plt.title("Loss")
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight");
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch");
    plt.ylabel("accuracy");
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.savefig(out_dir / "acc_curve.png", dpi=150, bbox_inches="tight");
    plt.close()

def evaluate_and_plot(model, loader, device, out_dir: Path, split="test"):
    model.eval();
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            pred = (prob >= 0.5).astype(np.int64)
            ys.extend(yb.numpy().ravel().astype(np.int64))
            ps.extend(pred)
    ys, ps = np.array(ys), np.array(ps)
    acc = accuracy_score(ys, ps)
    cm = confusion_matrix(ys, ps, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["neg(0)", "pos(1)"]).plot(cmap="Blues", colorbar=False)
    plt.title(f"{split} Confusion Matrix (acc={acc:.3f})")
    plt.savefig(out_dir / f"{split}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {split} accuracy = {acc:.4f}")
    return acc

# ----------------------------
# 텍스트 → 예측
# ----------------------------
def encode_sentence(text: str, vocab: Vocab, max_len: int, pad_id: int):
    toks = basic_tokenize(text)
    ids = [vocab[t] for t in toks]
    return pad_truncate(ids, max_len, pad_id)


def predict_sentence(model, text_or_path: str, vocab: Vocab, max_len: int, pad_id: int, device, out_dir: Path):
    # 파일 경로가 들어오면 내용 읽어서 예측
    if os.path.exists(text_or_path):
        text = Path(text_or_path).read_text(encoding="utf-8", errors="ignore")
    else:
        text = text_or_path
    model.eval()
    ids = encode_sentence(text, vocab, max_len, pad_id)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    label = 1 if p >= 0.5 else 0
    msg = f"'{text[:80]}...' → pred={label} (p(pos)={p:.4f})"
    print("[PREDICT]", msg)
    (out_dir/"my_sentence_pred.txt").write_text(msg+"\n", encoding="utf-8")

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs_imdb")
    ap.add_argument("--model", type=str, default="lstm", choices=SUPPORTED)
    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader num_workers (Windows issue 시 0)")
    ap.add_argument("--predict_text", type=str, default="C:/Users/KYH/Desktop/딥러닝프로그래밍/행복해.txt",
                    help="예측할 임의 문장(문자열 또는 텍스트 파일 경로)")
    args, _ = ap.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # 1) 데이터 로드 (datasets IMDB)
    ds = load_dataset("imdb")
    train_raw = list(zip(ds["train"]["label"], ds["train"]["text"]))  # (label:int, text:str)
    test_raw = list(zip(ds["test"]["label"], ds["test"]["text"]))

    # 2) 학습/검증 분리(라벨은 0/1 그대로 사용)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_raw))
    val_size = int(0.1 * len(train_raw))  # 2,500
    val_idx = set(perm[:val_size])
    train_spl = [train_raw[i] for i in range(len(train_raw)) if i not in val_idx]
    val_spl = [train_raw[i] for i in val_idx]

    # 3) 어휘사전(훈련셋만으로 구축)
    counter = Counter()
    for y, txt in train_spl:
        counter.update(basic_tokenize(txt))
    vocab = Vocab(counter, max_tokens=MAX_VOCAB, min_freq=MIN_FREQ)
    pad_id = vocab[PAD_TOKEN]

    # 4) 텍스트 → ids 변환
    def to_ids(split):
        data = []
        for y, txt in split:
            toks = basic_tokenize(txt)
            ids = [vocab[t] for t in toks]
            data.append((int(y), ids))
        return data

    train_ids = to_ids(train_spl)
    val_ids = to_ids(val_spl)
    test_ids = to_ids(test_raw)

    # 5) DataLoader
    collate = PadCollate(args.max_len, pad_id)
    use_cuda = (device == "cuda")

    # Windows 안전 설정: persistent_workers=False 권장
    train_loader = DataLoader(
        TextDataset(train_ids), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=use_cuda, collate_fn=collate,
        persistent_workers=False if args.workers > 0 else False
    )
    val_loader = DataLoader(
        TextDataset(val_ids), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=use_cuda, collate_fn=collate,
        persistent_workers=False if args.workers > 0 else False
    )
    test_loader = DataLoader(
        TextDataset(test_ids), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=use_cuda, collate_fn=collate,
        persistent_workers=False if args.workers > 0 else False
    )

    # 6) 모델
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        rnn_type=args.model,
        num_layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=0.5,
        pad_idx=pad_id
    ).to(device)
    print("[INFO] 랜덤 임베딩 사용 (사전학습 임베딩 미사용)")

    # 7) 학습
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, patience, patience_lim = 0.0, 0, 3
    best_path = out_dir / "best_imdb.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"[EPOCH {ep:02d}] train {tr_acc * 100:.2f}%/{tr_loss:.3f} | val {val_acc * 100:.2f}%/{val_loss:.3f}")

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

    # 8) 테스트 (best 로드 후)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}")
        test_acc = evaluate_and_plot(model, test_loader, device, out_dir, split="test")
        print(f"[RESULT] Test accuracy = {test_acc:.4f}")

    # 9) 내 문장 예측(파일 경로 입력도 지원)
    if args.predict_text:
        predict_sentence(model, args.predict_text, vocab, args.max_len, pad_id, device, out_dir)
        print(f"[INFO] Saved: my_sentence_pred.txt at {out_dir}")
    else:
        samples = [
            "this movie was surprisingly touching and beautifully acted.",
            "what a waste of time. the plot makes no sense at all.",
            "average film, but the soundtrack was great."
        ]
        with open(out_dir / "sample_predictions.txt", "w", encoding="utf-8") as f:
            for s in samples:
                ids = encode_sentence(s, vocab, args.max_len, pad_id)
                x = torch.tensor(ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model(x)).item()
                f.write(f"{s}\n→ pred={1 if p >= 0.5 else 0} (p(pos)={p:.4f})\n\n")
        print(f"[INFO] Saved: sample_predictions.txt")


if __name__ == "__main__":
    main()
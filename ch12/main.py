import argparse
import os
import time
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple\

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import pandas as pd
from tqdm import tqdm


# =========================
#  Text Tokenizer (Char-level CTC)
# =========================

class CharTokenizer:
    """
    단순 문자 단위 토크나이저.
    - blank 토큰 id = 0 (CTC 용도)
    - 나머지는 문자 사전 순서대로 1부터 부여
    """

    def __init__(self, vocab: List[str] = None):
        if vocab is None:
            # 첫 토큰은 항상 <blank>
            self.id2ch = ["<blank>"]
            self.ch2id = {"<blank>": 0}
        else:
            self.id2ch = vocab
            self.ch2id = {ch: i for i, ch in enumerate(vocab)}

    @staticmethod
    def build_from_texts(texts: List[str]) -> "CharTokenizer":
        chars = set()
        for t in texts:
            chars.update(list(t))
        chars = sorted(list(chars))
        vocab = ["<blank>"] + chars
        return CharTokenizer(vocab=vocab)

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch in self.ch2id:
                ids.append(self.ch2id[ch])
            # 모르는 문자는 생략 (필요하면 <unk> 추가 가능)
        return ids

    def decode(self, ids: List[int]) -> str:
        # CTC greedy에서 blank와 반복문자 제거는 따로 처리하므로 여기서는 단순 매핑
        chars = []
        for i in ids:
            if 0 <= i < len(self.id2ch):
                ch = self.id2ch[i]
                if ch != "<blank>":
                    chars.append(ch)
        return "".join(chars)

    def to_dict(self) -> Dict:
        return {"vocab": self.id2ch}

    @staticmethod
    def from_dict(d: Dict) -> "CharTokenizer":
        return CharTokenizer(vocab=d["vocab"])


# =========================
#  Audio & Feature Processing
# =========================

@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg

        n_fft = int(cfg.sample_rate * cfg.win_length_ms / 1000)
        hop_length = int(cfg.sample_rate * cfg.hop_length_ms / 1000)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=cfg.n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, T) 또는 (B, T)
        출력: (time, n_mels)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        with torch.no_grad():
            melspec = self.melspec(wav)  # (B, n_mels, time)
            melspec = torch.log1p(melspec)  # log(1 + x)

            # 발화별 평균/표준편차 정규화
            mean = melspec.mean(dim=-1, keepdim=True)
            std = melspec.std(dim=-1, keepdim=True) + 1e-5
            melspec = (melspec - mean) / std

        # (B, n_mels, time) -> (time, n_mels) (batch=1 가정)
        return melspec[0].transpose(0, 1)  # (time, n_mels)


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(path)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 2:
        # stereo -> mono
        wav = wav.mean(dim=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


# =========================
#  Model: Conv + BiLSTM + CTC
# =========================

@dataclass
class ModelConfig:
    input_dim: int = 80
    cnn_channels: int = 128
    cnn_kernel: int = 3
    cnn_layers: int = 2
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 100  # 실제 vocab 크기 + 1(blank)


class ConvBiLSTMCTC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        cnn_layers = []
        in_ch = 1
        for i in range(cfg.cnn_layers):
            out_ch = cfg.cnn_channels
            cnn_layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(cfg.cnn_kernel, 3),
                    padding=(cfg.cnn_kernel // 2, 1),
                    stride=(1, 2),  # 시간축:1, 주파수축:2 다운샘플
                )
            )
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # CNN 출력 차원: (B, C, T, F') -> LSTM input_dim = C * F'
        # F'는 input_dim(=n_mels)을 2^(cnn_layers)로 나눈 값 근사
        freq_after = cfg.input_dim // (2 ** cfg.cnn_layers)
        lstm_input_dim = cfg.cnn_channels * freq_after

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.lstm_hidden * 2, cfg.num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, F)
        return: log_probs (T', B, C)
        """
        B, T, feat_dim = feats.shape
        x = feats.unsqueeze(1)  # (B, 1, T, F)
        x = self.cnn(x)  # (B, C, T', F')

        B, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T', C, F')
        x = x.view(B, T2, C * F2)  # (B, T', C*F')

        x, _ = self.lstm(x)  # (B, T', 2*hidden)
        x = self.dropout(x)
        logits = self.fc(x)  # (B, T', num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T', C)
        log_probs = log_probs.permute(1, 0, 2)  # (T', B, C) for CTC loss
        return log_probs


# =========================
#  CTC Greedy Decoder
# =========================

def ctc_greedy_decode(
    log_probs: torch.Tensor, blank_id: int = 0
) -> List[List[int]]:
    """
    log_probs: (T, B, C)
    return: List of sequences (ids) for each batch element
    """
    probs = log_probs.exp()  # (T, B, C)
    max_ids = probs.argmax(dim=-1)  # (T, B)
    max_ids = max_ids.cpu().numpy()

    results = []
    T, B = max_ids.shape
    for b in range(B):
        prev = blank_id
        seq = []
        for t in range(T):
            idx = int(max_ids[t, b])
            if idx != blank_id and idx != prev:
                seq.append(idx)
            prev = idx
        results.append(seq)
    return results


# =========================
#  Metrics (CER & WER)
# =========================

def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0
    return levenshtein(list(ref), list(hyp)) / len(ref)


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


# =========================
#  Checkpoint IO
# =========================

def save_checkpoint(path: str, model: ConvBiLSTMCTC, tokenizer: CharTokenizer, feat_cfg: FeatureConfig, model_cfg: ModelConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "feature_config": asdict(feat_cfg),
        "model_config": asdict(model_cfg),
    }
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    feat_cfg = FeatureConfig(**ckpt["feature_config"])
    model_cfg = ModelConfig(**ckpt["model_config"])
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])

    model_cfg.num_classes = len(tokenizer.id2ch)
    model = ConvBiLSTMCTC(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, feat_cfg


# =========================
#  Inference: single file
# =========================

def infer_one_file(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, feat_cfg = load_checkpoint(args.ckpt, device)
    feat_extractor = FeatureExtractor(feat_cfg)

    wav = load_audio(args.wav, target_sr=feat_cfg.sample_rate)
    duration = len(wav) / feat_cfg.sample_rate

    feats = feat_extractor(wav)  # (T, F)
    feats = feats.unsqueeze(0).to(device)  # (1, T, F)

    with torch.no_grad():
        log_probs = model(feats)  # (T', 1, C)
        hyp_ids = ctc_greedy_decode(log_probs, blank_id=0)[0]

        # 🔻 아무 글자도 안 나왔을 때 응급처치
        if len(hyp_ids) == 0:
            # 각 time step에서 argmax 뽑아서 그냥 디코딩
            frame_ids = log_probs.argmax(dim=-1)[:, 0].cpu().tolist()
            hyp_text = tokenizer.decode(frame_ids)
        else:
            hyp_text = tokenizer.decode(hyp_ids)

        hyp_text = tokenizer.decode(hyp_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    transcripts_csv = os.path.join(args.out_dir, "transcripts.csv")

    row = {
        "wav_path": args.wav,
        "duration_sec": duration,
        "mode": "greedy",
        "transcript": hyp_text,
    }

    if os.path.exists(transcripts_csv):
        df = pd.read_csv(transcripts_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(transcripts_csv, index=False, encoding="utf-8")

    print(f"[RESULT] {args.wav}")
    print(hyp_text)


# =========================
#  Inference: directory
# =========================

def infer_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, feat_cfg = load_checkpoint(args.ckpt, device)
    feat_extractor = FeatureExtractor(feat_cfg)

    wav_paths = []
    for root, _, files in os.walk(args.in_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3")):
                wav_paths.append(os.path.join(root, f))

    os.makedirs(args.out_dir, exist_ok=True)
    transcripts_csv = os.path.join(args.out_dir, "transcripts.csv")

    rows = []
    for path in tqdm(wav_paths, desc="infer-dir"):
        wav = load_audio(path, target_sr=feat_cfg.sample_rate)
        duration = len(wav) / feat_cfg.sample_rate

        feats = feat_extractor(wav)  # (T, F)
        feats = feats.unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(feats)
            hyp_ids = ctc_greedy_decode(log_probs, blank_id=0)[0]
            hyp_text = tokenizer.decode(hyp_ids)

        rows.append(
            {
                "wav_path": path,
                "duration_sec": duration,
                "mode": "greedy",
                "transcript": hyp_text,
            }
        )

    if os.path.exists(transcripts_csv):
        df_prev = pd.read_csv(transcripts_csv)
        df = pd.concat([df_prev, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(transcripts_csv, index=False, encoding="utf-8")
    print(f"[INFO] Saved transcripts to {transcripts_csv}")


# =========================
#  Training / Finetune
# =========================

def load_wav_txt_pairs(in_wav_dir: str, in_txt_dir: str) -> List[Tuple[str, str]]:
    wav_files = []
    for f in os.listdir(in_wav_dir):
        if f.lower().endswith((".wav", ".flac", ".mp3")):
            wav_files.append(f)

    pairs = []
    for wf in wav_files:
        base = os.path.splitext(wf)[0]
        txt_path = os.path.join(in_txt_dir, base + ".txt")
        wav_path = os.path.join(in_wav_dir, wf)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            pairs.append((wav_path, text))
    return pairs


def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    pairs = load_wav_txt_pairs(args.in_wav, args.in_txt)
    if len(pairs) == 0:
        print("[ERROR] No wav/txt pairs found.")
        return

    texts = [t for _, t in pairs]
    tokenizer = CharTokenizer.build_from_texts(texts)
    feat_cfg = FeatureConfig()

    model_cfg = ModelConfig()
    model_cfg.input_dim = feat_cfg.n_mels
    model_cfg.num_classes = len(tokenizer.id2ch)

    model = ConvBiLSTMCTC(model_cfg).to(device)
    feat_extractor = FeatureExtractor(feat_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, math.ceil(len(pairs) / args.batch))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
    )
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # 아주 단순한 미니배치 구현 (정렬/패딩 없이 one-by-one)
        for i in tqdm(range(0, len(pairs), args.batch), desc=f"epoch {epoch+1}"):
            batch_pairs = pairs[i: i + args.batch]

            optimizer.zero_grad()
            batch_loss = 0.0

            for wav_path, text in batch_pairs:
                wav = load_audio(wav_path, target_sr=feat_cfg.sample_rate)
                feats = feat_extractor(wav)  # (T, F)
                feats = feats.unsqueeze(0).to(device)  # (1, T, F)

                log_probs = model(feats)  # (T', 1, C)
                input_lengths = torch.tensor(
                    [log_probs.size(0)], dtype=torch.long, device=device
                )

                target_ids = torch.tensor(
                    tokenizer.encode(text), dtype=torch.long, device=device
                )
                target_lengths = torch.tensor(
                    [len(target_ids)], dtype=torch.long, device=device
                )

                loss = ctc_loss_fn(
                    log_probs,  # (T', B, C)
                    target_ids.unsqueeze(0),  # (B, S)
                    input_lengths,
                    target_lengths,
                )
                loss.backward()
                batch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += batch_loss

        avg_loss = epoch_loss / max(1, (len(pairs) / args.batch))
        print(f"[EPOCH {epoch+1}] loss = {avg_loss:.4f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch {epoch+1}, loss {avg_loss:.4f}\n")

        # 매 epoch마다 last 저장, 가장 낮은 loss 기준 best 저장
        last_ckpt = os.path.join(args.out_dir, "checkpoints", "last.ckpt")
        save_checkpoint(last_ckpt, model, tokenizer, feat_cfg, model_cfg)

    # 마지막 epoch를 best로 취급 (간단 예시)
    best_ckpt = os.path.join(args.out_dir, "checkpoints", "best.ckpt")
    save_checkpoint(best_ckpt, model, tokenizer, feat_cfg, model_cfg)


# =========================
#  Evaluation
# =========================

def eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, feat_cfg = load_checkpoint(args.ckpt, device)
    feat_extractor = FeatureExtractor(feat_cfg)

    pairs = load_wav_txt_pairs(args.in_wav, args.in_txt)
    if len(pairs) == 0:
        print("[ERROR] No wav/txt pairs found.")
        return

    rows = []
    for wav_path, ref_text in tqdm(pairs, desc="eval"):
        wav = load_audio(wav_path, target_sr=feat_cfg.sample_rate)
        feats = feat_extractor(wav)
        feats = feats.unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(feats)
            hyp_ids = ctc_greedy_decode(log_probs, blank_id=0)[0]
            hyp_text = tokenizer.decode(hyp_ids)

        cer_val = cer(ref_text, hyp_text)
        wer_val = wer(ref_text, hyp_text)

        rows.append(
            {
                "wav_path": wav_path,
                "ref": ref_text,
                "hyp": hyp_text,
                "cer": cer_val,
                "wer": wer_val,
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    cer_wer_csv = os.path.join(args.out_dir, "cer_wer.csv")
    df = pd.DataFrame(rows)
    df.to_csv(cer_wer_csv, index=False, encoding="utf-8")
    print(f"[INFO] Saved CER/WER to {cer_wer_csv}")

    # 대표 20개 샘플 저장
    samples_path = os.path.join(args.out_dir, "samples_pred.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in rows[:20]:
            f.write(f"[REF] {row['ref']}\n")
            f.write(f"[HYP] {row['hyp']}\n")
            f.write("\n")
    print(f"[INFO] Saved samples to {samples_path}")


# =========================
#  CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Simple STT (Conv+BiLSTM+CTC)")
    subparsers = parser.add_subparsers(dest="command")

    # A. 단일 파일 전사
    p_infer = subparsers.add_parser("infer")
    p_infer.add_argument("--ckpt", type=str, required=True)
    p_infer.add_argument("--wav", type=str, required=True)
    p_infer.add_argument("--out_dir", type=str, default="runs_stt")
    # max_len, overlap, vad는 여기서는 구현 단순화를 위해 사용하지 않음
    p_infer.set_defaults(func=infer_one_file)

    # B. 폴더 배치 전사
    p_infer_dir = subparsers.add_parser("infer-dir")
    p_infer_dir.add_argument("--ckpt", type=str, required=True)
    p_infer_dir.add_argument("--in_dir", type=str, required=True)
    p_infer_dir.add_argument("--out_dir", type=str, default="runs_stt")
    p_infer_dir.set_defaults(func=infer_dir)

    # D. 소규모 미세조정
    p_ft = subparsers.add_parser("finetune")
    p_ft.add_argument("--in_wav", type=str, required=True)
    p_ft.add_argument("--in_txt", type=str, required=True)
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--batch", type=int, default=4)
    p_ft.add_argument("--lr", type=float, default=2e-4)
    p_ft.add_argument("--out_dir", type=str, default="runs_stt")
    p_ft.set_defaults(func=finetune)

    # 평가
    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--in_wav", type=str, required=True)
    p_eval.add_argument("--in_txt", type=str, required=True)
    p_eval.add_argument("--out_dir", type=str, default="runs_stt")
    p_eval.set_defaults(func=eval_model)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    start = time.time()
    args.func(args)
    end = time.time()
    print(f"[INFO] Elapsed: {end - start:.2f} sec")


if __name__ == "__main__":
    main()
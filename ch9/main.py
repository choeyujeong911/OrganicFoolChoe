# -*- coding: utf-8 -*-
"""
Fast NMT(EN→FR) for class demo
- Dataset: Tatoeba en-fr (short pairs); fallback to tiny in-memory pairs
- Tokenizer: SentencePiece BPE(shared, vocab≈2k), max_len=40 (BOS/EOS 포함)
- Model: BiGRU Encoder + Luong(dot) Attention + GRU Decoder + tied embeddings
- Tricks: teacher forcing(0.7), beam search(4), AMP, grad clip
- Outputs: runs_nmt/loss_curve.png, bleu_curve.png, attention_example.png,
best_fr_mt.pt, my_sentence_translation.txt
"""
import os, argparse, random, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------- optional HF datasets -----------
try:
    from datasets import load_dataset
    HF_OK = True
except Exception:
    HF_OK = False

import sentencepiece as spm
import sacrebleu

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED);
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ---------- tiny fallback (contains “I like apples.”) ----------
F_EN = [
    "I like apples .", "He is a student .", "We love deep learning .",
    "This book is interesting .", "She lives in Paris .",
    "The weather is good today .", "I want to learn French .",
    "They are playing football .", "This movie is very funny .",
    "Do you understand this lesson ?",
    "I am happy today .", "This is a small cat .", "Where is the station ?",
    "I like music and books .", "Thank you very much .", "See you tomorrow .",
]
F_FR = [
    "J'aime les pommes .", "Il est étudiant .", "Nous aimons l'apprentissage profond.",
    "Ce livre est intéressant .", "Elle vit à Paris .",
    "Il fait beau aujourd'hui .", "Je veux apprendre le français .",
    "Ils jouent au football .", "Ce film est très drôle .",
    "Comprends-tu cette leçon ?",
    "Je suis heureux aujourd'hui .", "C'est un petit chat .", "Où est la gare ?",
    "J'aime la musique et les livres .", "Merci beaucoup .", "À demain .",
]

# ---------------------------- data ----------------------------
def load_enfr(source="tatoeba", subset=6000, val_ratio=0.1, test_ratio=0.1) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]], List[Tuple[str,str]]]:
    pairs = []
    if HF_OK and source == "tatoeba":
        try:
            ds = load_dataset("tatoeba", "eng-fra") # small, short, clean
            for split in ["train", "validation", "test"]:
                if split in ds:
                    for r in ds[split]:
                        en = r["source_sentence"]; fr = r["target_sentence"]
                        if en and fr and 1 <= len(en) <= 120 and 1 <= len(fr) <= 120:
                            pairs.append((en.strip(), fr.strip()))
        except Exception as e:
            print(f"[WARN] HF Tatoeba load failed: {e}")
            pairs = []

    if not pairs:
        # very small fallback (replicate to a few thousand to stabilize training)
        pairs = (list(zip(F_EN, F_FR)) * 200) # ~3.2k
    random.shuffle(pairs)
    if subset and len(pairs) > subset:
        pairs = pairs[:subset]
    n = len(pairs)
    n_test = max(50, int(n * test_ratio))
    n_val = max(50, int(n * val_ratio))
    test = pairs[:n_test]
    val = pairs[n_test:n_test+n_val]
    train= pairs[n_test+n_val:]
    return train, val, test


# ----------------------- SentencePiece ------------------------
def train_sentencepiece(train_pairs: List[Tuple[str,str]], out_dir: Path, vocab_size=2000, prefix="spm"):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "train_text.txt"
    with open(raw, "w", encoding="utf-8") as f:
        for en, fr in train_pairs:
            f.write(en.replace("\n"," ") + "\n")
            f.write(fr.replace("\n"," ") + "\n")
    model_path = out_dir / f"{prefix}.model"
    if model_path.exists():
        print("[INFO] SPM 재사용:", model_path)
        return str(model_path)
    spm.SentencePieceTrainer.Train(
        input=str(raw),
        model_prefix=str(out_dir / prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, unk_id=UNK_ID
    )
    return str(model_path)



# --------------------- dataset / collate ----------------------
class NMTDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], sp, max_len: int = 40):
        self.pairs, self.sp, self.max_len = pairs, sp, max_len
    def __len__(self): return len(self.pairs)
    def encode(self, text: str) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        ids = ids[:max(0, self.max_len - 2)]
        return [BOS_ID] + ids + [EOS_ID]
    def __getitem__(self, idx):
        en, fr = self.pairs[idx]
        return torch.tensor(self.encode(en)), torch.tensor(self.encode(fr))


def pad_sequences(seqs: List[torch.Tensor], pad_id=0):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lens = []
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        lens.append(len(s))
    return out, torch.tensor(lens, dtype=torch.long)


class PadCollate:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id
    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        src_pad, src_len = pad_sequences(list(srcs), self.pad_id)
        tgt_pad, tgt_len = pad_sequences(list(tgts), self.pad_id)
        return src_pad, src_len, tgt_pad, tgt_len


# --------------------- model components ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers=1, dropout=0.2, bidir=True, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.bidir = bidir; self.hid_dim = hid_dim
    def forward(self, src, src_len):
        x = self.dropout(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) #[B,L,H*D]
        return out, h


class LuongDotAttention(nn.Module):
    def forward(self, dec_hidden, enc_out, mask):
        if dec_hidden.dim()==2:
            dec_hidden = dec_hidden.unsqueeze(1) # [B,1,H]
        score = torch.bmm(dec_hidden, enc_out.transpose(1,2)) # [B,1,L]
        score = score.masked_fill(~mask.unsqueeze(1), torch.finfo(score.dtype).min)
        attn = F.softmax(score, dim=-1) # [B,1,L]
        ctx = torch.bmm(attn, enc_out).squeeze(1) # [B,H]
        return ctx, attn.squeeze(1) # [B,H], [B,L]


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers=1, dropout=0.2, pad_idx=0, tie_weights=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=layers,batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.fc = nn.Linear(hid_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn = LuongDotAttention()
        self.tie = tie_weights
        if tie_weights:
            # projection to emb_dim before tying if needed
            if emb_dim != hid_dim:
                self.proj = nn.Linear(hid_dim, emb_dim, bias=False)
                self.fc_weight = self.emb.weight
            else:
                self.fc.weight = self.emb.weight
                self.proj = None
        else:
            self.proj = None

    def forward(self, y_prev, h_prev, enc_out, mask):
        emb = self.dropout(self.emb(y_prev)).unsqueeze(1) # [B,1,E]
        top = h_prev[-1] # [B,H]
        ctx, attn = self.attn(top, enc_out, mask) # [B,H]
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1) # [B,1,E+H]
        out, h = self.rnn(rnn_in, h_prev) # out:[B,1,H]
        h_t = out.squeeze(1) # [B,H]
        if self.tie and self.proj is not None:
            logits = F.linear(self.proj(h_t), self.emb.weight) # tie with projection
        else:
            logits = self.fc(h_t)
        return logits, h, attn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=256, layers=1, dropout=0.2, pad_idx=0, bidir=True):
        super().__init__()
        self.enc = Encoder(vocab_size, emb_dim, hid_dim, layers, dropout, bidir, pad_idx)
        self.dec = Decoder(vocab_size, emb_dim, hid_dim, layers, dropout, pad_idx, tie_weights=True)
        self.pad_idx = pad_idx; self.bidir = bidir
        self.bridge = nn.Linear(hid_dim*(2 if bidir else 1), hid_dim)
        self.enc_out_proj = nn.Linear(hid_dim*(2 if bidir else 1), hid_dim)

    def make_mask(self, src):
        return (src != self.pad_idx).to(src.device)

    def init_dec_h(self, enc_h):
        if self.bidir:
            L = enc_h.size(0)//2
            merged = []
            for i in range(L):
                fw, bw = enc_h[2*i], enc_h[2*i+1]
                merged.append(self.bridge(torch.cat([fw,bw], dim=-1)))
            h = torch.stack(merged, dim=0)
        else:
            h = enc_h
        return torch.tanh(h)

    def forward(self, src, src_len, tgt, teacher_forcing=0.7):
        B, Lt = tgt.size()
        mask = self.make_mask(src)
        enc_out, enc_h = self.enc(src, src_len)
        enc_out = self.enc_out_proj(enc_out)
        dec_h = self.init_dec_h(enc_h)
        y = tgt[:,0] # BOS
        outs = []
        for t in range(1, Lt):
            logits, dec_h, attn = self.dec(y, dec_h, enc_out, mask)
            outs.append(logits.unsqueeze(1))
            y = tgt[:,t] if (random.random() < teacher_forcing) else logits.argmax(-1)
        return torch.cat(outs, dim=1) # [B,L-1,V]


# ---------------- AMP scaler ----------------
SCALER = torch.amp.GradScaler('cuda', enabled=(DEVICE.type=="cuda"))

# ---------------- train / eval --------------
def train_one_epoch(model, loader, opt, crit, clip=1.0):
    model.train(); tot = 0.0
    for src, src_len, tgt, _ in loader:
        src, src_len, tgt = src.to(DEVICE), src_len.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(DEVICE.type=="cuda")):
            logits = model(src, src_len, tgt, teacher_forcing=0.7)
            gold = tgt[:,1:].contiguous()
            loss = crit(logits.reshape(-1, logits.size(-1)), gold.view(-1))
        SCALER.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        SCALER.step(opt); SCALER.update()
        tot += loss.item()
    return tot / max(1, len(loader))

@torch.no_grad()
def eval_loss(model, loader, crit):
    model.eval(); tot = 0.0
    for src, src_len, tgt, _ in loader:
        src, src_len, tgt = src.to(DEVICE), src_len.to(DEVICE), tgt.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE.type=="cuda")):
            logits = model(src, src_len, tgt, teacher_forcing=0.0)
            gold = tgt[:,1:].contiguous()
            loss = crit(logits.reshape(-1, logits.size(-1)), gold.view(-1))
        tot += loss.item()
    return tot / max(1, len(loader))


# ---------------- decoding ------------------
@torch.no_grad()
def beam_search_decode(model, sp, src_texts: List[str], max_len=60, beam_size=4):
    model.eval()
    # encode batch = 1 only (for attention viz simplicity)
    outs = []
    for s in src_texts:
        ids = [BOS_ID] + sp.EncodeAsIds(s) + [EOS_ID]
        src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        src_len = torch.tensor([len(ids)], dtype=torch.long, device=DEVICE)
        mask = model.make_mask(src)
        enc_out, enc_h = model.enc(src, src_len)
        enc_out = model.enc_out_proj(enc_out)
        dec_h = model.init_dec_h(enc_h)
        # beams: (logprob, seq(list), dec_h, attn_hist)
        beams = [(0.0, [BOS_ID], dec_h)]
        finished = []
        for _ in range(max_len):
            new_beams = []
            for lp, seq, h in beams:
                y = torch.tensor([seq[-1]], device=DEVICE)
                logits, h2, attn = model.dec(y, h, enc_out, mask)
                logp = F.log_softmax(logits, dim=-1).squeeze(0)
                topk = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    tok = topk.indices[k].item()
                    nlp = lp + float(topk.values[k].item())
                    nseq = seq + [tok]
                    if tok == EOS_ID:
                        finished.append((nlp, nseq))
                    else:
                        new_beams.append((nlp, nseq, h2))
            beams = sorted(new_beams, key=lambda x: -x[0])[:beam_size]
            if len(finished) >= beam_size: break
        if not finished: finished = [(b[0], b[1]) for b in beams]
        best = max(finished, key=lambda x: x[0])[1]
        outs.append(sp.DecodeIds([t for t in best if t not in (BOS_ID, EOS_ID)]))
    return outs


@torch.no_grad()
def evaluate_bleu(model, sp, pairs, batch=64):
    refs, hyps = [], []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i+batch]
        en = [x[0] for x in chunk]; fr = [x[1] for x in chunk]
        # greedy is fine for eval speed
        pred = beam_search_decode(model, sp, en, max_len=60, beam_size=4)
        refs.extend(fr); hyps.extend(pred)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    ex = list(zip(en[:5], fr[:5], pred[:5])) if len(pairs)>0 else []
    return bleu, ex


def plot_curves(history: Dict[str, List[float]], out_dir: Path):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("NMT Loss"); plt.grid(True, alpha=0.3);
    plt.legend()
    plt.savefig(out_dir/"loss_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    if "val_bleu" in history and len(history["val_bleu"])>0:
        plt.figure()
        plt.plot(history["val_bleu"], label="val BLEU")
        plt.xlabel("epoch"); plt.ylabel("BLEU"); plt.title("Validation BLEU"); plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(out_dir/"bleu_curve.png", dpi=150, bbox_inches="tight"); plt.close()


# ---------------- attention viz ----------------
@torch.no_grad()
def plot_attention_example(model, sp, src_text: str, tgt_text: str, out_dir: Path):
    # run step-by-step greedy to record attentions
    ids = [BOS_ID] + sp.EncodeAsIds(src_text) + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    src_len = torch.tensor([len(ids)], dtype=torch.long, device=DEVICE)
    mask = model.make_mask(src)
    enc_out, enc_h = model.enc(src, src_len); enc_out = model.enc_out_proj(enc_out)
    dec_h = model.init_dec_h(enc_h)
    y = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)

    attns = []
    pieces_src = ["▁"+p for p in sp.EncodeAsPieces(src_text)]
    tgt_pieces = []
    for _ in range(30):
        logits, dec_h, attn = model.dec(y, dec_h, enc_out, mask)
        nxt = logits.argmax(-1)
        if nxt.item() == EOS_ID: break
        tgt_pieces.append(sp.IdToPiece(int(nxt.item())))
        attns.append(attn.squeeze(0).detach().cpu().numpy())
        y = nxt
    if not attns: return
    import numpy as np
    A = np.stack(attns, axis=0) # [Lt, Ls]
    plt.figure(figsize=(7,4))
    plt.imshow(A, aspect="auto", origin="upper")
    plt.xticks(range(min(len(pieces_src),A.shape[1])), pieces_src[:A.shape[1]], rotation=45, ha="right")
    plt.yticks(range(min(len(tgt_pieces),A.shape[0])), tgt_pieces[:A.shape[0]])
    plt.colorbar(); plt.title("Attention Heatmap")
    plt.xlabel("Source (EN)"); plt.ylabel("Target (FR)")
    plt.tight_layout()
    plt.savefig(out_dir/"attention_example.png", dpi=150); plt.close()


# ---------------- predict util ----------------
@torch.no_grad()
def predict_sentence(model, sp, text_or_path: str, out_dir: Path):
    if os.path.exists(text_or_path):
        text = Path(text_or_path).read_text(encoding="utf-8", errors="ignore")
    else:
        text = text_or_path
    hyp = beam_search_decode(model, sp, [text], max_len=60, beam_size=4)[0]
    msg = f"[PREDICT] EN: {text[:120]}...\n FR: {hyp}"
    print(msg)
    (out_dir/"my_sentence_translation.txt").write_text(msg + "\n", encoding="utf-8")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs_nmt")
    ap.add_argument("--vocab_size", type=int, default=500)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hid_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bidirectional", action="store_true", help="use Bi encoder(default off)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--subset", type=int, default=6000, help="pairs used (quick demo)")
    ap.add_argument("--source", type=str, default="tatoeba", choices=["tatoeba","opus_books"])
    ap.add_argument("--do_train", action="store_true", help="full training; otherwise quick save after a few epochs")
    ap.add_argument("--predict_text", type=str, default="I like apples!", help="string or file path")
    args, _ = ap.parse_known_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "cpu"
    print(f"[INFO] device={DEVICE} ({gpu})")
    train_pairs, val_pairs, test_pairs = load_enfr(args.source, subset=args.subset, val_ratio=0.1, test_ratio=0.1)
    print(f"[INFO] data sizes: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    spm_path = train_sentencepiece(train_pairs, out_dir, vocab_size=args.vocab_size, prefix="spm")
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    vocab_size = sp.get_piece_size(); print(f"[INFO] vocab_size={vocab_size}")
    collate = PadCollate(pad_id=PAD_ID)
    train_ld = DataLoader(NMTDataset(train_pairs, sp, args.max_len), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=(DEVICE.type=="cuda"), collate_fn=collate, persistent_workers=False if args.workers>0 else False)
    val_ld = DataLoader(NMTDataset(val_pairs, sp, args.max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=(DEVICE.type=="cuda"), collate_fn=collate, persistent_workers=False if args.workers>0 else False)
    test_ld = DataLoader(NMTDataset(test_pairs, sp, args.max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=(DEVICE.type=="cuda"), collate_fn=collate, persistent_workers=False if args.workers>0 else False)
    model = Seq2Seq(vocab_size, emb_dim=args.emb_dim, hid_dim=args.hid_dim, layers=args.layers, dropout=args.dropout, pad_idx=PAD_ID, bidir=args.bidirectional).to(DEVICE)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID) # no label smoothing for tiny data
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_loss":[], "val_loss":[], "val_bleu":[]}
    best_bleu, best_path, bad = -1.0, out_dir/"best_fr_mt.pt", 0
    print("[INFO] training start")
    max_epochs = (args.epochs if args.do_train else max(3, min(6, args.epochs)))
    for ep in range(1, max_epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_ld, opt, crit, clip=args.clip)
        va_loss = eval_loss(model, val_ld, crit)
        bleu_val, _ = evaluate_bleu(model, sp, val_pairs, batch=64)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss);history["val_bleu"].append(bleu_val)
        print(f"[EPOCH {ep:02d}] train={tr_loss:.3f} val={va_loss:.3f} valBLEU={bleu_val:.2f} ({time.time()-t0:.1f}s)")
        if bleu_val > best_bleu:
            best_bleu, bad = bleu_val, 0
            torch.save({"model": model.state_dict(), "spm": str(spm_path)}, best_path)
        else:
            bad += 1
            if bad >= args.patience: print("[INFO] early stop"); break

    plot_curves(history, out_dir)

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE);model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded best: {best_path}")
    bleu_test, samples = evaluate_bleu(model, sp, test_pairs, batch=64)
    print(f"[TEST] sacreBLEU = {bleu_test:.2f}")
    with open(out_dir/"test_samples.txt", "w", encoding="utf-8") as f:
        f.write(f"[TEST] sacreBLEU = {bleu_test:.2f}\n\n")
        for en, ref, hyp in samples:
            f.write(f"- EN : {en}\n REF: {ref}\n HYP: {hyp}\n\n")
    # predict + attention viz
    predict_sentence(model, sp, args.predict_text, out_dir)
    plot_attention_example(model, sp, "I like apples .", "J'aime les pommes .", out_dir)
    print(f"[INFO] Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
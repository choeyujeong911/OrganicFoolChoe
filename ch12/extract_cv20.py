import os
import csv
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_root", type=str, required=True,
                        help="Common Voice 한국어 데이터가 풀려 있는 디렉토리 (train.tsv, clips/ 등이 있는 곳)")
    parser.add_argument("--out_root", type=str, default="data",
                        help="우리 STT 코드에서 쓸 data 루트 (data/train_mp3, data/train_txt 생성)")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="추출할 샘플 개수")
    args = parser.parse_args()

    cv_root = args.cv_root
    clips_dir = os.path.join(cv_root, "clips")
    train_tsv = os.path.join(cv_root, "train.tsv")

    if not os.path.exists(train_tsv):
        raise FileNotFoundError(f"train.tsv not found at {train_tsv}")
    if not os.path.isdir(clips_dir):
        raise NotADirectoryError(f"clips dir not found at {clips_dir}")

    out_wav = os.path.join(args.out_root, "train_mp3")
    out_txt = os.path.join(args.out_root, "train_txt")
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)

    # Common Voice TSV는 헤더가 있고, path / sentence(또는 text) 컬럼이 있음
    # 버전에 따라 컬럼 이름이 'sentence' 또는 'text'일 수 있어서 둘 다 처리
    selected = []
    with open(train_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # 문장 컬럼 이름 찾기
            text = row.get("sentence", None)
            if text is None:
                text = row.get("text", None)
            path = row.get("path", None)

            if text is None or path is None:
                continue

            # txt가 비어있는 경우는 스킵
            text = text.strip()
            if not text:
                continue

            selected.append((path, text))
            if len(selected) >= args.num_samples:
                break

    if len(selected) == 0:
        raise RuntimeError("No valid samples found in train.tsv")

    print(f"Selected {len(selected)} samples. Copying files...")

    for i, (rel_path, text) in enumerate(selected, start=1):
        src_audio = os.path.join(clips_dir, rel_path)
        if not os.path.exists(src_audio):
            print(f"[WARN] audio not found: {src_audio}, skip")
            continue

        base = f"cvko_{i:02d}"
        # Common Voice는 보통 .mp3라서 확장자 그대로 가져오기
        ext = os.path.splitext(rel_path)[1]
        dst_audio = os.path.join(out_wav, base + ext)
        dst_txt = os.path.join(out_txt, base + ".txt")

        shutil.copy2(src_audio, dst_audio)
        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write(text.strip())

        print(f"  -> {dst_audio}, {dst_txt}")

    print("\nDone!")
    print(f"train_mp3: {out_wav}")
    print(f"train_txt: {out_txt}")

if __name__ == "__main__":
    main()

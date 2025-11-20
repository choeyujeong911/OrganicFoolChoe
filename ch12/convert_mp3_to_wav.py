import os
import subprocess
import argparse

def convert_mp3_to_wav(mp3_dir, wav_dir, sr=16000):
    os.makedirs(wav_dir, exist_ok=True)

    mp3_files = [f for f in os.listdir(mp3_dir) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("[WARN] No MP3 files found.")
        return

    print(f"[INFO] Found {len(mp3_files)} MP3 files.")
    print(f"[INFO] Saving WAV files to: {wav_dir}")

    for mp3_name in mp3_files:
        mp3_path = os.path.join(mp3_dir, mp3_name)
        base = os.path.splitext(mp3_name)[0]
        wav_path = os.path.join(wav_dir, base + ".wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ar", str(sr),     # sampling rate 16kHz
            "-ac", "1",         # mono
            wav_path
        ]

        print(f"[CONVERT] {mp3_path} → {wav_path}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[DONE] Conversion completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp3_dir", type=str, default="./data/train_mp3",
                        help="Directory containing mp3 files")
    parser.add_argument("--wav_dir", type=str, default="./data/train_mp3",
                        help="Output directory for wav files")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Output sampling rate (default: 16kHz)")
    args = parser.parse_args()

    convert_mp3_to_wav(args.mp3_dir, args.wav_dir, args.sr)

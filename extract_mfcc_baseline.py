
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = PROJECT_DIR / "features_mfcc80.csv"

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_mfcc(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])


def main():
    audio_files = sorted(DATA_DIR.rglob("*.wav"))

    if not audio_files:
        print(f"No WAV files found in {DATA_DIR}")
        return

    rows = []

    print(f"Found {len(audio_files)} audio files.")

    for i, file_path in enumerate(audio_files, start=1):
        try:
            parts = file_path.stem.split("-")

            if len(parts) != 7 or parts[2] not in EMOTIONS:
                continue

            features = extract_mfcc(file_path)

            row = {
                f"mfcc_mean_{j + 1}": float(features[j])
                for j in range(40)
            }

            row.update({
                f"mfcc_std_{j + 1}": float(features[j + 40])
                for j in range(40)
            })

            row["emotion"] = EMOTIONS[parts[2]]
            row["actor_id"] = parts[6]
            row["file_name"] = file_path.name

            rows.append(row)

            if i % 100 == 0 or i == len(audio_files):
                print(f"Processed {i}/{len(audio_files)}")

        except Exception as error:
            print(f"Skipped {file_path.name}: {error}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nMFCC baseline features completed.")
    print(f"Samples: {len(df)}")
    print("Features: 80")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = PROJECT_DIR / "features.csv"

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

SAMPLE_RATE = 22050
N_MFCC = 40


# ==========================================
# 2. Feature Extraction Helpers
# ==========================================

def add_statistics(features, name, values):
    """Add mean and standard deviation for each feature."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1)

    for i, value in enumerate(means):
        features[f"{name}_mean_{i + 1}"] = float(value)

    for i, value in enumerate(stds):
        features[f"{name}_std_{i + 1}"] = float(value)


def extract_features(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    features = {}

    # --------------------------------------
    # A. MFCC: 40 coefficients
    # --------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    add_statistics(features, "mfcc", mfcc)

    # --------------------------------------
    # B. Delta MFCC: changes over time
    # --------------------------------------

    delta_mfcc = librosa.feature.delta(mfcc, order=1)

    add_statistics(features, "delta_mfcc", delta_mfcc)

    # --------------------------------------
    # C. Chroma: pitch class distribution
    # --------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )

    add_statistics(features, "chroma", chroma)

    # --------------------------------------
    # D. Energy and spectral characteristics
    # --------------------------------------

    rms = librosa.feature.rms(y=audio)
    add_statistics(features, "rms", rms)

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )
    add_statistics(features, "centroid", centroid)

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )
    add_statistics(features, "bandwidth", bandwidth)

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )
    add_statistics(features, "rolloff", rolloff)

    zero_crossing = librosa.feature.zero_crossing_rate(audio)
    add_statistics(features, "zcr", zero_crossing)

    return features


# ==========================================
# 3. Process Dataset
# ==========================================

def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Dataset folder not found: {DATA_DIR}")
        return

    audio_files = sorted(DATA_DIR.rglob("*.wav"))

    if not audio_files:
        print("ERROR: No WAV files found.")
        return

    print(f"Found {len(audio_files)} audio files.")
    print("Extracting enhanced audio features...\n")

    rows = []
    skipped = 0

    for index, file_path in enumerate(audio_files, start=1):
        try:
            parts = file_path.stem.split("-")

            if len(parts) != 7:
                skipped += 1
                continue

            emotion_code = parts[2]
            emotion = EMOTIONS.get(emotion_code)

            if emotion is None:
                skipped += 1
                continue

            # The final filename field identifies the actor.
            actor_id = parts[6]

            row = extract_features(file_path)
            row["emotion"] = emotion
            row["file_name"] = file_path.name
            row["actor_id"] = actor_id

            rows.append(row)

            if index % 100 == 0 or index == len(audio_files):
                print(
                    f"Processed {index}/{len(audio_files)}"
                )

        except Exception as error:
            skipped += 1
            print(f"Skipped {file_path.name}: {error}")

    if not rows:
        print("ERROR: No features extracted.")
        return

    df = pd.DataFrame(rows)

    # Ensure that all feature values are valid.
    feature_columns = [
        col for col in df.columns
        if col not in ("emotion", "file_name", "actor_id")
    ]

    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if df[feature_columns].isna().any().any():
        print("ERROR: Missing or invalid feature values found.")
        print("Inspect the source audio files before training.")
        return

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 45)
    print("ENHANCED FEATURE EXTRACTION COMPLETED")
    print("=" * 45)
    print(f"Audio files found: {len(audio_files)}")
    print(f"Successfully processed: {len(rows)}")
    print(f"Skipped files: {skipped}")
    print(f"Features per recording: {len(feature_columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().to_string())


if __name__ == "__main__":
    main()

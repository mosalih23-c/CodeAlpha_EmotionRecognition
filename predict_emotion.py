
from pathlib import Path

import joblib
import librosa
import numpy as np

# ==========================================
# 1. Project Configuration
# ==========================================

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")

MODEL_FILE = PROJECT_DIR / "models" / "svm.joblib"
ENCODER_FILE = PROJECT_DIR / "models" / "label_encoder.joblib"

# ضع مسار ملف الصوت الذي تريد اختباره هنا
AUDIO_FILE = PROJECT_DIR / "data" / "Actor_01" / "03-01-05-01-01-01-01.wav"


# ==========================================
# 2. Extract MFCC Features
# ==========================================

def extract_mfcc(file_path):
    audio, sample_rate = librosa.load(
        str(file_path),
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mean_features = np.mean(mfcc, axis=1)
    std_features = np.std(mfcc, axis=1)

    # Must match the training feature order
    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features.reshape(1, -1)


# ==========================================
# 3. Predict Emotion
# ==========================================

def main():
    if not MODEL_FILE.exists():
        print(f"ERROR: Model not found: {MODEL_FILE}")
        print("Run train_model.py first.")
        return

    if not ENCODER_FILE.exists():
        print(f"ERROR: Label encoder not found: {ENCODER_FILE}")
        return

    if not AUDIO_FILE.exists():
        print(f"ERROR: Audio file not found: {AUDIO_FILE}")
        print("Update AUDIO_FILE with a valid WAV path.")
        return

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)

    print("Loading audio...")
    features = extract_mfcc(AUDIO_FILE)

    # Predict encoded emotion and convert to its label
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]

    print("\n" + "=" * 40)
    print("SPEECH EMOTION RECOGNITION")
    print("=" * 40)
    print(f"Audio file: {AUDIO_FILE.name}")
    print(f"Predicted emotion: {emotion.upper()}")

    # Display class scores when the model supports probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = encoder.inverse_transform(model.classes_)

        print("\nClass scores:")
        for label, score in sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{label:12s}: {score:.2%}")

    print("=" * 40)


if __name__ == "__main__":
    main()

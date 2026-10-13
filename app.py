
from pathlib import Path
import tempfile
import os

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")
MODEL_FILE = PROJECT_DIR / "models" / "svm.joblib"
ENCODER_FILE = PROJECT_DIR / "models" / "label_encoder.joblib"


# ==========================================
# 2. Page Setup
# ==========================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #172554;
}
.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
}
.result-card {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}
.emotion-result {
    font-size: 34px;
    font-weight: 800;
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. Load Trained Model
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not ENCODER_FILE.exists():
        raise FileNotFoundError(f"Encoder not found: {ENCODER_FILE}")

    return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE)


# ==========================================
# 4. Extract MFCC Features
# ==========================================

def extract_mfcc(audio_path):
    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=22050,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio recording is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    columns = (
        [f"mfcc_mean_{i + 1}" for i in range(40)]
        + [f"mfcc_std_{i + 1}" for i in range(40)]
    )

    return pd.DataFrame([features], columns=columns)


# ==========================================
# 5. Analyze Uploaded or Recorded Audio
# ==========================================

def analyze_audio(audio_file, model, encoder):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_path = temp_file.name

        with st.spinner("Analyzing audio and predicting emotion..."):
            features = extract_mfcc(temp_path)

            prediction = model.predict(features)
            emotion = encoder.inverse_transform(prediction)[0]

            st.success("Audio analysis completed!")

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="color:#64748b;font-size:17px;">
                        Predicted Emotion
                    </div>
                    <div class="emotion-result">
                        {emotion.upper()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # SVM decision scores are not probabilities.
            if hasattr(model, "decision_function"):
                scores = np.asarray(
                    model.decision_function(features)
                )

                if scores.ndim == 2:
                    classes = encoder.inverse_transform(model.classes_)

                    chart_data = pd.DataFrame({
                        "Emotion": classes,
                        "Decision Score": scores[0]
                    }).sort_values(
                        "Decision Score",
                        ascending=False
                    )

                    st.subheader("Emotion Decision Scores")
                    st.bar_chart(
                        chart_data.set_index("Emotion")
                    )

                    st.caption(
                        "These are SVM decision scores, not percentages "
                        "or calibrated confidence probabilities."
                    )

            st.info(
                "The result is a model prediction and may not always "
                "match the speaker's actual emotion."
            )

    except Exception as error:
        st.error(f"Could not analyze audio: {error}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 6. Main Interface
# ==========================================

st.markdown(
    '<div class="main-title">🎙️ Speech Emotion Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Record your voice or upload an audio file to predict emotion'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

try:
    model, encoder = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# Sidebar
with st.sidebar:
    st.title("Project Information")
    st.write("**Model:** SVM")
    st.write("**Audio Features:** MFCC")
    st.write("**Features per audio:** 80")
    st.write("**Dataset:** RAVDESS")
    st.write("**Emotion Classes:** 8")

    st.divider()
    st.caption("Machine Learning Project")


# Two input methods
tab_record, tab_upload = st.tabs([
    "🎙️ Record Audio",
    "📁 Upload Audio"
])


# ------------------------------------------
# Tab 1: Record directly from microphone
# ------------------------------------------

with tab_record:
    st.subheader("Record Your Voice")
    st.write(
        "Allow microphone access when your browser asks. "
        "Record a short voice sample, then analyze it."
    )

    recorded_audio = st.audio_input(
        "Click to record audio using your microphone",
        key="recorded_audio"
    )

    if recorded_audio is not None:
        st.audio(recorded_audio, format="audio/wav")

        if st.button(
            "🔍 Analyze Recording",
            type="primary",
            key="analyze_recording",
            use_container_width=True
        ):
            analyze_audio(recorded_audio, model, encoder)


# ------------------------------------------
# Tab 2: Upload audio file
# ------------------------------------------

with tab_upload:
    st.subheader("Upload an Audio File")
    st.write("Choose a WAV file from your computer.")

    uploaded_audio = st.file_uploader(
        "Select an audio file",
        type=["wav"],
        key="uploaded_audio"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")

        if st.button(
            "🔍 Analyze Uploaded Audio",
            type="primary",
            key="analyze_upload",
            use_container_width=True
        ):
            analyze_audio(uploaded_audio, model, encoder)


# Footer
st.divider()

st.markdown(
    "<div style='text-align:center;color:#64748b;'>"
    "Speech Emotion Recognition | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)

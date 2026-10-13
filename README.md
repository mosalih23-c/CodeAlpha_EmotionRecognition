# Speech Emotion Recognition

A machine learning project that classifies emotions from speech audio
using the RAVDESS dataset. The project extracts acoustic features from
audio recordings, trains classification models, and provides a Streamlit
interface for making predictions from uploaded WAV files or microphone
recordings.

## Project Status

-   Dataset preparation: completed for 1,440 WAV recordings.
-   Traditional machine-learning models: trained and evaluated.
-   Streamlit interface: implemented; verify it uses the selected model
    files before final submission.
-   CNN experiment using Mel-Spectrograms: completed as an experiment,
    but not selected because its held-out actor test accuracy was poor.
-   Final packaging and documentation: in progress.

## Dataset

This project uses the RAVDESS speech recordings organized into actor
folders.

-   Total recordings processed: 1,440
-   Audio format: WAV
-   Emotion classes: angry, calm, disgust, fearful, happy, neutral, sad,
    surprised

The data is split by actor for the CNN experiment so that actors in the
training, validation, and test sets do not overlap. This provides a more
demanding evaluation of generalization to unseen speakers.

## Models and Evaluation

Several traditional models were tested, including Support Vector Machine
(SVM), Random Forest, and K-Nearest Neighbors (KNN).

An actor-based evaluation of an SVM using the 80-feature MFCC baseline
reported:

-   Test accuracy: **51.33%**
-   Balanced accuracy: **50.94%**
-   Group cross-validation accuracy: **44.81%**

A separate tuned SVM experiment using a broader feature set reported
test accuracy of **53.33%** and balanced accuracy of **53.13%**. These
results come from different feature configurations and should not be
treated as a direct apples-to-apples comparison without rerunning them
under identical conditions.

The CNN Mel-Spectrogram experiment achieved **13.33% test accuracy** on
the actor-held-out test set and is not the recommended model at this
stage.

These are experimental results, not a claim of production-level
accuracy. Performance on real-world audio may differ from performance on
RAVDESS.

## Project Structure

The project directory is expected to look broadly like this:

``` text
CodeAlpha_EmotionRecognition/
├── data/
│   └── Actor_XX/
├── models/
│   ├── svm_mfcc80_optimized.joblib
│   └── label_encoder_mfcc80.joblib
├── reports/
├── cnn_data/                  # generated CNN experiment data
├── extract_features.py
├── extract_mfcc_baseline.py
├── train_model.py
├── train_svm_optimized.py
├── predict_emotion.py
├── prepare_cnn_data.py
├── train_cnn.py
├── app.py
└── README.md
```

The exact files in `models/` may vary. Confirm the model and
label-encoder filenames expected by `predict_emotion.py` and `app.py`
before deleting or renaming any model files.

## Requirements

Use Python 3.10 or another Python version compatible with the installed
TensorFlow and audio-processing packages.

Common dependencies used by this project include:

-   librosa
-   numpy
-   pandas
-   scikit-learn
-   matplotlib
-   seaborn
-   joblib
-   streamlit
-   tensorflow

Install the dependencies in your active virtual environment:

``` bash
python -m pip install librosa numpy pandas scikit-learn matplotlib seaborn joblib streamlit tensorflow
```

If your environment already has these packages installed, you do not
need to reinstall them. If TensorFlow reports a
`MessageFactory.GetPrototype` error, check compatibility between the
installed `protobuf` and TensorFlow versions before changing package
versions.

## Running the Project

Run commands from the project root directory.

### 1. Predict an emotion from an audio file

``` bash
python predict_emotion.py
```

Follow the script's prompts, if any, to provide a WAV file. The
prediction script must load the model and label encoder whose paths are
configured in that script.

### 2. Launch the Streamlit interface

``` bash
streamlit run app.py
```

Open the local URL printed by Streamlit. The interface supports WAV
upload and microphone input, subject to browser and device permissions.

Before using the interface, ensure `app.py` and `predict_emotion.py`
load the intended SVM model and matching label encoder. Do not pair a
model with a label encoder from a different experiment.

### 3. Recreate the CNN experiment (optional)

Prepare Mel-Spectrogram arrays:

``` bash
python prepare_cnn_data.py
```

Train the CNN:

``` bash
python train_cnn.py
```

The generated arrays and reports are experimental artifacts. The CNN is
not the recommended model based on the current evaluation.

## Evaluation Methodology

For a fair estimate of generalization to unseen speakers, split the data
by actor rather than randomly splitting individual recordings. Record
the split strategy, feature configuration, model settings, and metrics
together when reporting results.

Use the validation set for model selection. Keep the test set for final
evaluation; repeated tuning against the same test set can make the
reported test score optimistic.

Useful metrics include:

-   **Accuracy:** fraction of all predictions that are correct.
-   **Balanced accuracy:** average recall across classes.
-   **Macro F1-score:** F1-score averaged equally across emotion
    classes.
-   **Confusion matrix:** shows which emotions are frequently confused.

## Limitations

-   Speech emotion is subjective, and some emotions have acoustically
    similar patterns.
-   The current actor-held-out SVM results are moderate and should be
    presented transparently.
-   RAVDESS is a controlled dataset; results may not transfer directly
    to spontaneous speech, other languages, microphones, or noisy
    environments.
-   Predictions should be treated as model estimates, not definitive
    assessments of a speaker's emotional state.

## Future Improvements

-   Verify the final SVM model and label encoder are consistently used
    by both the prediction script and Streamlit app.
-   Evaluate all candidate models under the same actor-based split and
    feature setup.
-   Inspect the confusion matrix and per-class metrics to identify
    difficult emotion classes.
-   Test with carefully selected recordings from speakers not used
    during training.
-   Improve dependency reproducibility by recording package versions in
    a `requirements.txt` file.

## Acknowledgements

-   Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech
    and Song).
-   Project developed as a machine-learning practice project associated
    with CodeAlpha.

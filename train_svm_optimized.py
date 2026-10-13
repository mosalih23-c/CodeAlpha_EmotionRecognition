
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")
DATA_FILE = PROJECT_DIR / "features_mfcc80.csv"

MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. Load Data
# ==========================================

def main():
    df = pd.read_csv(DATA_FILE)

    excluded = {"emotion", "actor_id", "file_name"}
    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]

    X = df[feature_columns].astype(float)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion"].astype(str))

    groups = df["actor_id"].astype(str)

    print("Dataset loaded.")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Actors: {groups.nunique()}")

    # ==========================================
    # 3. Split by Actor
    # ==========================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    assert set(groups_train).isdisjoint(set(groups_test))

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training actors: {groups_train.nunique()}")
    print(f"Testing actors: {groups_test.nunique()}")

    # ==========================================
    # 4. SVM Pipeline and Hyperparameter Search
    # ==========================================

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10, 100, 1000],
        "classifier__gamma": [
            "scale", 0.1, 0.01, 0.001, 0.0001
        ]
    }

    cv = GroupKFold(
        n_splits=min(5, groups_train.nunique())
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("\nSearching for the best SVM settings...")

    search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        f"Best group-CV accuracy: "
        f"{search.best_score_:.2%}"
    )

    # ==========================================
    # 5. Evaluate on Unseen Actors
    # ==========================================

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 45)
    print("FINAL TEST RESULTS")
    print("=" * 45)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")

    print("\nClassification Report:")
    report_text = classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0
    )

    print(report_text)

    # ==========================================
    # 6. Save Model and Results
    # ==========================================

    model_path = MODELS_DIR / "svm_mfcc80_optimized.joblib"
    encoder_path = MODELS_DIR / "label_encoder_mfcc80.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(encoder, encoder_path)

    with open(
        REPORTS_DIR / "svm_mfcc80_results.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Best parameters: {search.best_params_}\n")
        file.write(f"Group CV accuracy: {search.best_score_:.4f}\n")
        file.write(f"Test accuracy: {accuracy:.4f}\n")
        file.write(
            f"Balanced accuracy: {balanced_accuracy:.4f}\n\n"
        )
        file.write(report_text)

    with open(
        REPORTS_DIR / "svm_mfcc80_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "model": "SVM RBF",
            "features": feature_columns,
            "emotions": encoder.classes_.tolist(),
            "best_parameters": search.best_params_,
            "group_cv_accuracy": float(search.best_score_),
            "test_accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "test_actors": sorted(groups_test.unique().tolist()),
            "random_state": RANDOM_STATE
        }, file, indent=4, ensure_ascii=False)

    print(f"\nModel saved: {model_path}")
    print(f"Results saved: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

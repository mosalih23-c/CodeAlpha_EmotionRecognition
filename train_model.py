
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(r"D:\CodeAlpha_EmotionRecognition")
FEATURES_FILE = PROJECT_DIR / "features.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. Load Dataset
# ==========================================

def main():
    if not FEATURES_FILE.exists():
        print(f"ERROR: File not found: {FEATURES_FILE}")
        return

    df = pd.read_csv(FEATURES_FILE)

    required_columns = {"emotion", "actor_id"}
    if not required_columns.issubset(df.columns):
        print("ERROR: Missing emotion or actor_id columns.")
        print("Run the updated extract_features.py first.")
        return

    excluded_columns = {"emotion", "file_name", "actor_id"}
    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    if not feature_columns:
        print("ERROR: No feature columns found.")
        return

    X = df[feature_columns].astype(float)
    y_text = df["emotion"].astype(str)
    groups = df["actor_id"].astype(str)

    if X.isna().any().any():
        print("ERROR: Missing values found in features.")
        return

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    print("Dataset loaded successfully.")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Actors: {groups.nunique()}")
    print(f"Classes: {list(encoder.classes_)}")

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

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training actors: {groups_train.nunique()}")
    print(f"Testing actors: {groups_test.nunique()}")

    overlap = set(groups_train) & set(groups_test)
    assert not overlap, "Actor leakage detected!"

    # ==========================================
    # 4. Tune SVM using GroupKFold
    # ==========================================

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.001]
    }

    n_splits = min(5, groups_train.nunique())

    if n_splits < 2:
        print("ERROR: Not enough training actors for cross-validation.")
        return

    group_cv = GroupKFold(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=group_cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    models = {}

    print("\n" + "=" * 50)
    print("TUNING SVM")
    print("=" * 50)

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    models["SVM Tuned"] = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best group-CV accuracy: {grid_search.best_score_:.4f}")

    # ==========================================
    # 5. Train Baseline Models
    # ==========================================

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])

    # ==========================================
    # 6. Evaluate on Unseen Actors
    # ==========================================

    results = []
    predictions_by_model = {}

    for name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Evaluating: {name}")
        print("=" * 50)

        if name != "SVM Tuned":
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        predictions_by_model[name] = predictions

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            zero_division=0
        ))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy
        })

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(
            model,
            MODELS_DIR / f"{safe_name}.joblib"
        )

    # ==========================================
    # 7. Select Best Model
    # ==========================================

    results_df = pd.DataFrame(results).sort_values(
        "Accuracy",
        ascending=False
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    best_predictions = predictions_by_model[best_name]
    best_accuracy = accuracy_score(y_test, best_predictions)

    # Save the selected model and encoder.
    joblib.dump(
        best_model,
        MODELS_DIR / "best_model.joblib"
    )

    joblib.dump(
        encoder,
        MODELS_DIR / "label_encoder.joblib"
    )

    # Maintain compatibility with existing SVM-based app.
    # svm.joblib is the tuned SVM.
    joblib.dump(
        models["SVM Tuned"],
        MODELS_DIR / "svm.joblib"
    )

    with open(
        MODELS_DIR / "model_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "selected_model": best_name,
            "selected_model_accuracy": float(best_accuracy),
            "tuned_svm_best_params": grid_search.best_params_,
            "tuned_svm_group_cv_accuracy": float(grid_search.best_score_),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "emotions": encoder.classes_.tolist(),
            "split_strategy": "GroupShuffleSplit by actor_id",
            "random_state": RANDOM_STATE
        }, file, indent=4, ensure_ascii=False)

    # ==========================================
    # 8. Save Comparison Results
    # ==========================================

    results_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy"
    )
    plt.ylim(0, 1)
    plt.title("Emotion Recognition - Unseen Actor Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "model_comparison.png",
        dpi=300
    )
    plt.close()

    # ==========================================
    # 9. Best Model Confusion Matrix
    # ==========================================

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=list(range(len(encoder.classes_)))
    )

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # Save the final classification report.
    report = classification_report(
        y_test,
        best_predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "classification_report.csv"
    )

    # ==========================================
    # 10. Final Summary
    # ==========================================

    print("\n" + "=" * 50)
    print("TRAINING AND EVALUATION COMPLETED")
    print("=" * 50)

    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Test accuracy: {best_accuracy:.2%}")
    print(f"SVM group-CV accuracy: {grid_search.best_score_:.2%}")

    print("\nSaved models:", MODELS_DIR)
    print("Saved reports:", REPORTS_DIR)

    print(
        "\nNote: This test evaluates generalization to held-out "
        "actors. Compare results with the previous random split "
        "carefully because the evaluation protocols differ."
    )


if __name__ == "__main__":
    main()

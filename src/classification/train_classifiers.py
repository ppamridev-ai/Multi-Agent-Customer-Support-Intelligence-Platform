"""Train and validate ticket classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import (
    ARTIFACTS_DIR,
    CATEGORY_MODEL_PATH,
    MODELS_DIR,
    PRIORITY_MODEL_PATH,
    SENTIMENT_MODEL_PATH,
    TRAIN_PATH,
    VALIDATION_METRICS_PATH,
    VALIDATION_PATH,
)


TARGET_MODEL_PATHS = {
    "ticket_category": CATEGORY_MODEL_PATH,
    "priority": PRIORITY_MODEL_PATH,
    "sentiment": SENTIMENT_MODEL_PATH,
}


def build_pipeline() -> Pipeline:
    """Build a word-and-character TF-IDF classifier."""

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )

def calculate_scores(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        ),
    }

def train_target(train: pd.DataFrame,validation: pd.DataFrame,target: str,model_path: Path) -> dict:
    """Train and evaluate one target classifier."""

    print(f"\n{'=' * 70}")
    print(f"Training target: {target}")
    print("=" * 70)

    x_train = train["ticket_text"]
    y_train = train[target]

    x_validation = validation["ticket_text"]
    y_validation = validation[target]

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    validation_predictions = pipeline.predict(x_validation)

    train_scores = calculate_scores(actual=y_train,predicted=train_predictions)

    validation_scores = calculate_scores(actual=y_validation,predicted=validation_predictions)

    report = classification_report(y_validation,validation_predictions,output_dict=True,zero_division=0)

    model_path.parent.mkdir(parents=True,exist_ok=True)

    joblib.dump(pipeline, model_path)

    accuracy_gap = train_scores["accuracy"] - validation_scores["accuracy"]

    print("\nTraining metrics:")
    print(f"Accuracy:    {train_scores['accuracy']:.4f}")
    print(f"Macro F1:    {train_scores['macro_f1']:.4f}")
    print(f"Weighted F1: {train_scores['weighted_f1']:.4f}")
    print("\nValidation metrics:")
    print(f"Accuracy:    {validation_scores['accuracy']:.4f}")
    print(f"Macro F1:    {validation_scores['macro_f1']:.4f}")
    print(f"Weighted F1: {validation_scores['weighted_f1']:.4f}")

    print(f"\nAccuracy gap: {accuracy_gap:.4f}")
    print(f"Saved model: {model_path}")

    return {
        "target": target,
        "training": train_scores,
        "validation": validation_scores,
        "accuracy_gap": accuracy_gap,
        "classification_report": report,
        "model_path": str(model_path),
    }


def save_metrics(metrics: dict,output_path: Path) -> None:
    """Save validation metrics as JSON."""
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open("w",encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    metrics: dict[str, dict] = {}

    for target, model_path in TARGET_MODEL_PATHS.items():
        metrics[target] = train_target(train=train,validation=validation,target=target,model_path=model_path)

    save_metrics(metrics=metrics,output_path=VALIDATION_METRICS_PATH)

    print(f"\nValidation metrics saved to: {VALIDATION_METRICS_PATH}")


if __name__ == "__main__":
    main()
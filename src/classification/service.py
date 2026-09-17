"""Service for loading and running ticket classifiers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import (
    CATEGORY_MODEL_PATH,
    PRIORITY_MODEL_PATH,
    SENTIMENT_MODEL_PATH,
    settings,
)
from src.schemas import ClassificationPrediction


class ClassificationService:
    """Load and run all ticket classification models."""

    def __init__(
        self,
        category_model_path: Path = CATEGORY_MODEL_PATH,
        priority_model_path: Path = PRIORITY_MODEL_PATH,
        sentiment_model_path: Path = SENTIMENT_MODEL_PATH,
    ) -> None:
        self.category_model = self._load_model(category_model_path)
        self.priority_model = self._load_model(priority_model_path)
        self.sentiment_model = self._load_model(sentiment_model_path)

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        """Load one saved scikit-learn pipeline."""

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run train_classifiers.py first."
            )

        return joblib.load(model_path)

    @staticmethod
    def _predict_with_confidence(model: Any,ticket_text: str,) -> tuple[str, float]:
        """Predict one label and its maximum probability."""

        probabilities = model.predict_proba([ticket_text])[0]

        best_index = int(np.argmax(probabilities))
        predicted_label = str(model.classes_[best_index])
        confidence = float(probabilities[best_index])

        return predicted_label, confidence

    def predict(self,ticket_text: str) -> ClassificationPrediction:
        """Predict category, priority, and sentiment."""

        normalized_text = " ".join(ticket_text.split())

        if len(normalized_text) < 5:
            raise ValueError("Ticket text must contain at least 5 characters.")

        category, category_confidence = self._predict_with_confidence(self.category_model,normalized_text)
        priority, priority_confidence = self._predict_with_confidence(self.priority_model,normalized_text)
        sentiment, sentiment_confidence = self._predict_with_confidence(self.sentiment_model,normalized_text)

        low_confidence_targets: list[str] = []

        if category_confidence < settings.category_confidence_threshold:
            low_confidence_targets.append("category")

        if priority_confidence < settings.priority_confidence_threshold:
            low_confidence_targets.append("priority")

        if sentiment_confidence < settings.sentiment_confidence_threshold:
            low_confidence_targets.append("sentiment")

        return ClassificationPrediction(
            category=category,
            priority=priority,
            sentiment=sentiment,
            category_confidence=category_confidence,
            priority_confidence=priority_confidence,
            sentiment_confidence=sentiment_confidence,
            requires_review=bool(low_confidence_targets),
            low_confidence_targets=low_confidence_targets,
        )


@lru_cache(maxsize=1)
def get_classification_service() -> ClassificationService:
    """Return one cached service instance."""

    return ClassificationService()
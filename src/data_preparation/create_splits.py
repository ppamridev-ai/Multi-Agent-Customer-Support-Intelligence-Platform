"""Create reproducible train, validation, and test splits."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TEST_PATH,
    TICKETS_PROCESSED_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

TARGET_COLUMNS = [
    "ticket_category",
    "priority",
    "sentiment",
]


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """Create a joint label for multi-target stratification."""
    return df["ticket_category"]+ " | " + df["priority"] + " | " + df["sentiment"]


def choose_stratification_key(df: pd.DataFrame) -> pd.Series:
    """
    Select a safe stratification key.

    Joint category, priority, and sentiment stratification is preferred.
    If a combination has too few samples, category and priority are used.
    """

    full_key = create_stratification_key(df)
    minimum_full_count = full_key.value_counts().min()

    if minimum_full_count >= 10:
        print("Stratifying by category + priority + sentiment.")
        print("Smallest joint group:", minimum_full_count)
        return full_key

    fallback_key = df["ticket_category"] + " | " + df["priority"]

    minimum_fallback_count = fallback_key.value_counts().min()

    print("Some category-priority-sentiment groups are too small.")
    print("Stratifying by category + priority instead.")
    print("Smallest fallback group:", minimum_fallback_count)

    return fallback_key


def display_distribution(df: pd.DataFrame,split_name: str) -> None:
    """Print normalized distributions for every target."""

    print(f"\n{split_name}: {len(df):,} rows")

    for target in TARGET_COLUMNS:
        distribution = df[target].value_counts(normalize=True).mul(100).round(2)
        print(f"\n{target}:")
        print(distribution.to_string())


def main() -> None:
    tickets = pd.read_csv(TICKETS_PROCESSED_PATH)
    stratification_key = choose_stratification_key(tickets)

    train, temporary = train_test_split(
        tickets,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratification_key,
    )

    temporary_key = choose_stratification_key(temporary)

    validation, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temporary_key,
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(TRAIN_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    display_distribution(train, "Train")
    display_distribution(validation, "Validation")
    display_distribution(test, "Test")

    print("\nFiles created:")
    print(f"Train: {TRAIN_PATH}")
    print(f"Validation: {VALIDATION_PATH}")
    print(f"Test: {TEST_PATH}")


if __name__ == "__main__":
    main()
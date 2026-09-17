"""Inspect raw project datasets without modifying them."""

from pathlib import Path

import pandas as pd

from src.config import FAQS_RAW_PATH, TICKETS_RAW_PATH



def inspect_csv(path: Path, name: str) -> None:
    """Print the structure and basic quality information for a CSV file."""

    print(f"\n{'=' * 70}")
    print(f"{name}: {path}")
    print("=" * 70)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:>2}. {column}")

    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].to_string() or "No missing values")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nFirst three rows:")
    print(df.head(3).to_string(index=False))

    print("\nUnique values for low-cardinality columns:")

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 20:
            print(f"\n{column} ({unique_count} values):")
            print(df[column].value_counts(dropna=False).to_string())


def main() -> None:
    inspect_csv(TICKETS_RAW_PATH, "Support tickets")
    inspect_csv(FAQS_RAW_PATH, "FAQs")


if __name__ == "__main__":
    main()
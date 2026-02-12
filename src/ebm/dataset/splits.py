"""Deterministic train/val/test splitting for the Sudoku dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    val_size: int = 500_000,
    test_size: int = 500_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/val/test sets deterministically.

    Args:
        df: Full dataset with 'puzzle' and 'solution' columns.
        val_size: Number of validation samples.
        test_size: Number of test samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) DataFrames.

    """
    holdout_size = val_size + test_size
    train_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=seed)

    val_fraction = val_size / holdout_size
    val_df, test_df = train_test_split(holdout_df, test_size=1.0 - val_fraction, random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

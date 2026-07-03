"""Shared feature preparation utilities for model training and inference."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


TARGET_COLUMNS = ["decision_type", "correct_decision"]
NON_FEATURE_COLUMNS = {
    "correct_decision",
    "decision_type",
    "available_moves",
    "hero_hand",
    "action_sequence",
    "hand_notation",
    "hero_holding",
    "prev_line",
    "hero_pos",
    "instruction",
    "output",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize decision labels to lowercase action tokens."""
    return series.apply(
        lambda value: value.split()[0].lower()
        if isinstance(value, str) and value.strip()
        else "unknown"
    )


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Select numeric feature columns, excluding targets and metadata."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = NON_FEATURE_COLUMNS | {target_col}
    feature_cols = [col for col in numeric_cols if col not in exclude]

    # Guard against duplicate column names from upstream feature engineering.
    seen = set()
    unique_cols = []
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)

    return unique_cols


def prepare_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    label_encoder: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Optional[LabelEncoder]]:
    """
    Prepare model features and labels.

    Args:
        df: Input dataframe.
        feature_names: Expected feature columns when fit=False.
        label_encoder: Existing label encoder when fit=False.
        fit: Whether to infer feature columns and fit the label encoder.

    Returns:
        Tuple of (X, y, feature_names, label_encoder).
    """
    target_col = next((col for col in TARGET_COLUMNS if col in df.columns), None)
    if target_col is None:
        raise ValueError(
            f"No target column found. Expected one of: {TARGET_COLUMNS}"
        )

    y = normalize_labels(df[target_col].copy())

    if fit:
        selected_features = select_feature_columns(df, target_col)
        if not selected_features:
            raise ValueError("No numeric feature columns found for training.")
        feature_names = selected_features
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
    else:
        if feature_names is None:
            raise ValueError("feature_names must be provided when fit=False.")
        if label_encoder is None:
            raise ValueError("label_encoder must be provided when fit=False.")

    X = df.reindex(columns=feature_names, fill_value=0).copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    return X, y, feature_names, label_encoder

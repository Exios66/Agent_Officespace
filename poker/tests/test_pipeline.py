"""Unit tests for poker pipeline bug fixes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import PokerDataPreprocessor
from src.features.engineering import PokerFeatureEngineer
from src.features.feature_utils import prepare_features


@pytest.fixture
def sample_preflop_row():
    return {
        "hero_pos": "BTN",
        "hero_holding": "AhKh",
        "prev_line": "UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold",
        "correct_decision": "call",
        "num_players": 4,
        "num_bets": 3,
        "pot_size": 117.0,
    }


def test_action_sequence_parsing():
    preprocessor = PokerDataPreprocessor()
    actions = preprocessor.parse_action_sequence(
        "UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold"
    )

    assert len(actions) == 6
    assert actions[0]["action"] == "bet"
    assert actions[0]["amount"] == 2.0
    assert actions[3]["action"] == "allin"
    assert actions[4]["action"] == "fold"


def test_action_count_columns(sample_preflop_row):
    preprocessor = PokerDataPreprocessor()
    df = preprocessor.extract_preflop_features(pd.DataFrame([sample_preflop_row]))

    assert df.loc[0, "num_fold"] == 2
    assert df.loc[0, "num_call"] == 1
    assert df.loc[0, "num_bet"] == 1
    assert df.loc[0, "num_raise"] == 1
    assert df.loc[0, "num_allin"] == 1


def test_no_duplicate_pot_size_column(sample_preflop_row):
    preprocessor = PokerDataPreprocessor()
    engineer = PokerFeatureEngineer()

    df = preprocessor.extract_preflop_features(pd.DataFrame([sample_preflop_row]))
    df = engineer.engineer_features(df)

    assert df.columns.duplicated().sum() == 0
    assert isinstance(df.loc[0, "pot_size"], (float, np.floating, int))


def test_prepare_features_preserves_training_columns():
    train_df = pd.DataFrame(
        {
            "decision_type": ["call", "fold"],
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "decision_type": ["call"],
            "feat_a": [1.0],
        }
    )

    _, _, feature_names, label_encoder = prepare_features(train_df, fit=True)
    X_test, y_test, _, _ = prepare_features(
        test_df,
        feature_names=feature_names,
        label_encoder=label_encoder,
        fit=False,
    )

    assert list(X_test.columns) == feature_names
    assert X_test.loc[0, "feat_b"] == 0
    assert y_test.iloc[0] == "call"


def test_action_sequence_from_parquet_array(sample_preflop_row, tmp_path):
    """Parquet can deserialize action lists as numpy arrays."""
    preprocessor = PokerDataPreprocessor()
    engineer = PokerFeatureEngineer()

    df = preprocessor.extract_preflop_features(pd.DataFrame([sample_preflop_row]))
    parquet_path = tmp_path / "test_actions.parquet"
    df.to_parquet(parquet_path)
    df = pd.read_parquet(parquet_path)

    engineered = engineer.engineer_features(df)
    assert "aggression_factor" in engineered.columns


def test_label_normalization_handles_bet_sizing():
    labels = pd.Series(["Call", "bet 18", "Raise 24"])
    _, y, _, _ = prepare_features(
        pd.DataFrame({"correct_decision": labels, "feat": [1, 2, 3]}),
        fit=True,
    )

    assert list(y) == ["call", "bet", "raise"]

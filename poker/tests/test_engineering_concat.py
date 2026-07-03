"""Regression tests for `PokerFeatureEngineer.engineer_features`.

- Feature dicts for pot / hand / position may share column names with the
  incoming DataFrame (notably ``pot_size``). Naive ``pd.concat`` would create
  duplicate columns that ``to_parquet`` rejects.
- ``action_sequence`` deserialised from parquet is a ``np.ndarray`` object,
  which used to blow up ``if not actions:`` inside the encoder.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import PokerDataPreprocessor  # noqa: E402
from src.features.engineering import PokerFeatureEngineer  # noqa: E402


def _sample_frame():
    raw = pd.DataFrame(
        [
            {
                "prev_line": "UTG/2.0bb/HJ/fold",
                "hero_pos": "BTN",
                "hero_holding": "AhKh",
                "correct_decision": "raise",
                "num_players": 5,
                "num_bets": 1,
                "available_moves": "['fold', 'call', 'raise']",
                "pot_size": 4.5,
            },
            {
                "prev_line": "",
                "hero_pos": "UTG",
                "hero_holding": "7c2d",
                "correct_decision": "fold",
                "num_players": 6,
                "num_bets": 0,
                "available_moves": "['fold', 'call']",
                "pot_size": 1.5,
            },
        ]
    )
    pre = PokerDataPreprocessor()
    return pre.extract_preflop_features(raw)


def test_engineer_features_no_duplicate_columns(tmp_path):
    engineered = PokerFeatureEngineer().engineer_features(_sample_frame())
    assert engineered.columns.is_unique, (
        "engineered features should not have duplicate column names; "
        f"duplicates = {engineered.columns[engineered.columns.duplicated()].tolist()}"
    )
    # parquet write is the strictest check for duplicates.
    parquet_path = tmp_path / "features.parquet"
    engineered.to_parquet(parquet_path, index=False)
    assert parquet_path.is_file()


def test_encode_action_sequence_handles_numpy_and_empty():
    from src.features.engineering import ActionSequenceEncoder

    enc = ActionSequenceEncoder()

    # empty list
    empty = enc.encode_action_sequence([])
    assert empty["action_count"] == 0

    # numpy object array (how parquet re-hydrates lists of dicts)
    arr = np.array([{"action": "bet", "amount": 2.0}, {"action": "call", "amount": 2.0}], dtype=object)
    feats = enc.encode_action_sequence(arr)
    assert feats["action_count"] == 2
    assert feats["call_count"] == 1

    # None / NaN
    assert enc.encode_action_sequence(None)["action_count"] == 0
    assert enc.encode_action_sequence(float("nan"))["action_count"] == 0

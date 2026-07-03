# `poker_predictor.features`

Deterministic feature engineering. Every function here maps a
`PreflopSample` (or a sub-piece of one) to a flat `dict[str, float]`,
which [`build.py`](build.py) then stitches into a pandas DataFrame for
training and evaluation.

## Modules

| Module | Exports | Notes |
|---|---|---|
| [`cards.py`](cards.py) | `hand_class`, `hand_class_index`, `all_hand_classes`, `chen_strength`, `card_features`, plus `is_pair`, `is_suited`, `is_connector`, `is_broadway`, `gap`, `high_low`. | Every 2-card starting hand collapses to one of 169 canonical classes (13 pairs + 78 suited + 78 offsuit). `hand_class_index` gives a stable integer in `[0, 169)` suitable for embedding tables. |
| [`equity.py`](equity.py) | `preflop_equity_vs_random(hole)`. | Compact lookup keyed on the 169-class label with published HU-vs-random equity numbers. Missing classes fall back to a smooth Chen-based approximation. |
| [`position.py`](position.py) | `position_features(hero, num_players)`. | 6-max position index + `pos_is_blind`, `pos_is_btn`, `pos_is_early`, `seats_to_act_after`. |
| [`actions.py`](actions.py) | `action_features(events, hero)`. | Derives `num_raises`, `num_callers`, `is_3bet_pot`, `is_4bet_pot`, `is_squeeze`, `last_bet_bb`, `max_bet_bb`, `aggressor_pos_idx`, etc. |
| [`stacks.py`](stacks.py) | `stack_features(hero_stack_bb, pot_bb, facing_bet_bb)`. | `pot_odds`, `spr_proxy`, `allin_threshold` (≤25bb), `deep_stack` (≥100bb). |
| [`build.py`](build.py) | `sample_features(sample)` (one row), `build_feature_matrix(samples)` (DataFrame + labels), `canonical_action_label(raw)` (maps free-form PokerBench labels to `{fold, check, call, raise, allin}`), and the `ACTION_LABELS` list. | `canonical_action_label` also recognises the bare bet-size shorthand (`"3.0bb"`) that PokerBench sometimes uses, canonicalising it to `"raise"` so it does not spawn a singleton class in stratified splits. |

## Usage

```python
from poker_predictor.data.schemas import PreflopSample, Position
from poker_predictor.features.build import sample_features, build_feature_matrix

sample = PreflopSample(
    hero_pos=Position.BTN, hero_hole="AhKh",
    hero_stack_bb=100.0, num_players=6, pot_bb=1.5,
    action_sequence=[], available_moves=["fold", "call", "raise"],
)

# Single-sample feature dict (dozens of floats)
feats = sample_features(sample)
assert "chen_strength" in feats and "pot_odds" in feats

# Batched — DataFrame + label list, ready for sklearn
X, y = build_feature_matrix([sample])
```

## Tests

- [`../../tests/test_features.py`](../../tests/test_features.py)
- [`../../tests/test_canonical_label.py`](../../tests/test_canonical_label.py)

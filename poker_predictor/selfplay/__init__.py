"""Self-play data-generation pipeline for the poker LLM stack.

Compose an :class:`~poker_predictor.selfplay.runner.SelfPlayEngine` with a
list of :class:`~poker_predictor.selfplay.players.Player` instances to
generate PokerBench-compatible ``{"instruction", "output"}`` decisions from
simulated hands. The trajectories carry per-decision reward (final BB
delta) so the same file can drive weighted-SFT, best-of-N filtering, or
policy-gradient extensions.

The intended feedback loop:

.. code-block::

    gen_0:  base LLM + heuristic opponents  → data_0
    train:  data_0 → LLM_v1
    gen_1:  LLM_v1 vs LLM_v1 vs heuristic  → data_1  (harder distribution)
    train:  data_0 + data_1 → LLM_v2
    ...

See :mod:`poker_predictor.selfplay.cli` for the ``poker-predictor selfplay``
subcommands.
"""
from .engine import ActionKind, HandResult, NLHEEngine, Seat, Street
from .hand_eval import parse_card, parse_cards, score_5, score_hand
from .players import (
    HeuristicPlayer,
    LLMPlayer,
    LooseAggressivePlayer,
    Player,
    PlayerRoster,
    PolicyModelPlayer,
    RandomPlayer,
    TightAggressivePlayer,
)
from .prompts import DecisionPrompt, ParsedAction, parse_action_response, render_decision_prompt
from .reward import (
    HandTrajectory,
    TrajectoryDecision,
    compute_advantage,
    keep_positive_expectation,
    keep_showdown_actions,
    keep_winning_actions,
)
from .runner import (
    GenerationLog,
    SelfPlayConfig,
    SelfPlayEngine,
    prepare_sft_from_trajectories,
    run_generation_loop,
)

__all__ = [
    "NLHEEngine",
    "Seat",
    "Street",
    "ActionKind",
    "HandResult",
    "score_5",
    "score_hand",
    "parse_card",
    "parse_cards",
    "Player",
    "PlayerRoster",
    "RandomPlayer",
    "HeuristicPlayer",
    "TightAggressivePlayer",
    "LooseAggressivePlayer",
    "LLMPlayer",
    "PolicyModelPlayer",
    "DecisionPrompt",
    "ParsedAction",
    "render_decision_prompt",
    "parse_action_response",
    "TrajectoryDecision",
    "HandTrajectory",
    "keep_winning_actions",
    "keep_showdown_actions",
    "keep_positive_expectation",
    "compute_advantage",
    "SelfPlayEngine",
    "SelfPlayConfig",
    "GenerationLog",
    "prepare_sft_from_trajectories",
    "run_generation_loop",
]

"""Pydantic schemas for preflop poker samples.

We normalize inputs from PokerBench (CSV/JSON) and other hand-history sources
to a single :class:`PreflopSample`, which downstream feature engineering and
training code consume.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Position(str, Enum):
    """6-max positions. PokerBench uses these labels."""

    UTG = "UTG"
    HJ = "HJ"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"

    @property
    def order(self) -> int:
        """Preflop action order (SB/BB post blinds; UTG acts first preflop)."""
        return {
            Position.UTG: 0,
            Position.HJ: 1,
            Position.CO: 2,
            Position.BTN: 3,
            Position.SB: 4,
            Position.BB: 5,
        }[self]


class ActionType(str, Enum):
    """Canonical preflop action tokens."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALLIN = "allin"
    POST = "post"


class ActionEvent(BaseModel):
    """A single player action in the preflop action sequence."""

    position: Position
    action: ActionType
    amount_bb: float | None = Field(
        default=None, description="Bet/raise amount in big blinds, if applicable."
    )

    def token(self) -> str:
        if self.amount_bb is None:
            return f"{self.position.value}_{self.action.value}"
        return f"{self.position.value}_{self.action.value}_{self.amount_bb:g}bb"


class PreflopSample(BaseModel):
    """Normalized preflop decision point.

    All monetary values are in big blinds (BB) for scale-invariance across
    stakes.
    """

    hero_pos: Position
    hero_hole: str = Field(
        ..., min_length=4, max_length=4, description="Two cards concatenated, e.g. 'AhKs'."
    )
    hero_stack_bb: float = Field(default=100.0, ge=0.0)
    num_players: int = Field(..., ge=2, le=9)
    pot_bb: float = Field(..., ge=0.0)
    num_bets: int = Field(default=0, ge=0)

    action_sequence: list[ActionEvent] = Field(default_factory=list)
    available_moves: list[str] = Field(default_factory=list)

    correct_decision: str | None = Field(
        default=None, description="Solver-optimal decision label, if present."
    )

    raw: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)

    @field_validator("hero_hole")
    @classmethod
    def _validate_hole(cls, v: str) -> str:
        if len(v) != 4:
            raise ValueError(f"hero_hole must be 4 chars (e.g. 'AhKs'), got {v!r}")
        ranks = set("23456789TJQKA")
        suits = set("shdc")
        c1, c2 = v[:2], v[2:]
        for c in (c1, c2):
            if c[0] not in ranks or c[1] not in suits:
                raise ValueError(f"invalid card {c!r} in hole {v!r}")
        if c1 == c2:
            raise ValueError(f"duplicate card in hole {v!r}")
        return v

    @property
    def facing_bet_bb(self) -> float:
        """Largest raise/allin amount currently on the table (0 if unopened)."""
        amts = [e.amount_bb for e in self.action_sequence if e.amount_bb is not None]
        return max(amts) if amts else 0.0

    @property
    def is_open_pot(self) -> bool:
        return not any(e.action in (ActionType.RAISE, ActionType.ALLIN) for e in self.action_sequence)

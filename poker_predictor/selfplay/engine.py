"""No-Limit Texas Hold'em hand engine for self-play.

The engine is deliberately compact and single-hand-oriented: create one with
a seat roster, call :meth:`NLHEEngine.reset` to start a new hand, then loop
:meth:`legal_actions` + :meth:`apply_action` until :attr:`terminal` is
``True``.

Design choices:

- Positions follow the 6-max convention when 6 seats are configured
  (``UTG``, ``HJ``, ``CO``, ``BTN``, ``SB``, ``BB``). For 2-9 seats we
  degrade gracefully by dropping/renaming early positions.
- Bet sizes are tracked in *big blinds* (BB) to match the rest of the
  codebase.
- Side pots are computed at showdown using the *total-contribution* method
  (each pot layer is the difference between contribution tiers).
- The engine exposes both a structured trajectory (``events``) and a
  natural-language ``prev_line`` string that plugs into the PokerBench
  prompt renderer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from .hand_eval import RANK_ORDER, SUITS, Card, cards_to_str, score_hand


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALLIN = "allin"
    POST_SB = "post_sb"
    POST_BB = "post_bb"


DEFAULT_POSITIONS_BY_SEATS: dict[int, list[str]] = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "UTG+1", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "MP", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}


@dataclass
class Seat:
    seat_id: int
    name: str
    stack_bb: float
    position: str = ""
    hole: list[Card] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    contributed_this_street: float = 0.0
    total_contributed: float = 0.0
    last_action: ActionKind | None = None

    @property
    def is_live(self) -> bool:
        return not self.folded and not self.all_in

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in and self.stack_bb > 0


@dataclass
class ActionRecord:
    """A single action taken by a seat, with enough context to reconstruct state."""

    seat_id: int
    position: str
    street: Street
    action: ActionKind
    amount_bb: float | None = None
    total_after_bb: float | None = None
    stack_after_bb: float | None = None
    pot_after_bb: float | None = None


@dataclass
class HandResult:
    """Outcome of a completed hand."""

    winners: list[int]
    pots: list[dict]
    board: list[Card]
    net_deltas_bb: dict[int, float]
    reason: str
    showdown: bool


class NLHEEngine:
    """Single-hand No-Limit Hold'em engine, driven step by step."""

    def __init__(
        self,
        num_seats: int,
        starting_stack_bb: float = 100.0,
        small_blind_bb: float = 0.5,
        big_blind_bb: float = 1.0,
        min_raise_bb: float = 1.0,
        seat_names: list[str] | None = None,
        positions: list[str] | None = None,
    ) -> None:
        if num_seats < 2:
            raise ValueError("need at least 2 seats")
        if num_seats > 9:
            raise ValueError("engine supports up to 9 seats")

        self.num_seats = num_seats
        self.starting_stack_bb = float(starting_stack_bb)
        self.small_blind_bb = float(small_blind_bb)
        self.big_blind_bb = float(big_blind_bb)
        self.min_raise_bb_base = float(min_raise_bb)

        if positions is None:
            positions = list(DEFAULT_POSITIONS_BY_SEATS[num_seats])
        if len(positions) != num_seats:
            raise ValueError("positions must match num_seats")
        self.position_order: list[str] = list(positions)

        if seat_names is None:
            seat_names = [f"P{i}" for i in range(num_seats)]
        if len(seat_names) != num_seats:
            raise ValueError("seat_names must match num_seats")

        self.seats: list[Seat] = [
            Seat(seat_id=i, name=seat_names[i], stack_bb=self.starting_stack_bb, position="")
            for i in range(num_seats)
        ]

        self.button_idx: int = num_seats - 3 if num_seats >= 3 else 0
        self._assign_positions()
        self.street: Street = Street.PREFLOP
        self.board: list[Card] = []
        self.deck: list[Card] = []
        self.rng = random.Random()
        self.hand_id: int = 0

        self.current_bet_bb: float = 0.0
        self.min_raise_amount_bb: float = self.big_blind_bb
        self.last_aggressor: int | None = None
        self.actor_idx: int = 0
        self.terminal: bool = False
        self.result: HandResult | None = None

        self.events: list[ActionRecord] = []

    def reset(
        self,
        button_idx: int | None = None,
        seed: int | None = None,
        reset_stacks: bool = True,
        stacks_bb: list[float] | None = None,
    ) -> None:
        """Start a fresh hand: shuffle, deal, post blinds, set first actor."""
        self.hand_id += 1
        self.rng = random.Random(seed)
        self.deck = [(r, s) for r in range(2, 15) for s in SUITS]
        self.rng.shuffle(self.deck)
        self.board = []

        if button_idx is not None:
            self.button_idx = button_idx % self.num_seats
        self._assign_positions()

        for i, seat in enumerate(self.seats):
            if stacks_bb is not None:
                seat.stack_bb = float(stacks_bb[i])
            elif reset_stacks:
                seat.stack_bb = self.starting_stack_bb
            seat.hole = []
            seat.folded = False
            seat.all_in = False
            seat.contributed_this_street = 0.0
            seat.total_contributed = 0.0
            seat.last_action = None

        for _ in range(2):
            for seat in self._seat_order_from(self._first_seat_after_button()):
                if seat.stack_bb > 0:
                    seat.hole.append(self.deck.pop())

        self.street = Street.PREFLOP
        self.current_bet_bb = 0.0
        self.min_raise_amount_bb = self.big_blind_bb
        self.last_aggressor = None
        self.events = []
        self.terminal = False
        self.result = None

        sb_idx = self._sb_idx()
        bb_idx = self._bb_idx()
        self._post_blind(sb_idx, self.small_blind_bb, ActionKind.POST_SB)
        self._post_blind(bb_idx, self.big_blind_bb, ActionKind.POST_BB)
        self.current_bet_bb = self.big_blind_bb
        self.min_raise_amount_bb = self.big_blind_bb
        self.last_aggressor = bb_idx

        self.actor_idx = self._next_actor_after(bb_idx)

    def _assign_positions(self) -> None:
        """Reassign seat position labels so the button sits at ``self.button_idx``.

        For seat counts 3-9 we pivot ``DEFAULT_POSITIONS_BY_SEATS`` around the
        BTN label. For heads-up we place SB at the button and BB at the other
        seat.
        """
        base = self.position_order
        if self.num_seats == 2:
            self.seats[self.button_idx].position = "SB"
            self.seats[(self.button_idx + 1) % 2].position = "BB"
            return
        try:
            btn_offset = base.index("BTN")
        except ValueError:
            btn_offset = self.num_seats - 3
        for i, label in enumerate(base):
            seat_idx = (self.button_idx + (i - btn_offset)) % self.num_seats
            self.seats[seat_idx].position = label

    def _post_blind(self, seat_idx: int, amount: float, kind: ActionKind) -> None:
        seat = self.seats[seat_idx]
        pay = min(amount, seat.stack_bb)
        seat.stack_bb -= pay
        seat.contributed_this_street += pay
        seat.total_contributed += pay
        if seat.stack_bb == 0:
            seat.all_in = True
        seat.last_action = kind
        self.events.append(
            ActionRecord(
                seat_id=seat_idx,
                position=seat.position,
                street=self.street,
                action=kind,
                amount_bb=pay,
                total_after_bb=seat.contributed_this_street,
                stack_after_bb=seat.stack_bb,
                pot_after_bb=self.pot_bb,
            )
        )

    def _sb_idx(self) -> int:
        if self.num_seats == 2:
            return self.button_idx
        return (self.button_idx + 1) % self.num_seats

    def _bb_idx(self) -> int:
        if self.num_seats == 2:
            return (self.button_idx + 1) % self.num_seats
        return (self.button_idx + 2) % self.num_seats

    def _first_seat_after_button(self) -> int:
        return (self.button_idx + 1) % self.num_seats

    def _seat_order_from(self, start: int):
        for k in range(self.num_seats):
            yield self.seats[(start + k) % self.num_seats]

    def _next_actor_after(self, idx: int) -> int:
        for k in range(1, self.num_seats + 1):
            j = (idx + k) % self.num_seats
            if self.seats[j].can_act:
                return j
        return idx

    @property
    def pot_bb(self) -> float:
        return sum(s.total_contributed for s in self.seats)

    @property
    def actor(self) -> Seat:
        return self.seats[self.actor_idx]

    def _live_seats(self) -> list[Seat]:
        return [s for s in self.seats if not s.folded]

    def _actable_seats(self) -> list[Seat]:
        return [s for s in self.seats if s.can_act]

    def legal_actions(self) -> dict:
        """Return the set of legal actions for :attr:`actor_idx` as a dict.

        Keys always include ``"fold"`` and ``"allin"``. Depending on the
        state we also expose ``"check"``, ``"call"`` (with the required
        amount), and ``"raise"`` (with ``min`` and ``max`` amounts).
        """
        if self.terminal:
            return {}
        seat = self.actor
        to_call = max(0.0, self.current_bet_bb - seat.contributed_this_street)
        can_check = to_call <= 1e-9
        max_raise_to = seat.contributed_this_street + seat.stack_bb
        min_raise_to = self.current_bet_bb + max(self.min_raise_amount_bb, self.big_blind_bb)
        min_raise_to = min(min_raise_to, max_raise_to)

        actions: dict = {"fold": True}
        if can_check:
            actions["check"] = True
        else:
            actions["call"] = min(to_call, seat.stack_bb)
        if max_raise_to > self.current_bet_bb + 1e-9 and seat.stack_bb > 0:
            actions["raise"] = {
                "min_to_bb": round(min_raise_to, 6),
                "max_to_bb": round(max_raise_to, 6),
            }
        if seat.stack_bb > 0:
            actions["allin"] = round(max_raise_to, 6)
        return actions

    def apply_action(self, kind: str, amount_bb: float | None = None) -> ActionRecord:
        """Apply an action for :attr:`actor_idx`.

        ``kind`` is one of ``"fold"``, ``"check"``, ``"call"``, ``"raise"``,
        ``"allin"``. For ``"raise"``, ``amount_bb`` is the *raise-to* amount
        (i.e. the total street contribution the actor is raising to). We
        clamp against min/max and legality; illegal actions raise
        :class:`ValueError`.
        """
        if self.terminal:
            raise RuntimeError("hand is terminal")
        seat = self.actor
        legal = self.legal_actions()
        kind = kind.lower().strip()
        action = ActionKind(kind) if kind in {a.value for a in ActionKind} else None
        if action is None:
            raise ValueError(f"unknown action {kind!r}")

        record_amount: float | None = None
        if action is ActionKind.FOLD:
            seat.folded = True
        elif action is ActionKind.CHECK:
            if "check" not in legal:
                raise ValueError("check not legal — must call or fold")
        elif action is ActionKind.CALL:
            if "call" not in legal:
                raise ValueError("call not legal")
            pay = float(legal["call"])
            seat.stack_bb -= pay
            seat.contributed_this_street += pay
            seat.total_contributed += pay
            record_amount = seat.contributed_this_street
            if seat.stack_bb == 0:
                seat.all_in = True
        elif action is ActionKind.RAISE:
            if "raise" not in legal:
                raise ValueError("raise not legal")
            if amount_bb is None:
                raise ValueError("raise requires amount_bb (raise-to)")
            info = legal["raise"]
            raise_to = float(amount_bb)
            if raise_to > info["max_to_bb"] + 1e-6:
                raise ValueError(f"raise {raise_to} > max {info['max_to_bb']}")
            if raise_to < info["min_to_bb"] - 1e-6 and raise_to < seat.contributed_this_street + seat.stack_bb:
                raise ValueError(f"raise {raise_to} < min {info['min_to_bb']}")
            pay = raise_to - seat.contributed_this_street
            if pay > seat.stack_bb + 1e-6:
                raise ValueError("insufficient stack for raise")
            seat.stack_bb -= pay
            seat.contributed_this_street = raise_to
            seat.total_contributed += pay
            self.min_raise_amount_bb = max(self.min_raise_amount_bb, raise_to - self.current_bet_bb)
            self.current_bet_bb = raise_to
            self.last_aggressor = seat.seat_id
            record_amount = raise_to
            if seat.stack_bb == 0:
                seat.all_in = True
        elif action is ActionKind.ALLIN:
            all_in_to = seat.contributed_this_street + seat.stack_bb
            pay = seat.stack_bb
            seat.stack_bb = 0.0
            seat.contributed_this_street = all_in_to
            seat.total_contributed += pay
            seat.all_in = True
            if all_in_to > self.current_bet_bb + 1e-9:
                self.min_raise_amount_bb = max(self.min_raise_amount_bb, all_in_to - self.current_bet_bb)
                self.current_bet_bb = all_in_to
                self.last_aggressor = seat.seat_id
            record_amount = all_in_to
        else:
            raise ValueError(f"unhandled action {kind!r}")

        seat.last_action = action
        rec = ActionRecord(
            seat_id=seat.seat_id,
            position=seat.position,
            street=self.street,
            action=action,
            amount_bb=record_amount,
            total_after_bb=seat.contributed_this_street,
            stack_after_bb=seat.stack_bb,
            pot_after_bb=self.pot_bb,
        )
        self.events.append(rec)

        self._advance()
        return rec

    def _advance(self) -> None:
        live = [s for s in self.seats if not s.folded]
        if len(live) == 1:
            self._finish_by_fold(live[0])
            return

        if self._street_settled():
            actable = [s for s in self.seats if s.can_act]
            if len(actable) < 2:
                self._run_out_and_showdown()
                return
            self._next_street()
            return

        self.actor_idx = self._next_actor_after(self.actor_idx)

    def _street_settled(self) -> bool:
        live = [s for s in self.seats if not s.folded]
        actable = [s for s in live if not s.all_in]
        if not actable:
            return True
        target = self.current_bet_bb
        for s in actable:
            if abs(s.contributed_this_street - target) > 1e-9:
                return False
        if self.street is Street.PREFLOP:
            bb_idx = self._bb_idx()
            bb_seat = self.seats[bb_idx]
            if abs(self.current_bet_bb - self.big_blind_bb) < 1e-9 and bb_seat.last_action == ActionKind.POST_BB:
                return False
        for s in actable:
            if s.last_action is None or s.last_action in (ActionKind.POST_SB, ActionKind.POST_BB):
                return False
        return True

    def _next_street(self) -> None:
        for seat in self.seats:
            seat.contributed_this_street = 0.0
            if not seat.folded and not seat.all_in:
                seat.last_action = None
        self.current_bet_bb = 0.0
        self.min_raise_amount_bb = self.big_blind_bb
        self.last_aggressor = None

        if self.street is Street.PREFLOP:
            self.street = Street.FLOP
            self.board.extend(self.deck.pop() for _ in range(3))
        elif self.street is Street.FLOP:
            self.street = Street.TURN
            self.board.append(self.deck.pop())
        elif self.street is Street.TURN:
            self.street = Street.RIVER
            self.board.append(self.deck.pop())
        else:
            self._run_out_and_showdown()
            return

        first_after_btn = self._first_seat_after_button()
        for k in range(self.num_seats):
            j = (first_after_btn + k) % self.num_seats
            if self.seats[j].can_act:
                self.actor_idx = j
                return
        self._run_out_and_showdown()

    def _run_out_and_showdown(self) -> None:
        while len(self.board) < 5 and self.street != Street.RIVER:
            if self.street is Street.PREFLOP:
                self.street = Street.FLOP
                self.board.extend(self.deck.pop() for _ in range(3))
            elif self.street is Street.FLOP:
                self.street = Street.TURN
                self.board.append(self.deck.pop())
            elif self.street is Street.TURN:
                self.street = Street.RIVER
                self.board.append(self.deck.pop())
        while len(self.board) < 5:
            self.board.append(self.deck.pop())
        self.street = Street.SHOWDOWN
        self._settle_showdown()

    def _finish_by_fold(self, winner: Seat) -> None:
        pot = self.pot_bb
        winner.stack_bb += pot
        deltas = {s.seat_id: -s.total_contributed for s in self.seats}
        deltas[winner.seat_id] = pot - winner.total_contributed
        self.result = HandResult(
            winners=[winner.seat_id],
            pots=[{"amount_bb": pot, "eligible": [winner.seat_id], "winners": [winner.seat_id]}],
            board=list(self.board),
            net_deltas_bb=deltas,
            reason="fold-to-last",
            showdown=False,
        )
        self.terminal = True

    def _settle_showdown(self) -> None:
        live = [s for s in self.seats if not s.folded]
        if len(live) == 1:
            self._finish_by_fold(live[0])
            return

        contribs = sorted({s.total_contributed for s in self.seats if s.total_contributed > 0})
        prev = 0.0
        pots: list[dict] = []
        for tier in contribs:
            layer = tier - prev
            if layer <= 0:
                prev = tier
                continue
            eligible = [s for s in live if s.total_contributed >= tier]
            amount = 0.0
            for s in self.seats:
                if s.total_contributed >= prev:
                    amount += min(s.total_contributed, tier) - prev
            if amount > 1e-9 and eligible:
                pots.append({"amount_bb": amount, "eligible": [s.seat_id for s in eligible]})
            prev = tier

        deltas = {s.seat_id: -s.total_contributed for s in self.seats}
        pot_records: list[dict] = []
        for pot in pots:
            eligible_ids = pot["eligible"]
            eligible_seats = [self.seats[i] for i in eligible_ids]
            scores = {s.seat_id: score_hand(s.hole + self.board) for s in eligible_seats}
            best = max(scores.values())
            winners = [sid for sid, sc in scores.items() if sc == best]
            share = pot["amount_bb"] / len(winners)
            for wid in winners:
                self.seats[wid].stack_bb += share
                deltas[wid] += share
            pot_records.append(
                {"amount_bb": pot["amount_bb"], "eligible": eligible_ids, "winners": winners}
            )

        winners_final = sorted({w for pot in pot_records for w in pot["winners"]})
        self.result = HandResult(
            winners=winners_final,
            pots=pot_records,
            board=list(self.board),
            net_deltas_bb=deltas,
            reason="showdown",
            showdown=True,
        )
        self.terminal = True

    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot of the current engine state."""
        return {
            "hand_id": self.hand_id,
            "street": self.street.value,
            "board": [f"{RANK_ORDER[c[0] - 2]}{c[1]}" for c in self.board],
            "button_idx": self.button_idx,
            "actor_idx": self.actor_idx if not self.terminal else None,
            "current_bet_bb": self.current_bet_bb,
            "min_raise_amount_bb": self.min_raise_amount_bb,
            "pot_bb": self.pot_bb,
            "seats": [
                {
                    "seat_id": s.seat_id,
                    "name": s.name,
                    "position": s.position,
                    "stack_bb": s.stack_bb,
                    "hole": cards_to_str(s.hole),
                    "folded": s.folded,
                    "all_in": s.all_in,
                    "contributed_this_street": s.contributed_this_street,
                    "total_contributed": s.total_contributed,
                    "last_action": s.last_action.value if s.last_action else None,
                }
                for s in self.seats
            ],
            "terminal": self.terminal,
        }


__all__ = [
    "NLHEEngine",
    "Seat",
    "ActionRecord",
    "ActionKind",
    "Street",
    "HandResult",
    "DEFAULT_POSITIONS_BY_SEATS",
]

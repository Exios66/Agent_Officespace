"""Render engine snapshots into PokerBench-style natural-language prompts.

Also parses LLM free-form responses into a canonical action + raise amount,
using the same regexes / semantics as the rest of the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .engine import ActionKind, NLHEEngine, Street
from .hand_eval import RANK_NAMES, RANK_ORDER, SUIT_NAMES, cards_to_str

SYSTEM_PROMPT = (
    "You are a specialist in playing No Limit Texas Hold'em. You are given "
    "the current game scenario (positions, hole cards, betting history, "
    "stack sizes, and pot). Respond with the single optimal action from the "
    "available moves. Valid outputs: 'fold', 'call', 'check', 'raise <bb>', "
    "'allin'. Do not include any explanation."
)


def _card_to_english(card_str: str) -> str:
    """Convert ``"Ah"`` to ``"Ace of Heart"``."""
    if len(card_str) != 2:
        return card_str
    r, s = card_str[0].upper(), card_str[1].lower()
    rank_val = {c: i + 2 for i, c in enumerate(RANK_ORDER)}.get(r)
    if rank_val is None or s not in SUIT_NAMES:
        return card_str
    return f"{RANK_NAMES[rank_val]} of {SUIT_NAMES[s]}"


def _cards_english_list(cards_str: str) -> str:
    if not cards_str:
        return ""
    tokens = [cards_str[i : i + 2] for i in range(0, len(cards_str), 2)]
    english = [_card_to_english(t) for t in tokens]
    if len(english) == 1:
        return english[0]
    return " and ".join(english)


def _street_actions_prefix(street: Street) -> str:
    return {
        Street.PREFLOP: "Before the flop",
        Street.FLOP: "On the flop",
        Street.TURN: "On the turn",
        Street.RIVER: "On the river",
    }.get(street, "")


def _render_street_actions(engine: NLHEEngine, street: Street) -> str:
    """Render the action log for a single street as a slash-joined string."""
    parts: list[str] = []
    for ev in engine.events:
        if ev.street != street:
            continue
        if ev.action in (ActionKind.POST_SB, ActionKind.POST_BB):
            continue
        if ev.action is ActionKind.FOLD:
            parts.append(f"{ev.position}/fold")
        elif ev.action is ActionKind.CHECK:
            parts.append(f"{ev.position}/check")
        elif ev.action is ActionKind.CALL:
            parts.append(f"{ev.position}/call")
        elif ev.action is ActionKind.RAISE:
            amt = ev.amount_bb if ev.amount_bb is not None else 0.0
            parts.append(f"{ev.position}/{amt:g}bb")
        elif ev.action is ActionKind.ALLIN:
            if ev.amount_bb is not None:
                parts.append(f"{ev.position}/allin/{ev.amount_bb:g}bb")
            else:
                parts.append(f"{ev.position}/allin")
    return "/".join(parts)


def render_prev_line(engine: NLHEEngine) -> str:
    """Concatenate all street action logs into a single prev_line string."""
    chunks: list[str] = []
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        chunk = _render_street_actions(engine, street)
        if chunk:
            chunks.append(chunk)
    return " | ".join(chunks)


@dataclass
class DecisionPrompt:
    """A single decision point ready for a language model."""

    instruction: str
    system: str
    seat_id: int
    position: str
    street: str
    legal_actions: dict
    hero_hole: str
    board: str
    pot_bb: float
    to_call_bb: float
    hand_id: int
    prev_line: str


def render_decision_prompt(engine: NLHEEngine) -> DecisionPrompt:
    """Build a PokerBench-style prompt for the current actor."""
    if engine.terminal:
        raise RuntimeError("engine is terminal — no decision to render")

    seat = engine.actor
    positions_line = ", ".join(engine.position_order)
    hero_hole = cards_to_str(seat.hole)
    hero_hole_english = _cards_english_list(hero_hole)

    prev_line = render_prev_line(engine)
    street_desc = ""
    preflop_actions = _render_street_actions(engine, Street.PREFLOP)
    if preflop_actions:
        street_desc += (
            f"Before the flop, {preflop_actions}. "
            "Assume that all other players that is not mentioned folded. "
        )
    else:
        street_desc += "Before the flop, no action yet. "

    if engine.street != Street.PREFLOP and engine.board:
        board_english = ", ".join(_card_to_english(engine.snapshot()["board"][i]) for i in range(len(engine.board)))
        street_desc += f"The community cards so far are [{board_english}]. "
        for st in (Street.FLOP, Street.TURN, Street.RIVER):
            if st.value == engine.street.value:
                break
            actions = _render_street_actions(engine, st)
            if actions:
                street_desc += f"{_street_actions_prefix(st)}, {actions}. "
        cur_actions = _render_street_actions(engine, engine.street)
        if cur_actions:
            street_desc += f"{_street_actions_prefix(engine.street)}, {cur_actions}. "

    to_call = max(0.0, engine.current_bet_bb - seat.contributed_this_street)
    legal = engine.legal_actions()
    move_tokens: list[str] = ["fold"]
    if "check" in legal:
        move_tokens.append("check")
    if "call" in legal:
        move_tokens.append("call")
    if "raise" in legal:
        move_tokens.append("raise")
    if "allin" in legal:
        move_tokens.append("allin")
    moves_str = ", ".join(f"'{m}'" for m in move_tokens)

    instruction = (
        f"You are playing a {engine.num_seats}-handed No Limit Texas Hold'em cash game. "
        f"The player positions involved in this game are {positions_line}. "
        f"The small blind is {engine.small_blind_bb:g} chips and the big blind is "
        f"{engine.big_blind_bb:g} chips. Everyone started with {engine.starting_stack_bb:g} chips.\n"
        f"In this hand, your position is {seat.position}, and your holding is "
        f"[{hero_hole_english}]. "
        f"{street_desc}"
        f"The current pot size is {engine.pot_bb:g} chips. "
        f"You have {seat.stack_bb:g} chips remaining and need {to_call:g} chips to call. "
        f"Decide the optimal action from {moves_str}."
    )

    return DecisionPrompt(
        instruction=instruction,
        system=SYSTEM_PROMPT,
        seat_id=seat.seat_id,
        position=seat.position,
        street=engine.street.value,
        legal_actions=legal,
        hero_hole=hero_hole,
        board=cards_to_str(engine.board),
        pot_bb=engine.pot_bb,
        to_call_bb=to_call,
        hand_id=engine.hand_id,
        prev_line=prev_line,
    )


ACTION_PARSE_RE = re.compile(
    r"\b(?P<verb>fold|check|call|allin|all[\s\-]?in|raise|bet)"
    r"(?:\s*(?:to\s*)?(?P<amt>\d+(?:\.\d+)?)\s*(?:bb|chips)?)?",
    re.IGNORECASE,
)


@dataclass
class ParsedAction:
    kind: str
    amount_bb: float | None = None
    raw: str = ""

    def as_apply_args(self) -> tuple[str, float | None]:
        return self.kind, self.amount_bb


def parse_action_response(text: str, legal: dict, engine: NLHEEngine | None = None) -> ParsedAction:
    """Interpret a free-form LLM response as a canonical action.

    Illegal / unparseable responses default to a safe legal action:
    ``check`` if available, else ``call`` if free (pot-odds > breakeven),
    else ``fold``.
    """
    text = (text or "").strip().lower()
    for m in ACTION_PARSE_RE.finditer(text):
        verb = m.group("verb").replace(" ", "").replace("-", "")
        amt_s = m.group("amt")
        amount = float(amt_s) if amt_s else None

        if verb == "fold":
            return ParsedAction("fold", None, text)
        if verb == "check" and "check" in legal:
            return ParsedAction("check", None, text)
        if verb == "call":
            if "call" in legal:
                return ParsedAction("call", None, text)
            if "check" in legal:
                return ParsedAction("check", None, text)
        if verb == "allin" and "allin" in legal:
            return ParsedAction("allin", None, text)
        if verb in ("raise", "bet") and "raise" in legal:
            info = legal["raise"]
            if amount is None:
                amount = float(info["min_to_bb"])
            amount = max(float(info["min_to_bb"]), min(float(info["max_to_bb"]), amount))
            return ParsedAction("raise", amount, text)

    if "check" in legal:
        return ParsedAction("check", None, text)
    return ParsedAction("fold", None, text)


__all__ = [
    "SYSTEM_PROMPT",
    "DecisionPrompt",
    "ParsedAction",
    "render_decision_prompt",
    "render_prev_line",
    "parse_action_response",
]

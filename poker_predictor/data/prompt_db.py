"""Build a normalised SQL database of PokerBench natural-language prompts.

The PokerBench dataset ships two paired files per split:

* ``preflop_*_prompt_and_label.json`` — instruction/response pairs
  (natural-language "situation-stylized" prompts + the solver-optimal action).
* ``preflop_*_game_scenario_information.csv`` — the parallel structured
  fields used by our classical ML track (``prev_line``, ``hero_pos``,
  ``hero_holding``, ``correct_decision``, ``num_players``, ``num_bets``,
  ``available_moves``, ``pot_size``).

The two files are aligned by row index. This module fuses them into a
queryable SQL schema with:

* one row per situation (:table:`situations`) — the full prompt text,
  parsed slot values, canonical label, and derived fields
  (hand class, position category, decision type, facing bet, etc.);
* one row per action in the previous line (:table:`situation_actions`);
* one row per available move (:table:`situation_available_moves`);
* one row per position in play (:table:`situation_positions`);
* an enum of prompt template shells (:table:`prompt_templates`);
* a canonical-label taxonomy (:table:`label_taxonomy`);
* three convenience views (``v_situation_summary``,
  ``v_position_action_matrix``, ``v_hand_class_action_matrix``).

The schema is deliberately expressed as portable ANSI SQL with only two
SQLite-specific niceties (``INTEGER PRIMARY KEY`` and a ``WITHOUT ROWID``
option). :func:`postgres_schema` returns the same shape rewritten for
Postgres so the sandbox docker-compose can restore it.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..features.build import canonical_action_label
from ..features.cards import (
    hand_class,
    is_broadway,
    is_pair,
    is_suited,
)
from .parse_preflop import parse_prev_line
from .schemas import ActionType, Position


HOLDING_RE = re.compile(
    r"\[(?P<rank1>\w+)\s+of\s+(?P<suit1>\w+)\s+and\s+(?P<rank2>\w+)\s+of\s+(?P<suit2>\w+)\]",
    re.IGNORECASE,
)
POT_SIZE_RE = re.compile(r"current pot size is (?P<pot>[0-9]+(?:\.[0-9]+)?)\s*chips", re.IGNORECASE)
BLIND_RE = re.compile(
    r"small blind is (?P<sb>[0-9]+(?:\.[0-9]+)?)\s*chips? and the big blind is (?P<bb>[0-9]+(?:\.[0-9]+)?)\s*chips?",
    re.IGNORECASE,
)
STACK_RE = re.compile(r"Everyone started with (?P<stack>[0-9]+(?:\.[0-9]+)?)\s*chips", re.IGNORECASE)
TABLE_SIZE_RE = re.compile(r"(?P<n>\d+)-handed", re.IGNORECASE)
POSITIONS_LINE_RE = re.compile(
    r"player positions involved in this game are\s+(?P<positions>[A-Za-z0-9,\s]+?)\.",
    re.IGNORECASE,
)
HERO_LINE_RE = re.compile(
    r"your position is\s+(?P<pos>[A-Z]{2,3})\s*,",
    re.IGNORECASE,
)


RANK_WORDS = {
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "T",
    "jack": "J",
    "queen": "Q",
    "king": "K",
    "ace": "A",
}
SUIT_WORDS = {
    "spade": "s",
    "spades": "s",
    "heart": "h",
    "hearts": "h",
    "diamond": "d",
    "diamonds": "d",
    "club": "c",
    "clubs": "c",
}


def _english_card(rank_word: str, suit_word: str) -> str | None:
    r = RANK_WORDS.get(rank_word.lower())
    s = SUIT_WORDS.get(suit_word.lower())
    if r is None or s is None:
        return None
    return f"{r}{s}"


def parse_holding_from_prompt(prompt: str) -> str | None:
    """Extract the hero's two hole cards from the prompt ``[<rank> of <suit> and ...]``.

    Returns a 4-char string like ``"9s7s"`` or ``None`` if the prompt did
    not contain a parseable holding.
    """
    m = HOLDING_RE.search(prompt)
    if not m:
        return None
    c1 = _english_card(m.group("rank1"), m.group("suit1"))
    c2 = _english_card(m.group("rank2"), m.group("suit2"))
    if not c1 or not c2:
        return None
    return c1 + c2


def parse_prompt_slots(prompt: str) -> dict[str, Any]:
    """Extract every natural-language slot value from a PokerBench prompt.

    Returns a dict with keys ``table_size``, ``small_blind_chips``,
    ``big_blind_chips``, ``starting_stack_chips``, ``positions`` (list),
    ``hero_pos`` (str), ``hero_hole`` (str), ``pot_size_chips`` (float).
    Missing slots are returned as ``None``.
    """
    out: dict[str, Any] = {
        "table_size": None,
        "small_blind_chips": None,
        "big_blind_chips": None,
        "starting_stack_chips": None,
        "positions": None,
        "hero_pos": None,
        "hero_hole": None,
        "pot_size_chips": None,
    }

    m = TABLE_SIZE_RE.search(prompt)
    if m:
        out["table_size"] = int(m.group("n"))

    m = BLIND_RE.search(prompt)
    if m:
        out["small_blind_chips"] = float(m.group("sb"))
        out["big_blind_chips"] = float(m.group("bb"))

    m = STACK_RE.search(prompt)
    if m:
        out["starting_stack_chips"] = float(m.group("stack"))

    m = POSITIONS_LINE_RE.search(prompt)
    if m:
        raw = m.group("positions")
        out["positions"] = [tok.strip() for tok in raw.split(",") if tok.strip()]

    m = HERO_LINE_RE.search(prompt)
    if m:
        out["hero_pos"] = m.group("pos").upper()

    out["hero_hole"] = parse_holding_from_prompt(prompt)

    m = POT_SIZE_RE.search(prompt)
    if m:
        out["pot_size_chips"] = float(m.group("pot"))

    return out


TEMPLATE_MASKS: list[tuple[re.Pattern[str], str]] = [
    (HOLDING_RE, "[<HOLDING>]"),
    (TABLE_SIZE_RE, "<N>-handed"),
    (
        re.compile(
            r"small blind is [0-9.]+ chips? and the big blind is [0-9.]+ chips?",
            re.IGNORECASE,
        ),
        "small blind is <SB> chips and the big blind is <BB> chips",
    ),
    (
        re.compile(r"Everyone started with [0-9.]+ chips", re.IGNORECASE),
        "Everyone started with <STACK> chips",
    ),
    (
        re.compile(r"your position is\s+[A-Z]{2,3}", re.IGNORECASE),
        "your position is <POS>",
    ),
    (
        re.compile(
            r"player positions involved in this game are\s+[A-Za-z0-9,\s]+?\.",
            re.IGNORECASE,
        ),
        "player positions involved in this game are <POSITIONS>.",
    ),
    (
        re.compile(r"current pot size is [0-9.]+\s*chips", re.IGNORECASE),
        "current pot size is <POT> chips",
    ),
    (
        re.compile(
            r"Before the flop, .+?\. Assume that all other players that is not mentioned folded\.",
            re.IGNORECASE | re.DOTALL,
        ),
        "Before the flop, <PREV_LINE>. Assume that all other players that is not mentioned folded.",
    ),
]


def prompt_template_shell(prompt: str) -> str:
    """Return the prompt with all slot values masked, for template grouping."""
    shell = prompt
    for pat, repl in TEMPLATE_MASKS:
        shell = pat.sub(repl, shell)
    return re.sub(r"\s+", " ", shell).strip()


def prompt_template_hash(shell: str) -> str:
    return hashlib.sha256(shell.encode("utf-8")).hexdigest()[:16]


POSITION_CATEGORY = {
    "UTG": "early",
    "UTG+1": "early",
    "MP": "middle",
    "HJ": "middle",
    "LJ": "middle",
    "CO": "late",
    "BTN": "late",
    "SB": "blinds",
    "BB": "blinds",
}


def _position_category(pos: str | None) -> str | None:
    if pos is None:
        return None
    return POSITION_CATEGORY.get(pos.upper())


def _label_bet_bb(raw: str | None) -> float | None:
    """Extract the numeric raise-to size (in BB) from a raw label like ``"raise 13.1"``.

    Returns None for labels that don't carry a size.
    """
    if raw is None:
        return None
    m = re.search(r"(?P<amt>\d+(?:\.\d+)?)\s*bb?\b", raw, re.IGNORECASE)
    if m:
        return float(m.group("amt"))
    m = re.match(r"\s*(?:raise|bet)\s+(?P<amt>\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if m:
        return float(m.group("amt"))
    return None


def _decision_type(events: list, canonical: str | None) -> str:
    """Classify the *situation* the hero is facing, independent of their label.

    Values: ``unopened`` / ``open_raise_facing`` / ``3bet_facing`` /
    ``4bet+_facing`` / ``limped_facing`` / ``allin_facing`` / ``unknown``.
    """
    n_raises = sum(1 for e in events if e.action == ActionType.RAISE)
    n_allin = sum(1 for e in events if e.action == ActionType.ALLIN)
    n_calls = sum(1 for e in events if e.action == ActionType.CALL)
    if n_allin > 0:
        return "allin_facing"
    if n_raises == 0 and n_calls == 0:
        return "unopened"
    if n_raises == 0 and n_calls > 0:
        return "limped_facing"
    if n_raises == 1:
        return "open_raise_facing"
    if n_raises == 2:
        return "3bet_facing"
    if n_raises >= 3:
        return "4bet+_facing"
    return "unknown"


SCHEMA_SQLITE = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prompt_templates (
    template_id     INTEGER PRIMARY KEY,
    template_hash   TEXT NOT NULL UNIQUE,
    shell           TEXT NOT NULL,
    n_slots         INTEGER NOT NULL,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS label_taxonomy (
    raw_label           TEXT PRIMARY KEY,
    canonical_label     TEXT NOT NULL,
    bet_size_bb         REAL,
    n_occurrences       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS situations (
    situation_id            INTEGER PRIMARY KEY,
    split                   TEXT    NOT NULL,
    source_index            INTEGER NOT NULL,
    prompt_text             TEXT    NOT NULL,
    prompt_template_id      INTEGER NOT NULL REFERENCES prompt_templates(template_id),

    table_size              INTEGER,
    small_blind_chips       REAL,
    big_blind_chips         REAL,
    starting_stack_chips    REAL,

    hero_pos                TEXT    NOT NULL,
    hero_position_category  TEXT,
    hero_hole               TEXT    NOT NULL,
    hero_hand_class         TEXT    NOT NULL,
    hero_is_pair            INTEGER NOT NULL,
    hero_is_suited          INTEGER NOT NULL,
    hero_is_broadway        INTEGER NOT NULL,

    prev_line               TEXT,
    num_players             INTEGER NOT NULL,
    num_bets                INTEGER NOT NULL,
    pot_size_chips          REAL,
    pot_bb                  REAL    NOT NULL,
    facing_bet_bb           REAL    NOT NULL,
    is_open_pot             INTEGER NOT NULL,
    decision_type           TEXT    NOT NULL,

    raw_label               TEXT,
    canonical_label         TEXT,
    label_bet_bb            REAL,

    UNIQUE (split, source_index)
);
CREATE INDEX IF NOT EXISTS ix_situations_hero_pos ON situations(hero_pos);
CREATE INDEX IF NOT EXISTS ix_situations_hero_hand_class ON situations(hero_hand_class);
CREATE INDEX IF NOT EXISTS ix_situations_canonical_label ON situations(canonical_label);
CREATE INDEX IF NOT EXISTS ix_situations_decision_type ON situations(decision_type);
CREATE INDEX IF NOT EXISTS ix_situations_split ON situations(split);
CREATE INDEX IF NOT EXISTS ix_situations_template ON situations(prompt_template_id);

CREATE TABLE IF NOT EXISTS situation_positions (
    situation_id    INTEGER NOT NULL REFERENCES situations(situation_id) ON DELETE CASCADE,
    position        TEXT    NOT NULL,
    seat_order      INTEGER NOT NULL,
    PRIMARY KEY (situation_id, position)
);

CREATE TABLE IF NOT EXISTS situation_actions (
    situation_id    INTEGER NOT NULL REFERENCES situations(situation_id) ON DELETE CASCADE,
    seq_index       INTEGER NOT NULL,
    actor_pos       TEXT    NOT NULL,
    action_type     TEXT    NOT NULL,
    size_bb         REAL,
    PRIMARY KEY (situation_id, seq_index)
);
CREATE INDEX IF NOT EXISTS ix_situation_actions_actor ON situation_actions(actor_pos);
CREATE INDEX IF NOT EXISTS ix_situation_actions_type ON situation_actions(action_type);

CREATE TABLE IF NOT EXISTS situation_available_moves (
    situation_id    INTEGER NOT NULL REFERENCES situations(situation_id) ON DELETE CASCADE,
    move            TEXT    NOT NULL,
    PRIMARY KEY (situation_id, move)
);
"""

VIEWS_SQLITE = """
DROP VIEW IF EXISTS v_situation_summary;
CREATE VIEW v_situation_summary AS
SELECT
    s.situation_id,
    s.split,
    s.hero_pos,
    s.hero_position_category,
    s.hero_hand_class,
    s.hero_is_pair,
    s.hero_is_suited,
    s.hero_is_broadway,
    s.decision_type,
    s.num_players,
    s.num_bets,
    s.pot_bb,
    s.facing_bet_bb,
    s.is_open_pot,
    s.canonical_label,
    s.label_bet_bb,
    s.raw_label,
    (SELECT COUNT(*) FROM situation_actions a WHERE a.situation_id = s.situation_id) AS n_prev_actions,
    (SELECT COUNT(*) FROM situation_available_moves m WHERE m.situation_id = s.situation_id) AS n_available_moves
FROM situations s;

DROP VIEW IF EXISTS v_position_action_matrix;
CREATE VIEW v_position_action_matrix AS
SELECT hero_pos,
       canonical_label,
       COUNT(*) AS n
FROM situations
WHERE canonical_label IS NOT NULL
GROUP BY hero_pos, canonical_label;

DROP VIEW IF EXISTS v_hand_class_action_matrix;
CREATE VIEW v_hand_class_action_matrix AS
SELECT hero_hand_class,
       canonical_label,
       COUNT(*) AS n
FROM situations
WHERE canonical_label IS NOT NULL
GROUP BY hero_hand_class, canonical_label;

DROP VIEW IF EXISTS v_decision_type_mix;
CREATE VIEW v_decision_type_mix AS
SELECT decision_type,
       canonical_label,
       COUNT(*) AS n
FROM situations
WHERE canonical_label IS NOT NULL
GROUP BY decision_type, canonical_label;
"""


def sqlite_schema() -> str:
    return SCHEMA_SQLITE + "\n" + VIEWS_SQLITE


def postgres_schema() -> str:
    """Postgres-flavour DDL: same shape as SQLite, minus SQLite-only syntax.

    Emitted so the docker-compose sandbox can initialise a Postgres 16
    instance with the identical relational structure. The loader inserts
    explicit primary-key values (copied from the Parquet mirror), so PKs
    are plain ``BIGINT PRIMARY KEY`` rather than ``BIGSERIAL``. Booleans
    are stored as ``BOOLEAN`` (in SQLite they map onto ``INTEGER 0/1``).
    """
    pg = SCHEMA_SQLITE
    pg = pg.replace("PRAGMA foreign_keys = ON;", "")
    pg = pg.replace("INTEGER PRIMARY KEY", "BIGINT PRIMARY KEY")
    pg = pg.replace("REAL", "DOUBLE PRECISION")
    for col in (
        "hero_is_pair",
        "hero_is_suited",
        "hero_is_broadway",
        "is_open_pot",
    ):
        pg = re.sub(rf"({col})\s+INTEGER NOT NULL", rf"\1 BOOLEAN NOT NULL", pg)
    return pg + "\n" + VIEWS_SQLITE


@dataclass
class BuildStats:
    n_prompts_seen: int = 0
    n_situations_inserted: int = 0
    n_actions_inserted: int = 0
    n_moves_inserted: int = 0
    n_positions_inserted: int = 0
    n_templates: int = 0
    n_label_variants: int = 0
    per_split_counts: dict[str, int] = field(default_factory=dict)
    unparseable_prompts: int = 0


def _iter_split_rows(
    csv_path: Path, json_path: Path, split_name: str, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    csv_df = pd.read_csv(csv_path)
    with open(json_path) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError(f"expected list JSON at {json_path}, got {type(prompts)}")
    if len(prompts) != len(csv_df):
        # PokerBench occasionally drops the last few rows in one of the two
        # files; align on the shorter length rather than failing.
        n = min(len(prompts), len(csv_df))
    else:
        n = len(csv_df)
    if limit is not None:
        n = min(n, limit)

    for i in range(n):
        row = csv_df.iloc[i].to_dict()
        p = prompts[i]
        if not isinstance(p, dict):
            continue
        prompt_text = str(p.get("instruction", "") or "")
        raw_label = p.get("output")
        if isinstance(raw_label, str):
            raw_label = raw_label.strip()
        yield {
            "split": split_name,
            "source_index": i,
            "prompt_text": prompt_text,
            "raw_label": raw_label,
            "prev_line": row.get("prev_line"),
            "hero_pos": row.get("hero_pos"),
            "hero_holding": row.get("hero_holding"),
            "correct_decision": row.get("correct_decision"),
            "num_players": row.get("num_players"),
            "num_bets": row.get("num_bets"),
            "available_moves_raw": row.get("available_moves"),
            "pot_size": row.get("pot_size"),
        }


def _parse_available_moves(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    try:
        import ast

        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [tok.strip() for tok in s.strip("[]").split(",") if tok.strip()]


def _seat_order(positions: list[str]) -> dict[str, int]:
    """Assign a seat order using ``Position.order`` when possible, else input order."""
    known: dict[str, int] = {}
    for i, p in enumerate(positions):
        try:
            known[p] = Position(p).order
        except ValueError:
            known[p] = 100 + i
    return known


def build_sqlite_database(
    splits: dict[str, tuple[Path, Path]],
    db_path: Path,
    limit_per_split: int | None = None,
) -> BuildStats:
    """Materialise a SQLite DB from PokerBench (CSV + JSON) files.

    Parameters
    ----------
    splits:
        Mapping of split name (e.g. ``"train"``, ``"test"``) to
        ``(csv_path, prompt_json_path)``.
    db_path:
        Destination SQLite path. Overwritten if it exists.
    limit_per_split:
        Optional cap for smoke tests.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()
    templates: dict[str, int] = {}
    label_counts: dict[str, tuple[str, float | None, int]] = {}

    with closing(sqlite3.connect(db_path)) as con:
        con.executescript(sqlite_schema())
        cur = con.cursor()
        situation_id = 0

        for split_name, (csv_path, json_path) in splits.items():
            per_split = 0
            for row in _iter_split_rows(csv_path, json_path, split_name, limit_per_split):
                stats.n_prompts_seen += 1
                prompt = row["prompt_text"]
                slots = parse_prompt_slots(prompt)

                hero_pos = row.get("hero_pos") or slots.get("hero_pos")
                if hero_pos is None:
                    stats.unparseable_prompts += 1
                    continue
                hero_hole = row.get("hero_holding") or slots.get("hero_hole")
                if hero_hole is None or len(hero_hole) != 4:
                    stats.unparseable_prompts += 1
                    continue

                shell = prompt_template_shell(prompt)
                thash = prompt_template_hash(shell)
                if thash not in templates:
                    n_slots = shell.count("<") + shell.count("[<")
                    cur.execute(
                        "INSERT INTO prompt_templates (template_hash, shell, n_slots, description) VALUES (?, ?, ?, ?)",
                        (thash, shell, n_slots, f"auto-detected from {split_name}"),
                    )
                    templates[thash] = int(cur.lastrowid)
                template_id = templates[thash]

                raw_label = row.get("raw_label") or row.get("correct_decision")
                canonical = canonical_action_label(raw_label if isinstance(raw_label, str) else None)
                bet_bb = _label_bet_bb(raw_label if isinstance(raw_label, str) else None)
                if isinstance(raw_label, str):
                    key = raw_label.strip().lower()
                    prev = label_counts.get(key)
                    label_counts[key] = (canonical or "unknown", bet_bb, (prev[2] if prev else 0) + 1)

                prev_line = row.get("prev_line") or ""
                events = parse_prev_line(str(prev_line))
                facing_bet = 0.0
                for e in events:
                    if e.amount_bb is not None and e.amount_bb > facing_bet:
                        facing_bet = float(e.amount_bb)
                is_open_pot = not any(
                    e.action in (ActionType.RAISE, ActionType.ALLIN) for e in events
                )
                decision_type = _decision_type(events, canonical)

                sb = slots.get("small_blind_chips")
                bb = slots.get("big_blind_chips") or 1.0
                pot_chips = slots.get("pot_size_chips")
                pot_bb = float(row.get("pot_size")) if row.get("pot_size") is not None else (
                    (pot_chips or 0.0) / (bb or 1.0)
                )

                situation_id += 1
                cur.execute(
                    """
                    INSERT INTO situations (
                        situation_id, split, source_index, prompt_text, prompt_template_id,
                        table_size, small_blind_chips, big_blind_chips, starting_stack_chips,
                        hero_pos, hero_position_category, hero_hole, hero_hand_class,
                        hero_is_pair, hero_is_suited, hero_is_broadway,
                        prev_line, num_players, num_bets, pot_size_chips, pot_bb,
                        facing_bet_bb, is_open_pot, decision_type,
                        raw_label, canonical_label, label_bet_bb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        situation_id,
                        split_name,
                        int(row["source_index"]),
                        prompt,
                        template_id,
                        slots.get("table_size"),
                        sb,
                        bb,
                        slots.get("starting_stack_chips"),
                        hero_pos,
                        _position_category(hero_pos),
                        hero_hole,
                        hand_class(hero_hole),
                        1 if is_pair(hero_hole) else 0,
                        1 if is_suited(hero_hole) else 0,
                        1 if is_broadway(hero_hole) else 0,
                        prev_line,
                        int(row.get("num_players") or 0),
                        int(row.get("num_bets") or 0),
                        pot_chips,
                        float(pot_bb),
                        facing_bet,
                        1 if is_open_pot else 0,
                        decision_type,
                        raw_label if isinstance(raw_label, str) else None,
                        canonical,
                        bet_bb,
                    ),
                )
                stats.n_situations_inserted += 1

                positions = slots.get("positions") or []
                order = _seat_order(positions)
                for p in positions:
                    cur.execute(
                        "INSERT OR IGNORE INTO situation_positions (situation_id, position, seat_order) VALUES (?, ?, ?)",
                        (situation_id, p, order.get(p, 99)),
                    )
                    stats.n_positions_inserted += 1

                for seq, ev in enumerate(events):
                    cur.execute(
                        "INSERT INTO situation_actions (situation_id, seq_index, actor_pos, action_type, size_bb) VALUES (?, ?, ?, ?, ?)",
                        (
                            situation_id,
                            seq,
                            ev.position.value,
                            ev.action.value,
                            ev.amount_bb,
                        ),
                    )
                    stats.n_actions_inserted += 1

                for m in _parse_available_moves(row.get("available_moves_raw")):
                    cur.execute(
                        "INSERT OR IGNORE INTO situation_available_moves (situation_id, move) VALUES (?, ?)",
                        (situation_id, m.lower()),
                    )
                    stats.n_moves_inserted += 1

                per_split += 1
            stats.per_split_counts[split_name] = per_split

        cur.executemany(
            "INSERT INTO label_taxonomy (raw_label, canonical_label, bet_size_bb, n_occurrences) VALUES (?, ?, ?, ?)",
            [(k, v[0], v[1], v[2]) for k, v in label_counts.items()],
        )
        stats.n_label_variants = len(label_counts)
        stats.n_templates = len(templates)
        con.commit()

    return stats


def summarize_database(db_path: Path) -> dict[str, Any]:
    """Return a small dict of top-line stats over a built DB (row counts + samples)."""
    with closing(sqlite3.connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        def scalar(q: str) -> int:
            return int(cur.execute(q).fetchone()[0])

        summary = {
            "path": str(db_path),
            "size_bytes": Path(db_path).stat().st_size,
            "counts": {
                "situations": scalar("SELECT COUNT(*) FROM situations"),
                "prompt_templates": scalar("SELECT COUNT(*) FROM prompt_templates"),
                "label_taxonomy": scalar("SELECT COUNT(*) FROM label_taxonomy"),
                "situation_actions": scalar("SELECT COUNT(*) FROM situation_actions"),
                "situation_available_moves": scalar("SELECT COUNT(*) FROM situation_available_moves"),
                "situation_positions": scalar("SELECT COUNT(*) FROM situation_positions"),
            },
            "splits": [
                dict(r)
                for r in cur.execute(
                    "SELECT split, COUNT(*) AS n FROM situations GROUP BY split ORDER BY split"
                ).fetchall()
            ],
            "decision_types": [
                dict(r)
                for r in cur.execute(
                    "SELECT decision_type, COUNT(*) AS n FROM situations GROUP BY decision_type ORDER BY n DESC"
                ).fetchall()
            ],
            "canonical_labels": [
                dict(r)
                for r in cur.execute(
                    "SELECT canonical_label, COUNT(*) AS n FROM situations GROUP BY canonical_label ORDER BY n DESC"
                ).fetchall()
            ],
            "top_label_variants": [
                dict(r)
                for r in cur.execute(
                    "SELECT raw_label, canonical_label, n_occurrences FROM label_taxonomy ORDER BY n_occurrences DESC LIMIT 15"
                ).fetchall()
            ],
            "position_action_matrix": [
                dict(r)
                for r in cur.execute(
                    "SELECT hero_pos, canonical_label, n FROM v_position_action_matrix ORDER BY hero_pos, canonical_label"
                ).fetchall()
            ],
            "template_examples": [
                dict(r)
                for r in cur.execute(
                    "SELECT template_id, template_hash, n_slots, substr(shell,1,200) AS shell_prefix FROM prompt_templates ORDER BY template_id"
                ).fetchall()
            ],
        }

    return summary


__all__ = [
    "build_sqlite_database",
    "summarize_database",
    "sqlite_schema",
    "postgres_schema",
    "parse_prompt_slots",
    "parse_holding_from_prompt",
    "prompt_template_shell",
    "prompt_template_hash",
    "BuildStats",
]

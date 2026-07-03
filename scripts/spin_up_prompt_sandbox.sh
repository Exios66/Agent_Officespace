#!/usr/bin/env bash
# Spin up the PokerBench prompt SQL sandbox locally.
#
# What this does:
#   1. Ensures the raw PokerBench CSV+JSON files exist under
#      poker/data/raw/pokerbench/ (downloads them via the poker/ project's
#      download script if missing).
#   2. Builds a SQLite database at data/pokerbench_prompts.sqlite unless one
#      already exists (or --rebuild is passed).
#   3. Optionally opens the sqlite3 REPL (default) or launches the
#      Datasette web UI on http://localhost:8001 (--serve).
#
# Usage:
#   bash scripts/spin_up_prompt_sandbox.sh              # build + open REPL
#   bash scripts/spin_up_prompt_sandbox.sh --rebuild    # force full rebuild
#   bash scripts/spin_up_prompt_sandbox.sh --serve      # build + Datasette UI
#   bash scripts/spin_up_prompt_sandbox.sh --stats-only # just print stats
#
# Environment:
#   PB_RAW_DIR      override the raw-data directory
#   PB_DB_PATH      override the SQLite output path
#   PB_LIMIT        cap rows per split (smoke test)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DIR="${PB_RAW_DIR:-poker/data/raw/pokerbench}"
DB_PATH="${PB_DB_PATH:-data/pokerbench_prompts.sqlite}"
LIMIT_ARG=()
if [[ -n "${PB_LIMIT:-}" ]]; then
    LIMIT_ARG=(--limit "$PB_LIMIT")
fi

REBUILD=0
SERVE=0
STATS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --rebuild)    REBUILD=1 ;;
        --serve)      SERVE=1 ;;
        --stats-only) STATS_ONLY=1 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
    esac
done

TRAIN_CSV="$RAW_DIR/preflop_60k_train_set_game_scenario_information.csv"
TRAIN_JSON="$RAW_DIR/preflop_60k_train_set_prompt_and_label.json"
TEST_CSV="$RAW_DIR/preflop_1k_test_set_game_scenario_information.csv"
TEST_JSON="$RAW_DIR/preflop_1k_test_set_prompt_and_label.json"

if [[ ! -f "$TRAIN_CSV" || ! -f "$TRAIN_JSON" || ! -f "$TEST_CSV" || ! -f "$TEST_JSON" ]]; then
    echo "[spin_up] Raw PokerBench files missing under $RAW_DIR"
    if [[ -x "$(command -v python3)" ]]; then
        if [[ -f poker/scripts/download_data.py ]]; then
            echo "[spin_up] Downloading via poker/scripts/download_data.py ..."
            (cd poker && python3 scripts/download_data.py)
        else
            echo "[spin_up] No download script found; please populate $RAW_DIR manually." >&2
            exit 1
        fi
    else
        echo "[spin_up] python3 not available; cannot auto-download." >&2
        exit 1
    fi
fi

if [[ $REBUILD -eq 1 || ! -f "$DB_PATH" ]]; then
    echo "[spin_up] Building $DB_PATH from $RAW_DIR ..."
    mkdir -p "$(dirname "$DB_PATH")"
    python3 -m poker_predictor.data.prompt_db_cli build \
        --raw-dir "$RAW_DIR" \
        --db-path "$DB_PATH" \
        "${LIMIT_ARG[@]}"
else
    echo "[spin_up] Reusing existing DB at $DB_PATH (pass --rebuild to force)."
fi

python3 -m poker_predictor.data.prompt_db_cli stats --db-path "$DB_PATH"

if [[ $STATS_ONLY -eq 1 ]]; then
    exit 0
fi

if [[ $SERVE -eq 1 ]]; then
    if ! command -v datasette >/dev/null 2>&1; then
        echo "[spin_up] datasette is not installed. Install with: pip install datasette"
        exit 1
    fi
    echo "[spin_up] Serving Datasette UI at http://localhost:8001 ..."
    exec datasette serve "$DB_PATH" --host 0.0.0.0 --port 8001 --setting truncate_cells_html 200
fi

if command -v sqlite3 >/dev/null 2>&1; then
    echo "[spin_up] Opening sqlite3 REPL. Type '.tables', '.schema situations', or arbitrary SQL."
    echo "[spin_up] Example: SELECT hero_pos, canonical_label, COUNT(*) FROM situations GROUP BY 1,2;"
    exec sqlite3 "$DB_PATH"
else
    echo "[spin_up] sqlite3 CLI not installed. Query via:"
    echo "  python3 -m poker_predictor.data.prompt_db_cli query 'SELECT ...' --db-path $DB_PATH"
fi

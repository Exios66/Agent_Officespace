"""Load the PokerBench prompt sandbox into Postgres from the Parquet mirror.

Executed by the ``loader`` container in
``deploy/postgres-sandbox/docker-compose.yml``. Idempotent: drops and
recreates every table, then copies from the six Parquet files in
``PB_PARQUET_DIR``.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

from poker_predictor.data.prompt_db import postgres_schema


TABLES_IN_LOAD_ORDER = [
    "prompt_templates",
    "label_taxonomy",
    "situations",
    "situation_positions",
    "situation_actions",
    "situation_available_moves",
]


def _connect() -> psycopg.Connection:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg.connect(dsn, autocommit=False)
    return psycopg.connect(
        host=os.environ.get("PGHOST", "postgres"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "pokerbench"),
        password=os.environ.get("PGPASSWORD", "pokerbench"),
        dbname=os.environ.get("PGDATABASE", "pokerbench"),
        autocommit=False,
    )


def _apply_schema(conn: psycopg.Connection) -> None:
    ddl = postgres_schema()
    with conn.cursor() as cur:
        for table in reversed(TABLES_IN_LOAD_ORDER):
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute(ddl)
    conn.commit()
    print("[loader] schema applied")


def _coerce_booleans(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def _copy_dataframe(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ", ".join(df.columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as copy:
            copy.write(buf.read())
    return len(df)


def _load_table(conn: psycopg.Connection, table: str, parquet_dir: Path) -> int:
    path = parquet_dir / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing parquet mirror for table {table}: {path}")
    df = pd.read_parquet(path)
    if table == "situations":
        df = _coerce_booleans(df, ["hero_is_pair", "hero_is_suited", "hero_is_broadway", "is_open_pot"])
    n = _copy_dataframe(conn, table, df)
    print(f"[loader] {table}: copied {n:,} rows")
    return n


def main() -> int:
    parquet_dir = Path(os.environ.get("PB_PARQUET_DIR", "/workspace/data/pokerbench_prompts_parquet"))
    if not parquet_dir.exists():
        print(f"[loader] parquet mirror not found at {parquet_dir}; run "
              "`python -m poker_predictor.data.prompt_db_cli export-parquet` first", file=sys.stderr)
        return 1

    with _connect() as conn:
        _apply_schema(conn)
        total = 0
        for t in TABLES_IN_LOAD_ORDER:
            total += _load_table(conn, t, parquet_dir)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()

    print(f"[loader] done; total rows loaded = {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

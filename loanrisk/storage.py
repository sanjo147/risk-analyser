"""SQLite persistence of assessment runs (local file; nothing leaves the machine)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT,
    model_name TEXT,
    n_total INTEGER,
    n_assessed INTEGER
);
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER REFERENCES batches(id) ON DELETE CASCADE,
    applicant_id TEXT,
    default_probability REAL,
    risk_category TEXT,
    recommendation TEXT,
    record_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_assess_batch ON assessments(batch_id);
"""


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    c.executescript(SCHEMA)
    return c


def save_batch(results: pd.DataFrame, source: str, model_name: str) -> int:
    with closing(_conn()) as c, c:
        cur = c.execute("INSERT INTO batches(created_at, source, model_name, n_total, n_assessed) VALUES (?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), source, model_name, len(results),
                         int(results["default_probability"].notna().sum())))
        bid = cur.lastrowid
        recs = results.astype(object).where(results.notna(), None).to_dict("records")
        c.executemany("INSERT INTO assessments(batch_id, applicant_id, default_probability, risk_category, "
                      "recommendation, record_json) VALUES (?,?,?,?,?,?)",
                      [(bid, r["applicant_id"], r.get("default_probability"), r.get("risk_category"),
                        r.get("recommendation"), json.dumps(r, default=str)) for r in recs])
    return bid


def list_batches() -> pd.DataFrame:
    with closing(_conn()) as c:
        return pd.read_sql("SELECT * FROM batches ORDER BY id DESC", c)


def load_batch(batch_id: int) -> pd.DataFrame:
    with closing(_conn()) as c:
        rows = c.execute("SELECT record_json FROM assessments WHERE batch_id=?", (batch_id,)).fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def delete_all() -> None:
    with closing(_conn()) as c:
        with c:
            c.execute("DELETE FROM assessments")
            c.execute("DELETE FROM batches")
        c.execute("VACUUM")  # physically remove deleted applicant data from the file

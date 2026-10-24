"""The program database: every island, every program, its lineage and its scores (SQLite)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS islands (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    directive TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY,
    island_id INTEGER NOT NULL REFERENCES islands(id),
    parent_id INTEGER REFERENCES programs(id),
    generation INTEGER NOT NULL,
    mutator TEXT NOT NULL,
    code TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    why TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    error TEXT,
    gain REAL,
    breakdown TEXT,
    eval_seconds REAL,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS programs_by_island ON programs (island_id, status, gain);
"""


@dataclass(frozen=True)
class Program:
    id: int
    island_id: int
    parent_id: int | None
    generation: int
    mutator: str
    code: str
    title: str
    why: str
    status: str
    error: str | None
    gain: float | None
    breakdown: dict[str, float]
    cost_usd: float


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


class Database:
    def __init__(self, path: Path) -> None:
        self._con = sqlite3.connect(path)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)

    def close(self) -> None:
        self._con.close()

    def add_island(self, name: str, directive: str) -> int:
        cur = self._con.execute("INSERT INTO islands (name, directive) VALUES (?, ?)", (name, directive))
        self._con.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def islands(self) -> list[sqlite3.Row]:
        return list(self._con.execute("SELECT * FROM islands ORDER BY id"))

    def add_program(
        self,
        *,
        island_id: int,
        parent_id: int | None,
        generation: int,
        mutator: str,
        code: str,
        title: str = "",
        why: str = "",
        error: str | None = None,
        gain: float | None = None,
        breakdown: dict[str, float] | None = None,
        eval_seconds: float | None = None,
        cost_usd: float = 0.0,
    ) -> int:
        cur = self._con.execute(
            """INSERT INTO programs (island_id, parent_id, generation, mutator, code, code_hash, title, why, status,
               error, gain, breakdown, eval_seconds, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                island_id,
                parent_id,
                generation,
                mutator,
                code,
                code_hash(code),
                title,
                why,
                "failed" if error else "ok",
                error,
                gain,
                json.dumps(breakdown or {}),
                eval_seconds,
                cost_usd,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        self._con.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def _program(self, row: sqlite3.Row) -> Program:
        return Program(
            id=row["id"],
            island_id=row["island_id"],
            parent_id=row["parent_id"],
            generation=row["generation"],
            mutator=row["mutator"],
            code=row["code"],
            title=row["title"],
            why=row["why"],
            status=row["status"],
            error=row["error"],
            gain=row["gain"],
            breakdown=json.loads(row["breakdown"] or "{}"),
            cost_usd=row["cost_usd"],
        )

    def best(self, island_id: int, k: int) -> list[Program]:
        rows = self._con.execute(
            "SELECT * FROM programs WHERE island_id = ? AND status = 'ok' ORDER BY gain DESC, id LIMIT ?",
            (island_id, k),
        )
        return [self._program(r) for r in rows]

    def recent_failures(self, island_id: int, k: int) -> list[str]:
        rows = self._con.execute(
            """SELECT title, error FROM programs WHERE island_id = ? AND status = 'failed'
               AND error NOT LIKE 'mutation:%' ORDER BY id DESC LIMIT ?""",
            (island_id, k),
        )
        return [f"{r['title'] or 'untitled'}: {r['error']}" for r in rows]

    def has_code(self, island_id: int, code: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM programs WHERE island_id = ? AND code_hash = ? LIMIT 1", (island_id, code_hash(code))
        ).fetchone()
        return row is not None

    def get(self, program_id: int) -> Program:
        row = self._con.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
        if row is None:
            raise KeyError(program_id)
        return self._program(row)

    def all_programs(self) -> list[Program]:
        return [self._program(r) for r in self._con.execute("SELECT * FROM programs ORDER BY id")]

    def last_generation(self) -> int:
        row = self._con.execute("SELECT MAX(generation) FROM programs").fetchone()
        return int(row[0]) if row[0] is not None else -1

    def meta(self) -> dict[str, Any]:
        row = self._con.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'ok') AS ok, SUM(cost_usd) AS cost FROM programs"
        ).fetchone()
        return {"programs": row["n"], "valid": row["ok"], "cost_usd": row["cost"] or 0.0}

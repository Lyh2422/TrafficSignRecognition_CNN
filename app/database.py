"""Small SQLite repository for recognition history."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


class HistoryRepository:
    def __init__(self) -> None:
        default_path = Path(__file__).resolve().parent.parent / "data" / "recognitions.db"
        self.path = Path(os.getenv("TSR_DB_PATH", str(default_path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    name_zh TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    inference_ms REAL NOT NULL,
                    mode TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    top3_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )

    def add(self, filename: str, result: dict) -> int:
        prediction = result["prediction"]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recognitions (
                    filename, class_id, name_zh, confidence, inference_ms,
                    mode, width, height, top3_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    prediction["id"],
                    prediction["name_zh"],
                    prediction["confidence"],
                    result["inference_ms"],
                    result["mode"],
                    result["image"]["width"],
                    result["image"]["height"],
                    json.dumps(result["top3"], ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recognitions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["top3"] = json.loads(item.pop("top3_json"))
            records.append(item)
        return records

    def clear(self) -> int:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM recognitions").fetchone()[0]
            connection.execute("DELETE FROM recognitions")
            return int(count)


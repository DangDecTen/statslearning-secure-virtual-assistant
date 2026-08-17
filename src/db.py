"""SQLite-backed storage for enrolled speaker centroids.

Only the mean, L2-normalized embedding ("centroid") is persisted per
Requirement 2 / README section 8 — raw enrollment audio is not stored
here, since embeddings aren't trivially invertible back to intelligible
speech and this keeps the enrollment data privacy-minded by default.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    centroid TEXT NOT NULL,          -- JSON-encoded list[float], L2-normalized
    enrolled_at TEXT NOT NULL,
    n_enrollment_clips INTEGER NOT NULL
);
"""


@contextmanager
def _connect():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_user(user_id: str, centroid: np.ndarray, n_enrollment_clips: int) -> None:
    """Insert or overwrite a user's stored centroid (re-enrollment replaces it)."""
    payload = json.dumps(centroid.astype(float).tolist())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, centroid, enrolled_at, n_enrollment_clips)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                centroid=excluded.centroid,
                enrolled_at=excluded.enrolled_at,
                n_enrollment_clips=excluded.n_enrollment_clips
            """,
            (user_id, payload, datetime.now(timezone.utc).isoformat(), n_enrollment_clips),
        )


def get_user_centroid(user_id: str) -> np.ndarray | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT centroid FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return np.array(json.loads(row[0]), dtype=np.float32)


def get_all_centroids() -> dict[str, np.ndarray]:
    """Returns {user_id: centroid} for every enrolled user (for SID)."""
    with _connect() as conn:
        rows = conn.execute("SELECT user_id, centroid FROM users").fetchall()
    return {uid: np.array(json.loads(c), dtype=np.float32) for uid, c in rows}


def user_exists(user_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row is not None


def delete_user(user_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

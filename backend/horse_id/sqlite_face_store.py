"""SQLite-based face feature store — drop-in fallback when milvus-lite is unavailable (e.g. Windows).

Schema:
    rider_faces(id INTEGER PK, embedding BLOB, name TEXT, photo_path TEXT)

Vectors are stored as raw float32 bytes (512 * 4 = 2048 bytes per row).
Search uses brute-force inner-product (IP) — fast enough for < 10 k entries.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np

_DIM = 512
_FLOAT32_SIZE = 4


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class SQLiteFaceStore:
    """Minimal vector store backed by a plain SQLite database."""

    def __init__(self, db_path: str, collection_name: str = "rider_faces", dim: int = _DIM) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.collection_name = collection_name
        self.dim = dim
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                name TEXT NOT NULL,
                photo_path TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.commit()

    # ---- write ----

    def insert(self, embeddings: list[list[float]], names: list[str], photo_paths: list[str]) -> int:
        rows = [
            (_vec_to_blob(np.array(emb, dtype=np.float32)), name, path)
            for emb, name, path in zip(embeddings, names, photo_paths)
        ]
        self._conn.executemany(
            f"INSERT INTO {self.collection_name} (embedding, name, photo_path) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def drop_collection(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
        self._conn.commit()
        self._ensure_table()

    def has_collection(self) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (self.collection_name,),
        )
        return cur.fetchone() is not None

    def count(self) -> int:
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
        return cur.fetchone()[0]

    # ---- read ----

    def search(self, query_vec: list[float], limit: int = 1) -> list[dict[str, Any]]:
        """Brute-force IP (inner product) search. Returns list of {name, distance}."""
        q = np.array(query_vec, dtype=np.float32)
        cur = self._conn.execute(f"SELECT id, embedding, name FROM {self.collection_name}")
        scored: list[tuple[float, int, str]] = []
        for row_id, blob, name in cur:
            vec = _blob_to_vec(blob)
            score = float(np.dot(q, vec))
            scored.append((score, row_id, name))
        scored.sort(key=lambda x: -x[0])
        results: list[dict[str, Any]] = []
        for score, row_id, name in scored[:limit]:
            results.append({"id": row_id, "name": name, "distance": score})
        return results

    def get_all_names(self) -> set[str]:
        cur = self._conn.execute(f"SELECT DISTINCT name FROM {self.collection_name}")
        return {row[0] for row in cur if row[0]}

    def query_all(self, output_fields: list[str] | None = None) -> list[dict[str, Any]]:
        cur = self._conn.execute(f"SELECT id, embedding, name, photo_path FROM {self.collection_name}")
        results: list[dict[str, Any]] = []
        for row_id, blob, name, photo_path in cur:
            entry: dict[str, Any] = {"id": row_id, "name": name}
            if output_fields is None or "photo_path" in output_fields:
                entry["photo_path"] = photo_path
            if output_fields is None or "embedding" in output_fields:
                entry["embedding"] = _blob_to_vec(blob).tolist()
            results.append(entry)
        return results

    def close(self) -> None:
        self._conn.close()

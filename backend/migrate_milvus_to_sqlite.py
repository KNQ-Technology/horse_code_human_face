"""
Dump a Milvus Lite rider face DB to a plain SQLite file.

Needed because milvus-lite does not support Windows — on Windows the runtime uses
the SQLite backend (see horse_id/rider_identity.py). Run this once on a platform
where milvus-lite works (Linux) and copy the resulting `.sqlite.db` to Windows.

Usage:
    python migrate_milvus_to_sqlite.py --src rider.db [--dst rider.sqlite.db] [--collection rider_faces]

Output schema (matches save_rider_faces_to_milvus.py's SQLite writer, if added):
    rider_faces(id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,        -- float32 L2-normalized, `face_dim` values
                name TEXT NOT NULL,
                photo_path TEXT NOT NULL DEFAULT '')
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np


def _create_schema(con: sqlite3.Connection, collection: str) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {collection} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            name TEXT NOT NULL,
            photo_path TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _to_blob(vec) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr.tobytes()


def migrate(src: Path, dst: Path, collection: str) -> int:
    if not src.is_file():
        raise FileNotFoundError(f"milvus lite db not found: {src}")
    try:
        from pymilvus import MilvusClient
    except ImportError as e:
        raise RuntimeError(
            "pymilvus not installed. Run this script inside the backend venv "
            "on a platform where milvus-lite is supported (Linux)."
        ) from e

    client = MilvusClient(str(src.resolve()))
    rows = client.query(
        collection_name=collection,
        filter="",
        output_fields=["embedding", "name", "photo_path"],
        limit=1_000_000,
    )

    if dst.exists():
        dst.unlink()
    con = sqlite3.connect(dst)
    try:
        _create_schema(con, collection)
        inserted = 0
        with con:
            for row in rows:
                emb = row.get("embedding")
                name = row.get("name", "")
                photo = row.get("photo_path", "") or ""
                if emb is None or not name:
                    continue
                con.execute(
                    f"INSERT INTO {collection} (embedding, name, photo_path) VALUES (?, ?, ?)",
                    (_to_blob(emb), str(name), str(photo)),
                )
                inserted += 1
        return inserted
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Milvus Lite rider DB path (e.g. rider.db)")
    parser.add_argument("--dst", default=None, help="Output SQLite path (default: <src>.sqlite.db)")
    parser.add_argument("--collection", default="rider_faces", help="Milvus collection name")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else src.with_suffix(".sqlite.db")

    n = migrate(src, dst, args.collection)
    print(f"[migrate] {n} rows written to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

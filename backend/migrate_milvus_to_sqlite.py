"""Migrate rider.db (Milvus Lite protobuf format) → rider.sqlite.db (plain SQLite).

This script reads the Milvus Lite internal protobuf blobs directly from the
SQLite container, extracts the 512-d float32 vector and JSON metadata (name,
photo_path), and writes them into a standard SQLite database that the
SQLiteFaceStore backend can load on any platform (including Windows).

Usage:
    python migrate_milvus_to_sqlite.py [--src rider.db] [--dst rider.sqlite.db]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

_DIM = 512
_VEC_BYTES = _DIM * 4  # 2048


def _extract_record(blob: bytes) -> dict | None:
    """Extract vector, name, and photo_path from a Milvus Lite protobuf blob."""
    # --- extract name / photo_path from embedded JSON ---
    start = blob.rfind(b"{", 0, blob.rfind(b"name") if b"name" in blob else len(blob))
    end = blob.find(b"}", start) if start >= 0 else -1
    name = ""
    photo_path = ""
    if start >= 0 and end >= 0:
        try:
            js = json.loads(blob[start : end + 1])
            name = js.get("name", "")
            photo_path = js.get("photo_path", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if not name:
        return None

    # --- extract 512-d float32 vector ---
    # The vector sits after a protobuf header. We search for the offset where
    # 2048 consecutive bytes decode to float32 with L2-norm ≈ 1.0.
    vidx = blob.find(b"vector")
    if vidx < 0:
        return None

    # Milvus Lite protobuf: 'vector'(6 bytes) + 12 bytes header → raw float32 data
    _PROTO_HEADER = 18
    vec_start = vidx + _PROTO_HEADER
    if vec_start + _VEC_BYTES > len(blob):
        return None
    best_vec = np.frombuffer(blob[vec_start : vec_start + _VEC_BYTES], dtype=np.float32).copy()
    best_norm_diff = abs(float(np.linalg.norm(best_vec)) - 1.0)
    if best_norm_diff > 0.05:
        # Fallback: scan for correct offset
        for off in range(vidx + 6, min(vidx + 60, len(blob) - _VEC_BYTES)):
            vec = np.frombuffer(blob[off : off + _VEC_BYTES], dtype=np.float32).copy()
            diff = abs(float(np.linalg.norm(vec)) - 1.0)
            if diff < best_norm_diff:
                best_norm_diff = diff
                best_vec = vec
            if diff < 0.001:
                break
        if best_norm_diff > 0.05:
            return None

    return {"name": name, "photo_path": photo_path, "vector": best_vec}


def migrate(src: str, dst: str, collection_name: str = "rider_faces") -> int:
    src_path = Path(src).resolve()
    if not src_path.is_file():
        print(f"[migrate] source not found: {src_path}")
        return 0

    # Open source (Milvus Lite format)
    src_conn = sqlite3.connect(str(src_path))
    src_conn.text_factory = bytes
    cur = src_conn.cursor()

    # Check source has the table
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (collection_name.encode(),))
    if not cur.fetchone():
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (collection_name,))
        if not cur.fetchone():
            print(f"[migrate] table '{collection_name}' not found in {src_path}")
            src_conn.close()
            return 0

    cur.execute(f"SELECT data FROM {collection_name}")
    rows = cur.fetchall()
    src_conn.close()
    print(f"[migrate] read {len(rows)} rows from {src_path}")

    # Open/create destination (plain SQLite)
    dst_path = Path(dst).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    from horse_id.sqlite_face_store import SQLiteFaceStore
    store = SQLiteFaceStore(str(dst_path), collection_name)
    # Clear existing data
    store.drop_collection()

    embs: list[list[float]] = []
    names: list[str] = []
    paths: list[str] = []
    skipped = 0

    for (blob,) in rows:
        rec = _extract_record(blob)
        if rec is None:
            skipped += 1
            continue
        embs.append(rec["vector"].tolist())
        names.append(rec["name"])
        paths.append(rec["photo_path"])

    if embs:
        store.insert(embs, names, paths)

    n = store.count()
    all_names = store.get_all_names()
    store.close()
    print(f"[migrate] wrote {n} entries to {dst_path} (skipped {skipped})")
    print(f"[migrate] riders: {sorted(all_names)}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Milvus Lite rider.db → plain SQLite")
    parser.add_argument("--src", default="rider.db", help="Source Milvus Lite .db file")
    parser.add_argument("--dst", default="", help="Destination SQLite file (default: <stem>.sqlite.db)")
    parser.add_argument("--collection", default="rider_faces", help="Collection/table name")
    args = parser.parse_args()

    dst = args.dst
    if not dst:
        base = Path(args.src)
        dst = str(base.parent / f"{base.stem}.sqlite.db")

    n = migrate(args.src, dst, args.collection)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

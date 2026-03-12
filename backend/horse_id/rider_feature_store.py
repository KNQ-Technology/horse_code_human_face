from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


class RiderFeatureStore:
    """Persistent rider feature store for open-set identity management."""

    _FACE_CORRECTION_MIN_SCORE = 0.50

    def __init__(self, db_path: str | Path, known_names: set[str] | None = None) -> None:
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._known_names: set[str] = known_names or set()
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS riders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rider_code TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                face_name TEXT DEFAULT '',
                face_score REAL DEFAULT 0.0,
                color_proto_json TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'unknown',
                horse_hint TEXT DEFAULT '',
                seen_count INTEGER NOT NULL DEFAULT 0,
                first_seen_ts REAL NOT NULL,
                last_seen_ts REAL NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_riders_display_name ON riders(display_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_riders_face_name ON riders(face_name)")
        self.conn.commit()

    @staticmethod
    def _to_json_color(vec: np.ndarray | None) -> str:
        if vec is None:
            return ""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return ""
        return json.dumps(arr.tolist(), ensure_ascii=False)

    @staticmethod
    def _from_json_color(raw: str) -> np.ndarray | None:
        if not raw:
            return None
        try:
            arr = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-9:
            return None
        return arr / norm

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-9) -> float:
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _gen_new_code(self) -> str:
        row = self.conn.execute("SELECT id FROM riders ORDER BY id DESC LIMIT 1").fetchone()
        next_id = int(row["id"]) + 1 if row is not None else 1
        return f"R{next_id:06d}"

    def _row_to_dict(self, row: sqlite3.Row, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "id": int(row["id"]),
            "rider_code": str(row["rider_code"]),
            "display_name": str(row["display_name"]),
            "face_name": str(row["face_name"] or ""),
            "status": str(row["status"]),
            "horse_hint": str(row["horse_hint"] or ""),
            "seen_count": int(row["seen_count"]),
            "face_score": float(row["face_score"] or 0.0),
        }
        if extra:
            payload.update(extra)
        return payload

    def _create_rider(
        self,
        display_name: str,
        face_name: str,
        face_score: float,
        color_feat: np.ndarray | None,
        status: str,
        horse_hint: str,
        match_type: str,
    ) -> dict[str, Any]:
        now = float(time.time())
        rider_code = self._gen_new_code()
        color_json = self._to_json_color(color_feat)
        self.conn.execute(
            """
            INSERT INTO riders (
                rider_code, display_name, face_name, face_score, color_proto_json,
                status, horse_hint, seen_count, first_seen_ts, last_seen_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rider_code,
                display_name,
                face_name,
                float(face_score),
                color_json,
                status,
                horse_hint,
                1,
                now,
                now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM riders WHERE rider_code = ?", (rider_code,)).fetchone()
        return self._row_to_dict(row, {"is_new": True, "match_type": match_type})

    def _update_row(
        self,
        row: sqlite3.Row,
        face_name: str,
        face_score: float,
        color_feat: np.ndarray | None,
        horse_hint: str,
        match_type: str,
    ) -> dict[str, Any]:
        now = float(time.time())
        old_color = self._from_json_color(str(row["color_proto_json"] or ""))
        merged_color = old_color
        if color_feat is not None:
            if old_color is None:
                merged_color = color_feat
            else:
                merged = 0.85 * old_color + 0.15 * color_feat
                norm = float(np.linalg.norm(merged))
                merged_color = merged / norm if norm > 1e-9 else old_color
        next_face_name = face_name if face_name else str(row["face_name"] or "")
        next_face_score = max(float(row["face_score"] or 0.0), float(face_score))
        next_horse_hint = horse_hint if horse_hint else str(row["horse_hint"] or "")

        cur_display = str(row["display_name"] or "")
        if (
            next_face_name
            and next_face_name != cur_display
            and next_face_score >= self._FACE_CORRECTION_MIN_SCORE
            and self._is_known(next_face_name)
        ):
            next_display = next_face_name
        else:
            next_display = cur_display

        self.conn.execute(
            """
            UPDATE riders
            SET display_name = ?, face_name = ?, face_score = ?,
                color_proto_json = ?, horse_hint = ?,
                seen_count = seen_count + 1, last_seen_ts = ?
            WHERE id = ?
            """,
            (
                next_display,
                next_face_name,
                next_face_score,
                self._to_json_color(merged_color),
                next_horse_hint,
                now,
                int(row["id"]),
            ),
        )
        self.conn.commit()
        updated = self.conn.execute("SELECT * FROM riders WHERE id = ?", (int(row["id"]),)).fetchone()
        return self._row_to_dict(updated, {"is_new": False, "match_type": match_type})

    def get_by_code(self, rider_code: str) -> dict[str, Any] | None:
        code = (rider_code or "").strip()
        if not code:
            return None
        row = self.conn.execute("SELECT * FROM riders WHERE rider_code = ?", (code,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, {"is_new": False, "match_type": "cache"})

    def _unmatched_result(self, face_name: str, face_score: float, horse_hint: str) -> dict[str, Any]:
        return {
            "rider_code": "",
            "display_name": "",
            "face_name": face_name,
            "status": "unmatched",
            "horse_hint": horse_hint,
            "seen_count": 0,
            "face_score": float(face_score),
            "is_new": False,
            "match_type": "none",
        }

    def _is_known(self, name: str) -> bool:
        if not self._known_names:
            return False
        return name in self._known_names

    def observe(
        self,
        face_name: str,
        face_score: float,
        color_feat: np.ndarray | None,
        horse_hint: str,
        horse_map_name: str,
        cached_rider_code: str = "",
        color_match_threshold: float = 0.72,
        allowed_display_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Match or create a rider record only for names registered in the face DB.
        Priority:
        1) cached rider code
        2) horse_map_name exact match (must be in known_names to create)
        3) high-confidence face_name exact match (must be in known_names to create)
        4) color prototype nearest neighbor
        5) single horse-rider relation (must be in known_names to create)
        6) return unmatched — never create UNKNOWN entries
        """
        face_name = (face_name or "").strip()
        horse_hint = (horse_hint or "").strip()
        horse_map_name = (horse_map_name or "").strip()
        cached_rider_code = (cached_rider_code or "").strip()
        allowed_names = {
            str(name).strip()
            for name in (allowed_display_names or [])
            if str(name).strip()
        }

        def _is_allowed(name: str) -> bool:
            if not allowed_names:
                return True
            return name in allowed_names

        def _query_rows_for_allowed() -> list[sqlite3.Row]:
            if not allowed_names:
                return self.conn.execute("SELECT * FROM riders").fetchall()
            placeholders = ",".join("?" for _ in allowed_names)
            sql = f"SELECT * FROM riders WHERE display_name IN ({placeholders})"
            return self.conn.execute(sql, tuple(sorted(allowed_names))).fetchall()

        if color_feat is not None:
            color_feat = np.asarray(color_feat, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(color_feat))
            if norm > 1e-9:
                color_feat = color_feat / norm
            else:
                color_feat = None

        if cached_rider_code:
            row = self.conn.execute("SELECT * FROM riders WHERE rider_code = ?", (cached_rider_code,)).fetchone()
            if row is not None:
                return self._update_row(row, face_name, face_score, color_feat, horse_hint, "cache")

        if horse_map_name and _is_allowed(horse_map_name):
            row = self.conn.execute(
                "SELECT * FROM riders WHERE display_name = ? LIMIT 1",
                (horse_map_name,),
            ).fetchone()
            if row is not None:
                return self._update_row(row, face_name, face_score, color_feat, horse_hint, "horse_map")
            if self._is_known(horse_map_name):
                return self._create_rider(
                    display_name=horse_map_name,
                    face_name=face_name,
                    face_score=face_score,
                    color_feat=color_feat,
                    status="known",
                    horse_hint=horse_hint,
                    match_type="horse_map_new",
                )

        if face_name and float(face_score) > 0.0:
            if allowed_names:
                placeholders = ",".join("?" for _ in allowed_names)
                sql = (
                    f"SELECT * FROM riders WHERE (face_name = ? OR display_name = ?) "
                    f"AND display_name IN ({placeholders}) LIMIT 1"
                )
                row = self.conn.execute(sql, (face_name, face_name, *tuple(sorted(allowed_names)))).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM riders WHERE face_name = ? OR display_name = ? LIMIT 1",
                    (face_name, face_name),
                ).fetchone()
            if row is not None:
                return self._update_row(row, face_name, face_score, color_feat, horse_hint, "face")
            if float(face_score) >= 0.62 and _is_allowed(face_name) and self._is_known(face_name):
                return self._create_rider(
                    display_name=face_name,
                    face_name=face_name,
                    face_score=face_score,
                    color_feat=color_feat,
                    status="known",
                    horse_hint=horse_hint,
                    match_type="face_new",
                )

        if color_feat is not None:
            rows = _query_rows_for_allowed()
            best_row = None
            best_sim = 0.0
            for row in rows:
                proto = self._from_json_color(str(row["color_proto_json"] or ""))
                if proto is None:
                    continue
                sim = self._cosine_similarity(color_feat, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row
            if best_row is not None and best_sim >= float(color_match_threshold):
                data = self._update_row(best_row, face_name, face_score, color_feat, horse_hint, "color")
                data["color_similarity"] = float(best_sim)
                return data

        if allowed_names and len(allowed_names) == 1:
            only_name = next(iter(allowed_names))
            row = self.conn.execute(
                "SELECT * FROM riders WHERE display_name = ? LIMIT 1",
                (only_name,),
            ).fetchone()
            if row is not None:
                return self._update_row(row, face_name, face_score, color_feat, horse_hint, "horse_relation_single")
            if self._is_known(only_name):
                return self._create_rider(
                    display_name=only_name,
                    face_name=face_name,
                    face_score=face_score,
                    color_feat=color_feat,
                    status="known",
                    horse_hint=horse_hint,
                    match_type="horse_relation_new",
                )

        return self._unmatched_result(face_name, face_score, horse_hint)


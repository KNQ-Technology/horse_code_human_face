from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import redis


# =========================
# 基础：向量处理（识别准确率友好）
# =========================

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return vec
    return vec / n


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    # 前提：u、v 已经 L2 normalize，则 cosine = dot
    return float(np.dot(u, v))


# =========================
# Redis 存储约定
# =========================
# 每个人：face:{user_id} (hash)
#   vec   -> float32 bytes
#   dim   -> int
#   model -> str
#   ts    -> int timestamp
#
# 额外维护一个集合：face:ids (set) 存 user_id，便于 1:N 扫描


@dataclass
class FaceRecord:
    user_id: str
    vec: np.ndarray          # float32, L2-normalized
    model: str
    ts: int


class FaceEmbeddingStore:
    def __init__(
        self,
        redis_client: redis.Redis,
        ids_key: str = "face:ids",
        key_prefix: str = "face:",
    ) -> None:
        self.r = redis_client
        self.ids_key = ids_key
        self.key_prefix = key_prefix

    def _user_key(self, user_id: str) -> str:
        return f"{self.key_prefix}{user_id}"

    def upsert(self, user_id: str, emb: np.ndarray, model: str = "arcface", ts: Optional[int] = None) -> None:
        """
        存储 embedding（强烈建议：先L2 normalize，再存）
        """
        if ts is None:
            ts = int(time.time())

        vec = l2_normalize(emb).astype(np.float32)
        key = self._user_key(user_id)

        pipe = self.r.pipeline(transaction=True)
        pipe.hset(key, mapping={
            "vec": vec.tobytes(),
            "dim": vec.size,
            "model": model,
            "ts": ts,
        })
        pipe.sadd(self.ids_key, user_id)
        pipe.execute()

    def get(self, user_id: str) -> Optional[FaceRecord]:
        key = self._user_key(user_id)
        data = self.r.hgetall(key)
        if not data or b"vec" not in data:
            return None

        dim = int(data[b"dim"])
        vec = np.frombuffer(data[b"vec"], dtype=np.float32, count=dim).copy()
        # 再保险：确保是单位向量
        vec = l2_normalize(vec)

        model = data.get(b"model", b"").decode("utf-8", errors="ignore")
        ts = int(data.get(b"ts", b"0"))

        return FaceRecord(user_id=user_id, vec=vec, model=model, ts=ts)

    def delete(self, user_id: str) -> None:
        key = self._user_key(user_id)
        pipe = self.r.pipeline(transaction=True)
        pipe.delete(key)
        pipe.srem(self.ids_key, user_id)
        pipe.execute()

    def count(self) -> int:
        return int(self.r.scard(self.ids_key))

    def list_ids(self, limit: Optional[int] = None) -> List[str]:
        ids = list(self.r.smembers(self.ids_key))
        ids = [x.decode("utf-8", errors="ignore") for x in ids]
        if limit is not None:
            ids = ids[:limit]
        return ids

    # =========================
    # 查询：1:1 验证
    # =========================

    def verify_1v1(self, user_id: str, query_emb: np.ndarray) -> Optional[float]:
        """
        返回 cosine 相似度（越大越像），取值通常在 [-1, 1]，单位向量人脸多在 [0, 1]
        """
        rec = self.get(user_id)
        if rec is None:
            return None
        q = l2_normalize(query_emb)
        return cosine_sim(q, rec.vec)

    # =========================
    # 查询：1:N 识别（应用层 TopK）
    # =========================

    def search_1vn_topk(
        self,
        query_emb: np.ndarray,
        topk: int = 5,
        min_score: float = 0.0,
        batch_size: int = 512,
    ) -> List[Tuple[str, float]]:
        """
        在 Redis 中存的所有向量里，找与 query 最相似的 topk（cosine）。
        - 适合：几千~几万规模（取决于QPS/机器）
        - 更大规模：建议上 Redis Stack/RediSearch 向量索引

        返回：[(user_id, score), ...] score 越大越相似
        """
        q = l2_normalize(query_emb)
        ids = self.list_ids()

        best: List[Tuple[str, float]] = []

        # 批量读 vec，减少 RTT
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            pipe = self.r.pipeline(transaction=False)
            for uid in batch:
                pipe.hget(self._user_key(uid), "vec")
                pipe.hget(self._user_key(uid), "dim")
            raw = pipe.execute()

            # raw: [vec1, dim1, vec2, dim2, ...]
            for j, uid in enumerate(batch):
                vec_bytes = raw[2*j]
                dim_bytes = raw[2*j + 1]
                if vec_bytes is None or dim_bytes is None:
                    continue
                dim = int(dim_bytes)
                v = np.frombuffer(vec_bytes, dtype=np.float32, count=dim)
                # 默认库里已是单位向量；这里不再归一化以节省时间（但你也可加一行 l2_normalize）
                score = float(np.dot(q, v))
                if score < min_score:
                    continue
                best.append((uid, score))

        best.sort(key=lambda x: x[1], reverse=True)
        return best[:topk]


# =========================
# Demo：如何使用（可直接跑）
# =========================

def main():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    store = FaceEmbeddingStore(r)

    # 假设你的人脸模型输出 512 维 embedding
    dim = 512

    # ---- 1) 写入一些示例向量（实际替换为你的模型输出）----
    np.random.seed(0)
    for uid in ["u001", "u002", "u003"]:
        emb = np.random.randn(dim).astype(np.float32)
        store.upsert(uid, emb, model="demo_model")

    print("库中数量:", store.count())

    # ---- 2) 1:1 验证：query 与 u002 的相似度 ----
    query = np.random.randn(dim).astype(np.float32)
    score_1v1 = store.verify_1v1("u002", query)
    print("1:1 verify score(u002):", score_1v1)

    # ---- 3) 1:N 识别：找 topk ----
    topk = store.search_1vn_topk(query, topk=3, min_score=-1.0)
    print("1:N topk:", topk)

    # ---- 4) 一个更真实的演示：用 u003 的向量+微扰作为 query，应当更像 u003 ----
    base = store.get("u003").vec
    query2 = l2_normalize(base + 0.05 * np.random.randn(dim).astype(np.float32))
    topk2 = store.search_1vn_topk(query2, topk=3, min_score=0.0)
    print("1:N topk (query close to u003):", topk2)

    # 你可以据此设阈值：
    # 常见 ArcFace/InsightFace：同人 cosine 往往 > 0.35~0.6（取决于模型/数据/对齐）
    # 建议你用真实数据做 ROC/PR 找最佳阈值


if __name__ == "__main__":
    main()
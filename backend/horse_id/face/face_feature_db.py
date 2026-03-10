"""
人脸特征持久化：从本地 Milvus 库查询/注册，识别时用向量检索（IP 内积，L2 归一化）并返回匹配编号。
支持 Milvus 服务器（http）与 Milvus Lite（本地 .db 路径）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return vec
    return (vec / n).astype(np.float32)


def _is_lite_uri(uri: str) -> bool:
    u = (uri or "").strip()
    if not u:
        return False
    if u.startswith("http://") or u.startswith("https://"):
        return False
    return True


# 与 save_rider_faces_to_milvus 一致
DEFAULT_COLLECTION_NAME = "rider_faces"
DEFAULT_DIM = 512


class FaceFeatureDB:
    """
    人脸特征库：从本地 Milvus（或 Milvus Lite）查询/注册特征。
    - uri: Milvus 地址，本地路径（如 ./rider.db）为 Lite，http 为服务器。
    - query/query_or_register 返回的编号为 Milvus 主键（int）或匹配到的 id。
    """

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.uri = str(Path(db_path) if isinstance(db_path, str) and db_path else db_path)
        self.collection_name = collection_name
        self.dim = dim
        self._client = None
        self._collection = None
        self._use_lite = _is_lite_uri(self.uri)

    def _get_client(self):
        if self._client is not None:
            return self._client
        from pymilvus import MilvusClient

        path = str(Path(self.uri).resolve())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._client = MilvusClient(path)
        return self._client

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        from pymilvus import Collection, connections

        connections.connect(uri=self.uri)
        self._collection = Collection(name=self.collection_name)
        self._collection.load()
        return self._collection

    def load(self) -> None:
        """兼容旧接口：Milvus 无需预加载。"""
        pass

    def save(self) -> None:
        """兼容旧接口：Milvus 写入即持久化。"""
        pass

    def query(self, feature: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[int, float]]:
        """
        用向量检索在 Milvus 中比对（IP 内积，特征会 L2 归一化），
        若最大相似度 >= threshold 则返回 (主键 id, 置信度)，否则返回 None。
        """
        feature = _l2_normalize(np.asarray(feature).flatten().astype(np.float32))
        if feature.size != self.dim:
            return None

        try:
            if self._use_lite:
                client = self._get_client()
                if not client.has_collection(self.collection_name):
                    return None
                res = client.search(
                    collection_name=self.collection_name,
                    data=[feature.tolist()],
                    limit=1,
                    output_fields=["name"],
                )
                if not res or len(res) == 0 or len(res[0]) == 0:
                    return None
                hit = res[0][0]
                score = hit.get("distance") or hit.get("score")
                if score is None or score < threshold:
                    return None
                pk = hit.get("id")
                if pk is None:
                    return None
                return (int(pk), float(score))
            else:
                coll = self._get_collection()
                search_params = {"metric_type": "IP", "params": {"nprobe": 128}}
                results = coll.search(
                    data=[feature.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=1,
                    output_fields=["name"],
                )
                for hits in results:
                    if hits and len(hits) > 0:
                        hit = hits[0]
                        score = hit.distance
                        if score >= threshold:
                            return (int(hit.id), float(score))
                    break
        except Exception:
            pass
        return None

    def register(self, feature: np.ndarray) -> int:
        """
        将新特征插入 Milvus，返回分配的主键 id（auto_id）。
        """
        feature = _l2_normalize(np.asarray(feature).flatten().astype(np.float32))
        if feature.size != self.dim:
            raise ValueError(f"特征维度 {feature.size} 与集合 dim={self.dim} 不一致")

        if self._use_lite:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dim,
                    metric_type="IP",
                    auto_id=True,
                )
            data = [{"vector": feature.tolist()}]
            pk_list = client.insert(collection_name=self.collection_name, data=data)
            if not pk_list:
                raise RuntimeError("Milvus Lite insert 未返回主键")
            return int(pk_list[0])
        else:
            from pymilvus import Collection, connections

            connections.connect(uri=self.uri)
            coll = Collection(name=self.collection_name)
            insert_res = coll.insert([[feature.tolist()], [""], [""]])  # embedding, name, photo_path
            coll.flush()
            if hasattr(insert_res, "primary_keys") and insert_res.primary_keys:
                return int(insert_res.primary_keys[0])
            if hasattr(insert_res, "ids") and insert_res.ids:
                return int(insert_res.ids[0])
            return 0

    def query_or_register(self, feature: np.ndarray, threshold: float = 0.5) -> Tuple[int, bool, float]:
        """
        先查 Milvus；若匹配则返回 (id, True, 置信度)，否则插入并返回 (新 id, False, 0.0)。
        """
        result = self.query(feature, threshold)
        if result is not None:
            pid, confidence = result
            return pid, True, confidence
        new_id = self.register(feature)
        return new_id, False, 0.0

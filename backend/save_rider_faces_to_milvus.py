"""
从 rider 文件夹读取人脸图片（文件名=人名），提取人脸特征并写入 Milvus。

存储字段：id（主键）、特征值（L2 归一化向量）、姓名、照片路径。
依赖：
- horse_id.face (SCRFD / ArcFaceONNX / FaceDetector)
- pymilvus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import pkg_resources  # noqa: F401
except ImportError:
    try:
        import setuptools  # noqa: F401
    except ImportError:
        pass

import cv2
import numpy as np

from horse_id.face.arcface_onnx import ArcFaceONNX
from horse_id.face.face_detector import FaceDetector
from horse_id.face.scrfd import SCRFD

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility
from pymilvus.exceptions import MilvusException


def _is_lite_uri(uri: str) -> bool:
    """URI 为本地路径时使用 Milvus Lite（嵌入式，无需 Docker）。"""
    u = (uri or "").strip()
    if not u:
        return False
    if u.startswith("http://") or u.startswith("https://"):
        return False
    return True



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RIDER_DIR = PROJECT_ROOT / "rider"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "src" / "horse_id" / "models"
SCRFD_MODEL_NAME = "det_10g.onnx"
ARCFACE_MODEL_NAME = "w600k_r50.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Milvus 人脸集合默认配置
DEFAULT_COLLECTION_NAME = "rider_faces"
DEFAULT_DIM = 512  # ArcFace w600k_r50 输出维度
DEFAULT_VARCHAR_MAX_LENGTH = 512
DEFAULT_PHOTO_PATH_MAX_LENGTH = 1024


def _get_onnx_providers(device: str) -> list[str]:
    if device and str(device).strip().lower().startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def init_face_detector(device: str = "cpu", models_dir: Path | None = None) -> FaceDetector:
    models_dir = models_dir or DEFAULT_MODELS_DIR
    models_dir = models_dir.resolve()
    det_path = models_dir / SCRFD_MODEL_NAME
    rec_path = models_dir / ARCFACE_MODEL_NAME
    if not det_path.is_file():
        raise FileNotFoundError(
            f"人脸检测模型未找到: {det_path}\n"
            f"请将 {SCRFD_MODEL_NAME} 放到该目录，或通过 --models-dir 指定包含该文件的目录。"
        )
    if not rec_path.is_file():
        raise FileNotFoundError(
            f"人脸识别模型未找到: {rec_path}\n"
            f"请将 {ARCFACE_MODEL_NAME} 放到该目录，或通过 --models-dir 指定包含该文件的目录。"
        )
    providers = _get_onnx_providers(device)
    scrfd = SCRFD(model_file=str(det_path), providers=providers)
    arcface = ArcFaceONNX(model_file=str(rec_path), providers=providers)
    return FaceDetector(scrfd, arcface, probability=0.5)


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return vec
    return (vec / n).astype(np.float32)


def iter_image_files(root: Path, recursive: bool = True) -> Iterable[tuple[Path, str]]:
    """
    遍历目录下的图片，返回 (图片路径, 人名)。
    - recursive=False：仅 root 下直接子文件，人名 = 文件名（不含扩展名）。
    - recursive=True：递归所有子目录；若图片在 root 的直系子目录内（如 rider/张三/1.jpg），
      人名 = 子目录名（张三）；否则人名 = 文件名 stem（如 rider/张三.jpg -> 张三）。
    """
    root = root.resolve()
    if not root.is_dir():
        return

    if not recursive:
        for p in sorted(root.iterdir(), key=lambda x: x.name):
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p, p.stem
        return

    # 递归：先收集所有图片路径并排序
    all_paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        all_paths.extend(root.rglob(f"*{ext}"))
    all_paths.sort(key=lambda x: (x.relative_to(root).parts, x.name))

    for p in all_paths:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        # 若在 root 的直系子目录下（仅一层），用人名=子目录名；否则用人名=文件名 stem
        if len(rel.parts) == 2:
            # 例如 rider/张三/photo.jpg -> 人名 张三
            name = rel.parts[0]
        else:
            name = p.stem
        yield p, name


def extract_face_feature(detector: FaceDetector, img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.array([], dtype=np.float32)
    feat = detector.detect_features(img)
    if feat is None:
        return np.array([], dtype=np.float32)
    feat = np.asarray(feat)
    if feat.size == 0:
        return np.array([], dtype=np.float32)
    return feat.astype(np.float32)


def assess_face_quality(
    detector: FaceDetector,
    img: np.ndarray,
    det_thresh: float = 0.3,
) -> list[dict[str, Any]]:
    """Detect all faces in an image and return quality metrics for each."""
    bboxes, kpss = detector.detect_with_keypoints(img)
    if len(bboxes) == 0:
        bboxes_lo, kpss_lo = detector._face_detect_model.detect(
            img, input_size=(640, 640), thresh=det_thresh,
        )
        if bboxes_lo is None or len(bboxes_lo) == 0:
            return []
        bboxes, kpss = bboxes_lo, kpss_lo

    results: list[dict[str, Any]] = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = (
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),
        )
        face_w, face_h = x2 - x1, y2 - y1
        x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
        x2_i = min(img.shape[1], int(x2))
        y2_i = min(img.shape[0], int(y2))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        face_crop = img[y1_i:y2_i, x1_i:x2_i]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        entry: dict[str, Any] = {
            "index": i,
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "det_score": float(score),
            "face_size": [int(face_w), int(face_h)],
            "blur_score": blur,
        }
        if kpss is not None:
            entry["kps"] = kpss[i]
        results.append(entry)
    results.sort(key=lambda r: r["face_size"][0] * r["face_size"][1], reverse=True)
    return results


def _select_best_face(
    faces: list[dict[str, Any]],
    min_face_size: int = 40,
    min_blur_score: float = 50.0,
) -> dict[str, Any] | None:
    """Pick the best face from quality assessment results."""
    for f in faces:
        w, h = f["face_size"]
        if w < min_face_size or h < min_face_size:
            continue
        if f["blur_score"] < min_blur_score:
            continue
        if "kps" not in f:
            continue
        return f
    return None


def _connect_milvus(uri: str) -> None:
    """连接 Milvus，失败时抛出 MilvusException 并给出排查提示。"""
    try:
        connections.connect(uri=uri)
    except MilvusException as e:
        raise MilvusException(
            f"{e}\n"
            "排查建议：\n"
            "  1) 确认 Milvus 已启动（如 docker run -d ... milvusdb/milvus）。\n"
            "  2) 若 Milvus 在其它机器，使用 --uri 指定，例如: --uri http://192.168.0.17:19530"
        ) from e


def _ensure_collection(
    collection_name: str,
    dim: int,
    uri: str,
    name_max_length: int,
    photo_path_max_length: int,
    overwrite: bool,
) -> None:
    _connect_milvus(uri)

    if overwrite and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"[milvus] 已删除已有集合: {collection_name}")

    if utility.has_collection(collection_name):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="自增主键"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="人脸特征向量(L2归一化)"),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=name_max_length, description="姓名"),
        FieldSchema(name="photo_path", dtype=DataType.VARCHAR, max_length=photo_path_max_length, description="照片路径"),
    ]
    schema = CollectionSchema(fields=fields, description="骑手/人员人脸特征库")
    collection = Collection(name=collection_name, schema=schema)
    # 小数据量用 FLAT，避免 IVF 建簇带来的内存与 CPU 峰值，减少 Docker 下 OOM
    index_params = {
        "metric_type": "IP",
        "index_type": "FLAT",
        "params": {},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"[milvus] 集合已创建: {collection_name}, dim={dim}, index=FLAT")


def _ensure_collection_lite(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    overwrite: bool,
) -> None:
    """Milvus Lite：建表（quick setup，name/photo_path 存动态字段）。"""
    if overwrite and client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[milvus-lite] 已删除已有集合: {collection_name}")
    if client.has_collection(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="IP",
        auto_id=True,
    )
    print(f"[milvus-lite] 集合已创建: {collection_name}, dim={dim} (Lite 嵌入式)")


def save_rider_faces_to_milvus(
    rider_dir: Path,
    uri: str = "http://localhost:19530",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    dim: int = DEFAULT_DIM,
    device: str = "cpu",
    models_dir: Path | None = None,
    name_max_length: int = DEFAULT_VARCHAR_MAX_LENGTH,
    photo_path_max_length: int = DEFAULT_PHOTO_PATH_MAX_LENGTH,
    photo_path_style: str = "absolute",
    overwrite_collection: bool = False,
    recursive: bool = True,
    batch_size: int = 5,
    flush_timeout: float | None = 120.0,
    sleep_after_flush: float = 2.0,
    flush_at_end_only: bool = False,
    min_face_size: int = 40,
    min_blur_score: float = 50.0,
    dry_run: bool = False,
    validate: bool = True,
    report_path: str = "",
) -> dict[str, Any]:
    """
    遍历 rider_dir 下的人脸图片，提取特征并写入 Milvus。

    Returns enrollment report dict (also optionally saved to report_path).
    """
    rider_dir = rider_dir.resolve()
    report: dict[str, Any] = {
        "total_photos": 0, "enrolled_ok": 0, "enrolled_skip": 0,
        "dry_run": dry_run, "riders": {},
    }

    if not rider_dir.is_dir():
        print(f"[rider2milvus] 目录不存在: {rider_dir}")
        return report

    print(f"[rider2milvus] rider 目录: {rider_dir}")
    print(f"[rider2milvus] Milvus uri: {uri}, 集合: {collection_name}")
    if dry_run:
        print("[rider2milvus] *** DRY-RUN 模式 — 仅分析质量，不实际入库 ***")

    use_lite = _is_lite_uri(uri)
    use_sqlite_fallback = False
    client: MilvusClient | None = None
    collection: Collection | None = None
    sqlite_store = None

    if not dry_run:
        if use_lite:
            # Try Milvus Lite first; fall back to SQLite on Windows
            _milvus_lite_ok = False
            try:
                import milvus_lite  # noqa: F401
                _milvus_lite_ok = True
            except ImportError:
                pass

            if _milvus_lite_ok:
                try:
                    import pkg_resources  # noqa: F401
                except ImportError:
                    try:
                        import setuptools  # noqa: F401
                    except ImportError:
                        pass
                lite_path = Path(uri).resolve()
                lite_path.parent.mkdir(parents=True, exist_ok=True)
                client = MilvusClient(str(lite_path))
                _ensure_collection_lite(client, collection_name, dim, overwrite_collection)
            else:
                # SQLite fallback — write to <stem>.sqlite.db alongside the original URI
                from horse_id.sqlite_face_store import SQLiteFaceStore
                base = Path(uri).expanduser().resolve()
                sqlite_path = base.parent / f"{base.stem}.sqlite.db"
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                sqlite_store = SQLiteFaceStore(str(sqlite_path), collection_name, dim)
                if overwrite_collection:
                    sqlite_store.drop_collection()
                use_sqlite_fallback = True
                print(f"[rider2milvus] milvus-lite unavailable, using SQLite fallback: {sqlite_path}")
        else:
            _ensure_collection(
                collection_name=collection_name, dim=dim, uri=uri,
                name_max_length=name_max_length,
                photo_path_max_length=photo_path_max_length,
                overwrite=overwrite_collection,
            )
            _connect_milvus(uri)
            collection = Collection(name=collection_name)

    face_detector = init_face_detector(device=device, models_dir=models_dir)

    batch_size = max(1, int(batch_size))
    batch_emb: list[list[float]] = []
    batch_names: list[str] = []
    batch_paths: list[str] = []
    count_total = 0
    count_ok = 0
    total_inserted = 0
    batch_num = 0

    all_embeddings: list[tuple[str, np.ndarray]] = []

    def _insert_batch(do_flush: bool = True) -> None:
        nonlocal total_inserted, batch_num
        if not batch_emb:
            return
        batch_num += 1
        print(f"[rider2milvus] 第 {batch_num} 批: insert ({len(batch_emb)} 条)...", flush=True)
        try:
            if use_sqlite_fallback and sqlite_store is not None:
                sqlite_store.insert(batch_emb, batch_names, batch_paths)
                total_inserted += len(batch_emb)
                print(f"[rider2milvus] 已落盘(SQLite): 累计 {total_inserted} 条")
            elif use_lite and client is not None:
                data = [
                    {"vector": batch_emb[i], "name": batch_names[i], "photo_path": batch_paths[i]}
                    for i in range(len(batch_emb))
                ]
                client.insert(collection_name=collection_name, data=data)
                total_inserted += len(batch_emb)
                print(f"[rider2milvus] 已落盘(Lite): 累计 {total_inserted} 条")
            elif collection is not None:
                collection.insert([batch_emb, batch_names, batch_paths])
                total_inserted += len(batch_emb)
                if do_flush:
                    collection.flush()
                    print(f"[rider2milvus] 已落盘: 累计 {total_inserted} 条")
        except MilvusException as e:
            print(f"[rider2milvus] 插入失败 (已落盘约 {total_inserted} 条): {e}")
            raise

    for img_path, name in iter_image_files(rider_dir, recursive=recursive):
        count_total += 1
        if photo_path_style == "relative":
            photo_path = str(img_path.relative_to(rider_dir))
        else:
            photo_path = str(img_path.resolve())
        if len(photo_path) >= photo_path_max_length:
            photo_path = photo_path[: photo_path_max_length - 1]
        if len(name) >= name_max_length:
            name = name[: name_max_length - 1]

        rider_report = report["riders"].setdefault(name, {
            "photos": 0, "enrolled": 0,
            "avg_det_score": 0.0, "avg_blur_score": 0.0,
            "details": [],
        })
        rider_report["photos"] += 1

        print(f"[rider2milvus] ({count_total}) {img_path.name} (name={name})")

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            detail = {"file": img_path.name, "status": "read_failed"}
            rider_report["details"].append(detail)
            print(f"  -> 读取失败，跳过")
            continue

        faces = assess_face_quality(face_detector, img)
        detail: dict[str, Any] = {"file": img_path.name}

        if not faces:
            detail["status"] = "no_face_detected"
            rider_report["details"].append(detail)
            print(f"  -> 未检测到人脸，跳过")
            continue

        detail["num_faces"] = len(faces)
        if len(faces) > 1:
            print(f"  -> 检测到 {len(faces)} 张人脸, 选取最佳")

        best = _select_best_face(faces, min_face_size=min_face_size, min_blur_score=min_blur_score)
        if best is None:
            detail["status"] = "quality_too_low"
            top = faces[0]
            detail["best_face_size"] = top["face_size"]
            detail["best_det_score"] = top["det_score"]
            detail["best_blur_score"] = top["blur_score"]
            rider_report["details"].append(detail)
            print(f"  -> 质量不达标 (size={top['face_size']}, blur={top['blur_score']:.1f}), 跳过")
            continue

        detail["det_score"] = best["det_score"]
        detail["face_size"] = best["face_size"]
        detail["blur_score"] = round(best["blur_score"], 1)

        kps = best["kps"]
        feat = face_detector.get_embedding_from_kps(img, kps)
        if feat.size == 0:
            detail["status"] = "feature_extract_failed"
            rider_report["details"].append(detail)
            print(f"  -> 特征提取失败，跳过")
            continue

        emb = l2_normalize(feat)
        if emb.size != dim:
            detail["status"] = "dim_mismatch"
            rider_report["details"].append(detail)
            print(f"  -> 维度不匹配 ({emb.size} != {dim})，跳过")
            continue

        detail["status"] = "ok"
        rider_report["details"].append(detail)
        rider_report["enrolled"] += 1
        all_embeddings.append((name, emb.copy()))
        count_ok += 1

        print(
            f"  -> OK  det={best['det_score']:.2f}  size={best['face_size']}  "
            f"blur={best['blur_score']:.0f}"
        )

        if not dry_run:
            batch_emb.append(emb.tolist())
            batch_names.append(name)
            batch_paths.append(photo_path)
            if len(batch_emb) >= batch_size:
                _insert_batch(do_flush=not flush_at_end_only)
                batch_emb, batch_names, batch_paths = [], [], []
                if not flush_at_end_only and sleep_after_flush > 0:
                    time.sleep(sleep_after_flush)

    if not dry_run:
        _insert_batch(do_flush=not flush_at_end_only)
        if not use_lite and flush_at_end_only and total_inserted > 0 and collection is not None:
            print(f"[rider2milvus] 最终 flush ({total_inserted} 条)...", flush=True)
            collection.flush()

    report["total_photos"] = count_total
    report["enrolled_ok"] = count_ok
    report["enrolled_skip"] = count_total - count_ok

    for rname, rdata in report["riders"].items():
        details = rdata["details"]
        ok_details = [d for d in details if d.get("status") == "ok"]
        if ok_details:
            rdata["avg_det_score"] = round(
                sum(d["det_score"] for d in ok_details) / len(ok_details), 3,
            )
            rdata["avg_blur_score"] = round(
                sum(d["blur_score"] for d in ok_details) / len(ok_details), 1,
            )

    if validate and all_embeddings:
        _run_validation(report, all_embeddings, use_lite, client, collection, collection_name, dry_run, sqlite_store=sqlite_store)

    if not dry_run:
        print(f"[rider2milvus] 完成: 总数={count_total}, 入库={count_ok}, 跳过={count_total - count_ok}")
    else:
        print(f"[rider2milvus] DRY-RUN 完成: 总数={count_total}, 可入库={count_ok}, 不合格={count_total - count_ok}")

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[rider2milvus] 报告已保存: {rp}")

    return report


def _run_validation(
    report: dict[str, Any],
    all_embeddings: list[tuple[str, np.ndarray]],
    use_lite: bool,
    client: MilvusClient | None,
    collection: Collection | None,
    collection_name: str,
    dry_run: bool,
    sqlite_store: Any = None,
) -> None:
    """Post-enrollment validation: intra-class consistency + self-match test."""
    print("\n[validation] 开始入库自验证...")

    # --- Intra-class consistency ---
    by_name: dict[str, list[np.ndarray]] = {}
    for name, emb in all_embeddings:
        by_name.setdefault(name, []).append(emb)

    for rname, embs in by_name.items():
        rider_report = report["riders"].get(rname, {})
        if len(embs) < 2:
            rider_report["intra_similarity"] = 1.0
            continue
        sims: list[float] = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = float(np.dot(embs[i], embs[j]))
                sims.append(sim)
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        rider_report["intra_similarity"] = round(avg_sim, 3)
        if avg_sim < 0.5:
            print(f"  [WARNING] {rname}: 同人平均相似度仅 {avg_sim:.3f} — 可能有错误照片")
        else:
            print(f"  {rname}: 同人相似度 {avg_sim:.3f} ({len(embs)} 张)")

        if len(embs) >= 3:
            centroid = np.mean(embs, axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-9:
                centroid = centroid / norm
            for idx, emb in enumerate(embs):
                sim_to_centroid = float(np.dot(emb, centroid))
                if sim_to_centroid < avg_sim - 0.3:
                    detail_list = rider_report.get("details", [])
                    ok_details = [d for d in detail_list if d.get("status") == "ok"]
                    if idx < len(ok_details):
                        ok_details[idx]["outlier"] = True
                        print(f"    [OUTLIER] {rname} 第{idx+1}张 sim={sim_to_centroid:.3f}")

    # --- Self-match via search (only when not dry-run) ---
    if dry_run or (client is None and collection is None and sqlite_store is None):
        for rname in by_name:
            report["riders"].get(rname, {})["self_match_all_correct"] = None
        print("[validation] dry-run 模式跳过自匹配测试")
        return

    print("[validation] 自匹配测试 (每个 embedding 搜索 top-3)...")
    total_queries = 0
    total_correct = 0
    for rname, embs in by_name.items():
        rider_correct = 0
        for emb in embs:
            total_queries += 1
            top_name = None
            if sqlite_store is not None:
                hits = sqlite_store.search(emb.tolist(), limit=1)
                if hits:
                    top_name = hits[0].get("name", "")
            elif use_lite and client is not None:
                res = client.search(
                    collection_name=collection_name, data=[emb.tolist()],
                    limit=3, output_fields=["name"],
                )
                if res and len(res[0]) > 0:
                    hit = res[0][0]
                    top_name = hit.get("name", "")
            elif collection is not None:
                res = collection.search(
                    data=[emb.tolist()], anns_field="embedding",
                    param={"metric_type": "IP", "params": {"nprobe": 128}},
                    limit=3, output_fields=["name"],
                )
                for hits in res:
                    if hits and len(hits) > 0:
                        top_name = hits[0].get("name")
                        if top_name is None and hasattr(hits[0], "entity"):
                            top_name = getattr(hits[0].entity, "name", None)
            if isinstance(top_name, bytes):
                top_name = top_name.decode("utf-8", errors="replace")
            if str(top_name).strip() == rname:
                rider_correct += 1
                total_correct += 1

        all_ok = rider_correct == len(embs)
        report["riders"].get(rname, {})["self_match_all_correct"] = all_ok
        if not all_ok:
            print(f"  [WARNING] {rname}: 自匹配 {rider_correct}/{len(embs)}")

    print(f"[validation] 自匹配总体: {total_correct}/{total_queries}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 rider 文件夹中的人脸图片特征写入 Milvus（文件名=人名）")
    parser.add_argument(
        "--rider-dir", type=str, default="",
        help="rider 图片目录，默认使用项目根目录下的 rider/",
    )
    parser.add_argument(
        "--uri", type=str, default="http://localhost:19530",
        help="Milvus 地址：http 为服务器；本地路径(如 ./rider.db) 为 Milvus Lite 嵌入式，无需 Docker",
    )
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="集合名称")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="特征向量维度（需与模型一致）")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备: cpu 或 cuda / cuda:0")
    parser.add_argument(
        "--models-dir", type=str, default="",
        help="人脸模型目录（需包含 det_10g.onnx 与 w600k_r50.onnx），默认: 项目 src/horse_id/models",
    )
    parser.add_argument(
        "--photo-path-style", choices=["absolute", "relative"], default="absolute",
        help="照片路径存储方式: absolute 绝对路径, relative 相对 rider 目录",
    )
    parser.add_argument(
        "--overwrite-collection", action="store_true",
        help="若集合已存在则先删除再创建（慎用）",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="仅扫描 rider 目录下直接文件，不递归子目录",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="每批插入条数（默认 5）")
    parser.add_argument("--flush-timeout", type=float, default=120.0, help="flush 超时秒数")
    parser.add_argument("--sleep-after-flush", type=float, default=2.0, help="每批 flush 后休眠秒数")
    parser.add_argument("--flush-at-end-only", action="store_true", help="仅在全部插入完成后 flush 一次")
    parser.add_argument("--min-face-size", type=int, default=40, help="最小人脸像素（宽或高，默认 40）")
    parser.add_argument("--min-blur-score", type=float, default=50.0, help="最低清晰度分数（默认 50）")
    parser.add_argument("--validate", action="store_true", default=True, help="入库后运行自验证（默认开启）")
    parser.add_argument("--no-validate", action="store_true", help="跳过入库后自验证")
    parser.add_argument(
        "--report-path", type=str, default="outputs/enrollment_report.json",
        help="入库报告输出路径（默认 outputs/enrollment_report.json）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅分析照片质量，不实际入库")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rider_dir = Path(args.rider_dir) if args.rider_dir else DEFAULT_RIDER_DIR
    if not rider_dir.is_absolute():
        rider_dir = (PROJECT_ROOT / rider_dir).resolve()

    models_dir: Path | None = None
    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not models_dir.is_absolute():
            models_dir = (PROJECT_ROOT / models_dir).resolve()

    do_validate = args.validate and not args.no_validate

    save_rider_faces_to_milvus(
        rider_dir=rider_dir,
        uri=args.uri,
        collection_name=args.collection,
        dim=args.dim,
        device=args.device,
        models_dir=models_dir,
        photo_path_style=args.photo_path_style,
        overwrite_collection=args.overwrite_collection,
        recursive=not args.no_recursive,
        batch_size=args.batch_size,
        flush_timeout=args.flush_timeout,
        sleep_after_flush=args.sleep_after_flush,
        flush_at_end_only=args.flush_at_end_only,
        min_face_size=args.min_face_size,
        min_blur_score=args.min_blur_score,
        dry_run=args.dry_run,
        validate=do_validate,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()

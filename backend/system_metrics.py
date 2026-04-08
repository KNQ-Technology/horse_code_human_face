"""CPU / 内存 / GPU utilization for GET /api/system/metrics."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any


def _cpu_percent() -> float:
    import psutil

    return float(psutil.cpu_percent(interval=0.1))


def _memory_info() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    total = float(vm.total)
    used = float(vm.used)
    return {
        "memory_percent": round(float(vm.percent), 1),
        "memory_used_gb": round(used / (1024**3), 2),
        "memory_total_gb": round(total / (1024**3), 2),
    }


def _gpu_vram_aggregate(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Average VRAM % per card; summed used/total MiB for display."""
    if not gpus:
        return {
            "gpu_vram_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
        }
    ratios: list[float] = []
    used_sum = 0.0
    total_sum = 0.0
    for g in gpus:
        try:
            u = float(g.get("memory_used_mib") or 0)
            t = float(g.get("memory_total_mib") or 0)
        except (TypeError, ValueError):
            continue
        used_sum += u
        total_sum += t
        if t > 0:
            ratios.append(100.0 * u / t)
    return {
        "gpu_vram_percent": None if not ratios else round(sum(ratios) / len(ratios), 1),
        "gpu_memory_used_mib": None if used_sum <= 0 else round(used_sum, 1),
        "gpu_memory_total_mib": None if total_sum <= 0 else round(total_sum, 1),
    }


def _query_gpus_nvidia_smi() -> list[dict[str, Any]]:
    """Return one entry per physical GPU: index, utilization %, memory (MiB)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        args = [
            exe,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        kwargs: dict[str, Any] = {
            "args": args,
            "timeout": 3,
            "stderr": subprocess.DEVNULL,
            "text": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(**kwargs)
        rows: list[dict[str, Any]] = []
        for raw in out.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
            except ValueError:
                continue
            rows.append({
                "index": idx,
                "utilization_gpu": round(util, 1),
                "memory_used_mib": round(mem_used, 1),
                "memory_total_mib": round(mem_total, 1),
            })
        rows.sort(key=lambda r: r["index"])
        return rows
    except Exception:
        return []


def get_system_metrics() -> dict[str, Any]:
    cpu = _cpu_percent()
    mem = _memory_info()
    gpus = _query_gpus_nvidia_smi()
    gpu_avg: float | None = None
    if gpus:
        gpu_avg = sum(float(g["utilization_gpu"]) for g in gpus) / len(gpus)
    vram = _gpu_vram_aggregate(gpus)
    return {
        "cpu_percent": round(cpu, 1),
        **mem,
        "gpu_percent": None if gpu_avg is None else round(gpu_avg, 1),
        "gpu_available": len(gpus) > 0,
        "gpus": gpus,
        **vram,
    }

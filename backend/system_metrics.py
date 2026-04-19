"""CPU / 内存 / GPU utilization for GET /api/system/metrics.

GPU utilization on Windows has two distinct sources reported side-by-side:

- NVML (`nvidia-smi --query-gpu=utilization.gpu`): fraction of sampling window
  where >=1 CUDA kernel ran. Accurate signal that the GPU is busy doing compute
  work, but a tiny kernel firing each millisecond reads 100%.
- WDDM engine counters (`\\GPU Engine(*)\\Utilization Percentage`): what the
  Windows Task Manager shows — per-engine active-time sampled by the kernel
  graphics scheduler. Sums over all processes per engine type (3D, Compute,
  Copy, VideoDecode, VideoEncode, ...); the default "GPU" number in Task
  Manager is the max across those engine types.

The two can differ by 10–50% on the same workload. `gpu_percent` mirrors the
Task Manager number on Windows (WDDM max); `gpu_percent_nvml` exposes the NVML
value for reference.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
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


# --- Windows WDDM engine counters (Task-Manager-equivalent) ---------------

_WDDM_STATE: dict[str, Any] = {"lock": threading.Lock(), "ts": 0.0, "value": None}
_WDDM_STARTED = False
_WDDM_ENGTYPE_RE = re.compile(r"engtype_([a-zA-Z0-9]*)\)", re.IGNORECASE)


def _parse_wddm_csv_line(header_cols: list[str], value_cols: list[str]) -> dict[str, float] | None:
    """Turn one typeperf CSV sample into {engine_type: summed_percent}.

    The column path looks like:
      \\HOST\\GPU Engine(pid_12345_luid_..._eng_5_engtype_Compute)\\Utilization Percentage
    We strip the engtype name and sum all instances with that type.
    """
    if len(header_cols) != len(value_cols) or len(header_cols) < 2:
        return None
    totals: dict[str, float] = {}
    for h, v in zip(header_cols[1:], value_cols[1:]):  # col 0 is timestamp
        m = _WDDM_ENGTYPE_RE.search(h)
        if not m:
            continue
        etype = (m.group(1) or "other").lower() or "other"
        try:
            x = float(v.strip().strip('"'))
        except ValueError:
            continue
        if x <= 0:
            continue
        totals[etype] = totals.get(etype, 0.0) + x
    return totals


def _wddm_reader_loop() -> None:
    """Long-running thread: stream typeperf samples, update _WDDM_STATE cache."""
    if sys.platform != "win32":
        return
    exe = shutil.which("typeperf")
    if not exe:
        return
    while True:
        proc = None
        try:
            proc = subprocess.Popen(
                [exe, "\\GPU Engine(*)\\Utilization Percentage", "-si", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            header_cols: list[str] | None = None
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                # typeperf emits CSV; the first header row starts with `"(PDH-CSV ...)"`
                if line.startswith('"(PDH'):
                    header_cols = [c.strip('"') for c in line.split(",")]
                    continue
                if header_cols is None:
                    continue
                value_cols = [c.strip('"') for c in line.split(",")]
                totals = _parse_wddm_csv_line(header_cols, value_cols)
                if not totals:
                    continue
                overall_max = max(totals.values())
                compute = totals.get("compute", 0.0)
                with _WDDM_STATE["lock"]:
                    _WDDM_STATE["ts"] = time.time()
                    _WDDM_STATE["value"] = {
                        "max_percent": round(overall_max, 1),
                        "compute_percent": round(compute, 1),
                        "engines": {k: round(v, 1) for k, v in totals.items()},
                    }
        except Exception:
            pass
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        time.sleep(2.0)  # brief backoff before respawn if typeperf died


def _ensure_wddm_reader_started() -> None:
    global _WDDM_STARTED
    if _WDDM_STARTED or sys.platform != "win32":
        return
    _WDDM_STARTED = True
    t = threading.Thread(target=_wddm_reader_loop, daemon=True, name="wddm-gpu-reader")
    t.start()


def _wddm_snapshot(max_age_sec: float = 5.0) -> dict[str, Any] | None:
    _ensure_wddm_reader_started()
    with _WDDM_STATE["lock"]:
        ts = _WDDM_STATE["ts"]
        value = _WDDM_STATE["value"]
    if not value or (time.time() - ts) > max_age_sec:
        return None
    return dict(value)


def get_system_metrics() -> dict[str, Any]:
    cpu = _cpu_percent()
    mem = _memory_info()
    gpus = _query_gpus_nvidia_smi()
    gpu_nvml: float | None = None
    if gpus:
        gpu_nvml = sum(float(g["utilization_gpu"]) for g in gpus) / len(gpus)
    vram = _gpu_vram_aggregate(gpus)

    wddm = _wddm_snapshot()
    if wddm is not None:
        gpu_display: float | None = wddm["max_percent"]
        gpu_source = "wddm"
    elif gpu_nvml is not None:
        gpu_display = round(gpu_nvml, 1)
        gpu_source = "nvml"
    else:
        gpu_display = None
        gpu_source = "none"

    return {
        "cpu_percent": round(cpu, 1),
        **mem,
        "gpu_percent": gpu_display,
        "gpu_percent_source": gpu_source,
        "gpu_percent_nvml": None if gpu_nvml is None else round(gpu_nvml, 1),
        "gpu_percent_wddm_compute": wddm["compute_percent"] if wddm else None,
        "gpu_available": len(gpus) > 0,
        "gpus": gpus,
        **vram,
    }

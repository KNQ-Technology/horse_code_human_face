"""Add PyTorch's bundled CUDA/cuDNN/cuBLAS DLLs to the Windows DLL search path.

Why this exists:
- onnxruntime-gpu's `CUDAExecutionProvider` needs cublasLt64_12 / cudnn / cufft etc.
- Installing paddlepaddle-gpu used to drop nvidia-* pip packages alongside, but those
  conflict with torch's bundled CUDA runtime (see DEPLOY.md §4.2.1 — the standalone
  nvidia-* packages are uninstalled on Windows).
- That leaves torch/lib as the only source of these DLLs on disk. Windows only finds
  them automatically if torch has been imported (it calls os.add_dll_directory),
  but onnxruntime can be imported before torch depending on import order.

Import this module at the top of main.py and any other entrypoint BEFORE importing
onnxruntime / PaddleOCR, so SCRFD/ArcFace ONNX sessions actually use CUDA instead of
silently falling back to CPU.
"""

from __future__ import annotations

import os
import sys


def _add_torch_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib):
        return
    try:
        os.add_dll_directory(torch_lib)
    except (OSError, FileNotFoundError):
        pass
    # Also expose via PATH for libraries that use LoadLibrary without add_dll_directory.
    if torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_add_torch_cuda_dlls()

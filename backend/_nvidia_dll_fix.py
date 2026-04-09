"""Windows: ensure CUDA DLLs are findable before paddle/torch are imported.

PaddlePaddle-GPU expects NVIDIA DLLs (cudnn, cublas, etc.) on the search path.
On this system, PyTorch ships those DLLs in torch/lib. We register that
directory so paddle can find them too, avoiding '[WinError 127]' errors.
"""
import os
import sys

if sys.platform == "win32":
    _site_packages = os.path.normpath(
        os.path.join(os.path.dirname(os.__file__), "..", "Lib", "site-packages")
    )
    _torch_lib = os.path.normpath(os.path.join(_site_packages, "torch", "lib"))
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

"""
Build script for turboquant-kv.

Two extension modules:
  1. turboquant_kv._core  -- pure C++ via pybind11 (no PyTorch dependency)
  2. turboquant_kv._C     -- PyTorch C++/CUDA extension (optional, needs torch)
"""
import os
import platform
import shutil
import sys
from setuptools import setup


def _openmp_flags():
    """Return (compile_args, link_args) for OpenMP on the current platform."""
    if sys.platform == "darwin":
        # macOS: libomp from Homebrew/Xcode
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    elif sys.platform == "win32":
        return ["/openmp"], []
    else:
        return ["-fopenmp"], ["-fopenmp"]


def build_core_ext():
    """Build the pure-C++ _core extension using pybind11 (no torch needed)."""
    try:
        from pybind11.setup_helpers import Pybind11Extension
    except ImportError:
        # pybind11 not installed -- skip
        return []

    omp_compile, omp_link = _openmp_flags()

    sources = [
        os.path.join("csrc", "core", "turboquant_core.cpp"),
        os.path.join("csrc", "core", "bindings.cpp"),
    ]

    # Only include sources that exist
    sources = [s for s in sources if os.path.isfile(s)]
    if not sources:
        return []

    if sys.platform == "win32":
        extra_compile = ["/O2", "/std:c++17"] + omp_compile
        extra_link = omp_link
    else:
        extra_compile = ["-O3", "-std=c++17", "-fvisibility=hidden"] + omp_compile
        extra_link = omp_link

    ext = Pybind11Extension(
        "turboquant_kv._core",
        sources,
        include_dirs=[os.path.join("csrc", "core")],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", None)],
    )
    return [ext]


def build_torch_ext():
    """Build the PyTorch C++/CUDA _C extension (optional)."""
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError:
        return [], {}

    ext_modules = []
    cmdclass = {"build_ext": BuildExtension}

    cuda_available = torch.cuda.is_available() and shutil.which("nvcc") is not None

    cpu_sources = []
    cuda_sources = []

    cpu_dir = os.path.join("csrc", "cpu")
    cuda_dir = os.path.join("csrc", "cuda")
    registration = os.path.join("csrc", "registration.cpp")

    if os.path.isfile(registration):
        cpu_sources.append(registration)

    for f in sorted(os.listdir(cpu_dir)) if os.path.isdir(cpu_dir) else []:
        if f.endswith(".cpp"):
            cpu_sources.append(os.path.join(cpu_dir, f))

    if cuda_available:
        try:
            from torch.utils.cpp_extension import CUDAExtension

            for f in sorted(os.listdir(cuda_dir)) if os.path.isdir(cuda_dir) else []:
                if f.endswith((".cu", ".cpp")):
                    cuda_sources.append(os.path.join(cuda_dir, f))

            all_sources = cpu_sources + cuda_sources
            if all_sources:
                ext_modules.append(
                    CUDAExtension(
                        name="turboquant_kv._C",
                        sources=all_sources,
                        define_macros=[("WITH_CUDA", None)],
                        extra_compile_args={
                            "cxx": ["-O3"],
                            "nvcc": [
                                "-O3",
                                "--use_fast_math",
                                "--expt-relaxed-constexpr",
                            ],
                        },
                    )
                )
        except ImportError:
            cuda_available = False

    if not cuda_available and cpu_sources:
        ext_modules.append(
            CppExtension(
                name="turboquant_kv._C",
                sources=cpu_sources,
                extra_compile_args=["-O3"],
            )
        )

    return ext_modules, cmdclass


# Combine both extension types
core_exts = build_core_ext()
torch_exts, torch_cmdclass = build_torch_ext()

all_ext_modules = core_exts + torch_exts

# If we have torch extensions, we need BuildExtension as cmdclass.
# For pybind11-only builds, setuptools handles it natively.
cmdclass = torch_cmdclass if torch_cmdclass else {}

setup(
    ext_modules=all_ext_modules,
    cmdclass=cmdclass,
)

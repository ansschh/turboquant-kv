"""
Build script for turboquant-kv.
Uses CUDAExtension when nvcc is available, CPU-only fallback otherwise.
"""
import os
import shutil
from setuptools import setup

def build_ext():
    """Return the ext_modules list and cmdclass, or empty if torch is unavailable."""
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError:
        return [], {}

    ext_modules = []
    cmdclass = {"build_ext": BuildExtension}

    # Check for CUDA availability
    cuda_available = torch.cuda.is_available() and shutil.which("nvcc") is not None

    cpu_sources = []
    cuda_sources = []

    # Collect source files
    cpu_dir = os.path.join("csrc", "cpu")
    cuda_dir = os.path.join("csrc", "cuda")
    registration = os.path.join("csrc", "registration.cpp")

    if os.path.isfile(registration):
        cpu_sources.append(registration)

    for f in os.listdir(cpu_dir) if os.path.isdir(cpu_dir) else []:
        if f.endswith(".cpp"):
            cpu_sources.append(os.path.join(cpu_dir, f))

    if cuda_available:
        try:
            from torch.utils.cpp_extension import CUDAExtension

            for f in os.listdir(cuda_dir) if os.path.isdir(cuda_dir) else []:
                if f.endswith((".cu", ".cpp")):
                    cuda_sources.append(os.path.join(cuda_dir, f))

            all_sources = cpu_sources + cuda_sources
            if all_sources:
                ext_modules.append(
                    CUDAExtension(
                        name="turboquant_kv._C",
                        sources=all_sources,
                        extra_compile_args={
                            "cxx": ["-O3"],
                            "nvcc": ["-O3", "--use_fast_math"],
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


ext_modules, cmdclass = build_ext()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

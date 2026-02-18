"""Core shared modules for mobile_sife_cuda (CUDA-optimized)."""
from .patchcore_sife import PatchCoreSIFE
from .cuda_utils import (
    get_optimal_device,
    warmup_cuda,
    amp_autocast,
    clear_gpu_cache,
    gpu_memory_info,
    check_faiss_gpu,
)

__all__ = [
    "PatchCoreSIFE",
    "get_optimal_device",
    "warmup_cuda",
    "amp_autocast",
    "clear_gpu_cache",
    "gpu_memory_info",
    "check_faiss_gpu",
]

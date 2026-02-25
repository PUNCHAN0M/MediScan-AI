# core_shared/cuda_utils.py
"""
CUDA Acceleration Utilities.

Responsibilities:
- Device detection and optimal selection
- Mixed-precision (AMP) context management
- GPU memory management and monitoring
- CUDA warm-up for eliminating first-call latency
"""
import gc
import torch
from contextlib import contextmanager
from typing import Dict, Optional


# =========================================================
# DEVICE MANAGEMENT
# =========================================================

def get_optimal_device(prefer_cuda: bool = True) -> torch.device:
    """Select optimal compute device with CUDA preference."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def warmup_cuda(device: torch.device) -> None:
    """
    Warm up CUDA context to eliminate first-call latency.
    
    The first CUDA operation triggers context initialization (~200-500ms).
    Calling this during setup avoids latency during inference.
    """
    if device.type != "cuda":
        return
    dummy = torch.zeros(1, device=device)
    torch.cuda.synchronize()
    del dummy


# =========================================================
# MIXED PRECISION (AMP)
# =========================================================

@contextmanager
def amp_autocast(device: torch.device, dtype=torch.float16):
    """
    Context manager for automatic mixed precision.
    
    Enables FP16 inference on CUDA for ~1.5-2x speedup.
    Falls back to no-op on CPU.
    """
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def supports_amp(device: torch.device) -> bool:
    """Check if device supports AMP inference."""
    return device.type == "cuda"


# =========================================================
# MEMORY MANAGEMENT
# =========================================================

def clear_gpu_cache() -> None:
    """Release unused GPU memory back to CUDA allocator."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory statistics (in MB).
    
    Returns:
        Dict with allocated_mb, reserved_mb, max_allocated_mb, free_mb
    """
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    return {
        "available": True,
        "device_name": props.name,
        "total_mb": round(props.total_memory / 1024**2, 1),
        "allocated_mb": round(allocated / 1024**2, 1),
        "reserved_mb": round(reserved / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "free_mb": round((props.total_memory - allocated) / 1024**2, 1),
    }


def estimate_batch_size(
    single_image_mb: float = 12.0,
    max_batch: int = 32,
    safety_factor: float = 0.6,
) -> int:
    """
    Estimate safe batch size based on available GPU memory.
    
    Args:
        single_image_mb: Estimated GPU memory per image (MB)
        max_batch: Maximum allowed batch size
        safety_factor: Fraction of free memory to use (0.0-1.0)
    """
    if not torch.cuda.is_available():
        return max_batch

    free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    free_mb = free / 1024**2
    batch = int(free_mb * safety_factor / single_image_mb)
    return max(1, min(batch, max_batch))


# =========================================================
# FAISS GPU SUPPORT
# =========================================================

_FAISS_GPU_AVAILABLE: Optional[bool] = None


def check_faiss_gpu() -> bool:
    """Check if FAISS GPU support is available."""
    global _FAISS_GPU_AVAILABLE
    if _FAISS_GPU_AVAILABLE is not None:
        return _FAISS_GPU_AVAILABLE

    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            _FAISS_GPU_AVAILABLE = True
            del res
        else:
            _FAISS_GPU_AVAILABLE = False
    except Exception:
        _FAISS_GPU_AVAILABLE = False

    return _FAISS_GPU_AVAILABLE


def faiss_index_to_gpu(index):
    """
    Move FAISS index to GPU if supported, otherwise return CPU index.
    
    Args:
        index: FAISS CPU index
    Returns:
        GPU index if available, otherwise original CPU index
    """
    if not check_faiss_gpu():
        return index

    try:
        import faiss
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index)
    except Exception:
        return index

"""
SRT Router — Pooled Mahalanobis Distance with Welford-Hart incremental update.

GPU-accelerated via torch (CUDA). Falls back to numpy on CPU.
"""
from .pooled_mahalanobis import (
    PooledMahalanobisRouter,
    participation_ratio_np,
    _compute_distances_gpu,
    _SHRINKAGE_METHODS,
)
from .metrics import (
    compute_routing_accuracy,
    compute_confidence,
    baseline_cosine_routing,
    baseline_l2_routing,
    compute_per_task_accuracy,
)
from .adapters import (
    SMoLoRASRTAdapter,
    CLIPVisionSRTAdapter,
    HiDeLLaVASRTAdapter,
)

__all__ = [
    "PooledMahalanobisRouter",
    "participation_ratio_np",
    "_compute_distances_gpu",
    "_SHRINKAGE_METHODS",
    "compute_routing_accuracy",
    "compute_confidence",
    "baseline_cosine_routing",
    "baseline_l2_routing",
    "compute_per_task_accuracy",
    "SMoLoRASRTAdapter",
    "CLIPVisionSRTAdapter",
    "HiDeLLaVASRTAdapter",
]

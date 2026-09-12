"""GPU device configuration for M-RWGAN training/eval scripts.

The training scripts do nothing device-specific on their own — TensorFlow
auto-places ops on a visible GPU. This helper makes that explicit: it reports
what TF actually found (so a silent CPU fallback is visible in the logs) and
enables per-process memory growth so TF does not reserve all of VRAM up front.
"""
from __future__ import annotations

import tensorflow as tf


def configure_gpu(force_cpu: bool = False, memory_growth: bool = True) -> str:
    """Set up TF device visibility. Returns a human-readable summary string.

    - ``force_cpu=True``  -> hide all GPUs (CPU-only run).
    - otherwise           -> enable memory growth on every visible GPU so TF
      does not grab all of VRAM up front, and report what was found.
    """
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
        return "GPU disabled (force_cpu=True); running on CPU."

    # get_visible_devices (not list_physical_devices) so a prior force_cpu call
    # in the same process is respected — this makes a second call from train()
    # a genuine no-op.
    gpus = tf.config.get_visible_devices("GPU")
    if not gpus:
        if tf.config.list_physical_devices("GPU"):
            return "GPU present but hidden (force_cpu); running on CPU."
        return (
            "No GPU found by TensorFlow — running on CPU. "
            "Check that CUDA/cuDNN libs are installed and on LD_LIBRARY_PATH."
        )

    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # GPU already initialised — memory growth can no longer be set.
                pass

    return f"GPU(s) in use: {[gpu.name for gpu in gpus]}"

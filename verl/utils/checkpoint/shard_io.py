"""Synchronous shard persistence with optional, file-scoped cache advice."""

import os
import warnings


def save_shard(state, path, *, flush_reclaim=False):
    import torch

    if not flush_reclaim:
        torch.save(state, path)
        return
    # Keep PyTorch native path serialization (avoid Python buffer copies).
    torch.save(state, path)
    # torch.save has closed its writer. Do not truncate or rewrite the shard.
    with open(path, "r+b") as stream:
        os.fsync(stream.fileno())
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as exc:
                warnings.warn(f"Checkpoint saved but cache advice unavailable: {exc}", RuntimeWarning)
        else:
            warnings.warn("Checkpoint saved but cache advice unsupported", RuntimeWarning)

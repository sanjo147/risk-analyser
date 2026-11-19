"""Risk banding (model output -> category) kept separate from lending policy."""
from __future__ import annotations

import numpy as np


def categorize(prob, bands: list[dict]):
    """Map probabilities (0-1) to (category, band-default decision) arrays."""
    prob = np.atleast_1d(np.asarray(prob, dtype=float))
    uppers = np.array([b["max"] for b in bands])
    idx = np.searchsorted(uppers, prob, side="left")  # band i covers (upper[i-1], upper[i]]
    idx = np.clip(idx, 0, len(bands) - 1)
    cats = np.array([b["name"] for b in bands], dtype=object)[idx]
    decs = np.array([b["decision"] for b in bands], dtype=object)[idx]
    return cats, decs


def approve_threshold(bands: list[dict]) -> float:
    """Highest probability that still maps to APPROVE (used for confusion matrices)."""
    t = 0.0
    for b in bands:
        if b["decision"] == "APPROVE":
            t = b["max"]
        else:
            break
    return t


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

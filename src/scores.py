"""IDO-inspired relative suspicion ranks, never corruption probabilities.

Use complete stable-ID arrays from prior completed epochs, grouped only by
provided noisy labels. Freeze the returned scores for the following epoch.
No torch dependency or clean-label/corruption-mask inputs.
"""
from __future__ import annotations

import numpy as np


def _vector(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be a finite real vector")
    return values


def percentile_rank_by_noisy_class(values, noisy_labels):
    """One-based average tie ranks: (rank - 0.5) / class size.

    Singleton/all-tied classes receive 0.5; empty classes are skipped.
    Every complete nonempty class has mean 0.5 regardless of noise ratio.
    Array order (subset_position) is preserved. Empty input returns empty.
    """
    values = _vector(values, "values")
    labels = np.asarray(noisy_labels)
    if (labels.shape != values.shape or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= 10))):
        raise ValueError("Provided noisy labels must be aligned integers in [0,9]")
    scores = np.empty(len(values), dtype=np.float64)
    for label in range(10):
        ids = np.flatnonzero(labels == label)
        if not len(ids):
            continue
        order = np.argsort(values[ids], kind="stable")
        sorted_values = values[ids[order]]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        stops = np.r_[starts[1:], len(ids)]
        # Average one-based rank minus 0.5 for zero-based [start, stop).
        ranked = np.repeat((starts + stops) / (2.0 * len(ids)), stops - starts)
        scores[ids[order]] = ranked
    return scores


def suspicion_scores(we, aum, noisy_labels, *, observation_counts):
    """Independent WE-only, inverted-AUM and 50:50 hybrid arrays.

    Positive equal observation counts prevent ranking unobserved/partial passes.
    All inputs must share stable subset-ID order, not shuffled loader order.
    """
    we = _vector(we, "WE")
    aum = _vector(aum, "AUM")
    counts = np.asarray(observation_counts)
    if (not len(we) or aum.shape != we.shape or counts.shape != we.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0) or not np.all(counts == counts[0])
            or not np.issubdtype(we.dtype, np.integer)
            or np.any(we < 0) or np.any(we > counts)):
        raise ValueError("Expected aligned, fully observed cumulative WE/AUM statistics")
    s_we = percentile_rank_by_noisy_class(we, noisy_labels)
    s_aum = percentile_rank_by_noisy_class(-aum.astype(np.float64), noisy_labels)
    return {"ido_lite": s_we, "aum": s_aum, "hybrid": 0.5 * s_we + 0.5 * s_aum}


def heuristic_weights(suspicion, *, clean_weight_floor=0.1):
    """Experimental classification floor and parabolic difficulty, not IDO BMM."""
    suspicion = _vector(suspicion, "suspicion")
    if np.any((suspicion < 0) | (suspicion > 1)):
        raise ValueError("Suspicion must lie in [0,1]")
    if not np.isfinite(clean_weight_floor) or not 0 < clean_weight_floor <= 1:
        raise ValueError("clean_weight_floor must lie in (0,1]")
    return (clean_weight_floor + (1 - clean_weight_floor) * (1 - suspicion),
            4 * suspicion * (1 - suspicion))

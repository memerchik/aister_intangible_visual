"""Array-only primitives for the fixed Phase 5 v5 probability consensus."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


Array = np.ndarray


def _probability_matrix(values: Array, name: str) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty probability matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    totals = matrix.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one")
    return matrix


def centered_log_probabilities(probabilities: Array) -> Array:
    """Recover row-centered logits from normalized probabilities."""

    matrix = _probability_matrix(probabilities, "probabilities")
    clipped = np.clip(matrix, np.finfo(np.float64).tiny, 1.0)
    logits = np.log(clipped)
    return logits - logits.mean(axis=1, keepdims=True)


def fixed_probability_consensus(
    probability_heads: Sequence[Array],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Array:
    """Average centered head logits with fixed weights and return softmax rows."""

    heads = tuple(probability_heads)
    if len(heads) < 2:
        raise ValueError("At least two probability heads are required")
    matrices = tuple(
        _probability_matrix(head, f"probability_heads[{index}]")
        for index, head in enumerate(heads)
    )
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("All probability heads must have the same shape")
    if weights is None:
        normalized = np.full(len(matrices), 1.0 / len(matrices), dtype=np.float64)
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if normalized.ndim != 1 or len(normalized) != len(matrices):
            raise ValueError("weights must contain one value per head")
        if not np.isfinite(normalized).all() or np.any(normalized <= 0):
            raise ValueError("weights must contain finite positive values")
        if not np.isclose(float(normalized.sum()), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("weights must sum to one")
    combined = np.zeros_like(matrices[0], dtype=np.float64)
    for weight, matrix in zip(normalized, matrices):
        combined += float(weight) * centered_log_probabilities(matrix)
    shifted = combined - combined.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Fixed consensus produced non-finite probabilities")
    return probabilities

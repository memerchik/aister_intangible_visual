"""Pairwise ceramic correction primitives for the frozen Phase 5 v2 search.

This module is array-only.  It fits a binary head on two declared classes and
can modify only those two probability columns while preserving their combined
base mass.  It performs no path, cache, manifest, or sealed-evaluation access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .robustness import (
    CenteredLogitEnsemble,
    FittedProbe,
    compute_source_group_weights,
    fit_centered_logit_ensemble,
    fit_logistic_probe,
    ordinary_metrics,
)


Array = np.ndarray
FeatureBlocks = Union[Array, Sequence[Array]]
PairModel = Union[FittedProbe, CenteredLogitEnsemble]


def _feature_blocks(features: FeatureBlocks) -> Tuple[Array, ...]:
    raw = (features,) if isinstance(features, np.ndarray) else tuple(features)
    if not raw:
        raise ValueError("At least one pairwise feature block is required")
    blocks = []
    for index, value in enumerate(raw):
        block = np.asarray(value)
        if block.ndim != 2 or not block.shape[0] or not block.shape[1]:
            raise ValueError(f"Pairwise feature block {index} must be a matrix")
        if not np.issubdtype(block.dtype, np.number) or not np.isfinite(block).all():
            raise ValueError(f"Pairwise feature block {index} must be finite numeric")
        blocks.append(block)
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError("Pairwise feature blocks must have equal row counts")
    return tuple(blocks)


def _strings(values: Sequence[str], name: str) -> Array:
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty vector")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True)
class PairwiseCorrection:
    """A binary correction head with an explicit pair-class order."""

    model: PairModel
    pair_classes: Tuple[str, str]
    centered_logit_ensemble: bool

    @property
    def iterations(self) -> object:
        return self.model.iterations

    @property
    def converged(self) -> bool:
        return self.model.converged

    def predict_proba(self, features: FeatureBlocks) -> Array:
        return self.model.predict_proba(features)


def fit_pairwise_correction(
    features: FeatureBlocks,
    labels: Sequence[str],
    group_ids: Sequence[str],
    *,
    pair_classes: Sequence[str],
    c_value: float,
    source_group_exponent: float,
    centered_logit_ensemble: bool = False,
    seed: int = 0,
    max_iter: int = 10_000,
    tolerance: float = 1e-6,
) -> PairwiseCorrection:
    """Fit a fold-local binary probe using only the two pair classes."""

    blocks = _feature_blocks(features)
    label_array = _strings(labels, "labels")
    group_array = _strings(group_ids, "group_ids")
    if blocks[0].shape[0] != len(label_array) or len(group_array) != len(label_array):
        raise ValueError("features, labels, and group_ids must align")
    pair = tuple(pair_classes)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("pair_classes must contain exactly two unique labels")
    if set(pair) - set(label_array.tolist()):
        raise ValueError("Every pair class must occur in the supplied training rows")
    mask = np.isin(label_array, pair)
    pair_labels = label_array[mask]
    pair_groups = group_array[mask]
    pair_blocks = tuple(block[mask] for block in blocks)
    weights = compute_source_group_weights(
        pair_labels,
        pair_groups,
        source_group_exponent,
    )
    if centered_logit_ensemble:
        if len(pair_blocks) < 2:
            raise ValueError("Pairwise logit ensemble requires multiple blocks")
        model: PairModel = fit_centered_logit_ensemble(
            pair_blocks,
            pair_labels,
            c_value=c_value,
            sample_weight=weights,
            classes=pair,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    else:
        if len(pair_blocks) != 1:
            raise ValueError("Single pairwise probe requires exactly one block")
        model = fit_logistic_probe(
            pair_blocks[0],
            pair_labels,
            c_value=c_value,
            sample_weight=weights,
            classes=pair,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    return PairwiseCorrection(
        model=model,
        pair_classes=pair,  # type: ignore[arg-type]
        centered_logit_ensemble=centered_logit_ensemble,
    )


def apply_pairwise_logit_blend(
    base_probabilities: Array,
    base_classes: Sequence[str],
    pair_probabilities: Array,
    pair_probability_classes: Sequence[str],
    pair_classes: Sequence[str],
    blend_weight: float,
) -> Array:
    """Blend pair-conditional log-odds and preserve base pair mass exactly."""

    base = np.asarray(base_probabilities, dtype=np.float64)
    head = np.asarray(pair_probabilities, dtype=np.float64)
    base_order = tuple(base_classes)
    head_order = tuple(pair_probability_classes)
    pair = tuple(pair_classes)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("pair_classes must contain exactly two unique labels")
    if len(base_order) != len(set(base_order)) or set(pair) - set(base_order):
        raise ValueError("base_classes must uniquely contain both pair classes")
    if len(head_order) != 2 or set(head_order) != set(pair):
        raise ValueError("pair_probability_classes must exactly match the pair")
    if base.ndim != 2 or base.shape[1] != len(base_order):
        raise ValueError("base probability shape does not match base classes")
    if head.ndim != 2 or head.shape != (base.shape[0], 2):
        raise ValueError("pair probabilities must have one two-class row per sample")
    if (
        not np.isfinite(base).all()
        or not np.isfinite(head).all()
        or np.any(base < 0)
        or np.any(head < 0)
    ):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(base.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("base probability rows must sum to one")
    if not np.allclose(head.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("pair probability rows must sum to one")
    if (
        isinstance(blend_weight, bool)
        or not np.isfinite(blend_weight)
        or blend_weight < 0
        or blend_weight > 1
    ):
        raise ValueError("blend_weight must be in [0, 1]")

    base_positions = {label: index for index, label in enumerate(base_order)}
    head_positions = {label: index for index, label in enumerate(head_order)}
    first, second = pair
    first_index = base_positions[first]
    second_index = base_positions[second]
    pair_mass = base[:, first_index] + base[:, second_index]
    epsilon = 1e-12
    base_first = np.divide(
        base[:, first_index],
        pair_mass,
        out=np.full(base.shape[0], 0.5, dtype=np.float64),
        where=pair_mass > 0,
    )
    head_first = head[:, head_positions[first]]
    base_first = np.clip(base_first, epsilon, 1.0 - epsilon)
    head_first = np.clip(head_first, epsilon, 1.0 - epsilon)
    base_logit = np.log(base_first) - np.log1p(-base_first)
    head_logit = np.log(head_first) - np.log1p(-head_first)
    blended_logit = (1.0 - float(blend_weight)) * base_logit + float(
        blend_weight
    ) * head_logit
    blended_first = np.empty_like(blended_logit)
    positive = blended_logit >= 0
    blended_first[positive] = 1.0 / (1.0 + np.exp(-blended_logit[positive]))
    exponential = np.exp(blended_logit[~positive])
    blended_first[~positive] = exponential / (1.0 + exponential)

    combined = base.copy()
    combined[:, first_index] = pair_mass * blended_first
    combined[:, second_index] = pair_mass * (1.0 - blended_first)
    return combined


def harmonic_mean_many(values: Sequence[float]) -> float:
    """Return the generalized harmonic mean of finite non-negative values."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("Harmonic mean requires at least one value")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Harmonic-mean values must be finite and non-negative")
    if np.any(array == 0):
        return 0.0
    return float(len(array) / np.sum(1.0 / array))


@dataclass(frozen=True)
class BoundaryMetrics:
    opishnyan_recall: float
    bubnivka_recall: float
    bubnivka_precision: float
    ceramic_macro_f1: float
    boundary_score: float
    opishnyan_to_bubnivka_errors: int
    bubnivka_to_opishnyan_errors: int


def boundary_metrics(
    true_labels: Sequence[str],
    probabilities: Array,
    classes: Sequence[str],
    *,
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> BoundaryMetrics:
    """Calculate the exact v2 pair-boundary selection components."""

    pair = tuple(pair_classes)
    ceramics = tuple(ceramic_classes)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("pair_classes must contain two unique labels")
    if len(ceramics) != 3 or set(pair) - set(ceramics):
        raise ValueError("ceramic_classes must contain the pair and one other class")
    ordinary = ordinary_metrics(true_labels, probabilities, classes)
    truth = _strings(true_labels, "true_labels")
    class_array = np.asarray(tuple(classes), dtype=object)
    predictions = class_array[np.argmax(np.asarray(probabilities), axis=1)]
    opishnyan, bubnivka = pair
    ceramic_macro_f1 = float(
        np.mean([ordinary.per_class_f1[label] for label in ceramics])
    )
    components = (
        ordinary.per_class_recall[opishnyan],
        ordinary.per_class_recall[bubnivka],
        ordinary.per_class_precision[bubnivka],
        ceramic_macro_f1,
    )
    return BoundaryMetrics(
        opishnyan_recall=components[0],
        bubnivka_recall=components[1],
        bubnivka_precision=components[2],
        ceramic_macro_f1=components[3],
        boundary_score=harmonic_mean_many(components),
        opishnyan_to_bubnivka_errors=int(
            np.sum((truth == opishnyan) & (predictions == bubnivka))
        ),
        bubnivka_to_opishnyan_errors=int(
            np.sum((truth == bubnivka) & (predictions == opishnyan))
        ),
    )


@dataclass(frozen=True)
class CorrectionCandidateSummary:
    configuration_id: str
    boundary_score: float
    minimum_fold_boundary_score: float
    accuracy: float
    macro_f1: float
    source_robustness_score: float
    opishnyan_group_recall: float
    combined_view_count: int
    correction: str
    blend_weight: float
    c_value: float
    source_group_exponent: float
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class CorrectionEligibilityReference:
    accuracy: float
    macro_f1: float
    source_robustness_score: float
    opishnyan_group_recall: float
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class CorrectionEligibilityResult:
    eligible: bool
    failures: Tuple[str, ...]


def correction_eligibility(
    candidate: CorrectionCandidateSummary,
    reference: CorrectionEligibilityReference,
    *,
    other_class_recall_drops: Mapping[str, float],
    maximum_accuracy_drop: float = 0.005,
    maximum_macro_f1_drop: float = 0.005,
    maximum_source_robustness_drop: float = 0.005,
    maximum_opishnyan_group_recall_drop: float = 0.005,
) -> CorrectionEligibilityResult:
    """Apply the frozen v2 non-regression constraints."""

    limits = (
        maximum_accuracy_drop,
        maximum_macro_f1_drop,
        maximum_source_robustness_drop,
        maximum_opishnyan_group_recall_drop,
        *other_class_recall_drops.values(),
    )
    if any(not np.isfinite(value) or value < 0 for value in limits):
        raise ValueError("Eligibility drops must be finite and non-negative")
    failures = []
    if candidate.accuracy < reference.accuracy - maximum_accuracy_drop:
        failures.append("accuracy_drop")
    if candidate.macro_f1 < reference.macro_f1 - maximum_macro_f1_drop:
        failures.append("macro_f1_drop")
    if (
        candidate.source_robustness_score
        < reference.source_robustness_score - maximum_source_robustness_drop
    ):
        failures.append("source_robustness_drop")
    if (
        candidate.opishnyan_group_recall
        < reference.opishnyan_group_recall - maximum_opishnyan_group_recall_drop
    ):
        failures.append("opishnyan_group_recall_drop")
    for label, maximum_drop in other_class_recall_drops.items():
        if label not in candidate.per_class_recall or label not in reference.per_class_recall:
            raise ValueError(f"Eligibility recall is missing class {label!r}")
        if (
            candidate.per_class_recall[label]
            < reference.per_class_recall[label] - maximum_drop
        ):
            failures.append(f"recall_drop:{label}")
    return CorrectionEligibilityResult(
        eligible=not failures,
        failures=tuple(failures),
    )


def _tie_break_key(candidate: CorrectionCandidateSummary) -> Tuple[object, ...]:
    return (
        -candidate.minimum_fold_boundary_score,
        -candidate.source_robustness_score,
        -candidate.macro_f1,
        candidate.combined_view_count,
        0 if candidate.correction == "none" else 1,
        candidate.blend_weight,
        0 if np.isclose(candidate.c_value, 10.0, rtol=0.0, atol=1e-12) else 1,
        candidate.source_group_exponent,
        candidate.configuration_id,
    )


def rank_correction_candidates(
    candidates: Sequence[CorrectionCandidateSummary],
    *,
    tie_window: float = 0.002,
) -> Tuple[CorrectionCandidateSummary, ...]:
    """Rank correction candidates with score bands and frozen tie-breakers."""

    if not np.isfinite(tie_window) or tie_window < 0:
        raise ValueError("tie_window must be finite and non-negative")
    remaining = list(candidates)
    if not remaining:
        raise ValueError("At least one correction candidate is required")
    identifiers = [candidate.configuration_id for candidate in remaining]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Correction candidate IDs must be unique and non-empty")
    ranked = []
    while remaining:
        best = max(candidate.boundary_score for candidate in remaining)
        tied = [
            candidate
            for candidate in remaining
            if candidate.boundary_score >= best - tie_window
        ]
        tied.sort(key=_tie_break_key)
        ranked.extend(tied)
        tied_ids = {candidate.configuration_id for candidate in tied}
        remaining = [
            candidate
            for candidate in remaining
            if candidate.configuration_id not in tied_ids
        ]
    return tuple(ranked)

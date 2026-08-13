"""Array-only cross-encoder consensus primitives for Phase 5 v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _probabilities(values: Array, name: str) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or result.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty probability matrix")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"{name} rows must sum to one")
    return result


def centered_log_probabilities(probabilities: Array) -> Array:
    """Convert probability rows to centered log scores."""

    values = _probabilities(probabilities, "probabilities")
    scores = np.log(np.clip(values, 1e-12, 1.0))
    return scores - scores.mean(axis=1, keepdims=True)


def aggregate_head_scores(head_probabilities: Sequence[Array]) -> Array:
    """Average centered log-probability evidence across aligned heads."""

    heads = tuple(head_probabilities)
    if not heads:
        raise ValueError("At least one auxiliary head is required")
    scores = tuple(centered_log_probabilities(head) for head in heads)
    if len({score.shape for score in scores}) != 1:
        raise ValueError("Auxiliary head probabilities must have equal shapes")
    return np.mean(np.stack(scores, axis=0), axis=0)


def softmax_scores(scores: Array) -> Array:
    """Apply a stable row-wise softmax."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("scores must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def apply_multiclass_logit_fusion(
    base_probabilities: Array,
    head_probabilities: Sequence[Array],
    blend_weight: float,
) -> Array:
    """Blend base and auxiliary centered log evidence across every class."""

    base = _probabilities(base_probabilities, "base_probabilities")
    _validate_blend(blend_weight)
    auxiliary = aggregate_head_scores(head_probabilities)
    if auxiliary.shape != base.shape:
        raise ValueError("Base and auxiliary class probability shapes must match")
    combined = (
        (1.0 - float(blend_weight)) * centered_log_probabilities(base)
        + float(blend_weight) * auxiliary
    )
    return softmax_scores(combined)


def apply_pair_consensus(
    base_probabilities: Array,
    head_probabilities: Sequence[Array],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    blend_weight: float,
    *,
    majority_gated: bool,
) -> Tuple[Array, Array]:
    """Blend cross-encoder pair evidence on base-pair predictions only.

    Returns the new probabilities and the Boolean row mask on which the blend
    was applied.
    """

    base = _probabilities(base_probabilities, "base_probabilities")
    _validate_blend(blend_weight)
    class_order = tuple(classes)
    pair = tuple(pair_classes)
    if len(class_order) != base.shape[1] or len(set(class_order)) != len(class_order):
        raise ValueError("classes must uniquely match base probability columns")
    if len(pair) != 2 or len(set(pair)) != 2 or set(pair) - set(class_order):
        raise ValueError("pair_classes must contain two classes from classes")
    heads = tuple(_probabilities(head, "head_probabilities") for head in head_probabilities)
    if not heads or any(head.shape != base.shape for head in heads):
        raise ValueError("Every auxiliary head must align with base probabilities")

    first_index = class_order.index(pair[0])
    second_index = class_order.index(pair[1])
    base_prediction = np.argmax(base, axis=1)
    apply_mask = np.isin(base_prediction, (first_index, second_index))
    if majority_gated:
        votes = np.stack(
            [head[:, first_index] >= head[:, second_index] for head in heads], axis=1
        )
        first_votes = votes.sum(axis=1)
        required = int(np.ceil(2.0 * len(heads) / 3.0))
        agreement = np.maximum(first_votes, len(heads) - first_votes) >= required
        apply_mask &= agreement

    pair_mass = base[:, first_index] + base[:, second_index]
    epsilon = 1e-12
    base_first = np.divide(
        base[:, first_index],
        pair_mass,
        out=np.full(len(base), 0.5, dtype=np.float64),
        where=pair_mass > 0,
    )
    base_first = np.clip(base_first, epsilon, 1.0 - epsilon)
    base_logit = np.log(base_first) - np.log1p(-base_first)
    head_logits = []
    for head in heads:
        conditional = head[:, first_index] / (
            head[:, first_index] + head[:, second_index]
        )
        conditional = np.clip(conditional, epsilon, 1.0 - epsilon)
        head_logits.append(np.log(conditional) - np.log1p(-conditional))
    auxiliary_logit = np.mean(np.stack(head_logits, axis=1), axis=1)
    combined_logit = (
        (1.0 - float(blend_weight)) * base_logit
        + float(blend_weight) * auxiliary_logit
    )
    combined_first = np.empty_like(combined_logit)
    positive = combined_logit >= 0
    combined_first[positive] = 1.0 / (1.0 + np.exp(-combined_logit[positive]))
    exponential = np.exp(combined_logit[~positive])
    combined_first[~positive] = exponential / (1.0 + exponential)

    result = base.copy()
    result[apply_mask, first_index] = (
        pair_mass[apply_mask] * combined_first[apply_mask]
    )
    result[apply_mask, second_index] = (
        pair_mass[apply_mask] * (1.0 - combined_first[apply_mask])
    )
    return result, apply_mask


def harmonic_mean_many(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("Harmonic mean requires a non-empty vector")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Harmonic-mean values must be finite and non-negative")
    if np.any(array == 0):
        return 0.0
    return float(len(array) / np.sum(1.0 / array))


@dataclass(frozen=True)
class FusionCandidateSummary:
    configuration_id: str
    readiness_score: float
    minimum_fold_readiness_score: float
    accuracy: float
    macro_f1: float
    source_robustness_score: float
    opishnyan_group_recall: float
    intervention_rank: int
    blend_weight: float
    head_count: int
    encoder_count: int
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class FusionEligibilityReference:
    accuracy: float
    macro_f1: float
    source_robustness_score: float
    opishnyan_group_recall: float
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class FusionEligibilityResult:
    eligible: bool
    failures: Tuple[str, ...]


def fusion_eligibility(
    candidate: FusionCandidateSummary,
    reference: FusionEligibilityReference,
    *,
    other_class_recall_drops: Mapping[str, float],
    maximum_accuracy_drop: float = 0.003,
    maximum_macro_f1_drop: float = 0.003,
    maximum_source_robustness_drop: float = 0.003,
    maximum_opishnyan_group_recall_drop: float = 0.005,
) -> FusionEligibilityResult:
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
    if candidate.source_robustness_score < (
        reference.source_robustness_score - maximum_source_robustness_drop
    ):
        failures.append("source_robustness_drop")
    if candidate.opishnyan_group_recall < (
        reference.opishnyan_group_recall - maximum_opishnyan_group_recall_drop
    ):
        failures.append("opishnyan_group_recall_drop")
    for label, maximum_drop in other_class_recall_drops.items():
        if label not in candidate.per_class_recall or label not in reference.per_class_recall:
            raise ValueError(f"Eligibility recall is missing class {label!r}")
        if candidate.per_class_recall[label] < (
            reference.per_class_recall[label] - maximum_drop
        ):
            failures.append(f"recall_drop:{label}")
    return FusionEligibilityResult(not failures, tuple(failures))


def rank_fusion_candidates(
    candidates: Sequence[FusionCandidateSummary], *, tie_window: float
) -> Tuple[FusionCandidateSummary, ...]:
    if not np.isfinite(tie_window) or tie_window < 0:
        raise ValueError("tie_window must be finite and non-negative")
    remaining = list(candidates)
    if not remaining:
        raise ValueError("At least one fusion candidate is required")
    identifiers = [candidate.configuration_id for candidate in remaining]
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        raise ValueError("Fusion candidate identifiers must be unique and non-empty")
    ranked = []
    while remaining:
        best = max(candidate.readiness_score for candidate in remaining)
        tied = [
            candidate
            for candidate in remaining
            if candidate.readiness_score >= best - tie_window
        ]
        tied.sort(
            key=lambda candidate: (
                -candidate.minimum_fold_readiness_score,
                -candidate.source_robustness_score,
                -candidate.macro_f1,
                candidate.intervention_rank,
                candidate.blend_weight,
                candidate.head_count,
                candidate.encoder_count,
                candidate.configuration_id,
            )
        )
        ranked.extend(tied)
        tied_ids = {candidate.configuration_id for candidate in tied}
        remaining = [
            candidate
            for candidate in remaining
            if candidate.configuration_id not in tied_ids
        ]
    return tuple(ranked)


def _validate_blend(value: float) -> None:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError("blend_weight must be in [0, 1]")

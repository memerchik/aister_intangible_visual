"""Pure modelling and robustness metrics for Phase 5.

The functions in this module operate only on arrays supplied by the caller.
They deliberately perform no file or path access, which keeps fold-local
training statistics explicit and prevents reusable modelling code from
opening the sealed evaluation partition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits


Array = np.ndarray
FeatureBlocks = Union[Array, Sequence[Array]]


class ModelFitError(RuntimeError):
    """Raised when a probe cannot be fitted to the declared contract."""


def _string_vector(values: Sequence[str], name: str) -> Array:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if any(not isinstance(value, str) or not value for value in array):
        raise ValueError(f"{name} must contain non-empty strings")
    return array


def _feature_matrix(values: Array, name: str = "features") -> Array:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _feature_block_tuple(features: FeatureBlocks) -> Tuple[Array, ...]:
    if isinstance(features, np.ndarray):
        blocks = (_feature_matrix(features),)
    else:
        blocks = tuple(
            _feature_matrix(block, f"feature block {index}")
            for index, block in enumerate(features)
        )
        if not blocks:
            raise ValueError("At least one feature block is required")
    row_counts = {block.shape[0] for block in blocks}
    if len(row_counts) != 1:
        raise ValueError("All feature blocks must have the same row count")
    return blocks


def _declared_classes(
    labels: Array, classes: Optional[Sequence[str]], minimum: int = 2
) -> Tuple[str, ...]:
    observed = tuple(sorted(set(labels.tolist())))
    declared = observed if classes is None else tuple(classes)
    if len(declared) < minimum or len(set(declared)) != len(declared):
        raise ValueError(f"classes must contain at least {minimum} unique labels")
    if any(not isinstance(label, str) or not label for label in declared):
        raise ValueError("classes must contain non-empty strings")
    if set(observed) != set(declared):
        raise ValueError("Every declared class must occur and no undeclared class is allowed")
    return declared


def compute_source_group_weights(
    labels: Sequence[str],
    group_ids: Sequence[str],
    source_group_exponent: float,
) -> Array:
    """Compute fold-local source and class balancing weights.

    Group sizes and class normalizers are calculated solely from the rows
    passed to this function. For each row, the raw weight is its training-only
    group size raised to ``-source_group_exponent``. Raw weights are then
    normalized so each observed class has equal total mass and the global mean
    weight is exactly one (up to floating-point precision).
    """

    label_array = _string_vector(labels, "labels")
    group_array = _string_vector(group_ids, "group_ids")
    if len(label_array) != len(group_array):
        raise ValueError("labels and group_ids must have equal length")
    if (
        isinstance(source_group_exponent, bool)
        or not np.isfinite(source_group_exponent)
        or source_group_exponent < 0
    ):
        raise ValueError("source_group_exponent must be a finite non-negative number")

    group_counts: Dict[str, int] = {}
    for group_id in group_array:
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
    raw = np.asarray(
        [group_counts[group_id] ** (-float(source_group_exponent)) for group_id in group_array],
        dtype=np.float64,
    )

    classes = tuple(sorted(set(label_array.tolist())))
    target_class_mass = len(label_array) / len(classes)
    weights = raw.copy()
    for label in classes:
        mask = label_array == label
        raw_mass = float(raw[mask].sum())
        if raw_mass <= 0 or not np.isfinite(raw_mass):
            raise ValueError("Source weighting produced invalid class mass")
        weights[mask] *= target_class_mass / raw_mass

    weights /= float(weights.mean())
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Source weighting produced invalid sample weights")
    return weights


def _validate_sample_weight(sample_weight: Optional[Sequence[float]], rows: int) -> Optional[Array]:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != rows:
        raise ValueError("sample_weight must have one value per feature row")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample_weight must contain finite positive values")
    return weights


def _softmax(logits: Array) -> Array:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class FittedProbe:
    """A converged logistic probe with an explicit external class order."""

    estimator: LogisticRegression = field(repr=False)
    classes: Tuple[str, ...]
    iterations: Tuple[int, ...]
    converged: bool

    def _column_order(self) -> Array:
        positions = {label: index for index, label in enumerate(self.estimator.classes_)}
        return np.asarray([positions[label] for label in self.classes], dtype=np.int64)

    def decision_logits(self, features: Array) -> Array:
        matrix = _feature_matrix(features)
        if matrix.shape[1] != self.estimator.n_features_in_:
            raise ValueError("Feature dimension does not match the fitted probe")
        raw = np.asarray(self.estimator.decision_function(matrix), dtype=np.float64)
        if raw.ndim == 1:
            # sklearn stores one binary margin. [0, margin] has the same
            # probability ratio and becomes symmetric after row centering.
            raw = np.column_stack((np.zeros_like(raw), raw))
        ordered = raw[:, self._column_order()]
        return ordered - ordered.mean(axis=1, keepdims=True)

    def predict_proba(self, features: Array) -> Array:
        matrix = _feature_matrix(features)
        if matrix.shape[1] != self.estimator.n_features_in_:
            raise ValueError("Feature dimension does not match the fitted probe")
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        return probabilities[:, self._column_order()]

    def predict(self, features: Array) -> Array:
        probabilities = self.predict_proba(features)
        class_array = np.asarray(self.classes, dtype=object)
        return class_array[np.argmax(probabilities, axis=1)]


def fit_logistic_probe(
    features: Array,
    labels: Sequence[str],
    *,
    c_value: float,
    sample_weight: Optional[Sequence[float]] = None,
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
    max_iter: int = 10_000,
    tolerance: float = 1e-6,
) -> FittedProbe:
    """Fit a deterministic L2 logistic probe and fail on non-convergence."""

    matrix = _feature_matrix(features)
    label_array = _string_vector(labels, "labels")
    if len(label_array) != matrix.shape[0]:
        raise ValueError("features and labels must have equal row counts")
    declared = _declared_classes(label_array, classes)
    weights = _validate_sample_weight(sample_weight, matrix.shape[0])
    if isinstance(c_value, bool) or not np.isfinite(c_value) or c_value <= 0:
        raise ValueError("c_value must be a finite positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")

    estimator = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        tol=float(tolerance),
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            with threadpool_limits(limits=1):
                estimator.fit(matrix, label_array, sample_weight=weights)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise ModelFitError(f"Logistic probe fit failed: {error}") from error
    convergence_messages = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    iterations = tuple(int(value) for value in np.ravel(estimator.n_iter_))
    if convergence_messages or not iterations or max(iterations) >= max_iter:
        detail = "; ".join(convergence_messages) or "iteration limit reached"
        raise ModelFitError(f"Logistic probe did not converge: {detail}")
    if set(estimator.classes_) != set(declared):
        raise ModelFitError("Fitted probe class order does not match the declaration")
    return FittedProbe(
        estimator=estimator,
        classes=declared,
        iterations=iterations,
        converged=True,
    )


def predict_single_embedding(probe: FittedProbe, features: Array) -> Array:
    """Return class-aligned probabilities for one fitted embedding probe."""

    if not isinstance(probe, FittedProbe):
        raise TypeError("probe must be a FittedProbe")
    return probe.predict_proba(features)


@dataclass(frozen=True)
class CenteredLogitEnsemble:
    """Equal-weight mean of class-centered logits from multiple probes."""

    probes: Tuple[FittedProbe, ...]
    classes: Tuple[str, ...]

    @property
    def iterations(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(probe.iterations for probe in self.probes)

    @property
    def converged(self) -> bool:
        return bool(self.probes) and all(probe.converged for probe in self.probes)

    def decision_logits(self, feature_blocks: FeatureBlocks) -> Array:
        blocks = _feature_block_tuple(feature_blocks)
        if len(blocks) != len(self.probes):
            raise ValueError("Feature block count does not match the fitted ensemble")
        logits = [
            probe.decision_logits(block)
            for probe, block in zip(self.probes, blocks)
        ]
        return np.mean(np.stack(logits, axis=0), axis=0)

    def predict_proba(self, feature_blocks: FeatureBlocks) -> Array:
        return _softmax(self.decision_logits(feature_blocks))

    def predict(self, feature_blocks: FeatureBlocks) -> Array:
        probabilities = self.predict_proba(feature_blocks)
        class_array = np.asarray(self.classes, dtype=object)
        return class_array[np.argmax(probabilities, axis=1)]


def fit_centered_logit_ensemble(
    feature_blocks: Sequence[Array],
    labels: Sequence[str],
    *,
    c_value: float,
    sample_weight: Optional[Sequence[float]] = None,
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
    max_iter: int = 10_000,
    tolerance: float = 1e-6,
) -> CenteredLogitEnsemble:
    """Fit one probe per view for an equal-weight centered-logit ensemble."""

    blocks = _feature_block_tuple(feature_blocks)
    if len(blocks) < 2:
        raise ValueError("A centered-logit ensemble requires at least two feature blocks")
    label_array = _string_vector(labels, "labels")
    if blocks[0].shape[0] != len(label_array):
        raise ValueError("feature blocks and labels must have equal row counts")
    declared = _declared_classes(label_array, classes)
    weights = _validate_sample_weight(sample_weight, len(label_array))
    probes = tuple(
        fit_logistic_probe(
            block,
            label_array,
            c_value=c_value,
            sample_weight=weights,
            classes=declared,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        for block in blocks
    )
    return CenteredLogitEnsemble(probes=probes, classes=declared)


def predict_centered_logit_ensemble(
    ensemble: CenteredLogitEnsemble, feature_blocks: Sequence[Array]
) -> Array:
    """Return softmax probabilities from equally averaged centered logits."""

    if not isinstance(ensemble, CenteredLogitEnsemble):
        raise TypeError("ensemble must be a CenteredLogitEnsemble")
    return ensemble.predict_proba(feature_blocks)


ProbeModel = Union[FittedProbe, CenteredLogitEnsemble]


@dataclass(frozen=True)
class CeramicSpecialist:
    """A conditional model fitted only to the declared ceramic classes."""

    model: ProbeModel
    ceramic_classes: Tuple[str, str, str]

    @property
    def iterations(self) -> object:
        return self.model.iterations

    @property
    def converged(self) -> bool:
        return self.model.converged

    def predict_proba(self, features: FeatureBlocks) -> Array:
        return self.model.predict_proba(features)


def fit_ceramic_specialist(
    features: FeatureBlocks,
    labels: Sequence[str],
    group_ids: Sequence[str],
    *,
    ceramic_classes: Sequence[str],
    c_value: float,
    source_group_exponent: float,
    centered_logit_ensemble: bool = False,
    seed: int = 0,
    max_iter: int = 10_000,
    tolerance: float = 1e-6,
) -> CeramicSpecialist:
    """Fit a three-way ceramic probe using ceramic training rows only."""

    blocks = _feature_block_tuple(features)
    label_array = _string_vector(labels, "labels")
    group_array = _string_vector(group_ids, "group_ids")
    if blocks[0].shape[0] != len(label_array) or len(group_array) != len(label_array):
        raise ValueError("features, labels, and group_ids must have equal row counts")
    ceramics = tuple(ceramic_classes)
    if len(ceramics) != 3 or len(set(ceramics)) != 3:
        raise ValueError("ceramic_classes must contain exactly three unique labels")
    if any(label not in set(label_array.tolist()) for label in ceramics):
        raise ValueError("Every ceramic class must occur in the supplied training rows")

    mask = np.isin(label_array, ceramics)
    ceramic_labels = label_array[mask]
    ceramic_groups = group_array[mask]
    ceramic_blocks = tuple(block[mask] for block in blocks)
    weights = compute_source_group_weights(
        ceramic_labels, ceramic_groups, source_group_exponent
    )
    if centered_logit_ensemble:
        if len(ceramic_blocks) < 2:
            raise ValueError("Centered-logit specialist requires multiple feature blocks")
        model: ProbeModel = fit_centered_logit_ensemble(
            ceramic_blocks,
            ceramic_labels,
            c_value=c_value,
            sample_weight=weights,
            classes=ceramics,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    else:
        if len(ceramic_blocks) != 1:
            raise ValueError("Single-embedding specialist requires one feature matrix")
        model = fit_logistic_probe(
            ceramic_blocks[0],
            ceramic_labels,
            c_value=c_value,
            sample_weight=weights,
            classes=ceramics,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    return CeramicSpecialist(model=model, ceramic_classes=ceramics)  # type: ignore[arg-type]


def apply_ceramic_specialist(
    base_probabilities: Array,
    base_classes: Sequence[str],
    specialist_probabilities: Array,
    specialist_classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> Array:
    """Replace only the base model's conditional ceramic distribution.

    The total probability assigned by the base model to all ceramic classes is
    preserved exactly. Non-ceramic probabilities are left unchanged.
    """

    base = np.asarray(base_probabilities, dtype=np.float64)
    specialist = np.asarray(specialist_probabilities, dtype=np.float64)
    base_order = tuple(base_classes)
    specialist_order = tuple(specialist_classes)
    ceramics = tuple(ceramic_classes)
    if len(ceramics) != 3 or len(set(ceramics)) != 3:
        raise ValueError("ceramic_classes must contain exactly three unique labels")
    if len(base_order) != len(set(base_order)) or set(ceramics) - set(base_order):
        raise ValueError("base_classes must uniquely contain every ceramic class")
    if len(specialist_order) != 3 or set(specialist_order) != set(ceramics):
        raise ValueError("specialist_classes must exactly match ceramic_classes")
    if base.ndim != 2 or base.shape[1] != len(base_order):
        raise ValueError("base_probabilities shape does not match base_classes")
    if specialist.ndim != 2 or specialist.shape != (base.shape[0], 3):
        raise ValueError("specialist_probabilities must have one three-class row per sample")
    if (
        not np.isfinite(base).all()
        or not np.isfinite(specialist).all()
        or np.any(base < 0)
        or np.any(specialist < 0)
    ):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(base.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("base probability rows must sum to one")
    specialist_mass = specialist.sum(axis=1, keepdims=True)
    if np.any(specialist_mass <= 0):
        raise ValueError("specialist probability rows must have positive mass")
    conditional = specialist / specialist_mass

    base_positions = {label: index for index, label in enumerate(base_order)}
    specialist_positions = {
        label: index for index, label in enumerate(specialist_order)
    }
    ceramic_indices = [base_positions[label] for label in ceramics]
    ceramic_mass = base[:, ceramic_indices].sum(axis=1)
    combined = base.copy()
    for label in ceramics:
        combined[:, base_positions[label]] = (
            ceramic_mass * conditional[:, specialist_positions[label]]
        )
    return combined


def predict_three_way_ceramic_conditional(
    base_model: ProbeModel,
    specialist: CeramicSpecialist,
    features: FeatureBlocks,
) -> Array:
    """Predict with a base model and its fixed three-way ceramic specialist."""

    if tuple(base_model.classes) == specialist.ceramic_classes:
        raise ValueError("The base model must include non-ceramic classes")
    base = base_model.predict_proba(features)
    conditional = specialist.predict_proba(features)
    return apply_ceramic_specialist(
        base,
        base_model.classes,
        conditional,
        specialist.ceramic_classes,
        specialist.ceramic_classes,
    )


@dataclass(frozen=True)
class OrdinaryMetrics:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class_precision: Mapping[str, float]
    per_class_recall: Mapping[str, float]
    per_class_f1: Mapping[str, float]
    per_class_support: Mapping[str, int]


def ordinary_metrics(
    true_labels: Sequence[str], probabilities: Array, classes: Sequence[str]
) -> OrdinaryMetrics:
    """Calculate ordinary aggregate and per-class classification metrics."""

    labels = _string_vector(true_labels, "true_labels")
    class_order = tuple(classes)
    _declared_classes(labels, class_order)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape != (len(labels), len(class_order)):
        raise ValueError("probabilities shape does not match labels and classes")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(scores.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("probability rows must sum to one")
    predictions = np.asarray(class_order, dtype=object)[np.argmax(scores, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(class_order),
        zero_division=0,
    )
    return OrdinaryMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(np.mean(f1)),
        balanced_accuracy=float(np.mean(recall)),
        per_class_precision={
            label: float(value) for label, value in zip(class_order, precision)
        },
        per_class_recall={
            label: float(value) for label, value in zip(class_order, recall)
        },
        per_class_f1={label: float(value) for label, value in zip(class_order, f1)},
        per_class_support={
            label: int(value) for label, value in zip(class_order, support)
        },
    )


@dataclass(frozen=True)
class SourceGroupMetrics:
    per_class_group_recall: Mapping[str, float]
    source_group_balanced_accuracy: float
    ceramic_worst_group_recall: float
    robustness_score: float


def harmonic_mean(first: float, second: float) -> float:
    """Return the non-negative two-value harmonic mean."""

    if not np.isfinite(first) or not np.isfinite(second) or first < 0 or second < 0:
        raise ValueError("Harmonic-mean inputs must be finite and non-negative")
    if first == 0 or second == 0:
        return 0.0
    return float(2.0 * first * second / (first + second))


def source_group_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    group_ids: Sequence[str],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> SourceGroupMetrics:
    """Calculate class-balanced source-group recall and robustness score."""

    truth = _string_vector(true_labels, "true_labels")
    predictions = _string_vector(predicted_labels, "predicted_labels")
    groups = _string_vector(group_ids, "group_ids")
    if len(truth) != len(predictions) or len(truth) != len(groups):
        raise ValueError("true_labels, predicted_labels, and group_ids must align")
    class_order = tuple(classes)
    _declared_classes(truth, class_order)
    if any(label not in set(class_order) for label in predictions):
        raise ValueError("predicted_labels contains an undeclared class")
    ceramics = tuple(ceramic_classes)
    if len(ceramics) != 3 or len(set(ceramics)) != 3 or set(ceramics) - set(class_order):
        raise ValueError("ceramic_classes must be three declared classes")

    correct = predictions == truth
    per_class: Dict[str, float] = {}
    for label in class_order:
        class_mask = truth == label
        class_groups = tuple(sorted(set(groups[class_mask].tolist())))
        group_recalls = [
            float(np.mean(correct[class_mask & (groups == group_id)]))
            for group_id in class_groups
        ]
        per_class[label] = float(np.mean(group_recalls))
    source_group_balanced_accuracy = float(np.mean(list(per_class.values())))
    ceramic_worst_group_recall = float(min(per_class[label] for label in ceramics))
    return SourceGroupMetrics(
        per_class_group_recall=per_class,
        source_group_balanced_accuracy=source_group_balanced_accuracy,
        ceramic_worst_group_recall=ceramic_worst_group_recall,
        robustness_score=harmonic_mean(
            source_group_balanced_accuracy, ceramic_worst_group_recall
        ),
    )


@dataclass(frozen=True)
class CandidateSummary:
    configuration_id: str
    robustness_score: float
    minimum_inner_fold_source_group_balanced_accuracy: float
    macro_f1: float
    inference_cost_rank: int
    specialist: str
    c_value: float
    source_group_exponent: float
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class EligibilityReference:
    macro_f1: float
    per_class_recall: Mapping[str, float]


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    failures: Tuple[str, ...]


def candidate_eligibility(
    candidate: CandidateSummary,
    reference: EligibilityReference,
    *,
    ornek_class: str,
    petrykivka_class: str,
    maximum_macro_f1_drop: float = 0.01,
    maximum_ornek_recall_drop: float = 0.02,
    maximum_petrykivka_recall_drop: float = 0.02,
) -> EligibilityResult:
    """Apply the Phase 5 non-regression constraints to one candidate."""

    limits = (
        maximum_macro_f1_drop,
        maximum_ornek_recall_drop,
        maximum_petrykivka_recall_drop,
    )
    if any(not np.isfinite(value) or value < 0 for value in limits):
        raise ValueError("Maximum metric drops must be finite and non-negative")
    for label in (ornek_class, petrykivka_class):
        if label not in candidate.per_class_recall or label not in reference.per_class_recall:
            raise ValueError(f"Eligibility recall is missing class {label!r}")
    failures = []
    if candidate.macro_f1 < reference.macro_f1 - maximum_macro_f1_drop:
        failures.append("macro_f1_drop")
    if (
        candidate.per_class_recall[ornek_class]
        < reference.per_class_recall[ornek_class] - maximum_ornek_recall_drop
    ):
        failures.append("ornek_recall_drop")
    if (
        candidate.per_class_recall[petrykivka_class]
        < reference.per_class_recall[petrykivka_class]
        - maximum_petrykivka_recall_drop
    ):
        failures.append("petrykivka_recall_drop")
    return EligibilityResult(eligible=not failures, failures=tuple(failures))


def _validate_candidate(candidate: CandidateSummary) -> None:
    if not candidate.configuration_id:
        raise ValueError("Candidate configuration_id must be non-empty")
    finite_values = (
        candidate.robustness_score,
        candidate.minimum_inner_fold_source_group_balanced_accuracy,
        candidate.macro_f1,
        candidate.c_value,
        candidate.source_group_exponent,
    )
    if any(not np.isfinite(value) for value in finite_values):
        raise ValueError("Candidate ranking values must be finite")
    if candidate.inference_cost_rank < 0:
        raise ValueError("inference_cost_rank must be non-negative")


def _tie_break_key(candidate: CandidateSummary) -> Tuple[object, ...]:
    return (
        -candidate.minimum_inner_fold_source_group_balanced_accuracy,
        -candidate.macro_f1,
        candidate.inference_cost_rank,
        0 if candidate.specialist == "none" else 1,
        0 if np.isclose(candidate.c_value, 10.0, rtol=0.0, atol=1e-12) else 1,
        candidate.source_group_exponent,
        candidate.configuration_id,
    )


def rank_candidates(
    candidates: Sequence[CandidateSummary], *, tie_window: float = 0.005
) -> Tuple[CandidateSummary, ...]:
    """Rank candidates using score bands and the frozen deterministic ties.

    Within ``tie_window`` of the best remaining robustness score, the score is
    treated as tied and the declared tie-breakers are applied in order. The
    procedure is repeated for the remaining candidates, avoiding a
    non-transitive pairwise floating-point comparator.
    """

    if not np.isfinite(tie_window) or tie_window < 0:
        raise ValueError("tie_window must be finite and non-negative")
    remaining = list(candidates)
    if not remaining:
        raise ValueError("At least one candidate is required")
    for candidate in remaining:
        _validate_candidate(candidate)
    identifiers = [candidate.configuration_id for candidate in remaining]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate configuration IDs must be unique")

    ranked = []
    while remaining:
        best_score = max(candidate.robustness_score for candidate in remaining)
        tied = [
            candidate
            for candidate in remaining
            if candidate.robustness_score >= best_score - tie_window
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


def rank_eligible_candidates(
    candidates: Sequence[CandidateSummary],
    reference: EligibilityReference,
    *,
    ornek_class: str,
    petrykivka_class: str,
    tie_window: float = 0.005,
) -> Tuple[CandidateSummary, ...]:
    """Filter by the frozen non-regression gate, then deterministically rank."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate_eligibility(
            candidate,
            reference,
            ornek_class=ornek_class,
            petrykivka_class=petrykivka_class,
        ).eligible
    ]
    if not eligible:
        raise ValueError("No candidate satisfies the eligibility constraints")
    return rank_candidates(eligible, tie_window=tie_window)

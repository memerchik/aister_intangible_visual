#!/usr/bin/env python3
"""Run the frozen Phase 5 development-only source-robustness experiment.

This runner implements the nested source-atomic selection procedure declared in
``phases/phase_05_source_robustness/experiment_contract.json``.  It can only
load the canonical development contract and the explicitly allowlisted Phase 4
embedding arrays.  Final evaluation is intentionally outside this entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SOURCE_ROOT = STEP_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import threadpoolctl  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from ornament_classifier.contracts import (  # noqa: E402
    DevelopmentContract,
    DevelopmentRecord,
    load_development_contract,
)
from ornament_classifier.embeddings import (  # noqa: E402
    EmbeddingBlock,
    load_allowlisted_embeddings,
)
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    CandidateSummary,
    EligibilityReference,
    EligibilityResult,
    OrdinaryMetrics,
    SourceGroupMetrics,
    apply_ceramic_specialist,
    candidate_eligibility,
    compute_source_group_weights,
    fit_centered_logit_ensemble,
    fit_ceramic_specialist,
    fit_logistic_probe,
    ordinary_metrics,
    rank_candidates,
    source_group_metrics,
)


PHASE = "step02_phase_5_source_robustness"
EXPECTED_CONTRACT_SHA256 = (
    "55d0dd0d3665b4b84b33a9c9461763c6858452bd0b10b679fd9b77fbe630a87f"
)
# SHA-256 over canonical Phase 4 ``image_id\x1fpredicted_class\n`` rows.  The
# source OOF artifact SHA-256 was
# 6b3bc81bf37bebe03e58665be9d7b60c826cd0994e9ca386cc6b1cb2295edcd1.
# The runner pins the decision fingerprint directly and never opens that
# development-error artifact during Phase 5.
EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256 = (
    "d6f5b90c2c6b8af39fb8e6bffbe3e44ba3e6d99b3f8f069b8f1febfe95bab307"
)
CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract.json"
)
SCRIPT_PATH = Path(__file__).resolve()
MAX_ITERATIONS = 10_000
TOLERANCE = 1e-6

FeatureInput = Union[np.ndarray, Tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class FeatureFamily:
    family_id: str
    kind: str
    block_names: Tuple[str, ...]
    blocks: Tuple[np.ndarray, ...]
    inference_cost_rank: int


@dataclass(frozen=True)
class BaseConfiguration:
    configuration_id: str
    family: FeatureFamily
    c_value: float
    source_group_exponent: float


@dataclass(frozen=True)
class OOFResult:
    probabilities: np.ndarray
    evaluated_indices: np.ndarray
    fold_rows: Tuple[Mapping[str, object], ...]
    ordinary: OrdinaryMetrics
    source: SourceGroupMetrics
    summary: CandidateSummary
    maximum_iterations: int
    fit_count: int


@dataclass(frozen=True)
class SearchSelection:
    selected: OOFResult
    reference: EligibilityReference
    eligibility: Mapping[str, EligibilityResult]
    ranked_eligible_ids: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the frozen contract directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def code_provenance() -> Mapping[str, object]:
    """Fingerprint every local implementation input used by the Phase 5 run."""

    paths = (
        SCRIPT_PATH,
        STEP_ROOT / "src" / "ornament_classifier" / "robustness.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings.py",
        STEP_ROOT / "src" / "ornament_classifier" / "contracts.py",
        STEP_ROOT / "src" / "ornament_classifier" / "paths.py",
        STEP_ROOT / "requirements-phase5.txt",
    )
    files: Dict[str, str] = {}
    combined = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Phase 5 code-provenance input is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        combined.update(f"{relative}\x1f{digest}\n".encode("utf-8"))
    return {
        "files_sha256": files,
        "combined_sha256": combined.hexdigest(),
    }


def load_frozen_contract(paths: ProjectPaths) -> Tuple[Mapping[str, object], Path]:
    path = (paths.step_root / CONTRACT_RELATIVE_PATH).resolve()
    if not path.is_file():
        raise ValueError(f"Frozen Phase 5 experiment contract is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "Phase 5 experiment contract changed after it was frozen: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {actual_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load frozen Phase 5 experiment contract: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Frozen Phase 5 experiment contract must be a JSON object")
    if payload.get("status") != "frozen_before_fit":
        raise ValueError("Phase 5 experiment contract is not frozen before fit")
    scope = _mapping(payload, "scope")
    if scope.get("sealed_test_access") != "forbidden":
        raise ValueError("Phase 5 requires final-evaluation access to remain forbidden")
    return payload, path


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Contract field {key!r} must be an object")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Contract field {key!r} must be an array")
    return value


def _float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Contract field {key!r} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Contract field {key!r} must be finite")
    return result


def _int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Contract field {key!r} must be an integer")
    return value


def _text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Contract field {key!r} must be a non-empty string")
    return value


def _resolve_output_dir(
    paths: ProjectPaths,
    contract: Mapping[str, object],
    requested: Optional[Path],
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    relative = Path(_text(_mapping(contract, "output_contract"), "directory"))
    if relative.is_absolute() or not relative.parts:
        raise ValueError("Output-contract directory must be relative")
    repository_root = paths.step_root.parent.resolve()
    output_dir = (repository_root / relative).resolve()
    try:
        output_dir.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("Output-contract directory escapes the repository") from error
    return output_dir


def _validate_development(
    development: DevelopmentContract,
    contract: Mapping[str, object],
) -> None:
    scope = _mapping(contract, "scope")
    expected_count = _int(scope, "development_image_count")
    if len(development.records) != expected_count:
        raise ValueError("Development row count disagrees with Phase 5")
    expected_folds = tuple(str(value) for value in _sequence(scope, "outer_folds"))
    if development.fold_ids != expected_folds:
        raise ValueError("Development folds disagree with Phase 5")
    if any(
        record.source_atomic_split_group_id != record.split_group_id
        for record in development.records
    ):
        raise ValueError("Canonical split group must equal the source-atomic group")
    groups_to_folds: Dict[str, set] = {}
    for record in development.records:
        groups_to_folds.setdefault(record.source_atomic_split_group_id, set()).add(
            record.cv_fold
        )
    leaking = sorted(group for group, folds in groups_to_folds.items() if len(folds) != 1)
    if leaking:
        raise ValueError(
            "Source-atomic groups cross development folds: " + ", ".join(leaking[:5])
        )


def _embedding_requests(contract: Mapping[str, object]) -> Mapping[str, Tuple[str, str]]:
    requests: Dict[str, Tuple[str, str]] = {}
    for family in _sequence(contract, "base_feature_families"):
        if not isinstance(family, dict):
            raise ValueError("Each feature family must be an object")
        for reference in _sequence(family, "blocks"):
            if not isinstance(reference, str) or reference.count(".") != 1:
                raise ValueError(f"Invalid embedding block reference: {reference!r}")
            cache_name, array_name = reference.split(".", 1)
            requests[reference] = (cache_name, array_name)
    return requests


def build_feature_families(
    contract: Mapping[str, object],
    embedding_blocks: Mapping[str, EmbeddingBlock],
) -> Tuple[FeatureFamily, ...]:
    families: List[FeatureFamily] = []
    seen = set()
    for spec in _sequence(contract, "base_feature_families"):
        if not isinstance(spec, dict):
            raise ValueError("Each feature family must be an object")
        family_id = _text(spec, "id")
        kind = _text(spec, "kind")
        if family_id in seen:
            raise ValueError(f"Duplicate feature family: {family_id}")
        seen.add(family_id)
        block_names = tuple(str(value) for value in _sequence(spec, "blocks"))
        if not block_names or any(name not in embedding_blocks for name in block_names):
            raise ValueError(f"Feature family has unavailable blocks: {family_id}")
        blocks = tuple(embedding_blocks[name].values for name in block_names)
        if kind == "single_embedding" and len(blocks) != 1:
            raise ValueError("A single-embedding family must have exactly one block")
        if kind == "centered_logit_ensemble" and len(blocks) < 2:
            raise ValueError("A centered-logit ensemble must have multiple blocks")
        if kind not in ("single_embedding", "centered_logit_ensemble"):
            raise ValueError(f"Unsupported feature-family kind: {kind}")
        families.append(
            FeatureFamily(
                family_id=family_id,
                kind=kind,
                block_names=block_names,
                blocks=blocks,
                inference_cost_rank=_int(spec, "inference_cost_rank"),
            )
        )
    return tuple(families)


def _number_id(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(value, "g").replace(".", "p")


def build_base_configurations(
    contract: Mapping[str, object],
    families: Sequence[FeatureFamily],
) -> Tuple[BaseConfiguration, ...]:
    grid = _mapping(contract, "base_grid")
    c_values = tuple(float(value) for value in _sequence(grid, "c_values"))
    exponents = tuple(
        float(value) for value in _sequence(grid, "source_group_exponents")
    )
    configurations = tuple(
        BaseConfiguration(
            configuration_id=(
                f"{family.family_id}__c{_number_id(c_value)}"
                f"__source{_number_id(exponent)}"
            ),
            family=family,
            c_value=c_value,
            source_group_exponent=exponent,
        )
        for family in families
        for c_value in c_values
        for exponent in exponents
    )
    expected_count = _int(grid, "configuration_count")
    if len(configurations) != expected_count:
        raise ValueError(
            f"Base grid has {len(configurations)} configurations, expected {expected_count}"
        )
    return configurations


def _classes(records: Sequence[DevelopmentRecord]) -> Tuple[str, ...]:
    classes = tuple(sorted({record.ornament_label for record in records}))
    if len(classes) != 5:
        raise ValueError(f"Phase 5 requires exactly five classes, observed {len(classes)}")
    return classes


def _record_arrays(
    records: Sequence[DevelopmentRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([record.ornament_label for record in records], dtype=object)
    groups = np.asarray(
        [record.source_atomic_split_group_id for record in records], dtype=object
    )
    folds = np.asarray([record.cv_fold for record in records], dtype=object)
    return labels, groups, folds


def _assert_source_disjoint_partitions(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    groups: np.ndarray,
    context: str,
) -> None:
    """Make every modelling boundary actively prove source-group isolation."""

    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError(f"Training and validation rows overlap in {context}")
    train_groups = set(groups[train_indices].tolist())
    validation_groups = set(groups[validation_indices].tolist())
    overlap = sorted(train_groups & validation_groups)
    if overlap:
        raise ValueError(
            f"Source-atomic groups cross the {context} boundary: "
            + ", ".join(overlap[:5])
        )


def _slice_features(family: FeatureFamily, indices: np.ndarray) -> FeatureInput:
    sliced = tuple(block[indices] for block in family.blocks)
    return sliced[0] if family.kind == "single_embedding" else sliced


def _fit_base_model(
    configuration: BaseConfiguration,
    train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    seed: int,
):
    train_labels = labels[train_indices]
    weights = compute_source_group_weights(
        train_labels,
        groups[train_indices],
        configuration.source_group_exponent,
    )
    features = _slice_features(configuration.family, train_indices)
    with threadpool_limits(limits=1):
        if configuration.family.kind == "centered_logit_ensemble":
            return fit_centered_logit_ensemble(
                features,  # type: ignore[arg-type]
                train_labels,
                c_value=configuration.c_value,
                sample_weight=weights,
                classes=classes,
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        return fit_logistic_probe(
            features,  # type: ignore[arg-type]
            train_labels,
            c_value=configuration.c_value,
            sample_weight=weights,
            classes=classes,
            seed=seed,
            max_iter=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )


def _flatten_iterations(values: object) -> Tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        return (int(values),)
    if not isinstance(values, (tuple, list)):
        raise ValueError("Model iteration metadata has an invalid type")
    flattened: List[int] = []
    for value in values:
        flattened.extend(_flatten_iterations(value))
    if not flattened:
        raise ValueError("Model iteration metadata is empty")
    return tuple(flattened)


def _predict_labels(probabilities: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    return np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]


def _candidate_summary(
    configuration_id: str,
    configuration: BaseConfiguration,
    specialist: str,
    truth: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> Tuple[OrdinaryMetrics, SourceGroupMetrics, CandidateSummary, Tuple[Mapping[str, object], ...]]:
    predictions = _predict_labels(probabilities, classes)
    ordinary = ordinary_metrics(truth, probabilities, classes)
    source = source_group_metrics(
        truth, predictions, groups, classes, ceramic_classes
    )
    fold_rows: List[Mapping[str, object]] = []
    fold_group_scores = []
    for fold_id in sorted(set(folds.tolist())):
        mask = folds == fold_id
        fold_ordinary = ordinary_metrics(truth[mask], probabilities[mask], classes)
        fold_source = source_group_metrics(
            truth[mask], predictions[mask], groups[mask], classes, ceramic_classes
        )
        fold_group_scores.append(fold_source.source_group_balanced_accuracy)
        fold_rows.append(
            {
                "cv_fold": fold_id,
                "validation_examples": int(mask.sum()),
                "accuracy": fold_ordinary.accuracy,
                "macro_f1": fold_ordinary.macro_f1,
                "balanced_accuracy": fold_ordinary.balanced_accuracy,
                "source_group_balanced_accuracy": (
                    fold_source.source_group_balanced_accuracy
                ),
                "ceramic_worst_group_recall": fold_source.ceramic_worst_group_recall,
                "robustness_score": fold_source.robustness_score,
            }
        )
    summary = CandidateSummary(
        configuration_id=configuration_id,
        robustness_score=source.robustness_score,
        minimum_inner_fold_source_group_balanced_accuracy=float(min(fold_group_scores)),
        macro_f1=ordinary.macro_f1,
        inference_cost_rank=configuration.family.inference_cost_rank,
        specialist=specialist,
        c_value=configuration.c_value,
        source_group_exponent=configuration.source_group_exponent,
        per_class_recall=ordinary.per_class_recall,
    )
    return ordinary, source, summary, tuple(fold_rows)


def evaluate_base_oof(
    configuration: BaseConfiguration,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    seed: int,
) -> OOFResult:
    """Make pooled OOF predictions within the supplied fold universe."""

    labels, groups, folds = _record_arrays(records)
    fold_ids = tuple(eligible_fold_ids)
    if len(fold_ids) < 2 or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("OOF evaluation requires at least two unique folds")
    pool_mask = np.isin(folds, fold_ids)
    evaluated_indices = np.flatnonzero(pool_mask)
    probabilities = np.full((len(records), len(classes)), np.nan, dtype=np.float64)
    fit_iterations: List[int] = []
    for validation_fold in fold_ids:
        validation_indices = np.flatnonzero(pool_mask & (folds == validation_fold))
        train_indices = np.flatnonzero(pool_mask & (folds != validation_fold))
        if not len(validation_indices) or not len(train_indices):
            raise ValueError(f"Fold {validation_fold!r} has an empty OOF partition")
        _assert_source_disjoint_partitions(
            train_indices,
            validation_indices,
            groups,
            f"base OOF fold {validation_fold}",
        )
        model = _fit_base_model(
            configuration, train_indices, labels, groups, classes, seed
        )
        if not model.converged:
            raise RuntimeError(f"Base model did not converge: {configuration.configuration_id}")
        fit_iterations.extend(_flatten_iterations(model.iterations))
        probabilities[validation_indices] = model.predict_proba(
            _slice_features(configuration.family, validation_indices)
        )
    pooled = probabilities[evaluated_indices]
    if not np.isfinite(pooled).all():
        raise RuntimeError("OOF predictions are incomplete or non-finite")
    ordinary, source, summary, fold_rows = _candidate_summary(
        configuration.configuration_id,
        configuration,
        "none",
        labels[evaluated_indices],
        groups[evaluated_indices],
        folds[evaluated_indices],
        pooled,
        classes,
        ceramic_classes,
    )
    return OOFResult(
        probabilities=pooled,
        evaluated_indices=evaluated_indices,
        fold_rows=fold_rows,
        ordinary=ordinary,
        source=source,
        summary=summary,
        maximum_iterations=max(fit_iterations),
        fit_count=len(fit_iterations),
    )


def evaluate_specialist_oof(
    base_result: OOFResult,
    configuration: BaseConfiguration,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    seed: int,
) -> OOFResult:
    """Apply fold-local ceramic specialists to an existing base OOF result."""

    labels, groups, folds = _record_arrays(records)
    fold_ids = tuple(eligible_fold_ids)
    evaluated_indices = np.flatnonzero(np.isin(folds, fold_ids))
    if not np.array_equal(evaluated_indices, base_result.evaluated_indices):
        raise ValueError("Specialist and base OOF row scopes disagree")
    index_positions = {
        int(global_index): position
        for position, global_index in enumerate(evaluated_indices.tolist())
    }
    probabilities = np.array(base_result.probabilities, copy=True)
    fit_iterations: List[int] = []
    pool_mask = np.isin(folds, fold_ids)
    for validation_fold in fold_ids:
        validation_indices = np.flatnonzero(pool_mask & (folds == validation_fold))
        train_indices = np.flatnonzero(pool_mask & (folds != validation_fold))
        _assert_source_disjoint_partitions(
            train_indices,
            validation_indices,
            groups,
            f"specialist OOF fold {validation_fold}",
        )
        train_features = _slice_features(configuration.family, train_indices)
        validation_features = _slice_features(configuration.family, validation_indices)
        with threadpool_limits(limits=1):
            specialist = fit_ceramic_specialist(
                train_features,
                labels[train_indices],
                groups[train_indices],
                ceramic_classes=ceramic_classes,
                c_value=configuration.c_value,
                source_group_exponent=configuration.source_group_exponent,
                centered_logit_ensemble=(
                    configuration.family.kind == "centered_logit_ensemble"
                ),
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        if not specialist.converged:
            raise RuntimeError("Ceramic specialist did not converge")
        fit_iterations.extend(_flatten_iterations(specialist.iterations))
        specialist_probabilities = specialist.predict_proba(validation_features)
        local_positions = np.asarray(
            [index_positions[int(index)] for index in validation_indices], dtype=np.int64
        )
        probabilities[local_positions] = apply_ceramic_specialist(
            probabilities[local_positions],
            classes,
            specialist_probabilities,
            specialist.ceramic_classes,
            ceramic_classes,
        )
    identifier = configuration.configuration_id + "__specialist_three_way_conditional"
    ordinary, source, summary, fold_rows = _candidate_summary(
        identifier,
        configuration,
        "three_way_conditional",
        labels[evaluated_indices],
        groups[evaluated_indices],
        folds[evaluated_indices],
        probabilities,
        classes,
        ceramic_classes,
    )
    return OOFResult(
        probabilities=probabilities,
        evaluated_indices=evaluated_indices,
        fold_rows=fold_rows,
        ordinary=ordinary,
        source=source,
        summary=summary,
        maximum_iterations=max(fit_iterations),
        fit_count=len(fit_iterations),
    )


def _reference_configuration(
    configurations: Sequence[BaseConfiguration],
    contract: Mapping[str, object],
) -> BaseConfiguration:
    reference = _mapping(_mapping(contract, "base_grid"), "exact_phase_4_reference")
    family_id = _text(reference, "feature_family")
    c_value = _float(reference, "c_value")
    exponent = _float(reference, "source_group_exponent")
    matches = [
        configuration
        for configuration in configurations
        if configuration.family.family_id == family_id
        and np.isclose(configuration.c_value, c_value, rtol=0.0, atol=1e-12)
        and np.isclose(
            configuration.source_group_exponent, exponent, rtol=0.0, atol=1e-12
        )
    ]
    if len(matches) != 1:
        raise ValueError("Exact Phase 4 reference is not unique in the base grid")
    return matches[0]


def validate_phase4_reference_metrics(
    reference_result: OOFResult,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    contract: Mapping[str, object],
    paths: ProjectPaths,
) -> Mapping[str, object]:
    """Prove that the full-development exact reference preserves Phase 4 labels.

    The Phase 5 solver uses the tighter tolerance declared by this runner, so
    probabilities need not be byte-identical to Phase 4.  Selection depends on
    class decisions and metrics, however; those must remain identical.  Only
    the Phase 4 metrics file pinned by the frozen input contract is read here.
    """

    input_contract = _mapping(contract, "input_contract")
    relative = Path(_text(input_contract, "phase_4_metrics"))
    if relative.is_absolute() or not relative.parts:
        raise ValueError("Pinned Phase 4 metrics path must be relative")
    repository_root = paths.step_root.parent.resolve()
    path = (repository_root / relative).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("Pinned Phase 4 metrics path escapes the repository") from error
    expected_hash = _text(input_contract, "phase_4_metrics_sha256")
    if sha256_file(path) != expected_hash:
        raise ValueError("Pinned Phase 4 metrics hash changed")
    try:
        phase4 = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read pinned Phase 4 metrics: {error}") from error
    if not isinstance(phase4, dict) or tuple(phase4.get("class_names", ())) != tuple(classes):
        raise ValueError("Exact reference class order disagrees with Phase 4")

    tolerance = 1e-12

    def assert_close(name: str, actual: float, expected: object) -> None:
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise ValueError(f"Pinned Phase 4 metric is invalid: {name}")
        if not np.isclose(actual, float(expected), rtol=0.0, atol=tolerance):
            raise ValueError(
                f"Exact reference does not reproduce Phase 4 {name}: "
                f"{actual} != {expected}"
            )

    aggregate = phase4.get("aggregate_oof_metrics")
    if not isinstance(aggregate, dict):
        raise ValueError("Pinned Phase 4 aggregate metrics are missing")
    assert_close("accuracy", reference_result.ordinary.accuracy, aggregate.get("accuracy"))
    assert_close(
        "balanced_accuracy",
        reference_result.ordinary.balanced_accuracy,
        aggregate.get("balanced_accuracy"),
    )
    assert_close("macro_f1", reference_result.ordinary.macro_f1, aggregate.get("macro_f1"))

    phase4_per_class = phase4.get("per_class_metrics")
    if not isinstance(phase4_per_class, list):
        raise ValueError("Pinned Phase 4 per-class metrics are missing")
    expected_by_class = {
        row.get("class_name"): row
        for row in phase4_per_class
        if isinstance(row, dict) and isinstance(row.get("class_name"), str)
    }
    if set(expected_by_class) != set(classes):
        raise ValueError("Pinned Phase 4 per-class labels disagree with Phase 5")
    for label in classes:
        expected = expected_by_class[label]
        if reference_result.ordinary.per_class_support[label] != expected.get("support"):
            raise ValueError(f"Exact reference support disagrees for {label}")
        assert_close(
            f"precision:{label}",
            reference_result.ordinary.per_class_precision[label],
            expected.get("precision"),
        )
        assert_close(
            f"recall:{label}",
            reference_result.ordinary.per_class_recall[label],
            expected.get("recall"),
        )
        assert_close(
            f"f1:{label}",
            reference_result.ordinary.per_class_f1[label],
            expected.get("f1"),
        )

    phase4_folds = phase4.get("selected_fold_metrics")
    if not isinstance(phase4_folds, list):
        raise ValueError("Pinned Phase 4 fold metrics are missing")
    expected_by_fold = {
        str(row.get("cv_fold")): row for row in phase4_folds if isinstance(row, dict)
    }
    observed_by_fold = {str(row["cv_fold"]): row for row in reference_result.fold_rows}
    if set(observed_by_fold) != set(expected_by_fold):
        raise ValueError("Exact reference fold coverage disagrees with Phase 4")
    for fold_id in sorted(observed_by_fold):
        observed = observed_by_fold[fold_id]
        expected = expected_by_fold[fold_id]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert_close(
                f"fold_{fold_id}:{metric}",
                float(observed[metric]),
                expected.get(metric),
            )
    predictions = _predict_labels(reference_result.probabilities, classes)
    digest = hashlib.sha256()
    for index, prediction in zip(reference_result.evaluated_indices, predictions):
        digest.update(
            f"{records[int(index)].image_id}\x1f{prediction}\n".encode("utf-8")
        )
    observed_prediction_sha256 = digest.hexdigest()
    if observed_prediction_sha256 != EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256:
        raise ValueError(
            "Exact reference per-image argmax decisions disagree with Phase 4"
        )
    return {
        "validated": True,
        "comparison_scope": (
            "class_order_per_image_argmax_and_aggregate_fold_and_per_class_metrics"
        ),
        "probability_identity_required": False,
        "absolute_tolerance": tolerance,
        "expected_prediction_fingerprint_sha256": (
            EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256
        ),
        "observed_prediction_fingerprint_sha256": observed_prediction_sha256,
    }


def _eligibility_limits(contract: Mapping[str, object]) -> Mapping[str, float]:
    limits = _mapping(
        _mapping(contract, "selection_rule"),
        "eligibility_relative_to_exact_reference",
    )
    return {
        "maximum_macro_f1_drop": _float(limits, "maximum_macro_f1_drop"),
        "maximum_ornek_recall_drop": _float(limits, "maximum_ornek_recall_drop"),
        "maximum_petrykivka_recall_drop": _float(
            limits, "maximum_petrykivka_recall_drop"
        ),
    }


def select_search_candidates(
    candidates: Sequence[OOFResult],
    reference_result: OOFResult,
    contract: Mapping[str, object],
    *,
    ornek_class: str,
    petrykivka_class: str,
) -> SearchSelection:
    reference = EligibilityReference(
        macro_f1=reference_result.ordinary.macro_f1,
        per_class_recall=reference_result.ordinary.per_class_recall,
    )
    limits = _eligibility_limits(contract)
    eligibility: Dict[str, EligibilityResult] = {}
    eligible_summaries = []
    results_by_id = {result.summary.configuration_id: result for result in candidates}
    if len(results_by_id) != len(candidates):
        raise ValueError("Candidate result identifiers must be unique")
    for result in candidates:
        decision = candidate_eligibility(
            result.summary,
            reference,
            ornek_class=ornek_class,
            petrykivka_class=petrykivka_class,
            **limits,
        )
        eligibility[result.summary.configuration_id] = decision
        if decision.eligible:
            eligible_summaries.append(result.summary)
    if not eligible_summaries:
        raise ValueError("No candidate satisfies the frozen eligibility constraints")
    tie_window = _float(_mapping(contract, "selection_rule"), "tie_window")
    ranked = rank_candidates(eligible_summaries, tie_window=tie_window)
    ranked_ids = tuple(summary.configuration_id for summary in ranked)
    return SearchSelection(
        selected=results_by_id[ranked_ids[0]],
        reference=reference,
        eligibility=eligibility,
        ranked_eligible_ids=ranked_ids,
    )


def _no_specialist_result(base: OOFResult) -> OOFResult:
    summary = CandidateSummary(
        configuration_id=base.summary.configuration_id + "__specialist_none",
        robustness_score=base.summary.robustness_score,
        minimum_inner_fold_source_group_balanced_accuracy=(
            base.summary.minimum_inner_fold_source_group_balanced_accuracy
        ),
        macro_f1=base.summary.macro_f1,
        inference_cost_rank=base.summary.inference_cost_rank,
        specialist="none",
        c_value=base.summary.c_value,
        source_group_exponent=base.summary.source_group_exponent,
        per_class_recall=base.summary.per_class_recall,
    )
    return OOFResult(
        probabilities=base.probabilities,
        evaluated_indices=base.evaluated_indices,
        fold_rows=base.fold_rows,
        ordinary=base.ordinary,
        source=base.source,
        summary=summary,
        maximum_iterations=base.maximum_iterations,
        fit_count=base.fit_count,
    )


def _configuration_for_result(
    result: OOFResult, configurations: Sequence[BaseConfiguration]
) -> BaseConfiguration:
    matches = [
        configuration
        for configuration in configurations
        if result.summary.configuration_id.startswith(configuration.configuration_id)
    ]
    if len(matches) != 1:
        raise ValueError("Cannot resolve a selected result to one base configuration")
    return matches[0]


def _selection_rows(
    scope: str,
    stage: str,
    results: Sequence[OOFResult],
    selection: SearchSelection,
    reference_id: str,
    outer_fold: str = "",
) -> List[Dict[str, object]]:
    eligible_rank = {
        identifier: rank
        for rank, identifier in enumerate(selection.ranked_eligible_ids, start=1)
    }
    all_ranked = rank_candidates([result.summary for result in results])
    robustness_rank = {
        summary.configuration_id: rank
        for rank, summary in enumerate(all_ranked, start=1)
    }
    rows: List[Dict[str, object]] = []
    for result in results:
        summary = result.summary
        decision = selection.eligibility[summary.configuration_id]
        rows.append(
            {
                "scope": scope,
                "outer_fold": outer_fold,
                "stage": stage,
                "configuration_id": summary.configuration_id,
                "feature_family": _configuration_feature_id(summary.configuration_id),
                "specialist": summary.specialist,
                "c_value": summary.c_value,
                "source_group_exponent": summary.source_group_exponent,
                "inference_cost_rank": summary.inference_cost_rank,
                "is_exact_reference": summary.configuration_id
                in (reference_id, reference_id + "__specialist_none"),
                "eligible": decision.eligible,
                "eligibility_failures": "|".join(decision.failures),
                "robustness_rank": robustness_rank[summary.configuration_id],
                "eligible_selection_rank": eligible_rank.get(summary.configuration_id, ""),
                "selected": (
                    summary.configuration_id
                    == selection.selected.summary.configuration_id
                ),
                "evaluated_examples": len(result.evaluated_indices),
                "fit_count": result.fit_count,
                "maximum_iterations": result.maximum_iterations,
                "accuracy": result.ordinary.accuracy,
                "macro_f1": result.ordinary.macro_f1,
                "balanced_accuracy": result.ordinary.balanced_accuracy,
                "source_group_balanced_accuracy": (
                    result.source.source_group_balanced_accuracy
                ),
                "ceramic_worst_group_recall": (
                    result.source.ceramic_worst_group_recall
                ),
                "robustness_score": result.source.robustness_score,
                "minimum_fold_source_group_balanced_accuracy": (
                    summary.minimum_inner_fold_source_group_balanced_accuracy
                ),
                "per_class_recall_json": stable_json(
                    dict(result.ordinary.per_class_recall)
                ),
                "per_class_group_recall_json": stable_json(
                    dict(result.source.per_class_group_recall)
                ),
            }
        )
    return rows


def _configuration_feature_id(identifier: str) -> str:
    marker = "__c"
    return identifier.split(marker, 1)[0]


def _fit_outer_and_predict(
    configuration: BaseConfiguration,
    specialist_name: str,
    outer_fold: str,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    labels, groups, folds = _record_arrays(records)
    train_indices = np.flatnonzero(folds != outer_fold)
    validation_indices = np.flatnonzero(folds == outer_fold)
    _assert_source_disjoint_partitions(
        train_indices,
        validation_indices,
        groups,
        f"outer fold {outer_fold}",
    )
    model = _fit_base_model(
        configuration, train_indices, labels, groups, classes, seed
    )
    fit_iterations = list(_flatten_iterations(model.iterations))
    validation_features = _slice_features(configuration.family, validation_indices)
    probabilities = model.predict_proba(validation_features)
    if specialist_name == "three_way_conditional":
        with threadpool_limits(limits=1):
            specialist = fit_ceramic_specialist(
                _slice_features(configuration.family, train_indices),
                labels[train_indices],
                groups[train_indices],
                ceramic_classes=ceramic_classes,
                c_value=configuration.c_value,
                source_group_exponent=configuration.source_group_exponent,
                centered_logit_ensemble=(
                    configuration.family.kind == "centered_logit_ensemble"
                ),
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        fit_iterations.extend(_flatten_iterations(specialist.iterations))
        probabilities = apply_ceramic_specialist(
            probabilities,
            classes,
            specialist.predict_proba(validation_features),
            specialist.ceramic_classes,
            ceramic_classes,
        )
    elif specialist_name != "none":
        raise ValueError(f"Unsupported specialist: {specialist_name}")
    return validation_indices, probabilities, tuple(fit_iterations)


def _prediction_rows(
    records: Sequence[DevelopmentRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    evaluation_status: str,
    selected_by_fold: Optional[Mapping[str, Tuple[str, str]]] = None,
    selected_recipe: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, object]]:
    predictions = _predict_labels(probabilities, classes)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rows: List[Dict[str, object]] = []
    for position, global_index in enumerate(indices.tolist()):
        record = records[global_index]
        if selected_by_fold is not None:
            base_id, specialist = selected_by_fold[record.cv_fold]
        elif selected_recipe is not None:
            base_id, specialist = selected_recipe
        else:
            raise ValueError("Prediction provenance is missing")
        row: Dict[str, object] = {
            "image_id": record.image_id,
            "evaluation_status": evaluation_status,
            "relative_path": record.relative_path,
            "true_class": record.ornament_label,
            "predicted_class": str(predictions[position]),
            "is_correct": bool(predictions[position] == record.ornament_label),
            "uncalibrated_max_probability": float(probabilities[position].max()),
            "top_1_class": classes[int(order[position, 0])],
            "top_2_class": classes[int(order[position, 1])],
            "top_3_class": classes[int(order[position, 2])],
            "cv_fold": record.cv_fold,
            "production_split": record.production_split,
            "source_atomic_split_group_id": record.source_atomic_split_group_id,
            "source_atomic_cohort_ids": record.source_atomic_cohort_ids,
            "object_type": record.object_type,
            "motif_visibility": record.motif_visibility,
            "selected_base_configuration_id": base_id,
            "selected_specialist": specialist,
        }
        for class_index, class_name in enumerate(classes):
            row[f"probability_{class_name}"] = float(
                probabilities[position, class_index]
            )
        rows.append(row)
    return rows


def _per_class_rows(
    ordinary: OrdinaryMetrics, classes: Sequence[str], scope: str
) -> List[Dict[str, object]]:
    return [
        {
            "evaluation_scope": scope,
            "class_index": index,
            "class_name": label,
            "precision": ordinary.per_class_precision[label],
            "recall": ordinary.per_class_recall[label],
            "f1": ordinary.per_class_f1[label],
            "support": ordinary.per_class_support[label],
        }
        for index, label in enumerate(classes)
    ]


def _slice_row(
    dimension: str,
    value: str,
    mask: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    count = int(mask.sum())
    correct = int(np.sum(predictions[mask] == truth[mask])) if count else 0
    return {
        "dimension": dimension,
        "value": value,
        "image_count": count,
        "correct_count": correct,
        "error_count": count - correct,
        "accuracy": float(correct / count) if count else "",
        "mean_uncalibrated_max_probability": (
            float(np.mean(np.max(probabilities[mask], axis=1))) if count else ""
        ),
    }


def diagnostic_slice_rows(
    records: Sequence[DevelopmentRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    contract: Mapping[str, object],
) -> List[Dict[str, object]]:
    selected_records = [records[index] for index in indices]
    truth = np.asarray([record.ornament_label for record in selected_records], dtype=object)
    predictions = _predict_labels(probabilities, classes)
    folds = np.asarray([record.cv_fold for record in selected_records], dtype=object)
    cohorts = np.asarray(
        [record.source_atomic_cohort_ids for record in selected_records], dtype=object
    )
    objects = np.asarray([record.object_type for record in selected_records], dtype=object)
    visibility = np.asarray(
        [record.motif_visibility for record in selected_records], dtype=object
    )
    definitions = _mapping(contract, "diagnostic_slice_definitions")
    hard_cohorts = tuple(
        str(value) for value in _sequence(definitions, "hard_three_opishnyan_cohorts")
    )
    ceramic_classes = tuple(
        str(value) for value in _sequence(definitions, "ceramic_classes")
    )
    opishnyan = ceramic_classes[0]
    rows: List[Dict[str, object]] = []
    for fold_id in sorted(set(folds.tolist())):
        rows.append(
            _slice_row(
                "cv_fold", fold_id, folds == fold_id, truth, predictions, probabilities
            )
        )
    for cohort in sorted(value for value in set(cohorts.tolist()) if value):
        rows.append(
            _slice_row(
                "named_source_cohort",
                cohort,
                cohorts == cohort,
                truth,
                predictions,
                probabilities,
            )
        )
    for value in sorted(set(objects.tolist())):
        rows.append(
            _slice_row(
                "object_type", value, objects == value, truth, predictions, probabilities
            )
        )
    for value in sorted(set(visibility.tolist())):
        rows.append(
            _slice_row(
                "motif_visibility",
                value,
                visibility == value,
                truth,
                predictions,
                probabilities,
            )
        )
    hard_mask = np.isin(cohorts, hard_cohorts) & (truth == opishnyan)
    challenging_mask = (truth == opishnyan) & (np.isin(cohorts, hard_cohorts) | (cohorts == ""))
    rows.append(
        _slice_row(
            "contract_slice",
            "hard_three_opishnyan_pooled",
            hard_mask,
            truth,
            predictions,
            probabilities,
        )
    )
    rows.append(
        _slice_row(
            "contract_slice",
            "challenging_opishnyan",
            challenging_mask,
            truth,
            predictions,
            probabilities,
        )
    )
    large_minimum = _int(_mapping(contract, "promotion_gates"), "large_named_source_minimum_size")
    large_names = [
        cohort
        for cohort in sorted(value for value in set(cohorts.tolist()) if value)
        if int(np.sum(cohorts == cohort)) >= large_minimum
    ]
    rows.append(
        _slice_row(
            "contract_slice",
            "large_named_sources_pooled",
            np.isin(cohorts, large_names),
            truth,
            predictions,
            probabilities,
        )
    )
    return rows


def _accuracy_for_mask(
    truth: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> Optional[float]:
    return float(np.mean(truth[mask] == predictions[mask])) if np.any(mask) else None


def evaluate_promotion_gates(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
    ordinary: OrdinaryMetrics,
    outer_fold_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    stability_passed: bool,
) -> Mapping[str, object]:
    """Evaluate every frozen promotion gate on nested outer OOF predictions."""

    gates = _mapping(contract, "promotion_gates")
    definitions = _mapping(contract, "diagnostic_slice_definitions")
    ceramics = tuple(str(value) for value in _sequence(definitions, "ceramic_classes"))
    opishnyan, bubnivka, _ = ceramics
    hard_cohorts = tuple(
        str(value) for value in _sequence(definitions, "hard_three_opishnyan_cohorts")
    )
    truth = np.asarray([record.ornament_label for record in records], dtype=object)
    cohorts = np.asarray(
        [record.source_atomic_cohort_ids for record in records], dtype=object
    )
    predictions = _predict_labels(probabilities, classes)
    hard_mask = (truth == opishnyan) & np.isin(cohorts, hard_cohorts)
    challenging_mask = (truth == opishnyan) & (
        np.isin(cohorts, hard_cohorts) | (cohorts == "")
    )
    hard_accuracies = [
        _accuracy_for_mask(truth, predictions, (truth == opishnyan) & (cohorts == cohort))
        for cohort in hard_cohorts
    ]
    if any(value is None for value in hard_accuracies):
        hard_macro = None
    else:
        hard_macro = float(np.mean([float(value) for value in hard_accuracies]))
    large_minimum = _int(gates, "large_named_source_minimum_size")
    large_names = [
        cohort
        for cohort in sorted(value for value in set(cohorts.tolist()) if value)
        if int(np.sum(cohorts == cohort)) >= large_minimum
    ]
    large_accuracies = [
        _accuracy_for_mask(truth, predictions, cohorts == cohort) for cohort in large_names
    ]
    large_macro = (
        float(np.mean([float(value) for value in large_accuracies]))
        if large_accuracies and all(value is not None for value in large_accuracies)
        else None
    )
    fold_accuracies = np.asarray(
        [float(row["accuracy"]) for row in outer_fold_rows], dtype=np.float64
    )
    ceramic_macro_f1 = float(np.mean([ordinary.per_class_f1[label] for label in ceramics]))
    opish_to_bub = int(np.sum((truth == opishnyan) & (predictions == bubnivka)))

    results: List[Dict[str, object]] = []

    def minimum_gate(name: str, actual: Optional[float], threshold: float) -> None:
        results.append(
            {
                "gate": name,
                "comparison": ">=",
                "threshold": threshold,
                "actual": actual,
                "passed": actual is not None and actual >= threshold,
            }
        )

    def maximum_gate(name: str, actual: float, threshold: float) -> None:
        results.append(
            {
                "gate": name,
                "comparison": "<=",
                "threshold": threshold,
                "actual": actual,
                "passed": actual <= threshold,
            }
        )

    minimum_gate("overall_accuracy", ordinary.accuracy, _float(gates, "overall_accuracy_min"))
    minimum_gate("overall_macro_f1", ordinary.macro_f1, _float(gates, "overall_macro_f1_min"))
    minimum_gate("opishnyan_recall", ordinary.per_class_recall[opishnyan], _float(gates, "opishnyan_recall_min"))
    minimum_gate("bubnivka_precision", ordinary.per_class_precision[bubnivka], _float(gates, "bubnivka_precision_min"))
    minimum_gate("ceramic_macro_f1", ceramic_macro_f1, _float(gates, "ceramic_macro_f1_min"))
    maximum_gate("opishnyan_to_bubnivka_errors", float(opish_to_bub), _float(gates, "opishnyan_to_bubnivka_errors_max"))
    minimum_gate("challenging_opishnyan_accuracy", _accuracy_for_mask(truth, predictions, challenging_mask), _float(gates, "challenging_opishnyan_accuracy_min"))
    minimum_gate("hard_three_opishnyan_pooled_accuracy", _accuracy_for_mask(truth, predictions, hard_mask), _float(gates, "hard_three_opishnyan_pooled_accuracy_min"))
    minimum_gate("hard_three_opishnyan_cohort_macro_accuracy", hard_macro, _float(gates, "hard_three_opishnyan_cohort_macro_accuracy_min"))
    minimum_gate("large_named_source_macro_accuracy", large_macro, _float(gates, "large_named_source_macro_accuracy_min"))
    minimum_gate("worst_fold_accuracy", float(fold_accuracies.min()), _float(gates, "worst_fold_accuracy_min"))
    maximum_gate("fold_accuracy_std", float(fold_accuracies.std(ddof=1)), _float(gates, "fold_accuracy_std_max"))
    recall_floors = _mapping(gates, "other_class_recall_floors")
    for label in sorted(recall_floors):
        minimum_gate(
            f"recall_floor:{label}",
            ordinary.per_class_recall[label],
            _float(recall_floors, label),
        )
    results.append(
        {
            "gate": "base_recipe_stability",
            "comparison": "required",
            "threshold": True,
            "actual": stability_passed,
            "passed": stability_passed,
        }
    )
    all_passed = all(bool(result["passed"]) for result in results)
    return {
        "all_required": bool(gates.get("all_required")),
        "all_passed": all_passed,
        "promotion_decision": "promote" if all_passed else "reject",
        "gate_count": len(results),
        "passed_count": sum(bool(result["passed"]) for result in results),
        "failed_gates": [result["gate"] for result in results if not result["passed"]],
        "results": results,
        "supporting_values": {
            "hard_three_cohort_accuracies": dict(zip(hard_cohorts, hard_accuracies)),
            "large_named_source_cohorts": large_names,
            "large_named_source_cohort_accuracies": dict(zip(large_names, large_accuracies)),
        },
    }


def _confusion_pair_rows(
    matrix: np.ndarray, classes: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for true_index, true_class in enumerate(classes):
        support = int(matrix[true_index].sum())
        for predicted_index, predicted_class in enumerate(classes):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "error_count": count,
                    "fraction_of_true_class": float(count / support) if support else 0.0,
                }
            )
    rows.sort(key=lambda row: (-int(row["error_count"]), str(row["true_class"]), str(row["predicted_class"])))
    return rows


def _save_confusion_plot(
    matrix: np.ndarray, classes: Sequence[str], path: Path
) -> None:
    short = [label.split("_", 1)[1] if "_" in label else label for label in classes]
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(classes)), short, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), short)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Phase 5 nested development OOF confusion matrix")
    threshold = float(matrix.max()) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".png", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(
            temporary,
            dpi=180,
            format="png",
            metadata={"Software": "ornament-classifier-phase5"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _base_grid_rows(configurations: Sequence[BaseConfiguration]) -> List[Dict[str, object]]:
    return [
        {
            "configuration_id": configuration.configuration_id,
            "feature_family": configuration.family.family_id,
            "feature_kind": configuration.family.kind,
            "embedding_blocks": "|".join(configuration.family.block_names),
            "embedding_dimensions": "|".join(
                str(block.shape[1]) for block in configuration.family.blocks
            ),
            "c_value": configuration.c_value,
            "source_group_exponent": configuration.source_group_exponent,
            "inference_cost_rank": configuration.family.inference_cost_rank,
        }
        for configuration in configurations
    ]


def _required_output_names(contract: Mapping[str, object]) -> Tuple[str, ...]:
    required = tuple(
        str(value) for value in _sequence(_mapping(contract, "output_contract"), "required")
    )
    if len(required) != len(set(required)) or any(not value for value in required):
        raise ValueError("Required output filenames must be unique and non-empty")
    return required


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover()
    contract, contract_path = load_frozen_contract(paths)
    development = load_development_contract(paths)
    _validate_development(development, contract)
    records = development.records
    classes = _classes(records)
    seed = _int(contract, "seed")
    ceramic_spec = _mapping(contract, "ceramic_specialist")
    ceramic_classes = tuple(
        str(value) for value in _sequence(ceramic_spec, "classes")
    )
    if len(ceramic_classes) != 3 or set(ceramic_classes) - set(classes):
        raise ValueError("Frozen ceramic specialist classes disagree with development")
    ornek_class = "02_ornek"
    petrykivka_class = "04_petrykivka_painting"
    if ornek_class not in classes or petrykivka_class not in classes:
        raise ValueError("Frozen eligibility classes disagree with development")

    print("Loading and verifying frozen Phase 5 embedding caches...", flush=True)
    embedding_blocks = load_allowlisted_embeddings(
        _embedding_requests(contract), paths=paths
    )
    if any(
        block.provenance.get("experiment_contract_sha256")
        != EXPECTED_CONTRACT_SHA256
        for block in embedding_blocks.values()
    ):
        raise ValueError("Embedding provenance does not pin the frozen Phase 5 contract")
    families = build_feature_families(contract, embedding_blocks)
    configurations = build_base_configurations(contract, families)
    reference_configuration = _reference_configuration(configurations, contract)
    configurations_by_id = {
        configuration.configuration_id: configuration for configuration in configurations
    }
    fold_ids = development.fold_ids
    labels, groups, folds = _record_arrays(records)

    nested_probabilities = np.full((len(records), len(classes)), np.nan, dtype=np.float64)
    inner_rows: List[Dict[str, object]] = []
    outer_rows: List[Dict[str, object]] = []
    selected_by_outer: Dict[str, Tuple[str, str]] = {}
    outer_base_searches: Dict[str, Mapping[str, OOFResult]] = {}
    total_searches = len(fold_ids)

    for outer_number, outer_fold in enumerate(fold_ids, start=1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        print(
            f"Outer search {outer_number}/{total_searches}: hold out fold {outer_fold}",
            flush=True,
        )
        base_results: List[OOFResult] = []
        for candidate_number, configuration in enumerate(configurations, start=1):
            print(
                f"  base {candidate_number}/{len(configurations)} "
                f"{configuration.configuration_id}",
                flush=True,
            )
            base_results.append(
                evaluate_base_oof(
                    configuration,
                    records,
                    classes,
                    ceramic_classes,
                    inner_folds,
                    seed,
                )
            )
        base_by_id = {result.summary.configuration_id: result for result in base_results}
        outer_base_searches[outer_fold] = base_by_id
        reference_result = base_by_id[reference_configuration.configuration_id]
        base_selection = select_search_candidates(
            base_results,
            reference_result,
            contract,
            ornek_class=ornek_class,
            petrykivka_class=petrykivka_class,
        )
        inner_rows.extend(
            _selection_rows(
                "outer_inner_oof",
                "base",
                base_results,
                base_selection,
                reference_configuration.configuration_id,
                outer_fold,
            )
        )
        selected_base_configuration = configurations_by_id[
            base_selection.selected.summary.configuration_id
        ]
        specialist_results = [
            _no_specialist_result(base_selection.selected),
            evaluate_specialist_oof(
                base_selection.selected,
                selected_base_configuration,
                records,
                classes,
                ceramic_classes,
                inner_folds,
                seed,
            ),
        ]
        specialist_selection = select_search_candidates(
            specialist_results,
            reference_result,
            contract,
            ornek_class=ornek_class,
            petrykivka_class=petrykivka_class,
        )
        inner_rows.extend(
            _selection_rows(
                "outer_inner_oof",
                "specialist",
                specialist_results,
                specialist_selection,
                reference_configuration.configuration_id,
                outer_fold,
            )
        )
        selected_specialist = specialist_selection.selected.summary.specialist
        selected_by_outer[outer_fold] = (
            selected_base_configuration.configuration_id,
            selected_specialist,
        )
        validation_indices, probabilities, iterations = _fit_outer_and_predict(
            selected_base_configuration,
            selected_specialist,
            outer_fold,
            records,
            classes,
            ceramic_classes,
            seed,
        )
        nested_probabilities[validation_indices] = probabilities
        fold_ordinary = ordinary_metrics(labels[validation_indices], probabilities, classes)
        fold_predictions = _predict_labels(probabilities, classes)
        fold_source = source_group_metrics(
            labels[validation_indices],
            fold_predictions,
            groups[validation_indices],
            classes,
            ceramic_classes,
        )
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_base_configuration_id": (
                    selected_base_configuration.configuration_id
                ),
                "selected_feature_family": selected_base_configuration.family.family_id,
                "selected_specialist": selected_specialist,
                "c_value": selected_base_configuration.c_value,
                "source_group_exponent": (
                    selected_base_configuration.source_group_exponent
                ),
                "train_examples": int(np.sum(folds != outer_fold)),
                "validation_examples": len(validation_indices),
                "fit_count": len(iterations),
                "maximum_iterations": max(iterations),
                "accuracy": fold_ordinary.accuracy,
                "macro_f1": fold_ordinary.macro_f1,
                "balanced_accuracy": fold_ordinary.balanced_accuracy,
                "source_group_balanced_accuracy": (
                    fold_source.source_group_balanced_accuracy
                ),
                "ceramic_worst_group_recall": (
                    fold_source.ceramic_worst_group_recall
                ),
                "robustness_score": fold_source.robustness_score,
                "per_class_recall_json": stable_json(
                    dict(fold_ordinary.per_class_recall)
                ),
                "per_class_group_recall_json": stable_json(
                    dict(fold_source.per_class_group_recall)
                ),
            }
        )
    if not np.isfinite(nested_probabilities).all():
        raise RuntimeError("Nested OOF predictions are incomplete")

    print("Running full-development five-fold base selection...", flush=True)
    full_base_results = []
    for candidate_number, configuration in enumerate(configurations, start=1):
        print(
            f"  base {candidate_number}/{len(configurations)} "
            f"{configuration.configuration_id}",
            flush=True,
        )
        full_base_results.append(
            evaluate_base_oof(
                configuration,
                records,
                classes,
                ceramic_classes,
                fold_ids,
                seed,
            )
        )
    full_base_by_id = {
        result.summary.configuration_id: result for result in full_base_results
    }
    full_reference = full_base_by_id[reference_configuration.configuration_id]
    phase4_reference_validation = validate_phase4_reference_metrics(
        full_reference, records, classes, contract, paths
    )
    full_base_selection = select_search_candidates(
        full_base_results,
        full_reference,
        contract,
        ornek_class=ornek_class,
        petrykivka_class=petrykivka_class,
    )
    selected_full_base = configurations_by_id[
        full_base_selection.selected.summary.configuration_id
    ]
    full_specialist_results = [
        _no_specialist_result(full_base_selection.selected),
        evaluate_specialist_oof(
            full_base_selection.selected,
            selected_full_base,
            records,
            classes,
            ceramic_classes,
            fold_ids,
            seed,
        ),
    ]
    full_specialist_selection = select_search_candidates(
        full_specialist_results,
        full_reference,
        contract,
        ornek_class=ornek_class,
        petrykivka_class=petrykivka_class,
    )
    selected_full_result = full_specialist_selection.selected
    selected_full_specialist = selected_full_result.summary.specialist
    full_selection_rows = _selection_rows(
        "full_development_oof",
        "base",
        full_base_results,
        full_base_selection,
        reference_configuration.configuration_id,
    )
    full_selection_rows.extend(
        _selection_rows(
            "full_development_oof",
            "specialist",
            full_specialist_results,
            full_specialist_selection,
            reference_configuration.configuration_id,
        )
    )

    tie_window = _float(_mapping(contract, "selection_rule"), "tie_window")
    stable_outer_folds = []
    stability_details = []
    for outer_fold in fold_ids:
        search = outer_base_searches[outer_fold]
        reference_result = search[reference_configuration.configuration_id]
        search_selection = select_search_candidates(
            tuple(search.values()),
            reference_result,
            contract,
            ornek_class=ornek_class,
            petrykivka_class=petrykivka_class,
        )
        eligible_ids = search_selection.ranked_eligible_ids
        best_score = max(search[identifier].summary.robustness_score for identifier in eligible_ids)
        selected_result = search[selected_full_base.configuration_id]
        is_eligible = search_selection.eligibility[
            selected_full_base.configuration_id
        ].eligible
        within_window = bool(
            is_eligible
            and selected_result.summary.robustness_score >= best_score - tie_window
        )
        if within_window:
            stable_outer_folds.append(outer_fold)
        stability_details.append(
            {
                "outer_fold": outer_fold,
                "full_development_base_configuration_id": (
                    selected_full_base.configuration_id
                ),
                "eligible": is_eligible,
                "robustness_score": selected_result.summary.robustness_score,
                "best_eligible_robustness_score": best_score,
                "within_tie_window": within_window,
            }
        )
    stability_required = 4
    stability_passed = len(stable_outer_folds) >= stability_required

    nested_predictions = _predict_labels(nested_probabilities, classes)
    nested_ordinary = ordinary_metrics(labels, nested_probabilities, classes)
    nested_source = source_group_metrics(
        labels, nested_predictions, groups, classes, ceramic_classes
    )
    promotion = evaluate_promotion_gates(
        records,
        nested_probabilities,
        classes,
        nested_ordinary,
        outer_rows,
        contract,
        stability_passed,
    )

    output_dir = _resolve_output_dir(paths, contract, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_grid_rows = _base_grid_rows(configurations)
    nested_prediction_rows = _prediction_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested_probabilities,
        classes,
        "nested_outer_oof_adaptive_phase5_procedure",
        selected_by_fold=selected_by_outer,
    )
    selected_prediction_rows = _prediction_rows(
        records,
        selected_full_result.evaluated_indices,
        selected_full_result.probabilities,
        classes,
        "selection_conditional_full_development_oof",
        selected_recipe=(
            selected_full_base.configuration_id,
            selected_full_specialist,
        ),
    )
    per_class = _per_class_rows(nested_ordinary, classes, "nested_outer_oof")
    diagnostics = diagnostic_slice_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested_probabilities,
        classes,
        contract,
    )
    matrix = confusion_matrix(labels, nested_predictions, labels=list(classes))
    matrix_rows = [
        {
            "true_class": label,
            **{
                predicted_label: int(matrix[row_index, column_index])
                for column_index, predicted_label in enumerate(classes)
            },
        }
        for row_index, label in enumerate(classes)
    ]

    grid_fields = tuple(base_grid_rows[0].keys())
    search_fields = tuple(inner_rows[0].keys())
    outer_fields = tuple(outer_rows[0].keys())
    prediction_fields = tuple(nested_prediction_rows[0].keys())
    _write_csv(output_dir / "base_grid.csv", base_grid_rows, grid_fields)
    _write_csv(output_dir / "inner_search.csv", inner_rows, search_fields)
    _write_csv(output_dir / "outer_fold_metrics.csv", outer_rows, outer_fields)
    _write_csv(
        output_dir / "nested_oof_predictions.csv",
        nested_prediction_rows,
        prediction_fields,
    )
    _write_csv(
        output_dir / "full_development_selection.csv",
        full_selection_rows,
        tuple(full_selection_rows[0].keys()),
    )
    _write_csv(
        output_dir / "selected_oof_predictions.csv",
        selected_prediction_rows,
        tuple(selected_prediction_rows[0].keys()),
    )
    _write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        tuple(per_class[0].keys()),
    )
    _write_csv(
        output_dir / "diagnostic_slices.csv",
        diagnostics,
        tuple(diagnostics[0].keys()),
    )
    confusion_pairs = _confusion_pair_rows(matrix, classes)
    _write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pairs,
        tuple(confusion_pairs[0].keys()),
    )
    _write_csv(
        output_dir / "confusion_matrix.csv",
        matrix_rows,
        ("true_class", *classes),
    )
    _save_confusion_plot(matrix, classes, output_dir / "confusion_matrix.png")

    requested_outputs = _required_output_names(contract)
    metrics_payload: Dict[str, object] = {
        "phase": PHASE,
        "experiment_version": contract.get("experiment_version"),
        "experiment_contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "experiment_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "code_provenance": code_provenance(),
        "evaluation_scope": (
            "development_only_selection_aware_nested_source_atomic_cv"
        ),
        "sealed_test_evaluated": False,
        "performance_estimate": {
            "available": False,
            "reason": (
                "Nested OOF estimates the adaptive Phase 5 procedure, but the "
                "candidate family remains informed by Phase 4 development diagnostics."
            ),
            "nested_development_estimate_available": True,
        },
        "probability_policy": {
            "calibrated": False,
            "threshold_selection_used": False,
            "interpretation": "uncalibrated ranking scores only",
        },
        "split_version": development.split_version,
        "split_seed": development.split_seed,
        "split_assignment_fingerprint_sha256": development.audit.get(
            "assignment_fingerprint_sha256"
        ),
        "development_image_count": len(records),
        "class_names": classes,
        "ceramic_classes": ceramic_classes,
        "base_configuration_count": len(configurations),
        "outer_fold_count": len(fold_ids),
        "nested_selected_recipes_by_outer_fold": {
            fold: {"base_configuration_id": recipe[0], "specialist": recipe[1]}
            for fold, recipe in selected_by_outer.items()
        },
        "nested_aggregate_oof_metrics": {
            **asdict(nested_ordinary),
            **asdict(nested_source),
        },
        "nested_outer_fold_metrics": outer_rows,
        "full_development_selected_candidate": {
            "base_configuration_id": selected_full_base.configuration_id,
            "feature_family": selected_full_base.family.family_id,
            "feature_kind": selected_full_base.family.kind,
            "embedding_blocks": selected_full_base.family.block_names,
            "c_value": selected_full_base.c_value,
            "source_group_exponent": selected_full_base.source_group_exponent,
            "specialist": selected_full_specialist,
            "inference_cost_rank": selected_full_base.family.inference_cost_rank,
        },
        "full_development_selected_oof_metrics": {
            **asdict(selected_full_result.ordinary),
            **asdict(selected_full_result.source),
        },
        "exact_phase4_reference_validation": phase4_reference_validation,
        "base_recipe_stability": {
            "required_outer_searches_within_tie_window": stability_required,
            "observed_outer_searches_within_tie_window": len(stable_outer_folds),
            "passing_outer_folds": stable_outer_folds,
            "passed": stability_passed,
            "details": stability_details,
        },
        "promotion_gates": promotion,
        "promotion_decision": promotion["promotion_decision"],
        "model_fit_policy": {
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iterations": MAX_ITERATIONS,
            "tolerance": TOLERANCE,
            "convergence_warnings_treated_as_errors": True,
            "blas_threads_per_fit": 1,
            "seed": seed,
        },
        "embedding_provenance": {
            name: dict(block.provenance)
            for name, block in sorted(embedding_blocks.items())
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "required_outputs": requested_outputs,
    }
    _write_json(output_dir / "metrics.json", metrics_payload)
    missing = [name for name in requested_outputs if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Required Phase 5 outputs are missing: " + ", ".join(missing))

    print(
        json.dumps(
            {
                "nested_accuracy": nested_ordinary.accuracy,
                "nested_macro_f1": nested_ordinary.macro_f1,
                "nested_robustness_score": nested_source.robustness_score,
                "selected_base": selected_full_base.configuration_id,
                "selected_specialist": selected_full_specialist,
                "promotion_decision": promotion["promotion_decision"],
                "sealed_test_evaluated": False,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

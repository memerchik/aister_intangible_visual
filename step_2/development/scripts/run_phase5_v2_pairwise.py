#!/usr/bin/env python3
"""Run the frozen Phase 5 v2 pairwise ceramic-correction experiment.

V2 preserves the v1 nested base-selection procedure and evaluates a bounded
Opishnyan/Bubnivka conditional correction.  It uses canonical development
folds and explicitly allowlisted cached embeddings only.  Sealed evaluation,
calibration, confidence thresholds, and per-image v1 error routing are outside
this entry point.
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SOURCE_ROOT = STEP_ROOT / "src"
SCRIPT_ROOT = STEP_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import threadpoolctl  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import run_phase5_source_robustness as v1  # noqa: E402
from ornament_classifier.contracts import (  # noqa: E402
    DevelopmentContract,
    DevelopmentRecord,
    load_development_contract,
)
from ornament_classifier.embeddings import EmbeddingBlock  # noqa: E402
from ornament_classifier.embeddings_v2 import (  # noqa: E402
    load_allowlisted_embeddings_v2,
)
from ornament_classifier.pairwise import (  # noqa: E402
    BoundaryMetrics,
    CorrectionCandidateSummary,
    CorrectionEligibilityReference,
    CorrectionEligibilityResult,
    apply_pairwise_logit_blend,
    boundary_metrics,
    correction_eligibility,
    fit_pairwise_correction,
    rank_correction_candidates,
)
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    OrdinaryMetrics,
    SourceGroupMetrics,
    ordinary_metrics,
    source_group_metrics,
)


PHASE = "step02_phase_5_source_robustness_v2"
CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v2.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "481fcab8c19b412a59354f53735a3c6299da5c16ec06ed1c6e4ffbe49c1a5a68"
)
SCRIPT_PATH = Path(__file__).resolve()
MAX_ITERATIONS = 10_000
TOLERANCE = 1e-6
VIEW_MULTIPLICITY = {
    "center_crop": 1,
    "letterbox": 1,
    "global_fivecrop": 6,
}

FeatureInput = Union[np.ndarray, Tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class CorrectionFamily:
    family_id: str
    kind: str
    block_names: Tuple[str, ...]
    blocks: Tuple[np.ndarray, ...]
    required_views: Tuple[str, ...]


@dataclass(frozen=True)
class CorrectionConfiguration:
    configuration_id: str
    correction: str
    family: Optional[CorrectionFamily]
    c_value: float
    source_group_exponent: float
    blend_weight: float


@dataclass(frozen=True)
class HeadOOF:
    probabilities: np.ndarray
    evaluated_indices: np.ndarray
    maximum_iterations: int
    fit_count: int


@dataclass(frozen=True)
class CorrectionOOFResult:
    probabilities: np.ndarray
    evaluated_indices: np.ndarray
    fold_rows: Tuple[Mapping[str, object], ...]
    ordinary: OrdinaryMetrics
    source: SourceGroupMetrics
    boundary: BoundaryMetrics
    summary: CorrectionCandidateSummary
    maximum_iterations: int
    fit_count: int


@dataclass(frozen=True)
class CorrectionSelection:
    selected: CorrectionOOFResult
    reference: CorrectionEligibilityReference
    eligibility: Mapping[str, CorrectionEligibilityResult]
    ranked_eligible_ids: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the frozen v2 contract directory.",
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


def load_frozen_contract(paths: ProjectPaths) -> Tuple[Mapping[str, object], Path]:
    path = (paths.step_root / CONTRACT_RELATIVE_PATH).resolve()
    expected_path = paths.step_root / CONTRACT_RELATIVE_PATH
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"Frozen Phase 5 v2 contract is missing: {expected_path}")
    observed = sha256_file(path)
    if observed != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "Phase 5 v2 contract changed after it was frozen: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {observed}"
        )
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load frozen Phase 5 v2 contract: {error}") from error
    if not isinstance(contract, dict):
        raise ValueError("Frozen Phase 5 v2 contract must be an object")
    if contract.get("iteration") != "v2" or contract.get("status") != "frozen_before_fit":
        raise ValueError("Phase 5 v2 contract has an invalid iteration or status")
    scope = v1._mapping(contract, "scope")
    if scope.get("sealed_test_access") != "forbidden":
        raise ValueError("Phase 5 v2 requires sealed access to remain forbidden")
    return contract, path


def code_provenance() -> Mapping[str, object]:
    paths = (
        SCRIPT_PATH,
        STEP_ROOT / "scripts" / "run_phase5_source_robustness.py",
        STEP_ROOT / "src" / "ornament_classifier" / "pairwise.py",
        STEP_ROOT / "src" / "ornament_classifier" / "robustness.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings_v2.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings.py",
        STEP_ROOT / "src" / "ornament_classifier" / "contracts.py",
        STEP_ROOT / "src" / "ornament_classifier" / "paths.py",
        STEP_ROOT / "requirements-phase5.txt",
    )
    files: Dict[str, str] = {}
    combined = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Phase 5 v2 provenance input is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        combined.update(f"{relative}\x1f{digest}\n".encode("utf-8"))
    return {"files_sha256": files, "combined_sha256": combined.hexdigest()}


def _validate_development(
    development: DevelopmentContract, contract: Mapping[str, object]
) -> None:
    v1._validate_development(development, contract)
    if len(development.records) != 1693:
        raise ValueError("Phase 5 v2 requires all 1,693 development rows")


def _embedding_requests(
    contract: Mapping[str, object],
) -> Mapping[str, Tuple[str, str]]:
    references = set()
    base = v1._mapping(contract, "base_procedure")
    pair = v1._mapping(contract, "pairwise_correction")
    for container in (
        v1._sequence(base, "feature_families"),
        v1._sequence(pair, "feature_families"),
    ):
        for family in container:
            if not isinstance(family, dict):
                raise ValueError("Feature-family specs must be objects")
            for reference in v1._sequence(family, "blocks"):
                if not isinstance(reference, str) or reference.count(".") != 1:
                    raise ValueError(f"Invalid embedding reference: {reference!r}")
                references.add(reference)
    return {
        reference: tuple(reference.split(".", 1))  # type: ignore[arg-type]
        for reference in sorted(references)
    }


def build_base_configurations(
    contract: Mapping[str, object],
    embedding_blocks: Mapping[str, EmbeddingBlock],
) -> Tuple[v1.BaseConfiguration, ...]:
    procedure = v1._mapping(contract, "base_procedure")
    families: Dict[str, v1.FeatureFamily] = {}
    for spec in v1._sequence(procedure, "feature_families"):
        if not isinstance(spec, dict):
            raise ValueError("Base feature-family specs must be objects")
        family_id = v1._text(spec, "id")
        kind = v1._text(spec, "kind")
        block_names = tuple(str(value) for value in v1._sequence(spec, "blocks"))
        blocks = tuple(embedding_blocks[name].values for name in block_names)
        view_count = v1._int(spec, "view_count")
        families[family_id] = v1.FeatureFamily(
            family_id=family_id,
            kind=kind,
            block_names=block_names,
            blocks=blocks,
            inference_cost_rank=view_count,
        )
    configurations = []
    for spec in v1._sequence(procedure, "configurations"):
        if not isinstance(spec, dict):
            raise ValueError("Base configuration specs must be objects")
        family_id = v1._text(spec, "feature_family")
        configurations.append(
            v1.BaseConfiguration(
                configuration_id=v1._text(spec, "id"),
                family=families[family_id],
                c_value=v1._float(spec, "c_value"),
                source_group_exponent=v1._float(
                    spec, "source_group_exponent"
                ),
            )
        )
    if len(configurations) != 3 or len(
        {configuration.configuration_id for configuration in configurations}
    ) != 3:
        raise ValueError("Phase 5 v2 requires exactly three unique base configurations")
    return tuple(configurations)


def build_correction_families(
    contract: Mapping[str, object],
    embedding_blocks: Mapping[str, EmbeddingBlock],
) -> Tuple[CorrectionFamily, ...]:
    pair = v1._mapping(contract, "pairwise_correction")
    families = []
    for spec in v1._sequence(pair, "feature_families"):
        if not isinstance(spec, dict):
            raise ValueError("Correction feature-family specs must be objects")
        block_names = tuple(str(value) for value in v1._sequence(spec, "blocks"))
        family = CorrectionFamily(
            family_id=v1._text(spec, "id"),
            kind=v1._text(spec, "kind"),
            block_names=block_names,
            blocks=tuple(embedding_blocks[name].values for name in block_names),
            required_views=tuple(
                str(value) for value in v1._sequence(spec, "required_views")
            ),
        )
        if family.kind == "single_embedding" and len(family.blocks) != 1:
            raise ValueError("Single correction family must have one block")
        if family.kind == "centered_logit_ensemble" and len(family.blocks) < 2:
            raise ValueError("Correction ensemble must have multiple blocks")
        families.append(family)
    if len(families) != 4 or len({family.family_id for family in families}) != 4:
        raise ValueError("Phase 5 v2 requires exactly four correction families")
    return tuple(families)


def build_correction_configurations(
    contract: Mapping[str, object], families: Sequence[CorrectionFamily]
) -> Tuple[CorrectionConfiguration, ...]:
    pair = v1._mapping(contract, "pairwise_correction")
    configurations = [
        CorrectionConfiguration(
            configuration_id="pair_none",
            correction="none",
            family=None,
            c_value=10.0,
            source_group_exponent=0.0,
            blend_weight=0.0,
        )
    ]
    for family in families:
        for c_value in v1._sequence(pair, "c_values"):
            for exponent in v1._sequence(pair, "source_group_exponents"):
                for blend in v1._sequence(pair, "blend_weights"):
                    c_float = float(c_value)
                    exponent_float = float(exponent)
                    blend_float = float(blend)
                    identifier = (
                        f"{family.family_id}__c{v1._number_id(c_float)}"
                        f"__source{v1._number_id(exponent_float)}"
                        f"__blend{v1._number_id(blend_float)}"
                    )
                    configurations.append(
                        CorrectionConfiguration(
                            configuration_id=identifier,
                            correction="pairwise_conditional_logit_blend",
                            family=family,
                            c_value=c_float,
                            source_group_exponent=exponent_float,
                            blend_weight=blend_float,
                        )
                    )
    expected = v1._int(pair, "grid_configuration_count_including_none")
    if len(configurations) != expected or len(
        {configuration.configuration_id for configuration in configurations}
    ) != expected:
        raise ValueError("Correction grid does not match the frozen v2 count")
    return tuple(configurations)


def _record_arrays(
    records: Sequence[DevelopmentRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return v1._record_arrays(records)


def _slice_correction_features(
    family: CorrectionFamily, indices: np.ndarray
) -> FeatureInput:
    blocks = tuple(block[indices] for block in family.blocks)
    return blocks[0] if family.kind == "single_embedding" else blocks


def _base_required_views(configuration: v1.BaseConfiguration) -> Tuple[str, ...]:
    if configuration.family.family_id == "global_cls":
        return ("global_fivecrop",)
    if configuration.family.family_id == "three_view_cls_logit_ensemble":
        return ("center_crop", "letterbox", "global_fivecrop")
    raise ValueError(f"Unknown v2 base family: {configuration.family.family_id}")


def _combined_view_count(
    base: v1.BaseConfiguration, correction: CorrectionConfiguration
) -> int:
    views = set(_base_required_views(base))
    if correction.family is not None:
        views.update(correction.family.required_views)
    return sum(VIEW_MULTIPLICITY[view] for view in views)


def _fit_head_oof(
    family: CorrectionFamily,
    c_value: float,
    exponent: float,
    records: Sequence[DevelopmentRecord],
    pair_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    seed: int,
) -> HeadOOF:
    labels, groups, folds = _record_arrays(records)
    fold_ids = tuple(eligible_fold_ids)
    pool_mask = np.isin(folds, fold_ids)
    evaluated_indices = np.flatnonzero(pool_mask)
    probabilities = np.full((len(records), 2), np.nan, dtype=np.float64)
    iterations: List[int] = []
    for validation_fold in fold_ids:
        validation_indices = np.flatnonzero(pool_mask & (folds == validation_fold))
        train_indices = np.flatnonzero(pool_mask & (folds != validation_fold))
        v1._assert_source_disjoint_partitions(
            train_indices,
            validation_indices,
            groups,
            f"v2 pair OOF fold {validation_fold}",
        )
        with threadpool_limits(limits=1):
            head = fit_pairwise_correction(
                _slice_correction_features(family, train_indices),
                labels[train_indices],
                groups[train_indices],
                pair_classes=pair_classes,
                c_value=c_value,
                source_group_exponent=exponent,
                centered_logit_ensemble=(
                    family.kind == "centered_logit_ensemble"
                ),
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        if not head.converged:
            raise RuntimeError("Phase 5 v2 pairwise correction did not converge")
        iterations.extend(v1._flatten_iterations(head.iterations))
        probabilities[validation_indices] = head.predict_proba(
            _slice_correction_features(family, validation_indices)
        )
    pooled = probabilities[evaluated_indices]
    if not np.isfinite(pooled).all():
        raise RuntimeError("Pairwise OOF predictions are incomplete")
    return HeadOOF(
        probabilities=pooled,
        evaluated_indices=evaluated_indices,
        maximum_iterations=max(iterations),
        fit_count=len(iterations),
    )


def _correction_summary(
    probabilities: np.ndarray,
    evaluated_indices: np.ndarray,
    configuration: CorrectionConfiguration,
    base_configuration: v1.BaseConfiguration,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> Tuple[
    OrdinaryMetrics,
    SourceGroupMetrics,
    BoundaryMetrics,
    CorrectionCandidateSummary,
    Tuple[Mapping[str, object], ...],
]:
    labels, groups, folds = _record_arrays(records)
    truth = labels[evaluated_indices]
    selected_groups = groups[evaluated_indices]
    selected_folds = folds[evaluated_indices]
    predictions = v1._predict_labels(probabilities, classes)
    ordinary = ordinary_metrics(truth, probabilities, classes)
    source = source_group_metrics(
        truth, predictions, selected_groups, classes, ceramic_classes
    )
    boundary = boundary_metrics(
        truth,
        probabilities,
        classes,
        pair_classes=pair_classes,
        ceramic_classes=ceramic_classes,
    )
    fold_rows = []
    fold_boundary_scores = []
    for fold_id in sorted(set(selected_folds.tolist())):
        mask = selected_folds == fold_id
        fold_ordinary = ordinary_metrics(truth[mask], probabilities[mask], classes)
        fold_source = source_group_metrics(
            truth[mask],
            predictions[mask],
            selected_groups[mask],
            classes,
            ceramic_classes,
        )
        fold_boundary = boundary_metrics(
            truth[mask],
            probabilities[mask],
            classes,
            pair_classes=pair_classes,
            ceramic_classes=ceramic_classes,
        )
        fold_boundary_scores.append(fold_boundary.boundary_score)
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
                "robustness_score": fold_source.robustness_score,
                "boundary_score": fold_boundary.boundary_score,
                "opishnyan_recall": fold_boundary.opishnyan_recall,
                "bubnivka_recall": fold_boundary.bubnivka_recall,
                "bubnivka_precision": fold_boundary.bubnivka_precision,
            }
        )
    opishnyan = pair_classes[0]
    summary = CorrectionCandidateSummary(
        configuration_id=configuration.configuration_id,
        boundary_score=boundary.boundary_score,
        minimum_fold_boundary_score=float(min(fold_boundary_scores)),
        accuracy=ordinary.accuracy,
        macro_f1=ordinary.macro_f1,
        source_robustness_score=source.robustness_score,
        opishnyan_group_recall=source.per_class_group_recall[opishnyan],
        combined_view_count=_combined_view_count(
            base_configuration, configuration
        ),
        correction=configuration.correction,
        blend_weight=configuration.blend_weight,
        c_value=configuration.c_value,
        source_group_exponent=configuration.source_group_exponent,
        per_class_recall=ordinary.per_class_recall,
    )
    return ordinary, source, boundary, summary, tuple(fold_rows)


def evaluate_correction_oof_grid(
    base_result: v1.OOFResult,
    base_configuration: v1.BaseConfiguration,
    configurations: Sequence[CorrectionConfiguration],
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    seed: int,
) -> Tuple[CorrectionOOFResult, ...]:
    evaluated_indices = base_result.evaluated_indices
    results: List[CorrectionOOFResult] = []
    head_cache: Dict[Tuple[str, float, float], HeadOOF] = {}
    for configuration in configurations:
        if configuration.correction == "none":
            probabilities = np.array(base_result.probabilities, copy=True)
            maximum_iterations = base_result.maximum_iterations
            fit_count = 0
        else:
            if configuration.family is None:
                raise ValueError("Pairwise correction is missing its feature family")
            key = (
                configuration.family.family_id,
                configuration.c_value,
                configuration.source_group_exponent,
            )
            if key not in head_cache:
                head_cache[key] = _fit_head_oof(
                    configuration.family,
                    configuration.c_value,
                    configuration.source_group_exponent,
                    records,
                    pair_classes,
                    eligible_fold_ids,
                    seed,
                )
            head = head_cache[key]
            if not np.array_equal(head.evaluated_indices, evaluated_indices):
                raise ValueError("Base and pairwise OOF row scopes disagree")
            probabilities = apply_pairwise_logit_blend(
                base_result.probabilities,
                classes,
                head.probabilities,
                pair_classes,
                pair_classes,
                configuration.blend_weight,
            )
            maximum_iterations = head.maximum_iterations
            fit_count = head.fit_count
        ordinary, source, boundary, summary, fold_rows = _correction_summary(
            probabilities,
            evaluated_indices,
            configuration,
            base_configuration,
            records,
            classes,
            pair_classes,
            ceramic_classes,
        )
        results.append(
            CorrectionOOFResult(
                probabilities=probabilities,
                evaluated_indices=evaluated_indices,
                fold_rows=fold_rows,
                ordinary=ordinary,
                source=source,
                boundary=boundary,
                summary=summary,
                maximum_iterations=maximum_iterations,
                fit_count=fit_count,
            )
        )
    return tuple(results)


def _correction_limits(contract: Mapping[str, object]) -> Mapping[str, object]:
    selection = v1._mapping(contract, "correction_selection_rule")
    limits = v1._mapping(
        selection, "eligibility_relative_to_uncorrected_selected_base"
    )
    return {
        "maximum_accuracy_drop": v1._float(limits, "maximum_accuracy_drop"),
        "maximum_macro_f1_drop": v1._float(limits, "maximum_macro_f1_drop"),
        "maximum_source_robustness_drop": v1._float(
            limits, "maximum_source_robustness_drop"
        ),
        "maximum_opishnyan_group_recall_drop": v1._float(
            limits, "maximum_opishnyan_group_recall_drop"
        ),
        "other_class_recall_drops": {
            label: float(value)
            for label, value in v1._mapping(
                limits, "maximum_other_class_recall_drop"
            ).items()
        },
    }


def select_correction_candidates(
    results: Sequence[CorrectionOOFResult],
    contract: Mapping[str, object],
) -> CorrectionSelection:
    references = [
        result for result in results if result.summary.configuration_id == "pair_none"
    ]
    if len(references) != 1:
        raise ValueError("Correction grid must contain exactly one none reference")
    baseline = references[0]
    reference = CorrectionEligibilityReference(
        accuracy=baseline.ordinary.accuracy,
        macro_f1=baseline.ordinary.macro_f1,
        source_robustness_score=baseline.source.robustness_score,
        opishnyan_group_recall=baseline.summary.opishnyan_group_recall,
        per_class_recall=baseline.ordinary.per_class_recall,
    )
    limits = _correction_limits(contract)
    eligibility = {}
    eligible = []
    results_by_id = {result.summary.configuration_id: result for result in results}
    if len(results_by_id) != len(results):
        raise ValueError("Correction result identifiers must be unique")
    for result in results:
        decision = correction_eligibility(result.summary, reference, **limits)
        eligibility[result.summary.configuration_id] = decision
        if decision.eligible:
            eligible.append(result.summary)
    if not eligible:
        raise ValueError("No v2 correction satisfies non-regression constraints")
    tie_window = v1._float(
        v1._mapping(contract, "correction_selection_rule"), "tie_window"
    )
    ranked = rank_correction_candidates(eligible, tie_window=tie_window)
    ranked_ids = tuple(summary.configuration_id for summary in ranked)
    return CorrectionSelection(
        selected=results_by_id[ranked_ids[0]],
        reference=reference,
        eligibility=eligibility,
        ranked_eligible_ids=ranked_ids,
    )


def _base_search_rows(
    scope: str,
    outer_fold: str,
    results: Sequence[v1.OOFResult],
    selection: v1.SearchSelection,
    reference_id: str,
) -> List[Dict[str, object]]:
    eligible_rank = {
        identifier: rank
        for rank, identifier in enumerate(selection.ranked_eligible_ids, start=1)
    }
    rows = []
    for result in results:
        identifier = result.summary.configuration_id
        decision = selection.eligibility[identifier]
        rows.append(
            {
                "scope": scope,
                "outer_fold": outer_fold,
                "stage": "base",
                "configuration_id": identifier,
                "base_configuration_id": identifier,
                "correction_configuration_id": "",
                "feature_family": result.summary.configuration_id.split("__c", 1)[0],
                "correction": "not_applicable",
                "c_value": result.summary.c_value,
                "source_group_exponent": result.summary.source_group_exponent,
                "blend_weight": "",
                "combined_view_count": result.summary.inference_cost_rank,
                "is_exact_reference": identifier == reference_id,
                "eligible": decision.eligible,
                "eligibility_failures": "|".join(decision.failures),
                "eligible_selection_rank": eligible_rank.get(identifier, ""),
                "selected": identifier == selection.selected.summary.configuration_id,
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
                    result.summary.minimum_inner_fold_source_group_balanced_accuracy
                ),
                "boundary_score": "",
                "minimum_fold_boundary_score": "",
                "opishnyan_recall": result.ordinary.per_class_recall[
                    "01_opishnyan_ceramics"
                ],
                "bubnivka_recall": result.ordinary.per_class_recall[
                    "03_bubnivka_ceramics"
                ],
                "bubnivka_precision": result.ordinary.per_class_precision[
                    "03_bubnivka_ceramics"
                ],
                "ceramic_macro_f1": np.mean(
                    [
                        result.ordinary.per_class_f1[label]
                        for label in (
                            "01_opishnyan_ceramics",
                            "03_bubnivka_ceramics",
                            "05_kosiv_ceramics",
                        )
                    ]
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


def _correction_search_rows(
    scope: str,
    outer_fold: str,
    base_configuration: v1.BaseConfiguration,
    results: Sequence[CorrectionOOFResult],
    selection: CorrectionSelection,
) -> List[Dict[str, object]]:
    eligible_rank = {
        identifier: rank
        for rank, identifier in enumerate(selection.ranked_eligible_ids, start=1)
    }
    rows = []
    for result in results:
        summary = result.summary
        decision = selection.eligibility[summary.configuration_id]
        family_id = (
            "none"
            if summary.correction == "none"
            else summary.configuration_id.split("__c", 1)[0]
        )
        rows.append(
            {
                "scope": scope,
                "outer_fold": outer_fold,
                "stage": "correction",
                "configuration_id": summary.configuration_id,
                "base_configuration_id": base_configuration.configuration_id,
                "correction_configuration_id": summary.configuration_id,
                "feature_family": family_id,
                "correction": summary.correction,
                "c_value": summary.c_value,
                "source_group_exponent": summary.source_group_exponent,
                "blend_weight": summary.blend_weight,
                "combined_view_count": summary.combined_view_count,
                "is_exact_reference": summary.configuration_id == "pair_none",
                "eligible": decision.eligible,
                "eligibility_failures": "|".join(decision.failures),
                "eligible_selection_rank": eligible_rank.get(
                    summary.configuration_id, ""
                ),
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
                "minimum_fold_source_group_balanced_accuracy": "",
                "boundary_score": result.boundary.boundary_score,
                "minimum_fold_boundary_score": (
                    summary.minimum_fold_boundary_score
                ),
                "opishnyan_recall": result.boundary.opishnyan_recall,
                "bubnivka_recall": result.boundary.bubnivka_recall,
                "bubnivka_precision": result.boundary.bubnivka_precision,
                "ceramic_macro_f1": result.boundary.ceramic_macro_f1,
                "per_class_recall_json": stable_json(
                    dict(result.ordinary.per_class_recall)
                ),
                "per_class_group_recall_json": stable_json(
                    dict(result.source.per_class_group_recall)
                ),
            }
        )
    return rows


def _fit_outer_procedure(
    base_configuration: v1.BaseConfiguration,
    correction: CorrectionConfiguration,
    outer_fold: str,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    labels, groups, folds = _record_arrays(records)
    validation_indices, base_probabilities, base_iterations = v1._fit_outer_and_predict(
        base_configuration,
        "none",
        outer_fold,
        records,
        classes,
        ceramic_classes,
        seed,
    )
    if correction.correction == "none":
        return (
            validation_indices,
            base_probabilities,
            np.array(base_probabilities, copy=True),
            base_iterations,
        )
    if correction.family is None:
        raise ValueError("Selected pairwise correction is missing a family")
    train_indices = np.flatnonzero(folds != outer_fold)
    v1._assert_source_disjoint_partitions(
        train_indices,
        validation_indices,
        groups,
        f"v2 outer correction fold {outer_fold}",
    )
    with threadpool_limits(limits=1):
        head = fit_pairwise_correction(
            _slice_correction_features(correction.family, train_indices),
            labels[train_indices],
            groups[train_indices],
            pair_classes=pair_classes,
            c_value=correction.c_value,
            source_group_exponent=correction.source_group_exponent,
            centered_logit_ensemble=(
                correction.family.kind == "centered_logit_ensemble"
            ),
            seed=seed,
            max_iter=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )
    corrected = apply_pairwise_logit_blend(
        base_probabilities,
        classes,
        head.predict_proba(
            _slice_correction_features(correction.family, validation_indices)
        ),
        pair_classes,
        pair_classes,
        correction.blend_weight,
    )
    iterations = tuple(
        (*base_iterations, *v1._flatten_iterations(head.iterations))
    )
    return validation_indices, base_probabilities, corrected, iterations


def _decision_fingerprint(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> str:
    predictions = v1._predict_labels(probabilities, classes)
    digest = hashlib.sha256()
    for record, prediction in zip(records, predictions):
        digest.update(f"{record.image_id}\x1f{prediction}\n".encode("utf-8"))
    return digest.hexdigest()


def validate_v1_reproduction(
    records: Sequence[DevelopmentRecord],
    uncorrected_nested_probabilities: np.ndarray,
    selected_by_outer: Mapping[str, Tuple[str, str]],
    full_selected_base: v1.BaseConfiguration,
    contract: Mapping[str, object],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
) -> Mapping[str, object]:
    input_contract = v1._mapping(contract, "input_contract")
    expected_fingerprint = v1._text(
        input_contract, "phase_5_v1_nested_decision_fingerprint_sha256"
    )
    observed_fingerprint = _decision_fingerprint(
        records, uncorrected_nested_probabilities, classes
    )
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError("V2 uncorrected procedure does not reproduce v1 decisions")
    path = REPO_ROOT / v1._text(input_contract, "phase_5_v1_metrics")
    if sha256_file(path) != v1._text(
        input_contract, "phase_5_v1_metrics_sha256"
    ):
        raise RuntimeError("Pinned Phase 5 v1 metrics changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_recipes = payload["nested_selected_recipes_by_outer_fold"]
    for fold, (base_id, correction_id) in selected_by_outer.items():
        expected = expected_recipes[fold]
        if base_id != expected["base_configuration_id"]:
            raise RuntimeError(f"V2 base selection disagrees with v1 in fold {fold}")
        if correction_id != "pair_none" or expected["specialist"] != "none":
            raise RuntimeError("Uncorrected v2 baseline must match v1 no-specialist")
    expected_full_base = payload["full_development_selected_candidate"][
        "base_configuration_id"
    ]
    if full_selected_base.configuration_id != expected_full_base:
        raise RuntimeError("V2 full base selection disagrees with Phase 5 v1")
    labels, groups, _ = _record_arrays(records)
    predictions = v1._predict_labels(uncorrected_nested_probabilities, classes)
    ordinary = ordinary_metrics(labels, uncorrected_nested_probabilities, classes)
    source = source_group_metrics(
        labels, predictions, groups, classes, ceramic_classes
    )
    expected_metrics = payload["nested_aggregate_oof_metrics"]
    comparisons = {
        "accuracy": ordinary.accuracy,
        "macro_f1": ordinary.macro_f1,
        "balanced_accuracy": ordinary.balanced_accuracy,
        "robustness_score": source.robustness_score,
    }
    for name, actual in comparisons.items():
        if not np.isclose(
            actual, float(expected_metrics[name]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"V2 uncorrected metric differs from v1: {name}")
    return {
        "validated": True,
        "expected_decision_fingerprint_sha256": expected_fingerprint,
        "observed_decision_fingerprint_sha256": observed_fingerprint,
        "outer_base_recipes_match": True,
        "full_base_recipe_matches": True,
        "aggregate_metrics_match": True,
        "probability_identity_required": False,
    }


def _stability(
    fold_ids: Sequence[str],
    full_base: v1.BaseConfiguration,
    full_correction: CorrectionConfiguration,
    outer_base_results: Mapping[str, Mapping[str, v1.OOFResult]],
    outer_base_selections: Mapping[str, v1.SearchSelection],
    outer_correction_results: Mapping[str, Mapping[str, CorrectionOOFResult]],
    outer_correction_selections: Mapping[str, CorrectionSelection],
    contract: Mapping[str, object],
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    base_window = v1._float(
        v1._mapping(v1._mapping(contract, "base_procedure"), "selection_rule"),
        "tie_window",
    )
    correction_window = v1._float(
        v1._mapping(contract, "correction_selection_rule"), "tie_window"
    )
    base_details = []
    correction_details = []
    base_pass = []
    correction_pass = []
    for fold in fold_ids:
        base_results = outer_base_results[fold]
        base_selection = outer_base_selections[fold]
        base_result = base_results[full_base.configuration_id]
        best_base = max(
            base_results[identifier].summary.robustness_score
            for identifier in base_selection.ranked_eligible_ids
        )
        base_eligible = base_selection.eligibility[
            full_base.configuration_id
        ].eligible
        base_within = bool(
            base_eligible
            and base_result.summary.robustness_score >= best_base - base_window
        )
        if base_within:
            base_pass.append(fold)
        base_details.append(
            {
                "outer_fold": fold,
                "configuration_id": full_base.configuration_id,
                "eligible": base_eligible,
                "robustness_score": base_result.summary.robustness_score,
                "best_eligible_robustness_score": best_base,
                "within_tie_window": base_within,
            }
        )

        correction_results = outer_correction_results[fold]
        correction_selection = outer_correction_selections[fold]
        correction_result = correction_results[full_correction.configuration_id]
        best_correction = max(
            correction_results[identifier].summary.boundary_score
            for identifier in correction_selection.ranked_eligible_ids
        )
        correction_eligible = correction_selection.eligibility[
            full_correction.configuration_id
        ].eligible
        correction_within = bool(
            correction_eligible
            and correction_result.summary.boundary_score
            >= best_correction - correction_window
        )
        if correction_within:
            correction_pass.append(fold)
        correction_details.append(
            {
                "outer_fold": fold,
                "configuration_id": full_correction.configuration_id,
                "eligible": correction_eligible,
                "boundary_score": correction_result.summary.boundary_score,
                "best_eligible_boundary_score": best_correction,
                "within_tie_window": correction_within,
            }
        )
    required = 4
    return (
        {
            "required_outer_searches_within_tie_window": required,
            "observed_outer_searches_within_tie_window": len(base_pass),
            "passing_outer_folds": base_pass,
            "passed": len(base_pass) >= required,
            "details": base_details,
        },
        {
            "required_outer_searches_within_tie_window": required,
            "observed_outer_searches_within_tie_window": len(correction_pass),
            "passing_outer_folds": correction_pass,
            "passed": len(correction_pass) >= required,
            "details": correction_details,
        },
    )


def evaluate_v2_promotion_gates(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
    ordinary: OrdinaryMetrics,
    outer_fold_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    base_stability_passed: bool,
    correction_stability_passed: bool,
) -> Mapping[str, object]:
    base = dict(
        v1.evaluate_promotion_gates(
            records,
            probabilities,
            classes,
            ordinary,
            outer_fold_rows,
            contract,
            base_stability_passed,
        )
    )
    results = [dict(result) for result in base["results"]]
    results.append(
        {
            "gate": "correction_recipe_stability",
            "comparison": "required",
            "threshold": True,
            "actual": correction_stability_passed,
            "passed": correction_stability_passed,
        }
    )
    all_passed = all(bool(result["passed"]) for result in results)
    base.update(
        {
            "all_passed": all_passed,
            "promotion_decision": "promote" if all_passed else "reject",
            "gate_count": len(results),
            "passed_count": sum(bool(result["passed"]) for result in results),
            "failed_gates": [
                result["gate"] for result in results if not result["passed"]
            ],
            "results": results,
        }
    )
    return base


def _prediction_rows(
    records: Sequence[DevelopmentRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    evaluation_status: str,
    selected_by_fold: Optional[
        Mapping[str, Tuple[v1.BaseConfiguration, CorrectionConfiguration]]
    ] = None,
    selected_recipe: Optional[
        Tuple[v1.BaseConfiguration, CorrectionConfiguration]
    ] = None,
) -> List[Dict[str, object]]:
    predictions = v1._predict_labels(probabilities, classes)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rows = []
    for position, global_index in enumerate(indices.tolist()):
        record = records[global_index]
        if selected_by_fold is not None:
            base, correction = selected_by_fold[record.cv_fold]
        elif selected_recipe is not None:
            base, correction = selected_recipe
        else:
            raise ValueError("V2 prediction recipe provenance is missing")
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
            "selected_base_configuration_id": base.configuration_id,
            "selected_correction_configuration_id": (
                correction.configuration_id
            ),
            "selected_correction": correction.correction,
            "correction_feature_family": (
                correction.family.family_id if correction.family else "none"
            ),
            "correction_c_value": correction.c_value,
            "correction_source_group_exponent": (
                correction.source_group_exponent
            ),
            "correction_blend_weight": correction.blend_weight,
        }
        for class_index, class_name in enumerate(classes):
            row[f"probability_{class_name}"] = float(
                probabilities[position, class_index]
            )
        rows.append(row)
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
    axis.set_title("Phase 5 v2 nested development OOF confusion matrix")
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
            metadata={"Software": "ornament-classifier-phase5-v2"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _base_grid_rows(
    configurations: Sequence[v1.BaseConfiguration],
) -> List[Dict[str, object]]:
    return [
        {
            "configuration_id": configuration.configuration_id,
            "feature_family": configuration.family.family_id,
            "feature_kind": configuration.family.kind,
            "embedding_blocks": "|".join(configuration.family.block_names),
            "c_value": configuration.c_value,
            "source_group_exponent": configuration.source_group_exponent,
            "view_count": configuration.family.inference_cost_rank,
        }
        for configuration in configurations
    ]


def _correction_grid_rows(
    configurations: Sequence[CorrectionConfiguration],
) -> List[Dict[str, object]]:
    return [
        {
            "configuration_id": configuration.configuration_id,
            "correction": configuration.correction,
            "feature_family": (
                configuration.family.family_id if configuration.family else "none"
            ),
            "feature_kind": (
                configuration.family.kind if configuration.family else "none"
            ),
            "embedding_blocks": (
                "|".join(configuration.family.block_names)
                if configuration.family
                else ""
            ),
            "required_views": (
                "|".join(configuration.family.required_views)
                if configuration.family
                else ""
            ),
            "c_value": configuration.c_value,
            "source_group_exponent": configuration.source_group_exponent,
            "blend_weight": configuration.blend_weight,
        }
        for configuration in configurations
    ]


def _required_output_names(contract: Mapping[str, object]) -> Tuple[str, ...]:
    required = tuple(
        str(value)
        for value in v1._sequence(
            v1._mapping(contract, "output_contract"), "required"
        )
    )
    if len(required) != 13 or len(set(required)) != 13:
        raise ValueError("Phase 5 v2 requires exactly 13 unique outputs")
    return required


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover()
    contract, contract_path = load_frozen_contract(paths)
    development = load_development_contract(paths)
    _validate_development(development, contract)
    records = development.records
    classes = v1._classes(records)
    seed = v1._int(contract, "seed")
    pair_spec = v1._mapping(contract, "pairwise_correction")
    pair_classes = tuple(str(value) for value in v1._sequence(pair_spec, "classes"))
    diagnostic = v1._mapping(contract, "diagnostic_slice_definitions")
    ceramic_classes = tuple(
        str(value) for value in v1._sequence(diagnostic, "ceramic_classes")
    )
    if len(pair_classes) != 2 or set(pair_classes) - set(ceramic_classes):
        raise ValueError("V2 pair classes disagree with the ceramic declaration")

    print("Loading and verifying frozen Phase 5 v2 embedding arrays...", flush=True)
    embedding_blocks = load_allowlisted_embeddings_v2(
        _embedding_requests(contract), paths=paths
    )
    if any(
        block.provenance.get("experiment_contract_sha256")
        != EXPECTED_CONTRACT_SHA256
        for block in embedding_blocks.values()
    ):
        raise ValueError("V2 embedding provenance does not pin the frozen contract")
    base_configurations = build_base_configurations(contract, embedding_blocks)
    correction_families = build_correction_families(contract, embedding_blocks)
    correction_configurations = build_correction_configurations(
        contract, correction_families
    )
    base_by_id = {
        configuration.configuration_id: configuration
        for configuration in base_configurations
    }
    correction_by_id = {
        configuration.configuration_id: configuration
        for configuration in correction_configurations
    }
    reference_base_id = "global_cls__c10__source0"
    reference_base = base_by_id[reference_base_id]
    fold_ids = development.fold_ids
    labels, groups, folds = _record_arrays(records)

    nested_probabilities = np.full((len(records), len(classes)), np.nan)
    uncorrected_nested_probabilities = np.full_like(nested_probabilities, np.nan)
    inner_rows: List[Dict[str, object]] = []
    outer_rows: List[Dict[str, object]] = []
    selected_by_outer: Dict[
        str, Tuple[v1.BaseConfiguration, CorrectionConfiguration]
    ] = {}
    outer_base_results: Dict[str, Mapping[str, v1.OOFResult]] = {}
    outer_base_selections: Dict[str, v1.SearchSelection] = {}
    outer_correction_results: Dict[
        str, Mapping[str, CorrectionOOFResult]
    ] = {}
    outer_correction_selections: Dict[str, CorrectionSelection] = {}

    base_selection_contract = {
        "base_grid": {
            "exact_phase_4_reference": {
                "feature_family": "global_cls",
                "c_value": 10.0,
                "source_group_exponent": 0.0,
            }
        },
        "selection_rule": v1._mapping(
            v1._mapping(contract, "base_procedure"), "selection_rule"
        ),
    }

    for outer_number, outer_fold in enumerate(fold_ids, start=1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        print(
            f"Outer v2 search {outer_number}/{len(fold_ids)}: hold out fold {outer_fold}",
            flush=True,
        )
        base_results = tuple(
            v1.evaluate_base_oof(
                configuration,
                records,
                classes,
                ceramic_classes,
                inner_folds,
                seed,
            )
            for configuration in base_configurations
        )
        fold_base_by_id = {
            result.summary.configuration_id: result for result in base_results
        }
        base_selection = v1.select_search_candidates(
            base_results,
            fold_base_by_id[reference_base_id],
            base_selection_contract,
            ornek_class="02_ornek",
            petrykivka_class="04_petrykivka_painting",
        )
        selected_base = base_by_id[base_selection.selected.summary.configuration_id]
        correction_results = evaluate_correction_oof_grid(
            base_selection.selected,
            selected_base,
            correction_configurations,
            records,
            classes,
            pair_classes,
            ceramic_classes,
            inner_folds,
            seed,
        )
        correction_selection = select_correction_candidates(
            correction_results, contract
        )
        selected_correction = correction_by_id[
            correction_selection.selected.summary.configuration_id
        ]
        outer_base_results[outer_fold] = fold_base_by_id
        outer_base_selections[outer_fold] = base_selection
        outer_correction_results[outer_fold] = {
            result.summary.configuration_id: result
            for result in correction_results
        }
        outer_correction_selections[outer_fold] = correction_selection
        selected_by_outer[outer_fold] = (selected_base, selected_correction)
        inner_rows.extend(
            _base_search_rows(
                "outer_inner_oof",
                outer_fold,
                base_results,
                base_selection,
                reference_base_id,
            )
        )
        inner_rows.extend(
            _correction_search_rows(
                "outer_inner_oof",
                outer_fold,
                selected_base,
                correction_results,
                correction_selection,
            )
        )
        validation_indices, base_probabilities, corrected, iterations = (
            _fit_outer_procedure(
                selected_base,
                selected_correction,
                outer_fold,
                records,
                classes,
                pair_classes,
                ceramic_classes,
                seed,
            )
        )
        uncorrected_nested_probabilities[validation_indices] = base_probabilities
        nested_probabilities[validation_indices] = corrected
        fold_ordinary = ordinary_metrics(
            labels[validation_indices], corrected, classes
        )
        fold_predictions = v1._predict_labels(corrected, classes)
        fold_source = source_group_metrics(
            labels[validation_indices],
            fold_predictions,
            groups[validation_indices],
            classes,
            ceramic_classes,
        )
        fold_boundary = boundary_metrics(
            labels[validation_indices],
            corrected,
            classes,
            pair_classes=pair_classes,
            ceramic_classes=ceramic_classes,
        )
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_base_configuration_id": selected_base.configuration_id,
                "selected_correction_configuration_id": (
                    selected_correction.configuration_id
                ),
                "selected_correction": selected_correction.correction,
                "correction_feature_family": (
                    selected_correction.family.family_id
                    if selected_correction.family
                    else "none"
                ),
                "correction_c_value": selected_correction.c_value,
                "correction_source_group_exponent": (
                    selected_correction.source_group_exponent
                ),
                "correction_blend_weight": selected_correction.blend_weight,
                "combined_view_count": _combined_view_count(
                    selected_base, selected_correction
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
                "boundary_score": fold_boundary.boundary_score,
                "opishnyan_recall": fold_boundary.opishnyan_recall,
                "bubnivka_recall": fold_boundary.bubnivka_recall,
                "bubnivka_precision": fold_boundary.bubnivka_precision,
                "ceramic_macro_f1": fold_boundary.ceramic_macro_f1,
                "per_class_recall_json": stable_json(
                    dict(fold_ordinary.per_class_recall)
                ),
                "per_class_group_recall_json": stable_json(
                    dict(fold_source.per_class_group_recall)
                ),
            }
        )
    if not np.isfinite(nested_probabilities).all() or not np.isfinite(
        uncorrected_nested_probabilities
    ).all():
        raise RuntimeError("Phase 5 v2 nested predictions are incomplete")

    print("Running full-development Phase 5 v2 selection...", flush=True)
    full_base_results = tuple(
        v1.evaluate_base_oof(
            configuration,
            records,
            classes,
            ceramic_classes,
            fold_ids,
            seed,
        )
        for configuration in base_configurations
    )
    full_base_by_id = {
        result.summary.configuration_id: result for result in full_base_results
    }
    full_base_selection = v1.select_search_candidates(
        full_base_results,
        full_base_by_id[reference_base_id],
        base_selection_contract,
        ornek_class="02_ornek",
        petrykivka_class="04_petrykivka_painting",
    )
    full_selected_base = base_by_id[
        full_base_selection.selected.summary.configuration_id
    ]
    full_correction_results = evaluate_correction_oof_grid(
        full_base_selection.selected,
        full_selected_base,
        correction_configurations,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        fold_ids,
        seed,
    )
    full_correction_selection = select_correction_candidates(
        full_correction_results, contract
    )
    full_selected_correction = correction_by_id[
        full_correction_selection.selected.summary.configuration_id
    ]
    full_selected_result = full_correction_selection.selected
    full_rows = _base_search_rows(
        "full_development_oof",
        "",
        full_base_results,
        full_base_selection,
        reference_base_id,
    )
    full_rows.extend(
        _correction_search_rows(
            "full_development_oof",
            "",
            full_selected_base,
            full_correction_results,
            full_correction_selection,
        )
    )

    v1_reproduction = validate_v1_reproduction(
        records,
        uncorrected_nested_probabilities,
        {
            fold: (base.configuration_id, "pair_none")
            for fold, (base, _) in selected_by_outer.items()
        },
        full_selected_base,
        contract,
        classes,
        ceramic_classes,
    )
    base_stability, correction_stability = _stability(
        fold_ids,
        full_selected_base,
        full_selected_correction,
        outer_base_results,
        outer_base_selections,
        outer_correction_results,
        outer_correction_selections,
        contract,
    )

    nested_predictions = v1._predict_labels(nested_probabilities, classes)
    nested_ordinary = ordinary_metrics(labels, nested_probabilities, classes)
    nested_source = source_group_metrics(
        labels, nested_predictions, groups, classes, ceramic_classes
    )
    nested_boundary = boundary_metrics(
        labels,
        nested_probabilities,
        classes,
        pair_classes=pair_classes,
        ceramic_classes=ceramic_classes,
    )
    promotion = evaluate_v2_promotion_gates(
        records,
        nested_probabilities,
        classes,
        nested_ordinary,
        outer_rows,
        contract,
        bool(base_stability["passed"]),
        bool(correction_stability["passed"]),
    )

    output_dir = v1._resolve_output_dir(paths, contract, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_grid_rows = _base_grid_rows(base_configurations)
    correction_grid_rows = _correction_grid_rows(correction_configurations)
    nested_prediction_rows = _prediction_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested_probabilities,
        classes,
        "nested_outer_oof_adaptive_phase5_v2_procedure",
        selected_by_fold=selected_by_outer,
    )
    selected_prediction_rows = _prediction_rows(
        records,
        full_selected_result.evaluated_indices,
        full_selected_result.probabilities,
        classes,
        "selection_conditional_full_development_oof_v2",
        selected_recipe=(full_selected_base, full_selected_correction),
    )
    per_class_rows = v1._per_class_rows(
        nested_ordinary, classes, "nested_outer_oof_v2"
    )
    diagnostic_rows = v1.diagnostic_slice_rows(
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
                predicted: int(matrix[row_index, column_index])
                for column_index, predicted in enumerate(classes)
            },
        }
        for row_index, label in enumerate(classes)
    ]
    confusion_pairs = v1._confusion_pair_rows(matrix, classes)

    v1._write_csv(
        output_dir / "base_grid.csv",
        base_grid_rows,
        tuple(base_grid_rows[0]),
    )
    v1._write_csv(
        output_dir / "correction_grid.csv",
        correction_grid_rows,
        tuple(correction_grid_rows[0]),
    )
    v1._write_csv(
        output_dir / "inner_search.csv", inner_rows, tuple(inner_rows[0])
    )
    v1._write_csv(
        output_dir / "outer_fold_metrics.csv", outer_rows, tuple(outer_rows[0])
    )
    v1._write_csv(
        output_dir / "nested_oof_predictions.csv",
        nested_prediction_rows,
        tuple(nested_prediction_rows[0]),
    )
    v1._write_csv(
        output_dir / "full_development_selection.csv",
        full_rows,
        tuple(full_rows[0]),
    )
    v1._write_csv(
        output_dir / "selected_oof_predictions.csv",
        selected_prediction_rows,
        tuple(selected_prediction_rows[0]),
    )
    v1._write_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        tuple(per_class_rows[0]),
    )
    v1._write_csv(
        output_dir / "diagnostic_slices.csv",
        diagnostic_rows,
        tuple(diagnostic_rows[0]),
    )
    v1._write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pairs,
        tuple(confusion_pairs[0]),
    )
    v1._write_csv(
        output_dir / "confusion_matrix.csv", matrix_rows, ("true_class", *classes)
    )
    _save_confusion_plot(matrix, classes, output_dir / "confusion_matrix.png")

    required_outputs = _required_output_names(contract)
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
            "nested_development_estimate_available": True,
            "reason": (
                "Nested OOF estimates the adaptive v2 procedure, but v1 development "
                "errors and class metrics informed the correction family."
            ),
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
        "pair_classes": pair_classes,
        "ceramic_classes": ceramic_classes,
        "base_configuration_count": len(base_configurations),
        "correction_configuration_count": len(correction_configurations),
        "nested_selected_recipes_by_outer_fold": {
            fold: {
                "base_configuration_id": base.configuration_id,
                "correction_configuration_id": correction.configuration_id,
            }
            for fold, (base, correction) in selected_by_outer.items()
        },
        "nested_aggregate_oof_metrics": {
            **asdict(nested_ordinary),
            **asdict(nested_source),
            **asdict(nested_boundary),
        },
        "nested_outer_fold_metrics": outer_rows,
        "full_development_selected_candidate": {
            "base_configuration_id": full_selected_base.configuration_id,
            "base_feature_family": full_selected_base.family.family_id,
            "base_c_value": full_selected_base.c_value,
            "base_source_group_exponent": (
                full_selected_base.source_group_exponent
            ),
            "correction_configuration_id": (
                full_selected_correction.configuration_id
            ),
            "correction": full_selected_correction.correction,
            "correction_feature_family": (
                full_selected_correction.family.family_id
                if full_selected_correction.family
                else "none"
            ),
            "correction_c_value": full_selected_correction.c_value,
            "correction_source_group_exponent": (
                full_selected_correction.source_group_exponent
            ),
            "correction_blend_weight": full_selected_correction.blend_weight,
            "combined_view_count": _combined_view_count(
                full_selected_base, full_selected_correction
            ),
        },
        "full_development_selected_oof_metrics": {
            **asdict(full_selected_result.ordinary),
            **asdict(full_selected_result.source),
            **asdict(full_selected_result.boundary),
        },
        "v1_reproduction_validation": v1_reproduction,
        "base_recipe_stability": base_stability,
        "correction_recipe_stability": correction_stability,
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
        "required_outputs": required_outputs,
    }
    v1._write_json(output_dir / "metrics.json", metrics_payload)
    missing = [name for name in required_outputs if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Required Phase 5 v2 outputs are missing: " + ", ".join(missing))

    print(
        json.dumps(
            {
                "nested_accuracy": nested_ordinary.accuracy,
                "nested_macro_f1": nested_ordinary.macro_f1,
                "nested_boundary_score": nested_boundary.boundary_score,
                "selected_base": full_selected_base.configuration_id,
                "selected_correction": full_selected_correction.configuration_id,
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

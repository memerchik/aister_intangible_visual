#!/usr/bin/env python3
"""Run the frozen Phase 5 v4 motif-localized representation experiment.

V4 preserves the exact Phase 5 v1 base selection and then compares no change
against a bounded family of label-free motif descriptor augmentations. All
selection is nested inside source-atomic development folds. Calibration,
sealed evaluation, and Phase 6 packaging are outside this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


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
import run_phase5_v2_pairwise as v2  # noqa: E402
import run_phase5_v3_consensus as v3  # noqa: E402
from ornament_classifier.consensus import (  # noqa: E402
    FusionCandidateSummary,
    FusionEligibilityReference,
    FusionEligibilityResult,
    fusion_eligibility,
    harmonic_mean_many,
    rank_fusion_candidates,
)
from ornament_classifier.contracts import (  # noqa: E402
    DevelopmentRecord,
    load_development_contract,
)
from ornament_classifier.embeddings import EmbeddingBlock  # noqa: E402
from ornament_classifier.embeddings_v3 import load_allowlisted_embeddings_v3  # noqa: E402
from ornament_classifier.embeddings_v4 import (  # noqa: E402
    MotifEmbeddingCache,
    load_motif_embeddings_v4,
)
from ornament_classifier.motif import augment_feature_blocks  # noqa: E402
from ornament_classifier.pairwise import BoundaryMetrics, boundary_metrics  # noqa: E402
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    OrdinaryMetrics,
    SourceGroupMetrics,
    compute_source_group_weights,
    fit_centered_logit_ensemble,
    fit_logistic_probe,
    ordinary_metrics,
    source_group_metrics,
)


PHASE = "step02_phase_5_source_robustness_v4"
CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v4.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da"
)
SCRIPT_PATH = Path(__file__).resolve()
MAX_ITERATIONS = 10_000
TOLERANCE = 1e-6


@dataclass(frozen=True)
class MotifConfiguration:
    configuration_id: str
    descriptor_name: Optional[str]
    c_value: float
    source_group_exponent: float
    complexity_rank: int
    descriptor_dimensions: int


@dataclass(frozen=True)
class MotifOOFResult:
    probabilities: np.ndarray
    applied_mask: np.ndarray
    evaluated_indices: np.ndarray
    fold_rows: Tuple[Mapping[str, object], ...]
    ordinary: OrdinaryMetrics
    source: SourceGroupMetrics
    boundary: BoundaryMetrics
    readiness_score: float
    hard_three_accuracy: float
    summary: FusionCandidateSummary
    maximum_iterations: int
    fit_count: int


@dataclass(frozen=True)
class MotifSelection:
    selected: MotifOOFResult
    reference: FusionEligibilityReference
    eligibility: Mapping[str, FusionEligibilityResult]
    ranked_eligible_ids: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the frozen v4 contract directory.",
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
    expected_path = paths.step_root / CONTRACT_RELATIVE_PATH
    path = expected_path.resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"Frozen Phase 5 v4 contract is missing: {expected_path}")
    observed = sha256_file(path)
    if observed != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "Phase 5 v4 contract changed after it was frozen: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {observed}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Frozen Phase 5 v4 contract must be an object")
    if contract.get("iteration") != "v4" or contract.get("status") != "frozen_before_fit":
        raise ValueError("Phase 5 v4 contract has an invalid iteration or status")
    if v1._mapping(contract, "scope").get("sealed_test_access") != "forbidden":
        raise ValueError("Phase 5 v4 requires sealed access to remain forbidden")
    return contract, path


def code_provenance() -> Mapping[str, object]:
    paths = (
        SCRIPT_PATH,
        STEP_ROOT / "scripts" / "run_phase5_source_robustness.py",
        STEP_ROOT / "scripts" / "run_phase5_v2_pairwise.py",
        STEP_ROOT / "scripts" / "run_phase5_v3_consensus.py",
        STEP_ROOT / "scripts" / "run_phase5_v4_motif_extraction.py",
        STEP_ROOT / "src" / "ornament_classifier" / "motif.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings_v4.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings_v3.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings.py",
        STEP_ROOT / "src" / "ornament_classifier" / "consensus.py",
        STEP_ROOT / "src" / "ornament_classifier" / "pairwise.py",
        STEP_ROOT / "src" / "ornament_classifier" / "robustness.py",
        STEP_ROOT / "src" / "ornament_classifier" / "contracts.py",
        STEP_ROOT / "src" / "ornament_classifier" / "paths.py",
        STEP_ROOT / "requirements-phase5.txt",
    )
    files: Dict[str, str] = {}
    combined = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Phase 5 v4 provenance input is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        combined.update(f"{relative}\x1f{digest}\n".encode("utf-8"))
    return {"files_sha256": files, "combined_sha256": combined.hexdigest()}


def load_base_embeddings(paths: ProjectPaths) -> Mapping[str, EmbeddingBlock]:
    requests = {
        "dinov3_center.embedding__cls": ("dinov3_center", "embedding__cls"),
        "dinov3_letterbox.embedding__cls": ("dinov3_letterbox", "embedding__cls"),
        "dinov3_global.embedding__cls": ("dinov3_global", "embedding__cls"),
    }
    return load_allowlisted_embeddings_v3(requests, paths=paths)


def build_motif_configurations(
    contract: Mapping[str, object], cache: MotifEmbeddingCache
) -> Tuple[MotifConfiguration, ...]:
    grid = v1._mapping(contract, "motif_grid")
    descriptors = tuple(str(value) for value in v1._sequence(grid, "descriptors"))
    if set(descriptors) != set(cache.embeddings):
        raise ValueError("V4 motif descriptor grid disagrees with the frozen cache")
    c_values = tuple(float(value) for value in v1._sequence(grid, "c_values"))
    exponent = v1._float(grid, "source_group_exponent")
    ranks = v1._mapping(grid, "complexity_rank")
    configurations = [MotifConfiguration("motif_none", None, 0.0, exponent, 0, 0)]
    for descriptor in descriptors:
        dimensions = int(cache.embeddings[descriptor].shape[1])
        for c_value in c_values:
            configurations.append(
                MotifConfiguration(
                    configuration_id=(
                        f"motif_{descriptor}__c{v1._number_id(c_value)}"
                        f"__source{v1._number_id(exponent)}"
                    ),
                    descriptor_name=descriptor,
                    c_value=c_value,
                    source_group_exponent=exponent,
                    complexity_rank=int(ranks[descriptor]),
                    descriptor_dimensions=dimensions,
                )
            )
    expected = v1._int(grid, "configuration_count_including_none")
    if len(configurations) != expected or len({item.configuration_id for item in configurations}) != expected:
        raise ValueError("V4 motif grid does not match the frozen count")
    return tuple(configurations)


def _record_arrays(
    records: Sequence[DevelopmentRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return v1._record_arrays(records)


def _augmented_blocks(
    base: v1.BaseConfiguration,
    configuration: MotifConfiguration,
    cache: MotifEmbeddingCache,
) -> Tuple[np.ndarray, ...]:
    if configuration.descriptor_name is None:
        raise ValueError("The no-motif configuration has no augmented blocks")
    return augment_feature_blocks(
        base.family.blocks, cache.embeddings[configuration.descriptor_name]
    )


def _slice_augmented(
    blocks: Sequence[np.ndarray], base: v1.BaseConfiguration, indices: np.ndarray
):
    sliced = tuple(block[indices] for block in blocks)
    return sliced[0] if base.family.kind == "single_embedding" else sliced


def _fit_augmented_model(
    base: v1.BaseConfiguration,
    configuration: MotifConfiguration,
    cache: MotifEmbeddingCache,
    train: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    seed: int,
):
    blocks = _augmented_blocks(base, configuration, cache)
    features = _slice_augmented(blocks, base, train)
    weights = compute_source_group_weights(
        labels[train], groups[train], configuration.source_group_exponent
    )
    with threadpool_limits(limits=1):
        if base.family.kind == "centered_logit_ensemble":
            return fit_centered_logit_ensemble(
                features,
                labels[train],
                c_value=configuration.c_value,
                sample_weight=weights,
                classes=classes,
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        return fit_logistic_probe(
            features,
            labels[train],
            c_value=configuration.c_value,
            sample_weight=weights,
            classes=classes,
            seed=seed,
            max_iter=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )


def _summarize(
    configuration: MotifConfiguration,
    probabilities: np.ndarray,
    evaluated_indices: np.ndarray,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    contract: Mapping[str, object],
    iterations: Sequence[int],
) -> MotifOOFResult:
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
    hard = v3._hard_three_accuracy(records, evaluated_indices, predictions, contract)
    if hard is None:
        raise RuntimeError("Pooled v4 selection rows have no hard-three support")
    readiness = harmonic_mean_many(
        (
            ordinary.accuracy,
            ordinary.macro_f1,
            source.source_group_balanced_accuracy,
            source.ceramic_worst_group_recall,
            boundary.opishnyan_recall,
            boundary.bubnivka_recall,
            boundary.bubnivka_precision,
            boundary.ceramic_macro_f1,
            hard,
        )
    )
    fold_rows = []
    fold_readiness = []
    for fold in sorted(set(selected_folds.tolist())):
        mask = selected_folds == fold
        fold_indices = evaluated_indices[mask]
        fold_probabilities = probabilities[mask]
        fold_predictions = predictions[mask]
        fold_ordinary = ordinary_metrics(truth[mask], fold_probabilities, classes)
        fold_source = source_group_metrics(
            truth[mask], fold_predictions, selected_groups[mask], classes, ceramic_classes
        )
        fold_boundary = boundary_metrics(
            truth[mask],
            fold_probabilities,
            classes,
            pair_classes=pair_classes,
            ceramic_classes=ceramic_classes,
        )
        fold_hard = v3._hard_three_accuracy(
            records, fold_indices, fold_predictions, contract
        )
        components = [
            fold_ordinary.accuracy,
            fold_ordinary.macro_f1,
            fold_source.source_group_balanced_accuracy,
            fold_source.ceramic_worst_group_recall,
            fold_boundary.opishnyan_recall,
            fold_boundary.bubnivka_recall,
            fold_boundary.bubnivka_precision,
            fold_boundary.ceramic_macro_f1,
        ]
        if fold_hard is not None:
            components.append(fold_hard)
        fold_score = harmonic_mean_many(components)
        fold_readiness.append(fold_score)
        fold_rows.append(
            {
                "cv_fold": fold,
                "validation_examples": int(mask.sum()),
                "accuracy": fold_ordinary.accuracy,
                "macro_f1": fold_ordinary.macro_f1,
                "source_group_balanced_accuracy": fold_source.source_group_balanced_accuracy,
                "robustness_score": fold_source.robustness_score,
                "readiness_score": fold_score,
                "hard_three_accuracy": fold_hard,
            }
        )
    summary = FusionCandidateSummary(
        configuration_id=configuration.configuration_id,
        readiness_score=readiness,
        minimum_fold_readiness_score=min(fold_readiness),
        accuracy=ordinary.accuracy,
        macro_f1=ordinary.macro_f1,
        source_robustness_score=source.robustness_score,
        opishnyan_group_recall=source.per_class_group_recall[pair_classes[0]],
        intervention_rank=configuration.complexity_rank,
        blend_weight=0.0,
        head_count=0 if configuration.descriptor_name is None else 1,
        encoder_count=0 if configuration.descriptor_name is None else 1,
        per_class_recall=dict(ordinary.per_class_recall),
    )
    applied = np.full(
        len(evaluated_indices), configuration.descriptor_name is not None, dtype=bool
    )
    flattened = tuple(iterations)
    return MotifOOFResult(
        probabilities=probabilities,
        applied_mask=applied,
        evaluated_indices=evaluated_indices,
        fold_rows=tuple(fold_rows),
        ordinary=ordinary,
        source=source,
        boundary=boundary,
        readiness_score=readiness,
        hard_three_accuracy=hard,
        summary=summary,
        maximum_iterations=max(flattened) if flattened else 0,
        fit_count=len(flattened),
    )


def evaluate_motif_oof_grid(
    base_result: v1.OOFResult,
    base: v1.BaseConfiguration,
    configurations: Sequence[MotifConfiguration],
    cache: MotifEmbeddingCache,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> Tuple[MotifOOFResult, ...]:
    labels, groups, folds = _record_arrays(records)
    fold_ids = tuple(eligible_fold_ids)
    pool = np.isin(folds, fold_ids)
    evaluated = np.flatnonzero(pool)
    if not np.array_equal(evaluated, base_result.evaluated_indices):
        raise RuntimeError("V4 base and motif OOF rows disagree")
    results = []
    for configuration in configurations:
        if configuration.descriptor_name is None:
            results.append(
                _summarize(
                    configuration,
                    np.array(base_result.probabilities, copy=True),
                    evaluated,
                    records,
                    classes,
                    pair_classes,
                    ceramic_classes,
                    contract,
                    (),
                )
            )
            continue
        probabilities = np.full((len(records), len(classes)), np.nan, dtype=np.float64)
        iterations: List[int] = []
        blocks = _augmented_blocks(base, configuration, cache)
        for validation_fold in fold_ids:
            validation = np.flatnonzero(pool & (folds == validation_fold))
            train = np.flatnonzero(pool & (folds != validation_fold))
            v1._assert_source_disjoint_partitions(
                train,
                validation,
                groups,
                f"v4 motif OOF {configuration.configuration_id} fold {validation_fold}",
            )
            model = _fit_augmented_model(
                base, configuration, cache, train, labels, groups, classes, seed
            )
            if not model.converged:
                raise RuntimeError(
                    f"V4 motif model did not converge: {configuration.configuration_id}"
                )
            iterations.extend(v1._flatten_iterations(model.iterations))
            probabilities[validation] = model.predict_proba(
                _slice_augmented(blocks, base, validation)
            )
        pooled = probabilities[evaluated]
        if not np.isfinite(pooled).all():
            raise RuntimeError("V4 motif OOF predictions are incomplete")
        results.append(
            _summarize(
                configuration,
                pooled,
                evaluated,
                records,
                classes,
                pair_classes,
                ceramic_classes,
                contract,
                iterations,
            )
        )
    return tuple(results)


def select_motif_candidates(
    results: Sequence[MotifOOFResult], contract: Mapping[str, object]
) -> MotifSelection:
    references = [
        result for result in results if result.summary.configuration_id == "motif_none"
    ]
    if len(references) != 1:
        raise ValueError("V4 grid must contain exactly one no-motif reference")
    baseline = references[0]
    reference = FusionEligibilityReference(
        accuracy=baseline.ordinary.accuracy,
        macro_f1=baseline.ordinary.macro_f1,
        source_robustness_score=baseline.source.robustness_score,
        opishnyan_group_recall=baseline.summary.opishnyan_group_recall,
        per_class_recall=baseline.ordinary.per_class_recall,
    )
    rule = v1._mapping(contract, "motif_selection_rule")
    limits = v1._mapping(rule, "eligibility_relative_to_uncorrected_selected_base")
    kwargs = {
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
    eligibility: Dict[str, FusionEligibilityResult] = {}
    eligible = []
    by_id = {result.summary.configuration_id: result for result in results}
    if len(by_id) != len(results):
        raise ValueError("V4 motif result IDs must be unique")
    for result in results:
        decision = fusion_eligibility(result.summary, reference, **kwargs)
        eligibility[result.summary.configuration_id] = decision
        if decision.eligible:
            eligible.append(result.summary)
    ranked = rank_fusion_candidates(
        eligible, tie_window=v1._float(rule, "tie_window")
    )
    ranked_ids = tuple(summary.configuration_id for summary in ranked)
    return MotifSelection(by_id[ranked_ids[0]], reference, eligibility, ranked_ids)


def fit_outer_procedure(
    base: v1.BaseConfiguration,
    motif: MotifConfiguration,
    cache: MotifEmbeddingCache,
    outer_fold: str,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    labels, groups, folds = _record_arrays(records)
    validation, base_probabilities, base_iterations = v1._fit_outer_and_predict(
        base, "none", outer_fold, records, classes, ceramic_classes, seed
    )
    if motif.descriptor_name is None:
        return (
            validation,
            base_probabilities,
            np.array(base_probabilities, copy=True),
            np.zeros(len(validation), dtype=bool),
            base_iterations,
        )
    train = np.flatnonzero(folds != outer_fold)
    v1._assert_source_disjoint_partitions(
        train, validation, groups, f"v4 outer motif {motif.configuration_id}"
    )
    model = _fit_augmented_model(
        base, motif, cache, train, labels, groups, classes, seed
    )
    blocks = _augmented_blocks(base, motif, cache)
    probabilities = model.predict_proba(_slice_augmented(blocks, base, validation))
    iterations = tuple(v1._flatten_iterations(model.iterations))
    return (
        validation,
        base_probabilities,
        probabilities,
        np.ones(len(validation), dtype=bool),
        iterations,
    )


def _search_row(
    scope: str,
    outer_fold: str,
    stage: str,
    base_id: str,
    motif_id: str,
    descriptor: str,
    c_value: object,
    source_exponent: object,
    dimensions: int,
    complexity_rank: int,
    exact_reference: bool,
    eligible: bool,
    failures: str,
    rank: object,
    selected: bool,
    result: object,
) -> Dict[str, object]:
    ordinary = result.ordinary
    source = result.source
    boundary = getattr(result, "boundary", None)
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "stage": stage,
        "configuration_id": motif_id if stage == "motif" else base_id,
        "base_configuration_id": base_id,
        "motif_configuration_id": motif_id,
        "motif_descriptor": descriptor,
        "c_value": c_value,
        "source_group_exponent": source_exponent,
        "descriptor_dimensions": dimensions,
        "complexity_rank": complexity_rank,
        "is_exact_reference": exact_reference,
        "eligible": eligible,
        "eligibility_failures": failures,
        "eligible_selection_rank": rank,
        "selected": selected,
        "evaluated_examples": len(result.evaluated_indices),
        "fit_count": result.fit_count,
        "maximum_iterations": result.maximum_iterations,
        "accuracy": ordinary.accuracy,
        "macro_f1": ordinary.macro_f1,
        "balanced_accuracy": ordinary.balanced_accuracy,
        "source_group_balanced_accuracy": source.source_group_balanced_accuracy,
        "ceramic_worst_group_recall": source.ceramic_worst_group_recall,
        "robustness_score": source.robustness_score,
        "readiness_score": getattr(result, "readiness_score", ""),
        "minimum_fold_readiness_score": getattr(result.summary, "minimum_fold_readiness_score", ""),
        "hard_three_accuracy": getattr(result, "hard_three_accuracy", ""),
        "opishnyan_recall": boundary.opishnyan_recall if boundary else ordinary.per_class_recall["01_opishnyan_ceramics"],
        "bubnivka_recall": boundary.bubnivka_recall if boundary else ordinary.per_class_recall["03_bubnivka_ceramics"],
        "bubnivka_precision": boundary.bubnivka_precision if boundary else "",
        "ceramic_macro_f1": boundary.ceramic_macro_f1 if boundary else "",
        "per_class_recall_json": stable_json(dict(ordinary.per_class_recall)),
        "per_class_group_recall_json": stable_json(dict(source.per_class_group_recall)),
    }


def base_search_rows(
    scope: str,
    outer_fold: str,
    results: Sequence[v1.OOFResult],
    selection: v1.SearchSelection,
    reference_id: str,
) -> List[Dict[str, object]]:
    ranks = {value: index for index, value in enumerate(selection.ranked_eligible_ids, 1)}
    rows = []
    for result in results:
        identifier = result.summary.configuration_id
        decision = selection.eligibility[identifier]
        rows.append(
            _search_row(
                scope,
                outer_fold,
                "base",
                identifier,
                "",
                "not_applicable",
                result.summary.c_value,
                result.summary.source_group_exponent,
                0,
                result.summary.inference_cost_rank,
                identifier == reference_id,
                decision.eligible,
                "|".join(decision.failures),
                ranks.get(identifier, ""),
                identifier == selection.selected.summary.configuration_id,
                result,
            )
        )
    return rows


def motif_search_rows(
    scope: str,
    outer_fold: str,
    base: v1.BaseConfiguration,
    configurations_by_id: Mapping[str, MotifConfiguration],
    results: Sequence[MotifOOFResult],
    selection: MotifSelection,
) -> List[Dict[str, object]]:
    ranks = {value: index for index, value in enumerate(selection.ranked_eligible_ids, 1)}
    rows = []
    for result in results:
        identifier = result.summary.configuration_id
        configuration = configurations_by_id[identifier]
        decision = selection.eligibility[identifier]
        rows.append(
            _search_row(
                scope,
                outer_fold,
                "motif",
                base.configuration_id,
                identifier,
                configuration.descriptor_name or "none",
                configuration.c_value if configuration.descriptor_name else "",
                configuration.source_group_exponent,
                configuration.descriptor_dimensions,
                configuration.complexity_rank,
                configuration.descriptor_name is None,
                decision.eligible,
                "|".join(decision.failures),
                ranks.get(identifier, ""),
                identifier == selection.selected.summary.configuration_id,
                result,
            )
        )
    return rows


def stability(
    fold_ids: Sequence[str],
    full_base: v1.BaseConfiguration,
    full_motif: MotifConfiguration,
    outer_base_results: Mapping[str, Mapping[str, v1.OOFResult]],
    outer_base_selections: Mapping[str, v1.SearchSelection],
    outer_motif_results: Mapping[str, Mapping[str, MotifOOFResult]],
    outer_motif_selections: Mapping[str, MotifSelection],
    contract: Mapping[str, object],
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    base_window = v1._float(
        v1._mapping(v1._mapping(contract, "base_procedure"), "selection_rule"),
        "tie_window",
    )
    motif_window = v1._float(v1._mapping(contract, "motif_selection_rule"), "tie_window")
    required = v1._int(
        v1._mapping(contract, "stability_rule"),
        "required_outer_searches_within_tie_window",
    )
    base_details = []
    motif_details = []
    base_pass = []
    motif_pass = []
    for fold in fold_ids:
        base_result = outer_base_results[fold][full_base.configuration_id]
        base_selection = outer_base_selections[fold]
        best_base = max(
            outer_base_results[fold][identifier].summary.robustness_score
            for identifier in base_selection.ranked_eligible_ids
        )
        base_eligible = base_selection.eligibility[full_base.configuration_id].eligible
        base_within = base_eligible and base_result.summary.robustness_score >= best_base - base_window
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
        motif_result = outer_motif_results[fold][full_motif.configuration_id]
        motif_selection = outer_motif_selections[fold]
        best_motif = max(
            outer_motif_results[fold][identifier].readiness_score
            for identifier in motif_selection.ranked_eligible_ids
        )
        motif_eligible = motif_selection.eligibility[full_motif.configuration_id].eligible
        motif_within = motif_eligible and motif_result.readiness_score >= best_motif - motif_window
        if motif_within:
            motif_pass.append(fold)
        motif_details.append(
            {
                "outer_fold": fold,
                "configuration_id": full_motif.configuration_id,
                "eligible": motif_eligible,
                "readiness_score": motif_result.readiness_score,
                "best_eligible_readiness_score": best_motif,
                "within_tie_window": motif_within,
            }
        )

    def payload(passing: List[str], details: List[Mapping[str, object]]) -> Mapping[str, object]:
        return {
            "required_outer_searches_within_tie_window": required,
            "observed_outer_searches_within_tie_window": len(passing),
            "passing_outer_folds": passing,
            "passed": len(passing) >= required,
            "details": details,
        }

    return payload(base_pass, base_details), payload(motif_pass, motif_details)


def evaluate_promotion_gates(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
    ordinary: OrdinaryMetrics,
    source: SourceGroupMetrics,
    outer_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    base_stability_passed: bool,
    motif_stability_passed: bool,
) -> Mapping[str, object]:
    base = dict(
        v1.evaluate_promotion_gates(
            records,
            probabilities,
            classes,
            ordinary,
            outer_rows,
            contract,
            base_stability_passed,
        )
    )
    results = [dict(value) for value in base["results"]]
    results.append(
        {
            "gate": "motif_recipe_stability",
            "comparison": "required",
            "threshold": True,
            "actual": motif_stability_passed,
            "passed": motif_stability_passed,
        }
    )
    non_regression = v1._mapping(
        v1._mapping(contract, "promotion_gates"), "v1_non_regression"
    )
    for name, actual, threshold_key in (
        ("v1_accuracy_non_regression", ordinary.accuracy, "accuracy_min"),
        ("v1_macro_f1_non_regression", ordinary.macro_f1, "macro_f1_min"),
        (
            "v1_source_robustness_non_regression",
            source.robustness_score,
            "source_robustness_min",
        ),
    ):
        threshold = v1._float(non_regression, threshold_key)
        results.append(
            {
                "gate": name,
                "comparison": ">=",
                "threshold": threshold,
                "actual": actual,
                "passed": actual >= threshold,
            }
        )
    all_passed = all(bool(result["passed"]) for result in results)
    base.update(
        {
            "all_passed": all_passed,
            "promotion_decision": "promote" if all_passed else "reject",
            "gate_count": len(results),
            "passed_count": sum(bool(result["passed"]) for result in results),
            "failed_gates": [result["gate"] for result in results if not result["passed"]],
            "results": results,
        }
    )
    return base


def prediction_rows(
    records: Sequence[DevelopmentRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
    applied: np.ndarray,
    classes: Sequence[str],
    evaluation_status: str,
    *,
    selected_by_fold: Optional[
        Mapping[str, Tuple[v1.BaseConfiguration, MotifConfiguration]]
    ] = None,
    selected_recipe: Optional[
        Tuple[v1.BaseConfiguration, MotifConfiguration]
    ] = None,
) -> List[Dict[str, object]]:
    predictions = v1._predict_labels(probabilities, classes)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rows = []
    for position, global_index in enumerate(indices.tolist()):
        record = records[global_index]
        if selected_by_fold is not None:
            base, motif = selected_by_fold[record.cv_fold]
        elif selected_recipe is not None:
            base, motif = selected_recipe
        else:
            raise ValueError("V4 prediction recipe provenance is missing")
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
            "selected_motif_configuration_id": motif.configuration_id,
            "motif_descriptor": motif.descriptor_name or "none",
            "motif_c_value": motif.c_value if motif.descriptor_name else "",
            "motif_applied": bool(applied[position]),
        }
        for class_index, class_name in enumerate(classes):
            row[f"probability_{class_name}"] = float(probabilities[position, class_index])
        rows.append(row)
    return rows


def proposal_rows(
    records: Sequence[DevelopmentRecord], cache: MotifEmbeddingCache
) -> List[Dict[str, object]]:
    rows = []
    for image_index, record in enumerate(records):
        for rank in range(cache.proposal_ids.shape[1]):
            left, top, right, bottom = cache.proposal_boxes[image_index, rank]
            x_position, y_position = cache.proposal_positions[image_index, rank]
            rows.append(
                {
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "proposal_rank": rank + 1,
                    "proposal_id": str(cache.proposal_ids[image_index, rank]),
                    "left": int(left),
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "scale": float(cache.proposal_scales[image_index, rank]),
                    "x_position": float(x_position),
                    "y_position": float(y_position),
                    "texture_score": float(cache.proposal_scores[image_index, rank]),
                }
            )
    return rows


def save_confusion_plot(
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
    axis.set_title("Phase 5 v4 nested development OOF confusion matrix")
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
            metadata={"Software": "ornament-classifier-phase5-v4"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def previous_version_comparison(
    contract: Mapping[str, object], nested: OrdinaryMetrics, source: SourceGroupMetrics
) -> Mapping[str, object]:
    inputs = v1._mapping(contract, "input_contract")
    result: Dict[str, object] = {}
    for version in ("v1", "v2", "v3"):
        path_key = f"phase_5_{version}_metrics"
        hash_key = f"phase_5_{version}_metrics_sha256"
        path = REPO_ROOT / v1._text(inputs, path_key)
        if sha256_file(path) != v1._text(inputs, hash_key):
            raise RuntimeError(f"Pinned Phase 5 {version} metrics changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["nested_aggregate_oof_metrics"]
        result[version] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "source_robustness_score": metrics["robustness_score"],
            "promotion_decision": payload["promotion_decision"],
        }
    result["v4"] = {
        "accuracy": nested.accuracy,
        "macro_f1": nested.macro_f1,
        "balanced_accuracy": nested.balanced_accuracy,
        "source_robustness_score": source.robustness_score,
    }
    return result


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover()
    contract, contract_path = load_frozen_contract(paths)
    development = load_development_contract(paths)
    v1._validate_development(development, contract)
    if len(development.records) != 1693:
        raise ValueError("Phase 5 v4 requires all 1,693 development rows")
    records = development.records
    classes = v1._classes(records)
    seed = v1._int(contract, "seed")
    pair_classes = tuple(
        str(value)
        for value in v1._sequence(
            v1._mapping(contract, "diagnostic_slice_definitions"), "pair_classes"
        )
    )
    ceramic_classes = tuple(
        str(value)
        for value in v1._sequence(
            v1._mapping(contract, "diagnostic_slice_definitions"), "ceramic_classes"
        )
    )
    print("Loading and verifying frozen v4 motif and base embeddings...", flush=True)
    base_blocks = load_base_embeddings(paths)
    motif_cache = load_motif_embeddings_v4(paths=paths)
    base_configurations = v3.build_base_configurations(contract, base_blocks)
    motif_configurations = build_motif_configurations(contract, motif_cache)
    base_by_id = {item.configuration_id: item for item in base_configurations}
    motif_by_id = {item.configuration_id: item for item in motif_configurations}
    reference_base_id = "global_cls__c10__source0"
    fold_ids = development.fold_ids
    labels, groups, folds = _record_arrays(records)
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

    nested = np.full((len(records), len(classes)), np.nan)
    uncorrected = np.full_like(nested, np.nan)
    nested_applied = np.zeros(len(records), dtype=bool)
    inner_rows: List[Dict[str, object]] = []
    outer_rows: List[Dict[str, object]] = []
    selected_by_outer: Dict[
        str, Tuple[v1.BaseConfiguration, MotifConfiguration]
    ] = {}
    outer_base_results: Dict[str, Mapping[str, v1.OOFResult]] = {}
    outer_base_selections: Dict[str, v1.SearchSelection] = {}
    outer_motif_results: Dict[str, Mapping[str, MotifOOFResult]] = {}
    outer_motif_selections: Dict[str, MotifSelection] = {}

    for number, outer_fold in enumerate(fold_ids, 1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        print(
            f"Outer v4 search {number}/{len(fold_ids)}: hold out fold {outer_fold}",
            flush=True,
        )
        base_results = tuple(
            v1.evaluate_base_oof(
                configuration, records, classes, ceramic_classes, inner_folds, seed
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
        motif_results = evaluate_motif_oof_grid(
            base_selection.selected,
            selected_base,
            motif_configurations,
            motif_cache,
            records,
            classes,
            pair_classes,
            ceramic_classes,
            inner_folds,
            contract,
            seed,
        )
        motif_selection = select_motif_candidates(motif_results, contract)
        selected_motif = motif_by_id[motif_selection.selected.summary.configuration_id]
        selected_by_outer[outer_fold] = (selected_base, selected_motif)
        outer_base_results[outer_fold] = fold_base_by_id
        outer_base_selections[outer_fold] = base_selection
        outer_motif_results[outer_fold] = {
            result.summary.configuration_id: result for result in motif_results
        }
        outer_motif_selections[outer_fold] = motif_selection
        inner_rows.extend(
            base_search_rows(
                "outer_inner_oof",
                outer_fold,
                base_results,
                base_selection,
                reference_base_id,
            )
        )
        inner_rows.extend(
            motif_search_rows(
                "outer_inner_oof",
                outer_fold,
                selected_base,
                motif_by_id,
                motif_results,
                motif_selection,
            )
        )
        validation, base_probabilities, probabilities, applied, iterations = fit_outer_procedure(
            selected_base,
            selected_motif,
            motif_cache,
            outer_fold,
            records,
            classes,
            ceramic_classes,
            seed,
        )
        uncorrected[validation] = base_probabilities
        nested[validation] = probabilities
        nested_applied[validation] = applied
        ordinary = ordinary_metrics(labels[validation], probabilities, classes)
        predictions = v1._predict_labels(probabilities, classes)
        source = source_group_metrics(
            labels[validation], predictions, groups[validation], classes, ceramic_classes
        )
        boundary = boundary_metrics(
            labels[validation],
            probabilities,
            classes,
            pair_classes=pair_classes,
            ceramic_classes=ceramic_classes,
        )
        hard = v3._hard_three_accuracy(records, validation, predictions, contract)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_base_configuration_id": selected_base.configuration_id,
                "selected_motif_configuration_id": selected_motif.configuration_id,
                "motif_descriptor": selected_motif.descriptor_name or "none",
                "motif_c_value": selected_motif.c_value if selected_motif.descriptor_name else "",
                "motif_applied_count": int(applied.sum()),
                "train_examples": int(np.sum(folds != outer_fold)),
                "validation_examples": len(validation),
                "fit_count": len(iterations),
                "maximum_iterations": max(iterations),
                "accuracy": ordinary.accuracy,
                "macro_f1": ordinary.macro_f1,
                "balanced_accuracy": ordinary.balanced_accuracy,
                "source_group_balanced_accuracy": source.source_group_balanced_accuracy,
                "ceramic_worst_group_recall": source.ceramic_worst_group_recall,
                "robustness_score": source.robustness_score,
                "hard_three_accuracy": hard,
                "opishnyan_recall": boundary.opishnyan_recall,
                "bubnivka_recall": boundary.bubnivka_recall,
                "bubnivka_precision": boundary.bubnivka_precision,
                "ceramic_macro_f1": boundary.ceramic_macro_f1,
                "per_class_recall_json": stable_json(dict(ordinary.per_class_recall)),
                "per_class_group_recall_json": stable_json(dict(source.per_class_group_recall)),
            }
        )
    if not np.isfinite(nested).all() or not np.isfinite(uncorrected).all():
        raise RuntimeError("Phase 5 v4 nested predictions are incomplete")

    print("Running full-development Phase 5 v4 selection...", flush=True)
    full_base_results = tuple(
        v1.evaluate_base_oof(
            configuration, records, classes, ceramic_classes, fold_ids, seed
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
    full_base = base_by_id[full_base_selection.selected.summary.configuration_id]
    full_motif_results = evaluate_motif_oof_grid(
        full_base_selection.selected,
        full_base,
        motif_configurations,
        motif_cache,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        fold_ids,
        contract,
        seed,
    )
    full_motif_selection = select_motif_candidates(full_motif_results, contract)
    full_motif = motif_by_id[full_motif_selection.selected.summary.configuration_id]
    full_selected = full_motif_selection.selected
    full_rows = base_search_rows(
        "full_development_oof",
        "",
        full_base_results,
        full_base_selection,
        reference_base_id,
    )
    full_rows.extend(
        motif_search_rows(
            "full_development_oof",
            "",
            full_base,
            motif_by_id,
            full_motif_results,
            full_motif_selection,
        )
    )
    reproduction = v2.validate_v1_reproduction(
        records,
        uncorrected,
        {
            fold: (base.configuration_id, "pair_none")
            for fold, (base, _) in selected_by_outer.items()
        },
        full_base,
        contract,
        classes,
        ceramic_classes,
    )
    base_stability, motif_stability = stability(
        fold_ids,
        full_base,
        full_motif,
        outer_base_results,
        outer_base_selections,
        outer_motif_results,
        outer_motif_selections,
        contract,
    )
    nested_predictions = v1._predict_labels(nested, classes)
    nested_ordinary = ordinary_metrics(labels, nested, classes)
    nested_source = source_group_metrics(
        labels, nested_predictions, groups, classes, ceramic_classes
    )
    nested_boundary = boundary_metrics(
        labels,
        nested,
        classes,
        pair_classes=pair_classes,
        ceramic_classes=ceramic_classes,
    )
    nested_hard = v3._hard_three_accuracy(
        records,
        np.arange(len(records), dtype=np.int64),
        nested_predictions,
        contract,
    )
    if nested_hard is None:
        raise RuntimeError("V4 nested result has no hard-three support")
    nested_readiness = harmonic_mean_many(
        (
            nested_ordinary.accuracy,
            nested_ordinary.macro_f1,
            nested_source.source_group_balanced_accuracy,
            nested_source.ceramic_worst_group_recall,
            nested_boundary.opishnyan_recall,
            nested_boundary.bubnivka_recall,
            nested_boundary.bubnivka_precision,
            nested_boundary.ceramic_macro_f1,
            nested_hard,
        )
    )
    promotion = evaluate_promotion_gates(
        records,
        nested,
        classes,
        nested_ordinary,
        nested_source,
        outer_rows,
        contract,
        bool(base_stability["passed"]),
        bool(motif_stability["passed"]),
    )

    output_dir = v1._resolve_output_dir(paths, contract, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_grid_rows = v2._base_grid_rows(base_configurations)
    motif_grid_rows = [
        {
            "configuration_id": item.configuration_id,
            "descriptor": item.descriptor_name or "none",
            "c_value": item.c_value if item.descriptor_name else "",
            "source_group_exponent": item.source_group_exponent,
            "complexity_rank": item.complexity_rank,
            "descriptor_dimensions": item.descriptor_dimensions,
            "application": (
                "none"
                if item.descriptor_name is None
                else "concatenate descriptor to every selected base feature block"
            ),
        }
        for item in motif_configurations
    ]
    nested_prediction_rows = prediction_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested,
        nested_applied,
        classes,
        "nested_outer_oof_adaptive_phase5_v4_procedure",
        selected_by_fold=selected_by_outer,
    )
    selected_prediction_rows = prediction_rows(
        records,
        full_selected.evaluated_indices,
        full_selected.probabilities,
        full_selected.applied_mask,
        classes,
        "selection_conditional_full_development_oof_v4",
        selected_recipe=(full_base, full_motif),
    )
    per_class_rows = v1._per_class_rows(
        nested_ordinary, classes, "nested_outer_oof_v4"
    )
    diagnostic_rows = v1.diagnostic_slice_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested,
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
    writes = (
        ("base_grid.csv", base_grid_rows),
        ("motif_grid.csv", motif_grid_rows),
        ("motif_proposals.csv", proposal_rows(records, motif_cache)),
        ("inner_search.csv", inner_rows),
        ("outer_fold_metrics.csv", outer_rows),
        ("nested_oof_predictions.csv", nested_prediction_rows),
        ("full_development_selection.csv", full_rows),
        ("selected_oof_predictions.csv", selected_prediction_rows),
        ("per_class_metrics.csv", per_class_rows),
        ("diagnostic_slices.csv", diagnostic_rows),
        ("confusion_pairs.csv", confusion_pairs),
    )
    for name, rows in writes:
        v1._write_csv(output_dir / name, rows, tuple(rows[0]))
    v1._write_csv(
        output_dir / "confusion_matrix.csv", matrix_rows, ("true_class", *classes)
    )
    save_confusion_plot(matrix, classes, output_dir / "confusion_matrix.png")

    required = tuple(
        str(value)
        for value in v1._sequence(v1._mapping(contract, "output_contract"), "required")
    )
    metrics_payload: Dict[str, object] = {
        "phase": PHASE,
        "experiment_version": contract.get("experiment_version"),
        "experiment_contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "experiment_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "code_provenance": code_provenance(),
        "evaluation_scope": "development_only_selection_aware_nested_source_atomic_cv",
        "sealed_test_evaluated": False,
        "performance_estimate": {
            "available": False,
            "nested_development_estimate_available": True,
            "reason": "Nested OOF estimates the adaptive v4 procedure, but v1-v3 development findings informed the frozen motif family.",
        },
        "probability_policy": {
            "calibrated": False,
            "threshold_selection_used": False,
            "interpretation": "uncalibrated ranking scores only",
        },
        "phase_6_transition_allowed": promotion["promotion_decision"] == "promote",
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
        "motif_configuration_count": len(motif_configurations),
        "motif_proposal_count_per_image": motif_cache.proposal_ids.shape[1],
        "nested_selected_recipes_by_outer_fold": {
            fold: {
                "base_configuration_id": base.configuration_id,
                "motif_configuration_id": motif.configuration_id,
            }
            for fold, (base, motif) in selected_by_outer.items()
        },
        "nested_aggregate_oof_metrics": {
            **asdict(nested_ordinary),
            **asdict(nested_source),
            **asdict(nested_boundary),
            "readiness_score": nested_readiness,
            "hard_three_accuracy": nested_hard,
            "motif_applied_count": int(nested_applied.sum()),
        },
        "nested_outer_fold_metrics": outer_rows,
        "full_development_selected_candidate": {
            "base_configuration_id": full_base.configuration_id,
            "base_feature_family": full_base.family.family_id,
            "base_c_value": full_base.c_value,
            "base_source_group_exponent": full_base.source_group_exponent,
            "motif_configuration_id": full_motif.configuration_id,
            "motif_descriptor": full_motif.descriptor_name or "none",
            "motif_c_value": full_motif.c_value if full_motif.descriptor_name else None,
            "motif_source_group_exponent": full_motif.source_group_exponent,
            "descriptor_dimensions": full_motif.descriptor_dimensions,
        },
        "full_development_selected_oof_metrics": {
            **asdict(full_selected.ordinary),
            **asdict(full_selected.source),
            **asdict(full_selected.boundary),
            "readiness_score": full_selected.readiness_score,
            "hard_three_accuracy": full_selected.hard_three_accuracy,
            "motif_applied_count": int(full_selected.applied_mask.sum()),
        },
        "previous_phase5_comparison": previous_version_comparison(
            contract, nested_ordinary, nested_source
        ),
        "v1_reproduction_validation": reproduction,
        "base_recipe_stability": base_stability,
        "motif_recipe_stability": motif_stability,
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
        "base_embedding_provenance": {
            name: dict(block.provenance) for name, block in sorted(base_blocks.items())
        },
        "motif_embedding_provenance": motif_cache.provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "required_outputs": required,
    }
    v1._write_json(output_dir / "metrics.json", metrics_payload)
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Required Phase 5 v4 outputs are missing: " + ", ".join(missing))
    print(
        json.dumps(
            {
                "nested_accuracy": nested_ordinary.accuracy,
                "nested_macro_f1": nested_ordinary.macro_f1,
                "nested_readiness_score": nested_readiness,
                "selected_base": full_base.configuration_id,
                "selected_motif": full_motif.configuration_id,
                "promotion_decision": promotion["promotion_decision"],
                "phase_6_transition_allowed": promotion["promotion_decision"] == "promote",
                "sealed_test_evaluated": False,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen Phase 5 v3 cross-encoder consensus experiment.

V3 preserves the Phase 5 v1 nested base procedure, then compares a bounded
family of development-only cross-encoder consensus fusions. The sealed test,
calibration, thresholding, and Phase 6 packaging are outside this runner.
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
from ornament_classifier.consensus import (  # noqa: E402
    FusionCandidateSummary,
    FusionEligibilityReference,
    FusionEligibilityResult,
    apply_multiclass_logit_fusion,
    apply_pair_consensus,
    fusion_eligibility,
    harmonic_mean_many,
    rank_fusion_candidates,
)
from ornament_classifier.contracts import (  # noqa: E402
    DevelopmentContract,
    DevelopmentRecord,
    load_development_contract,
)
from ornament_classifier.embeddings import EmbeddingBlock  # noqa: E402
from ornament_classifier.embeddings_v3 import (  # noqa: E402
    load_allowlisted_embeddings_v3,
)
from ornament_classifier.pairwise import BoundaryMetrics, boundary_metrics  # noqa: E402
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    OrdinaryMetrics,
    SourceGroupMetrics,
    compute_source_group_weights,
    fit_logistic_probe,
    ordinary_metrics,
    source_group_metrics,
)


PHASE = "step02_phase_5_source_robustness_v3"
CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v3.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "828cc5aa45724518e10808929589722133ce336ee272cb8bda20b5089c0dc980"
)
SCRIPT_PATH = Path(__file__).resolve()
MAX_ITERATIONS = 10_000
TOLERANCE = 1e-6


@dataclass(frozen=True)
class FusionFamily:
    family_id: str
    block_names: Tuple[str, ...]
    blocks: Tuple[np.ndarray, ...]
    head_count: int
    encoder_count: int


@dataclass(frozen=True)
class FusionConfiguration:
    configuration_id: str
    mode: str
    family: Optional[FusionFamily]
    blend_weight: float
    intervention_rank: int


@dataclass(frozen=True)
class AuxiliaryOOF:
    probabilities_by_block: Mapping[str, np.ndarray]
    evaluated_indices: np.ndarray
    iterations_by_block: Mapping[str, Tuple[int, ...]]


@dataclass(frozen=True)
class FusionOOFResult:
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
class FusionSelection:
    selected: FusionOOFResult
    reference: FusionEligibilityReference
    eligibility: Mapping[str, FusionEligibilityResult]
    ranked_eligible_ids: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the frozen v3 contract directory.",
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
        raise ValueError(f"Frozen Phase 5 v3 contract is missing: {expected_path}")
    observed = sha256_file(path)
    if observed != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "Phase 5 v3 contract changed after it was frozen: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {observed}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Frozen Phase 5 v3 contract must be an object")
    if contract.get("iteration") != "v3" or contract.get("status") != "frozen_before_fit":
        raise ValueError("Phase 5 v3 contract has an invalid iteration or status")
    if v1._mapping(contract, "scope").get("sealed_test_access") != "forbidden":
        raise ValueError("Phase 5 v3 requires sealed access to remain forbidden")
    return contract, path


def code_provenance() -> Mapping[str, object]:
    paths = (
        SCRIPT_PATH,
        STEP_ROOT / "scripts" / "run_phase5_source_robustness.py",
        STEP_ROOT / "scripts" / "run_phase5_v2_pairwise.py",
        STEP_ROOT / "src" / "ornament_classifier" / "consensus.py",
        STEP_ROOT / "src" / "ornament_classifier" / "pairwise.py",
        STEP_ROOT / "src" / "ornament_classifier" / "robustness.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings_v3.py",
        STEP_ROOT / "src" / "ornament_classifier" / "embeddings.py",
        STEP_ROOT / "src" / "ornament_classifier" / "contracts.py",
        STEP_ROOT / "src" / "ornament_classifier" / "paths.py",
        STEP_ROOT / "requirements-phase5.txt",
    )
    files: Dict[str, str] = {}
    combined = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Phase 5 v3 provenance input is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        combined.update(f"{relative}\x1f{digest}\n".encode("utf-8"))
    return {"files_sha256": files, "combined_sha256": combined.hexdigest()}


def _embedding_requests(
    contract: Mapping[str, object],
) -> Mapping[str, Tuple[str, str]]:
    references = {
        "dinov3_center.embedding__cls",
        "dinov3_letterbox.embedding__cls",
        "dinov3_global.embedding__cls",
    }
    heads = v1._mapping(contract, "consensus_heads")
    for family in v1._sequence(heads, "families"):
        if not isinstance(family, dict):
            raise ValueError("Consensus family specs must be objects")
        for reference in v1._sequence(family, "blocks"):
            if not isinstance(reference, str) or reference.count(".") != 1:
                raise ValueError(f"Invalid v3 embedding reference: {reference!r}")
            references.add(reference)
    return {
        reference: tuple(reference.split(".", 1))  # type: ignore[arg-type]
        for reference in sorted(references)
    }


def build_base_configurations(
    contract: Mapping[str, object],
    blocks: Mapping[str, EmbeddingBlock],
) -> Tuple[v1.BaseConfiguration, ...]:
    global_family = v1.FeatureFamily(
        family_id="global_cls",
        kind="single_embedding",
        block_names=("dinov3_global.embedding__cls",),
        blocks=(blocks["dinov3_global.embedding__cls"].values,),
        inference_cost_rank=6,
    )
    three_view = v1.FeatureFamily(
        family_id="three_view_cls_logit_ensemble",
        kind="centered_logit_ensemble",
        block_names=(
            "dinov3_center.embedding__cls",
            "dinov3_letterbox.embedding__cls",
            "dinov3_global.embedding__cls",
        ),
        blocks=(
            blocks["dinov3_center.embedding__cls"].values,
            blocks["dinov3_letterbox.embedding__cls"].values,
            blocks["dinov3_global.embedding__cls"].values,
        ),
        inference_cost_rank=8,
    )
    families = {"global_cls": global_family, "three_view_cls_logit_ensemble": three_view}
    configurations = []
    for spec in v1._sequence(v1._mapping(contract, "base_procedure"), "configurations"):
        if not isinstance(spec, dict):
            raise ValueError("V3 base configuration specs must be objects")
        configurations.append(
            v1.BaseConfiguration(
                configuration_id=v1._text(spec, "id"),
                family=families[v1._text(spec, "feature_family")],
                c_value=v1._float(spec, "c_value"),
                source_group_exponent=v1._float(spec, "source_group_exponent"),
            )
        )
    if len(configurations) != 3 or len({item.configuration_id for item in configurations}) != 3:
        raise ValueError("Phase 5 v3 requires exactly three unique base configurations")
    return tuple(configurations)


def build_fusion_families(
    contract: Mapping[str, object], blocks: Mapping[str, EmbeddingBlock]
) -> Tuple[FusionFamily, ...]:
    families = []
    for spec in v1._sequence(v1._mapping(contract, "consensus_heads"), "families"):
        if not isinstance(spec, dict):
            raise ValueError("Consensus family specs must be objects")
        names = tuple(str(value) for value in v1._sequence(spec, "blocks"))
        family = FusionFamily(
            family_id=v1._text(spec, "id"),
            block_names=names,
            blocks=tuple(blocks[name].values for name in names),
            head_count=v1._int(spec, "head_count"),
            encoder_count=v1._int(spec, "encoder_count"),
        )
        if family.head_count != len(names) or len(set(names)) != len(names):
            raise ValueError("V3 family head count or block uniqueness is invalid")
        families.append(family)
    if len(families) != 3 or len({family.family_id for family in families}) != 3:
        raise ValueError("Phase 5 v3 requires exactly three consensus families")
    return tuple(families)


def build_fusion_configurations(
    contract: Mapping[str, object], families: Sequence[FusionFamily]
) -> Tuple[FusionConfiguration, ...]:
    fusion = v1._mapping(contract, "consensus_fusion")
    modes = {
        "multiclass_logit_fusion": ("multiclass", 3),
        "pair_logit_consensus": ("pair", 2),
        "pair_majority_gated_logit_consensus": ("pair_majority", 1),
    }
    configurations = [
        FusionConfiguration("fusion_none", "none", None, 0.0, 0)
    ]
    declared = tuple(str(value) for value in v1._sequence(fusion, "versions"))
    if set(declared) != {"none", *modes}:
        raise ValueError("V3 fusion versions disagree with the frozen contract")
    for family in families:
        for declared_mode in declared:
            if declared_mode == "none":
                continue
            short_mode, intervention_rank = modes[declared_mode]
            for value in v1._sequence(fusion, "blend_weights"):
                blend = float(value)
                identifier = (
                    f"{family.family_id}__{short_mode}"
                    f"__blend{v1._number_id(blend)}"
                )
                configurations.append(
                    FusionConfiguration(
                        identifier, short_mode, family, blend, intervention_rank
                    )
                )
    expected = v1._int(fusion, "grid_configuration_count_including_none")
    if len(configurations) != expected or len({item.configuration_id for item in configurations}) != expected:
        raise ValueError("Fusion grid does not match the frozen v3 count")
    return tuple(configurations)


def _record_arrays(
    records: Sequence[DevelopmentRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return v1._record_arrays(records)


def _unique_auxiliary_blocks(families: Sequence[FusionFamily]) -> Mapping[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for family in families:
        for name, block in zip(family.block_names, family.blocks):
            previous = result.get(name)
            if previous is not None and previous is not block:
                raise ValueError(f"Auxiliary block identity disagrees: {name}")
            result[name] = block
    return result


def fit_auxiliary_oof(
    families: Sequence[FusionFamily],
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> AuxiliaryOOF:
    labels, groups, folds = _record_arrays(records)
    fold_ids = tuple(eligible_fold_ids)
    pool_mask = np.isin(folds, fold_ids)
    evaluated = np.flatnonzero(pool_mask)
    heads = v1._mapping(contract, "consensus_heads")
    c_value = v1._float(heads, "c_value")
    exponent = v1._float(heads, "source_group_exponent")
    outputs: Dict[str, np.ndarray] = {}
    iterations_by_block: Dict[str, Tuple[int, ...]] = {}
    for name, block in _unique_auxiliary_blocks(families).items():
        probabilities = np.full((len(records), len(classes)), np.nan, dtype=np.float64)
        iterations: List[int] = []
        for validation_fold in fold_ids:
            validation = np.flatnonzero(pool_mask & (folds == validation_fold))
            train = np.flatnonzero(pool_mask & (folds != validation_fold))
            v1._assert_source_disjoint_partitions(
                train, validation, groups, f"v3 auxiliary OOF {name} fold {validation_fold}"
            )
            weights = compute_source_group_weights(labels[train], groups[train], exponent)
            with threadpool_limits(limits=1):
                model = fit_logistic_probe(
                    block[train],
                    labels[train],
                    c_value=c_value,
                    sample_weight=weights,
                    classes=classes,
                    seed=seed,
                    max_iter=MAX_ITERATIONS,
                    tolerance=TOLERANCE,
                )
            if not model.converged:
                raise RuntimeError(f"V3 auxiliary head did not converge: {name}")
            iterations.extend(v1._flatten_iterations(model.iterations))
            probabilities[validation] = model.predict_proba(block[validation])
        pooled = probabilities[evaluated]
        if not np.isfinite(pooled).all():
            raise RuntimeError(f"V3 auxiliary OOF predictions are incomplete: {name}")
        outputs[name] = pooled
        iterations_by_block[name] = tuple(iterations)
    return AuxiliaryOOF(outputs, evaluated, iterations_by_block)


def _apply_configuration(
    configuration: FusionConfiguration,
    base_probabilities: np.ndarray,
    auxiliary: AuxiliaryOOF,
    classes: Sequence[str],
    pair_classes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if configuration.mode == "none":
        return np.array(base_probabilities, copy=True), np.zeros(len(base_probabilities), dtype=bool)
    if configuration.family is None:
        raise ValueError("Non-none v3 fusion is missing a family")
    heads = tuple(
        auxiliary.probabilities_by_block[name]
        for name in configuration.family.block_names
    )
    if configuration.mode == "multiclass":
        return (
            apply_multiclass_logit_fusion(
                base_probabilities, heads, configuration.blend_weight
            ),
            np.ones(len(base_probabilities), dtype=bool),
        )
    if configuration.mode in {"pair", "pair_majority"}:
        return apply_pair_consensus(
            base_probabilities,
            heads,
            classes,
            pair_classes,
            configuration.blend_weight,
            majority_gated=configuration.mode == "pair_majority",
        )
    raise ValueError(f"Unknown v3 fusion mode: {configuration.mode}")


def _hard_three_accuracy(
    records: Sequence[DevelopmentRecord],
    indices: np.ndarray,
    predictions: np.ndarray,
    contract: Mapping[str, object],
) -> Optional[float]:
    definitions = v1._mapping(contract, "diagnostic_slice_definitions")
    hard = set(str(value) for value in v1._sequence(definitions, "hard_three_opishnyan_cohorts"))
    selected = [
        position
        for position, global_index in enumerate(indices.tolist())
        if records[global_index].ornament_label == "01_opishnyan_ceramics"
        and records[global_index].source_atomic_cohort_ids in hard
    ]
    if not selected:
        return None
    return float(
        np.mean(
            predictions[np.asarray(selected, dtype=np.int64)]
            == "01_opishnyan_ceramics"
        )
    )


def _summarize_fusion(
    configuration: FusionConfiguration,
    base_configuration: v1.BaseConfiguration,
    probabilities: np.ndarray,
    applied_mask: np.ndarray,
    evaluated_indices: np.ndarray,
    auxiliary: AuxiliaryOOF,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    contract: Mapping[str, object],
) -> FusionOOFResult:
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
    hard_accuracy = _hard_three_accuracy(
        records, evaluated_indices, predictions, contract
    )
    if hard_accuracy is None:
        raise RuntimeError("Pooled v3 selection rows have no hard-three support")
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
            hard_accuracy,
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
        fold_hard = _hard_three_accuracy(records, fold_indices, fold_predictions, contract)
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
                "applied_count": int(applied_mask[mask].sum()),
            }
        )
    family = configuration.family
    relevant_iterations = (
        tuple(
            iteration
            for name in family.block_names
            for iteration in auxiliary.iterations_by_block[name]
        )
        if family is not None
        else ()
    )
    summary = FusionCandidateSummary(
        configuration_id=configuration.configuration_id,
        readiness_score=readiness,
        minimum_fold_readiness_score=min(fold_readiness),
        accuracy=ordinary.accuracy,
        macro_f1=ordinary.macro_f1,
        source_robustness_score=source.robustness_score,
        opishnyan_group_recall=source.per_class_group_recall[pair_classes[0]],
        intervention_rank=configuration.intervention_rank,
        blend_weight=configuration.blend_weight,
        head_count=family.head_count if family else 0,
        encoder_count=family.encoder_count if family else 0,
        per_class_recall=dict(ordinary.per_class_recall),
    )
    return FusionOOFResult(
        probabilities=probabilities,
        applied_mask=applied_mask,
        evaluated_indices=evaluated_indices,
        fold_rows=tuple(fold_rows),
        ordinary=ordinary,
        source=source,
        boundary=boundary,
        readiness_score=readiness,
        hard_three_accuracy=hard_accuracy,
        summary=summary,
        maximum_iterations=max(relevant_iterations) if relevant_iterations else 0,
        fit_count=len(relevant_iterations),
    )


def evaluate_fusion_oof_grid(
    base_result: v1.OOFResult,
    base_configuration: v1.BaseConfiguration,
    configurations: Sequence[FusionConfiguration],
    families: Sequence[FusionFamily],
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    eligible_fold_ids: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> Tuple[FusionOOFResult, ...]:
    auxiliary = fit_auxiliary_oof(
        families, records, classes, eligible_fold_ids, contract, seed
    )
    if not np.array_equal(base_result.evaluated_indices, auxiliary.evaluated_indices):
        raise RuntimeError("V3 base and auxiliary OOF rows disagree")
    results = []
    for configuration in configurations:
        probabilities, applied = _apply_configuration(
            configuration,
            base_result.probabilities,
            auxiliary,
            classes,
            pair_classes,
        )
        results.append(
            _summarize_fusion(
                configuration,
                base_configuration,
                probabilities,
                applied,
                auxiliary.evaluated_indices,
                auxiliary,
                records,
                classes,
                pair_classes,
                ceramic_classes,
                contract,
            )
        )
    return tuple(results)


def select_fusion_candidates(
    results: Sequence[FusionOOFResult], contract: Mapping[str, object]
) -> FusionSelection:
    references = [result for result in results if result.summary.configuration_id == "fusion_none"]
    if len(references) != 1:
        raise ValueError("V3 grid must contain exactly one no-fusion reference")
    baseline = references[0]
    reference = FusionEligibilityReference(
        accuracy=baseline.ordinary.accuracy,
        macro_f1=baseline.ordinary.macro_f1,
        source_robustness_score=baseline.source.robustness_score,
        opishnyan_group_recall=baseline.summary.opishnyan_group_recall,
        per_class_recall=baseline.ordinary.per_class_recall,
    )
    rule = v1._mapping(contract, "fusion_selection_rule")
    limits = v1._mapping(rule, "eligibility_relative_to_uncorrected_selected_base")
    kwargs = {
        "maximum_accuracy_drop": v1._float(limits, "maximum_accuracy_drop"),
        "maximum_macro_f1_drop": v1._float(limits, "maximum_macro_f1_drop"),
        "maximum_source_robustness_drop": v1._float(limits, "maximum_source_robustness_drop"),
        "maximum_opishnyan_group_recall_drop": v1._float(limits, "maximum_opishnyan_group_recall_drop"),
        "other_class_recall_drops": {
            label: float(value)
            for label, value in v1._mapping(limits, "maximum_other_class_recall_drop").items()
        },
    }
    eligibility: Dict[str, FusionEligibilityResult] = {}
    eligible = []
    by_id = {result.summary.configuration_id: result for result in results}
    if len(by_id) != len(results):
        raise ValueError("V3 fusion result IDs must be unique")
    for result in results:
        decision = fusion_eligibility(result.summary, reference, **kwargs)
        eligibility[result.summary.configuration_id] = decision
        if decision.eligible:
            eligible.append(result.summary)
    ranked = rank_fusion_candidates(
        eligible, tie_window=v1._float(rule, "tie_window")
    )
    ranked_ids = tuple(summary.configuration_id for summary in ranked)
    return FusionSelection(by_id[ranked_ids[0]], reference, eligibility, ranked_ids)


def _fit_auxiliary_outer(
    family: FusionFamily,
    train: np.ndarray,
    validation: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
    heads = v1._mapping(contract, "consensus_heads")
    c_value = v1._float(heads, "c_value")
    exponent = v1._float(heads, "source_group_exponent")
    probabilities = []
    iterations = []
    for name, block in zip(family.block_names, family.blocks):
        v1._assert_source_disjoint_partitions(
            train, validation, groups, f"v3 outer auxiliary {name}"
        )
        weights = compute_source_group_weights(labels[train], groups[train], exponent)
        with threadpool_limits(limits=1):
            model = fit_logistic_probe(
                block[train],
                labels[train],
                c_value=c_value,
                sample_weight=weights,
                classes=classes,
                seed=seed,
                max_iter=MAX_ITERATIONS,
                tolerance=TOLERANCE,
            )
        if not model.converged:
            raise RuntimeError(f"V3 outer auxiliary head did not converge: {name}")
        probabilities.append(model.predict_proba(block[validation]))
        iterations.extend(v1._flatten_iterations(model.iterations))
    return tuple(probabilities), tuple(iterations)


def fit_outer_procedure(
    base: v1.BaseConfiguration,
    fusion: FusionConfiguration,
    outer_fold: str,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    pair_classes: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    labels, groups, folds = _record_arrays(records)
    validation, base_probabilities, base_iterations = v1._fit_outer_and_predict(
        base, "none", outer_fold, records, classes, ceramic_classes, seed
    )
    if fusion.mode == "none":
        return (
            validation,
            base_probabilities,
            np.array(base_probabilities, copy=True),
            np.zeros(len(validation), dtype=bool),
            base_iterations,
        )
    if fusion.family is None:
        raise ValueError("Selected v3 fusion is missing a family")
    train = np.flatnonzero(folds != outer_fold)
    heads, auxiliary_iterations = _fit_auxiliary_outer(
        fusion.family,
        train,
        validation,
        labels,
        groups,
        classes,
        contract,
        seed,
    )
    auxiliary = AuxiliaryOOF(
        {name: values for name, values in zip(fusion.family.block_names, heads)},
        validation,
        {},
    )
    corrected, applied = _apply_configuration(
        fusion, base_probabilities, auxiliary, classes, pair_classes
    )
    return validation, base_probabilities, corrected, applied, tuple((*base_iterations, *auxiliary_iterations))


def _search_row(
    scope: str,
    outer_fold: str,
    stage: str,
    configuration_id: str,
    base_id: str,
    fusion_id: str,
    family_id: str,
    mode: str,
    blend: object,
    head_count: int,
    encoder_count: int,
    intervention_rank: int,
    exact_reference: bool,
    eligible: bool,
    failures: str,
    rank: object,
    selected: bool,
    evaluated_examples: int,
    fit_count: int,
    maximum_iterations: int,
    ordinary: OrdinaryMetrics,
    source: SourceGroupMetrics,
    boundary: Optional[BoundaryMetrics],
    readiness: object,
    minimum_readiness: object,
    hard_three: object,
    applied_count: object,
) -> Dict[str, object]:
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "stage": stage,
        "configuration_id": configuration_id,
        "base_configuration_id": base_id,
        "fusion_configuration_id": fusion_id,
        "feature_family": family_id,
        "fusion_mode": mode,
        "blend_weight": blend,
        "head_count": head_count,
        "encoder_count": encoder_count,
        "intervention_rank": intervention_rank,
        "is_exact_reference": exact_reference,
        "eligible": eligible,
        "eligibility_failures": failures,
        "eligible_selection_rank": rank,
        "selected": selected,
        "evaluated_examples": evaluated_examples,
        "fit_count": fit_count,
        "maximum_iterations": maximum_iterations,
        "accuracy": ordinary.accuracy,
        "macro_f1": ordinary.macro_f1,
        "balanced_accuracy": ordinary.balanced_accuracy,
        "source_group_balanced_accuracy": source.source_group_balanced_accuracy,
        "ceramic_worst_group_recall": source.ceramic_worst_group_recall,
        "robustness_score": source.robustness_score,
        "readiness_score": readiness,
        "minimum_fold_readiness_score": minimum_readiness,
        "hard_three_accuracy": hard_three,
        "applied_count": applied_count,
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
                scope, outer_fold, "base", identifier, identifier, "",
                result.summary.configuration_id.split("__c", 1)[0],
                "not_applicable", "", 0, 0, 0,
                identifier == reference_id, decision.eligible,
                "|".join(decision.failures), ranks.get(identifier, ""),
                identifier == selection.selected.summary.configuration_id,
                len(result.evaluated_indices), result.fit_count,
                result.maximum_iterations, result.ordinary, result.source,
                None, "", "", "", "",
            )
        )
    return rows


def fusion_search_rows(
    scope: str,
    outer_fold: str,
    base: v1.BaseConfiguration,
    results: Sequence[FusionOOFResult],
    selection: FusionSelection,
) -> List[Dict[str, object]]:
    ranks = {value: index for index, value in enumerate(selection.ranked_eligible_ids, 1)}
    rows = []
    for result in results:
        summary = result.summary
        decision = selection.eligibility[summary.configuration_id]
        family = next(
            (
                configuration.family
                for configuration in FUSION_CONFIGURATIONS
                if configuration.configuration_id == summary.configuration_id
            ),
            None,
        )
        configuration = FUSION_BY_ID[summary.configuration_id]
        rows.append(
            _search_row(
                scope, outer_fold, "fusion", summary.configuration_id,
                base.configuration_id, summary.configuration_id,
                family.family_id if family else "none", configuration.mode,
                configuration.blend_weight, summary.head_count,
                summary.encoder_count, summary.intervention_rank,
                summary.configuration_id == "fusion_none", decision.eligible,
                "|".join(decision.failures), ranks.get(summary.configuration_id, ""),
                summary.configuration_id == selection.selected.summary.configuration_id,
                len(result.evaluated_indices), result.fit_count,
                result.maximum_iterations, result.ordinary, result.source,
                result.boundary, result.readiness_score,
                summary.minimum_fold_readiness_score, result.hard_three_accuracy,
                int(result.applied_mask.sum()),
            )
        )
    return rows


def stability(
    fold_ids: Sequence[str],
    full_base: v1.BaseConfiguration,
    full_fusion: FusionConfiguration,
    outer_base_results: Mapping[str, Mapping[str, v1.OOFResult]],
    outer_base_selections: Mapping[str, v1.SearchSelection],
    outer_fusion_results: Mapping[str, Mapping[str, FusionOOFResult]],
    outer_fusion_selections: Mapping[str, FusionSelection],
    contract: Mapping[str, object],
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    base_window = v1._float(
        v1._mapping(v1._mapping(contract, "base_procedure"), "selection_rule"),
        "tie_window",
    )
    fusion_window = v1._float(v1._mapping(contract, "fusion_selection_rule"), "tie_window")
    required = v1._int(v1._mapping(contract, "stability_rule"), "required_outer_searches_within_tie_window")
    base_details = []
    fusion_details = []
    base_pass = []
    fusion_pass = []
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
        fusion_result = outer_fusion_results[fold][full_fusion.configuration_id]
        fusion_selection = outer_fusion_selections[fold]
        best_fusion = max(
            outer_fusion_results[fold][identifier].readiness_score
            for identifier in fusion_selection.ranked_eligible_ids
        )
        fusion_eligible = fusion_selection.eligibility[full_fusion.configuration_id].eligible
        fusion_within = fusion_eligible and fusion_result.readiness_score >= best_fusion - fusion_window
        if fusion_within:
            fusion_pass.append(fold)
        fusion_details.append(
            {
                "outer_fold": fold,
                "configuration_id": full_fusion.configuration_id,
                "eligible": fusion_eligible,
                "readiness_score": fusion_result.readiness_score,
                "best_eligible_readiness_score": best_fusion,
                "within_tie_window": fusion_within,
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
    return payload(base_pass, base_details), payload(fusion_pass, fusion_details)


def evaluate_promotion_gates(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
    ordinary: OrdinaryMetrics,
    source: SourceGroupMetrics,
    outer_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    base_stability_passed: bool,
    fusion_stability_passed: bool,
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
            "gate": "fusion_recipe_stability",
            "comparison": "required",
            "threshold": True,
            "actual": fusion_stability_passed,
            "passed": fusion_stability_passed,
        }
    )
    non_regression = v1._mapping(v1._mapping(contract, "promotion_gates"), "v1_non_regression")
    for name, actual, threshold_key in (
        ("v1_accuracy_non_regression", ordinary.accuracy, "accuracy_min"),
        ("v1_macro_f1_non_regression", ordinary.macro_f1, "macro_f1_min"),
        ("v1_source_robustness_non_regression", source.robustness_score, "source_robustness_min"),
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
    selected_by_fold: Optional[Mapping[str, Tuple[v1.BaseConfiguration, FusionConfiguration]]] = None,
    selected_recipe: Optional[Tuple[v1.BaseConfiguration, FusionConfiguration]] = None,
) -> List[Dict[str, object]]:
    predictions = v1._predict_labels(probabilities, classes)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rows = []
    for position, global_index in enumerate(indices.tolist()):
        record = records[global_index]
        if selected_by_fold is not None:
            base, fusion = selected_by_fold[record.cv_fold]
        elif selected_recipe is not None:
            base, fusion = selected_recipe
        else:
            raise ValueError("V3 prediction recipe provenance is missing")
        family = fusion.family
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
            "selected_fusion_configuration_id": fusion.configuration_id,
            "fusion_mode": fusion.mode,
            "fusion_family": family.family_id if family else "none",
            "fusion_blend_weight": fusion.blend_weight,
            "fusion_applied": bool(applied[position]),
        }
        for class_index, class_name in enumerate(classes):
            row[f"probability_{class_name}"] = float(probabilities[position, class_index])
        rows.append(row)
    return rows


def save_confusion_plot(matrix: np.ndarray, classes: Sequence[str], path: Path) -> None:
    short = [label.split("_", 1)[1] if "_" in label else label for label in classes]
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(classes)), short, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), short)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Phase 5 v3 nested development OOF confusion matrix")
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
            metadata={"Software": "ornament-classifier-phase5-v3"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    global FUSION_CONFIGURATIONS, FUSION_BY_ID
    args = parse_args()
    paths = ProjectPaths.discover()
    contract, contract_path = load_frozen_contract(paths)
    development = load_development_contract(paths)
    v1._validate_development(development, contract)
    if len(development.records) != 1693:
        raise ValueError("Phase 5 v3 requires all 1,693 development rows")
    records = development.records
    classes = v1._classes(records)
    seed = v1._int(contract, "seed")
    pair_classes = tuple(
        str(value)
        for value in v1._sequence(v1._mapping(contract, "consensus_heads"), "pair_classes")
    )
    ceramic_classes = tuple(
        str(value)
        for value in v1._sequence(
            v1._mapping(contract, "diagnostic_slice_definitions"), "ceramic_classes"
        )
    )
    print("Loading and verifying frozen Phase 5 v3 embedding arrays...", flush=True)
    embedding_blocks = load_allowlisted_embeddings_v3(
        _embedding_requests(contract), paths=paths
    )
    if any(
        block.provenance.get("experiment_contract_sha256") != EXPECTED_CONTRACT_SHA256
        for block in embedding_blocks.values()
    ):
        raise ValueError("V3 embedding provenance does not pin the frozen contract")
    base_configurations = build_base_configurations(contract, embedding_blocks)
    families = build_fusion_families(contract, embedding_blocks)
    FUSION_CONFIGURATIONS = build_fusion_configurations(contract, families)
    FUSION_BY_ID = {item.configuration_id: item for item in FUSION_CONFIGURATIONS}
    base_by_id = {item.configuration_id: item for item in base_configurations}
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
    selected_by_outer: Dict[str, Tuple[v1.BaseConfiguration, FusionConfiguration]] = {}
    outer_base_results: Dict[str, Mapping[str, v1.OOFResult]] = {}
    outer_base_selections: Dict[str, v1.SearchSelection] = {}
    outer_fusion_results: Dict[str, Mapping[str, FusionOOFResult]] = {}
    outer_fusion_selections: Dict[str, FusionSelection] = {}

    for number, outer_fold in enumerate(fold_ids, 1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        print(f"Outer v3 search {number}/{len(fold_ids)}: hold out fold {outer_fold}", flush=True)
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
        fusion_results = evaluate_fusion_oof_grid(
            base_selection.selected,
            selected_base,
            FUSION_CONFIGURATIONS,
            families,
            records,
            classes,
            pair_classes,
            ceramic_classes,
            inner_folds,
            contract,
            seed,
        )
        fusion_selection = select_fusion_candidates(fusion_results, contract)
        selected_fusion = FUSION_BY_ID[fusion_selection.selected.summary.configuration_id]
        selected_by_outer[outer_fold] = (selected_base, selected_fusion)
        outer_base_results[outer_fold] = fold_base_by_id
        outer_base_selections[outer_fold] = base_selection
        outer_fusion_results[outer_fold] = {
            result.summary.configuration_id: result for result in fusion_results
        }
        outer_fusion_selections[outer_fold] = fusion_selection
        inner_rows.extend(
            base_search_rows(
                "outer_inner_oof", outer_fold, base_results, base_selection, reference_base_id
            )
        )
        inner_rows.extend(
            fusion_search_rows(
                "outer_inner_oof", outer_fold, selected_base, fusion_results, fusion_selection
            )
        )
        validation, base_probabilities, probabilities, applied, iterations = fit_outer_procedure(
            selected_base,
            selected_fusion,
            outer_fold,
            records,
            classes,
            ceramic_classes,
            pair_classes,
            contract,
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
            labels[validation], probabilities, classes,
            pair_classes=pair_classes, ceramic_classes=ceramic_classes
        )
        hard = _hard_three_accuracy(records, validation, predictions, contract)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_base_configuration_id": selected_base.configuration_id,
                "selected_fusion_configuration_id": selected_fusion.configuration_id,
                "fusion_mode": selected_fusion.mode,
                "fusion_family": selected_fusion.family.family_id if selected_fusion.family else "none",
                "fusion_blend_weight": selected_fusion.blend_weight,
                "fusion_applied_count": int(applied.sum()),
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
        raise RuntimeError("Phase 5 v3 nested predictions are incomplete")

    print("Running full-development Phase 5 v3 selection...", flush=True)
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
    full_fusion_results = evaluate_fusion_oof_grid(
        full_base_selection.selected,
        full_base,
        FUSION_CONFIGURATIONS,
        families,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        fold_ids,
        contract,
        seed,
    )
    full_fusion_selection = select_fusion_candidates(full_fusion_results, contract)
    full_fusion = FUSION_BY_ID[full_fusion_selection.selected.summary.configuration_id]
    full_selected = full_fusion_selection.selected
    full_rows = base_search_rows(
        "full_development_oof", "", full_base_results, full_base_selection, reference_base_id
    )
    full_rows.extend(
        fusion_search_rows(
            "full_development_oof", "", full_base, full_fusion_results, full_fusion_selection
        )
    )
    reproduction = v2.validate_v1_reproduction(
        records,
        uncorrected,
        {fold: (base.configuration_id, "pair_none") for fold, (base, _) in selected_by_outer.items()},
        full_base,
        contract,
        classes,
        ceramic_classes,
    )
    base_stability, fusion_stability = stability(
        fold_ids,
        full_base,
        full_fusion,
        outer_base_results,
        outer_base_selections,
        outer_fusion_results,
        outer_fusion_selections,
        contract,
    )
    nested_predictions = v1._predict_labels(nested, classes)
    nested_ordinary = ordinary_metrics(labels, nested, classes)
    nested_source = source_group_metrics(
        labels, nested_predictions, groups, classes, ceramic_classes
    )
    nested_boundary = boundary_metrics(
        labels, nested, classes, pair_classes=pair_classes, ceramic_classes=ceramic_classes
    )
    nested_hard = _hard_three_accuracy(
        records, np.arange(len(records), dtype=np.int64), nested_predictions, contract
    )
    if nested_hard is None:
        raise RuntimeError("V3 nested result has no hard-three support")
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
        bool(fusion_stability["passed"]),
    )

    output_dir = v1._resolve_output_dir(paths, contract, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_grid_rows = v2._base_grid_rows(base_configurations)
    fusion_grid_rows = [
        {
            "configuration_id": item.configuration_id,
            "fusion_mode": item.mode,
            "family": item.family.family_id if item.family else "none",
            "embedding_blocks": "|".join(item.family.block_names) if item.family else "",
            "blend_weight": item.blend_weight,
            "head_count": item.family.head_count if item.family else 0,
            "encoder_count": item.family.encoder_count if item.family else 0,
            "intervention_rank": item.intervention_rank,
        }
        for item in FUSION_CONFIGURATIONS
    ]
    nested_prediction_rows = prediction_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested,
        nested_applied,
        classes,
        "nested_outer_oof_adaptive_phase5_v3_procedure",
        selected_by_fold=selected_by_outer,
    )
    selected_prediction_rows = prediction_rows(
        records,
        full_selected.evaluated_indices,
        full_selected.probabilities,
        full_selected.applied_mask,
        classes,
        "selection_conditional_full_development_oof_v3",
        selected_recipe=(full_base, full_fusion),
    )
    per_class_rows = v1._per_class_rows(nested_ordinary, classes, "nested_outer_oof_v3")
    diagnostic_rows = v1.diagnostic_slice_rows(
        records, np.arange(len(records), dtype=np.int64), nested, classes, contract
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
        ("fusion_grid.csv", fusion_grid_rows),
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
    v1._write_csv(output_dir / "confusion_matrix.csv", matrix_rows, ("true_class", *classes))
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
            "reason": "Nested OOF estimates the adaptive v3 procedure, but v1/v2 development findings informed the frozen cross-encoder family.",
        },
        "probability_policy": {
            "calibrated": False,
            "threshold_selection_used": False,
            "interpretation": "uncalibrated ranking scores only",
        },
        "phase_6_transition_allowed": promotion["promotion_decision"] == "promote",
        "split_version": development.split_version,
        "split_seed": development.split_seed,
        "split_assignment_fingerprint_sha256": development.audit.get("assignment_fingerprint_sha256"),
        "development_image_count": len(records),
        "class_names": classes,
        "pair_classes": pair_classes,
        "ceramic_classes": ceramic_classes,
        "base_configuration_count": len(base_configurations),
        "fusion_configuration_count": len(FUSION_CONFIGURATIONS),
        "nested_selected_recipes_by_outer_fold": {
            fold: {
                "base_configuration_id": base.configuration_id,
                "fusion_configuration_id": fusion.configuration_id,
            }
            for fold, (base, fusion) in selected_by_outer.items()
        },
        "nested_aggregate_oof_metrics": {
            **asdict(nested_ordinary),
            **asdict(nested_source),
            **asdict(nested_boundary),
            "readiness_score": nested_readiness,
            "hard_three_accuracy": nested_hard,
            "fusion_applied_count": int(nested_applied.sum()),
        },
        "nested_outer_fold_metrics": outer_rows,
        "full_development_selected_candidate": {
            "base_configuration_id": full_base.configuration_id,
            "base_feature_family": full_base.family.family_id,
            "base_c_value": full_base.c_value,
            "base_source_group_exponent": full_base.source_group_exponent,
            "fusion_configuration_id": full_fusion.configuration_id,
            "fusion_mode": full_fusion.mode,
            "fusion_family": full_fusion.family.family_id if full_fusion.family else "none",
            "fusion_blend_weight": full_fusion.blend_weight,
            "head_count": full_fusion.family.head_count if full_fusion.family else 0,
            "encoder_count": full_fusion.family.encoder_count if full_fusion.family else 0,
        },
        "full_development_selected_oof_metrics": {
            **asdict(full_selected.ordinary),
            **asdict(full_selected.source),
            **asdict(full_selected.boundary),
            "readiness_score": full_selected.readiness_score,
            "hard_three_accuracy": full_selected.hard_three_accuracy,
            "fusion_applied_count": int(full_selected.applied_mask.sum()),
        },
        "v1_reproduction_validation": reproduction,
        "base_recipe_stability": base_stability,
        "fusion_recipe_stability": fusion_stability,
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
            name: dict(block.provenance) for name, block in sorted(embedding_blocks.items())
        },
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
        raise RuntimeError("Required Phase 5 v3 outputs are missing: " + ", ".join(missing))
    print(
        json.dumps(
            {
                "nested_accuracy": nested_ordinary.accuracy,
                "nested_macro_f1": nested_ordinary.macro_f1,
                "nested_readiness_score": nested_readiness,
                "selected_base": full_base.configuration_id,
                "selected_fusion": full_fusion.configuration_id,
                "promotion_decision": promotion["promotion_decision"],
                "phase_6_transition_allowed": promotion["promotion_decision"] == "promote",
                "sealed_test_evaluated": False,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


FUSION_CONFIGURATIONS: Tuple[FusionConfiguration, ...] = ()
FUSION_BY_ID: Mapping[str, FusionConfiguration] = {}


if __name__ == "__main__":
    main()

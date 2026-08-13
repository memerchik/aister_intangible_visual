#!/usr/bin/env python3
"""Run the final frozen Phase 5 v5 fixed motif-consensus experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


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
import run_phase5_v4_motif as v4  # noqa: E402
from ornament_classifier.consensus import (  # noqa: E402
    FusionEligibilityReference,
    FusionEligibilityResult,
    fusion_eligibility,
    harmonic_mean_many,
)
from ornament_classifier.contracts import (  # noqa: E402
    DevelopmentRecord,
    load_development_contract,
)
from ornament_classifier.embeddings_v4 import (  # noqa: E402
    MotifEmbeddingCache,
    load_motif_embeddings_v4,
)
from ornament_classifier.fixed_consensus import fixed_probability_consensus  # noqa: E402
from ornament_classifier.pairwise import boundary_metrics  # noqa: E402
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    OrdinaryMetrics,
    SourceGroupMetrics,
    ordinary_metrics,
    source_group_metrics,
)


PHASE = "step02_phase_5_source_robustness_v5"
CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v5.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "c9473a7e20c57049fc4bc648f1133c5269434a955f955f390f10594adb80a70d"
)
SCRIPT_PATH = Path(__file__).resolve()
MAX_ITERATIONS = 10_000
TOLERANCE = 1e-6
FIXED_ID = "fixed_base_plus_three_motif_heads__equal_centered_logits"
DESCRIPTORS = ("uniform_mean", "texture_weighted_mean", "texture_top2_mean")
HEAD_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the frozen v5 contract directory.",
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
    if not expected_path.is_file():
        raise ValueError(f"Frozen Phase 5 v5 contract is missing: {expected_path}")
    observed = sha256_file(expected_path)
    if observed != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "Phase 5 v5 contract changed after it was frozen: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {observed}"
        )
    contract = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Frozen Phase 5 v5 contract must be an object")
    if contract.get("iteration") != "v5" or contract.get("status") != "frozen_before_fit":
        raise ValueError("Phase 5 v5 contract has an invalid iteration or status")
    scope = v1._mapping(contract, "scope")
    if scope.get("sealed_test_access") != "forbidden":
        raise ValueError("Phase 5 v5 requires sealed access to remain forbidden")
    fixed = v1._mapping(contract, "fixed_consensus")
    if v1._int(fixed, "candidate_count") != 1 or fixed.get("selection_after_base") is not False:
        raise ValueError("Phase 5 v5 must retain exactly one fixed post-base recipe")
    if tuple(float(value) for value in v1._sequence(fixed, "head_weights")) != HEAD_WEIGHTS:
        raise ValueError("Phase 5 v5 fixed head weights changed")
    return contract, expected_path


def code_provenance() -> Mapping[str, object]:
    paths = (
        SCRIPT_PATH,
        STEP_ROOT / "scripts" / "run_phase5_source_robustness.py",
        STEP_ROOT / "scripts" / "run_phase5_v2_pairwise.py",
        STEP_ROOT / "scripts" / "run_phase5_v3_consensus.py",
        STEP_ROOT / "scripts" / "run_phase5_v4_motif.py",
        STEP_ROOT / "src" / "ornament_classifier" / "fixed_consensus.py",
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
            raise ValueError(f"Phase 5 v5 provenance input is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        combined.update(f"{relative}\x1f{digest}\n".encode("utf-8"))
    return {"files_sha256": files, "combined_sha256": combined.hexdigest()}


def descriptor_configurations(
    contract: Mapping[str, object], cache: MotifEmbeddingCache
) -> Tuple[v4.MotifConfiguration, ...]:
    declared = tuple(
        str(value)
        for value in v1._sequence(v1._mapping(contract, "motif_cache"), "descriptor_names")
    )
    if declared != DESCRIPTORS or any(name not in cache.embeddings for name in declared):
        raise ValueError("V5 descriptor heads disagree with the frozen motif cache")
    return tuple(
        v4.MotifConfiguration(
            configuration_id=f"v5_head_{name}__c10__source0p5",
            descriptor_name=name,
            c_value=10.0,
            source_group_exponent=0.5,
            complexity_rank=index,
            descriptor_dimensions=int(cache.embeddings[name].shape[1]),
        )
        for index, name in enumerate(declared, 1)
    )


def fixed_configuration(cache: MotifEmbeddingCache) -> v4.MotifConfiguration:
    dimensions = sum(int(cache.embeddings[name].shape[1]) for name in DESCRIPTORS)
    return v4.MotifConfiguration(
        configuration_id=FIXED_ID,
        descriptor_name="fixed_three_descriptor_consensus",
        c_value=10.0,
        source_group_exponent=0.5,
        complexity_rank=1,
        descriptor_dimensions=dimensions,
    )


def evaluate_fixed_oof(
    base_result: v1.OOFResult,
    base: v1.BaseConfiguration,
    head_configurations: Sequence[v4.MotifConfiguration],
    fixed: v4.MotifConfiguration,
    cache: MotifEmbeddingCache,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    pair_classes: Sequence[str],
    ceramic_classes: Sequence[str],
    fold_ids: Sequence[str],
    contract: Mapping[str, object],
    seed: int,
) -> v4.MotifOOFResult:
    head_results = v4.evaluate_motif_oof_grid(
        base_result,
        base,
        head_configurations,
        cache,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        fold_ids,
        contract,
        seed,
    )
    probabilities = fixed_probability_consensus(
        (base_result.probabilities, *(result.probabilities for result in head_results)),
        weights=HEAD_WEIGHTS,
    )
    synthetic_iterations: List[int] = [base_result.maximum_iterations] * base_result.fit_count
    for result in head_results:
        synthetic_iterations.extend([result.maximum_iterations] * result.fit_count)
    return v4._summarize(
        fixed,
        probabilities,
        base_result.evaluated_indices,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        contract,
        synthetic_iterations,
    )


def fixed_eligibility(
    result: v4.MotifOOFResult,
    base: v1.OOFResult,
    contract: Mapping[str, object],
) -> FusionEligibilityResult:
    reference = FusionEligibilityReference(
        accuracy=base.ordinary.accuracy,
        macro_f1=base.ordinary.macro_f1,
        source_robustness_score=base.source.robustness_score,
        opishnyan_group_recall=base.source.per_class_group_recall["01_opishnyan_ceramics"],
        per_class_recall=base.ordinary.per_class_recall,
    )
    limits = v1._mapping(contract, "fixed_consensus_eligibility")
    return fusion_eligibility(
        result.summary,
        reference,
        maximum_accuracy_drop=v1._float(limits, "maximum_accuracy_drop"),
        maximum_macro_f1_drop=v1._float(limits, "maximum_macro_f1_drop"),
        maximum_source_robustness_drop=v1._float(
            limits, "maximum_source_robustness_drop"
        ),
        maximum_opishnyan_group_recall_drop=v1._float(
            limits, "maximum_opishnyan_group_recall_drop"
        ),
        other_class_recall_drops={
            label: float(value)
            for label, value in v1._mapping(
                limits, "maximum_other_class_recall_drop"
            ).items()
        },
    )


def fit_outer_fixed(
    base: v1.BaseConfiguration,
    head_configurations: Sequence[v4.MotifConfiguration],
    cache: MotifEmbeddingCache,
    outer_fold: str,
    records: Sequence[DevelopmentRecord],
    classes: Sequence[str],
    ceramic_classes: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    labels, groups, folds = v1._record_arrays(records)
    validation, base_probabilities, base_iterations = v1._fit_outer_and_predict(
        base, "none", outer_fold, records, classes, ceramic_classes, seed
    )
    train = np.flatnonzero(folds != outer_fold)
    v1._assert_source_disjoint_partitions(
        train, validation, groups, f"v5 outer fixed consensus fold {outer_fold}"
    )
    heads = [base_probabilities]
    iterations = list(base_iterations)
    for configuration in head_configurations:
        with threadpool_limits(limits=1):
            model = v4._fit_augmented_model(
                base, configuration, cache, train, labels, groups, classes, seed
            )
        if not model.converged:
            raise RuntimeError(f"V5 head did not converge: {configuration.configuration_id}")
        blocks = v4._augmented_blocks(base, configuration, cache)
        heads.append(model.predict_proba(v4._slice_augmented(blocks, base, validation)))
        iterations.extend(v1._flatten_iterations(model.iterations))
    return (
        validation,
        base_probabilities,
        fixed_probability_consensus(tuple(heads), weights=HEAD_WEIGHTS),
        tuple(iterations),
    )


def stability(
    fold_ids: Sequence[str],
    full_base: v1.BaseConfiguration,
    outer_base_results: Mapping[str, Mapping[str, v1.OOFResult]],
    outer_base_selections: Mapping[str, v1.SearchSelection],
    outer_fixed_results: Mapping[str, v4.MotifOOFResult],
    outer_fixed_eligibility: Mapping[str, FusionEligibilityResult],
    contract: Mapping[str, object],
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    base_window = v1._float(
        v1._mapping(v1._mapping(contract, "base_procedure"), "selection_rule"),
        "tie_window",
    )
    required = v1._int(v1._mapping(contract, "stability_rule"), "required_outer_searches")
    base_details = []
    consensus_details = []
    base_passing = []
    consensus_passing = []
    for fold in fold_ids:
        selection = outer_base_selections[fold]
        base_result = outer_base_results[fold][full_base.configuration_id]
        best = max(
            outer_base_results[fold][identifier].summary.robustness_score
            for identifier in selection.ranked_eligible_ids
        )
        eligible = selection.eligibility[full_base.configuration_id].eligible
        within = eligible and base_result.summary.robustness_score >= best - base_window
        if within:
            base_passing.append(fold)
        base_details.append(
            {
                "outer_fold": fold,
                "configuration_id": full_base.configuration_id,
                "eligible": eligible,
                "robustness_score": base_result.summary.robustness_score,
                "best_eligible_robustness_score": best,
                "within_tie_window": within,
            }
        )
        fixed_result = outer_fixed_results[fold]
        fixed_decision = outer_fixed_eligibility[fold]
        if fixed_decision.eligible:
            consensus_passing.append(fold)
        consensus_details.append(
            {
                "outer_fold": fold,
                "configuration_id": FIXED_ID,
                "eligible": fixed_decision.eligible,
                "eligibility_failures": list(fixed_decision.failures),
                "readiness_score": fixed_result.readiness_score,
            }
        )

    def payload(passing: List[str], details: List[Mapping[str, object]]) -> Mapping[str, object]:
        return {
            "required_outer_searches": required,
            "observed_outer_searches": len(passing),
            "passing_outer_folds": passing,
            "passed": len(passing) >= required,
            "details": details,
        }

    return payload(base_passing, base_details), payload(consensus_passing, consensus_details)


def evaluate_promotion_gates(
    records: Sequence[DevelopmentRecord],
    probabilities: np.ndarray,
    classes: Sequence[str],
    ordinary: OrdinaryMetrics,
    source: SourceGroupMetrics,
    outer_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    base_stability_passed: bool,
    consensus_stability_passed: bool,
) -> Mapping[str, object]:
    payload = dict(
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
    results = [dict(value) for value in payload["results"]]
    results.append(
        {
            "gate": "fixed_consensus_stability",
            "comparison": "required",
            "threshold": True,
            "actual": consensus_stability_passed,
            "passed": consensus_stability_passed,
        }
    )
    floors = v1._mapping(v1._mapping(contract, "promotion_gates"), "v1_non_regression")
    for gate, actual, key in (
        ("v1_accuracy_non_regression", ordinary.accuracy, "accuracy_min"),
        ("v1_macro_f1_non_regression", ordinary.macro_f1, "macro_f1_min"),
        ("v1_source_robustness_non_regression", source.robustness_score, "source_robustness_min"),
    ):
        threshold = v1._float(floors, key)
        results.append(
            {
                "gate": gate,
                "comparison": ">=",
                "threshold": threshold,
                "actual": actual,
                "passed": actual >= threshold,
            }
        )
    if len(results) != 21:
        raise RuntimeError(f"Phase 5 v5 must evaluate exactly 21 gates, observed {len(results)}")
    all_passed = all(bool(result["passed"]) for result in results)
    payload.update(
        {
            "all_passed": all_passed,
            "promotion_decision": "promote" if all_passed else "reject",
            "gate_count": len(results),
            "passed_count": sum(bool(result["passed"]) for result in results),
            "failed_gates": [result["gate"] for result in results if not result["passed"]],
            "results": results,
        }
    )
    return payload


def fixed_search_row(
    scope: str,
    outer_fold: str,
    base: v1.BaseConfiguration,
    result: v4.MotifOOFResult,
    eligibility: FusionEligibilityResult,
) -> Dict[str, object]:
    return v4._search_row(
        scope,
        outer_fold,
        "motif",
        base.configuration_id,
        FIXED_ID,
        "+".join(DESCRIPTORS),
        10.0,
        0.5,
        1152,
        1,
        False,
        eligibility.eligible,
        "|".join(eligibility.failures),
        1 if eligibility.eligible else "",
        True,
        result,
    )


def save_confusion_plot(matrix: np.ndarray, classes: Sequence[str], path: Path) -> None:
    short = [label.split("_", 1)[1] if "_" in label else label for label in classes]
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(classes)), short, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), short)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Phase 5 v5 nested development OOF confusion matrix")
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
            metadata={"Software": "ornament-classifier-phase5-v5"},
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
    for version in ("v1", "v2", "v3", "v4"):
        path = REPO_ROOT / v1._text(inputs, f"phase_5_{version}_metrics")
        if sha256_file(path) != v1._text(inputs, f"phase_5_{version}_metrics_sha256"):
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
    result["v5"] = {
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
        raise ValueError("Phase 5 v5 requires all 1,693 development rows")
    records = development.records
    classes = v1._classes(records)
    seed = v1._int(contract, "seed")
    slices = v1._mapping(contract, "diagnostic_slice_definitions")
    pair_classes = tuple(str(value) for value in v1._sequence(slices, "pair_classes"))
    ceramic_classes = tuple(
        str(value) for value in v1._sequence(slices, "ceramic_classes")
    )
    print("Loading and verifying frozen v5 motif and base embeddings...", flush=True)
    base_blocks = v4.load_base_embeddings(paths)
    motif_cache = load_motif_embeddings_v4(paths=paths)
    base_configurations = v3.build_base_configurations(contract, base_blocks)
    heads = descriptor_configurations(contract, motif_cache)
    fixed = fixed_configuration(motif_cache)
    base_by_id = {item.configuration_id: item for item in base_configurations}
    reference_base_id = "global_cls__c10__source0"
    fold_ids = development.fold_ids
    labels, groups, folds = v1._record_arrays(records)
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
    inner_rows: List[Dict[str, object]] = []
    outer_rows: List[Dict[str, object]] = []
    selected_by_outer: Dict[str, Tuple[v1.BaseConfiguration, v4.MotifConfiguration]] = {}
    outer_base_results: Dict[str, Mapping[str, v1.OOFResult]] = {}
    outer_base_selections: Dict[str, v1.SearchSelection] = {}
    outer_fixed_results: Dict[str, v4.MotifOOFResult] = {}
    outer_fixed_eligibility: Dict[str, FusionEligibilityResult] = {}

    for number, outer_fold in enumerate(fold_ids, 1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        print(
            f"Outer v5 procedure {number}/{len(fold_ids)}: hold out fold {outer_fold}",
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
        fixed_result = evaluate_fixed_oof(
            base_selection.selected,
            selected_base,
            heads,
            fixed,
            motif_cache,
            records,
            classes,
            pair_classes,
            ceramic_classes,
            inner_folds,
            contract,
            seed,
        )
        eligibility = fixed_eligibility(fixed_result, base_selection.selected, contract)
        selected_by_outer[outer_fold] = (selected_base, fixed)
        outer_base_results[outer_fold] = fold_base_by_id
        outer_base_selections[outer_fold] = base_selection
        outer_fixed_results[outer_fold] = fixed_result
        outer_fixed_eligibility[outer_fold] = eligibility
        inner_rows.extend(
            v4.base_search_rows(
                "outer_inner_oof", outer_fold, base_results, base_selection, reference_base_id
            )
        )
        inner_rows.append(
            fixed_search_row(
                "outer_inner_oof", outer_fold, selected_base, fixed_result, eligibility
            )
        )
        validation, base_probabilities, probabilities, iterations = fit_outer_fixed(
            selected_base,
            heads,
            motif_cache,
            outer_fold,
            records,
            classes,
            ceramic_classes,
            seed,
        )
        uncorrected[validation] = base_probabilities
        nested[validation] = probabilities
        ordinary = ordinary_metrics(labels[validation], probabilities, classes)
        predictions = v1._predict_labels(probabilities, classes)
        source = source_group_metrics(
            labels[validation], predictions, groups[validation], classes, ceramic_classes
        )
        boundary = boundary_metrics(
            labels[validation], probabilities, classes,
            pair_classes=pair_classes, ceramic_classes=ceramic_classes,
        )
        hard = v3._hard_three_accuracy(records, validation, predictions, contract)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_base_configuration_id": selected_base.configuration_id,
                "fixed_consensus_configuration_id": FIXED_ID,
                "fixed_consensus_eligible_on_inner_oof": eligibility.eligible,
                "fixed_consensus_eligibility_failures": "|".join(eligibility.failures),
                "motif_applied_count": len(validation),
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
        raise RuntimeError("Phase 5 v5 nested predictions are incomplete")

    print("Running full-development Phase 5 v5 procedure...", flush=True)
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
    full_result = evaluate_fixed_oof(
        full_base_selection.selected,
        full_base,
        heads,
        fixed,
        motif_cache,
        records,
        classes,
        pair_classes,
        ceramic_classes,
        fold_ids,
        contract,
        seed,
    )
    full_eligibility = fixed_eligibility(full_result, full_base_selection.selected, contract)
    full_rows = v4.base_search_rows(
        "full_development_oof", "", full_base_results, full_base_selection, reference_base_id
    )
    full_rows.append(
        fixed_search_row(
            "full_development_oof", "", full_base, full_result, full_eligibility
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
    base_stability, consensus_stability = stability(
        fold_ids,
        full_base,
        outer_base_results,
        outer_base_selections,
        outer_fixed_results,
        outer_fixed_eligibility,
        contract,
    )
    nested_predictions = v1._predict_labels(nested, classes)
    nested_ordinary = ordinary_metrics(labels, nested, classes)
    nested_source = source_group_metrics(
        labels, nested_predictions, groups, classes, ceramic_classes
    )
    nested_boundary = boundary_metrics(
        labels, nested, classes,
        pair_classes=pair_classes, ceramic_classes=ceramic_classes,
    )
    nested_hard = v3._hard_three_accuracy(
        records, np.arange(len(records), dtype=np.int64), nested_predictions, contract
    )
    if nested_hard is None:
        raise RuntimeError("Phase 5 v5 nested result has no hard-three support")
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
        bool(consensus_stability["passed"]),
    )

    output_dir = v1._resolve_output_dir(paths, contract, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_grid_rows = v2._base_grid_rows(base_configurations)
    recipe_rows = [
        {
            "configuration_id": FIXED_ID,
            "candidate_count": 1,
            "selection_after_base": False,
            "head_count": 4,
            "head_ids": "base_anchor|" + "|".join(DESCRIPTORS),
            "head_weights": "0.25|0.25|0.25|0.25",
            "motif_c_value": 10.0,
            "motif_source_group_exponent": 0.5,
            "combination": "equal mean of centered log probabilities then softmax",
        }
    ]
    nested_prediction_rows = v4.prediction_rows(
        records,
        np.arange(len(records), dtype=np.int64),
        nested,
        np.ones(len(records), dtype=bool),
        classes,
        "nested_outer_oof_adaptive_base_fixed_consensus_v5",
        selected_by_fold=selected_by_outer,
    )
    selected_prediction_rows = v4.prediction_rows(
        records,
        full_result.evaluated_indices,
        full_result.probabilities,
        np.ones(len(full_result.evaluated_indices), dtype=bool),
        classes,
        "selection_conditional_full_development_oof_fixed_consensus_v5",
        selected_recipe=(full_base, fixed),
    )
    per_class_rows = v1._per_class_rows(nested_ordinary, classes, "nested_outer_oof_v5")
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
    writes = (
        ("base_grid.csv", base_grid_rows),
        ("consensus_recipe.csv", recipe_rows),
        ("motif_proposals.csv", v4.proposal_rows(records, motif_cache)),
        ("inner_search.csv", inner_rows),
        ("outer_fold_metrics.csv", outer_rows),
        ("nested_oof_predictions.csv", nested_prediction_rows),
        ("full_development_selection.csv", full_rows),
        ("selected_oof_predictions.csv", selected_prediction_rows),
        ("per_class_metrics.csv", per_class_rows),
        ("diagnostic_slices.csv", diagnostic_rows),
        ("confusion_pairs.csv", v1._confusion_pair_rows(matrix, classes)),
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
            "reason": "Nested OOF estimates the fixed v5 procedure, but repeated v1-v4 development findings informed it.",
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
        "fixed_consensus_candidate_count": 1,
        "fixed_consensus_configuration_id": FIXED_ID,
        "fixed_consensus_head_weights": HEAD_WEIGHTS,
        "motif_proposal_count_per_image": motif_cache.proposal_ids.shape[1],
        "nested_selected_base_by_outer_fold": {
            fold: base.configuration_id for fold, (base, _) in selected_by_outer.items()
        },
        "nested_aggregate_oof_metrics": {
            **asdict(nested_ordinary),
            **asdict(nested_source),
            **asdict(nested_boundary),
            "readiness_score": nested_readiness,
            "hard_three_accuracy": nested_hard,
            "motif_applied_count": len(records),
        },
        "nested_outer_fold_metrics": outer_rows,
        "full_development_candidate": {
            "base_configuration_id": full_base.configuration_id,
            "base_feature_family": full_base.family.family_id,
            "base_c_value": full_base.c_value,
            "base_source_group_exponent": full_base.source_group_exponent,
            "fixed_consensus_configuration_id": FIXED_ID,
            "fixed_consensus_eligible": full_eligibility.eligible,
            "fixed_consensus_eligibility_failures": full_eligibility.failures,
        },
        "full_development_oof_metrics": {
            **asdict(full_result.ordinary),
            **asdict(full_result.source),
            **asdict(full_result.boundary),
            "readiness_score": full_result.readiness_score,
            "hard_three_accuracy": full_result.hard_three_accuracy,
            "motif_applied_count": len(full_result.evaluated_indices),
        },
        "previous_phase5_comparison": previous_version_comparison(
            contract, nested_ordinary, nested_source
        ),
        "v1_reproduction_validation": reproduction,
        "base_recipe_stability": base_stability,
        "fixed_consensus_stability": consensus_stability,
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
        raise RuntimeError("Required Phase 5 v5 outputs are missing: " + ", ".join(missing))
    print(
        json.dumps(
            {
                "nested_accuracy": nested_ordinary.accuracy,
                "nested_macro_f1": nested_ordinary.macro_f1,
                "nested_readiness_score": nested_readiness,
                "selected_base": full_base.configuration_id,
                "fixed_consensus": FIXED_ID,
                "promotion_decision": promotion["promotion_decision"],
                "failed_gates": promotion["failed_gates"],
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

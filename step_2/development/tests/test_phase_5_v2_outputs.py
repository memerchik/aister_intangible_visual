from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import struct
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from relocation_support import assert_recorded_file


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SPLIT_ROOT = STEP_ROOT / "splits"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness_v2"
V1_OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness"
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract_v2.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "481fcab8c19b412a59354f53735a3c6299da5c16ec06ed1c6e4ffbe49c1a5a68"
)
EXPECTED_METRICS_SHA256 = (
    "6b01234a4b04985d26185bdded6fb5382f5186e04895d1571687ad1011b174d3"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssertionError(f"Expected serialized Boolean, observed {value!r}")


def harmonic_mean(values: list[float]) -> float:
    if any(value == 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def classification_metrics(
    rows: list[dict[str, str]], classes: list[str]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    support = Counter(row["true_class"] for row in rows)
    predicted = Counter(row["predicted_class"] for row in rows)
    true_positive = Counter(
        row["true_class"]
        for row in rows
        if row["true_class"] == row["predicted_class"]
    )
    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        precision = true_positive[label] / predicted[label] if predicted[label] else 0.0
        recall = true_positive[label] / support[label] if support[label] else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support[label]),
        }
    aggregate = {
        "accuracy": sum(true_positive.values()) / len(rows),
        "macro_f1": statistics.fmean(per_class[label]["f1"] for label in classes),
        "balanced_accuracy": statistics.fmean(
            per_class[label]["recall"] for label in classes
        ),
    }
    return aggregate, per_class


def source_metrics(
    rows: list[dict[str, str]], classes: list[str], ceramics: list[str]
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[int]]] = {
        label: defaultdict(lambda: [0, 0]) for label in classes
    }
    for row in rows:
        bucket = grouped[row["true_class"]][row["source_atomic_split_group_id"]]
        bucket[0] += row["true_class"] == row["predicted_class"]
        bucket[1] += 1
    per_class = {
        label: statistics.fmean(correct / total for correct, total in grouped[label].values())
        for label in classes
    }
    group_balanced = statistics.fmean(per_class.values())
    ceramic_worst = min(per_class[label] for label in ceramics)
    return {
        "per_class_group_recall": per_class,
        "source_group_balanced_accuracy": group_balanced,
        "ceramic_worst_group_recall": ceramic_worst,
        "robustness_score": harmonic_mean([group_balanced, ceramic_worst]),
    }


def correction_rank(rows: list[dict[str, str]], tie_window: float) -> list[str]:
    remaining = list(rows)
    ranked: list[str] = []
    while remaining:
        best = max(float(row["boundary_score"]) for row in remaining)
        tied = [
            row
            for row in remaining
            if float(row["boundary_score"]) >= best - tie_window
        ]
        tied.sort(
            key=lambda row: (
                -float(row["minimum_fold_boundary_score"]),
                -float(row["robustness_score"]),
                -float(row["macro_f1"]),
                int(row["combined_view_count"]),
                0 if row["correction"] == "none" else 1,
                float(row["blend_weight"]),
                0 if math.isclose(float(row["c_value"]), 10.0, abs_tol=1e-12) else 1,
                float(row["source_group_exponent"]),
                row["configuration_id"],
            )
        )
        ranked.extend(row["configuration_id"] for row in tied)
        tied_ids = {row["configuration_id"] for row in tied}
        remaining = [
            row for row in remaining if row["configuration_id"] not in tied_ids
        ]
    return ranked


class PhaseFiveV2OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUTPUT_ROOT / "metrics.json").is_file():
            raise unittest.SkipTest("canonical Phase 5 v2 outputs have not been generated")
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.metrics = json.loads(
            (OUTPUT_ROOT / "metrics.json").read_text(encoding="utf-8")
        )
        cls.development = read_csv(SPLIT_ROOT / "development.csv")
        cls.sealed_test = read_csv(SPLIT_ROOT / "test.csv")
        cls.base_grid = read_csv(OUTPUT_ROOT / "base_grid.csv")
        cls.correction_grid = read_csv(OUTPUT_ROOT / "correction_grid.csv")
        cls.inner = read_csv(OUTPUT_ROOT / "inner_search.csv")
        cls.outer = read_csv(OUTPUT_ROOT / "outer_fold_metrics.csv")
        cls.nested = read_csv(OUTPUT_ROOT / "nested_oof_predictions.csv")
        cls.full = read_csv(OUTPUT_ROOT / "full_development_selection.csv")
        cls.selected = read_csv(OUTPUT_ROOT / "selected_oof_predictions.csv")
        cls.per_class = read_csv(OUTPUT_ROOT / "per_class_metrics.csv")
        cls.slices = read_csv(OUTPUT_ROOT / "diagnostic_slices.csv")
        cls.confusion_pairs = read_csv(OUTPUT_ROOT / "confusion_pairs.csv")
        cls.confusion = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        cls.classes = list(cls.metrics["class_names"])
        cls.ceramics = list(cls.metrics["ceramic_classes"])

    def assert_close(self, observed: object, expected: float) -> None:
        self.assertIsNotNone(observed)
        self.assertAlmostEqual(float(observed), expected, places=12)

    def assert_metric_payload(
        self, rows: list[dict[str, str]], payload: dict[str, object]
    ) -> None:
        aggregate, per_class = classification_metrics(rows, self.classes)
        source = source_metrics(rows, self.classes, self.ceramics)
        for key, value in aggregate.items():
            self.assert_close(payload[key], value)
        for payload_key, class_key in (
            ("per_class_precision", "precision"),
            ("per_class_recall", "recall"),
            ("per_class_f1", "f1"),
            ("per_class_support", "support"),
        ):
            recorded = payload[payload_key]
            self.assertEqual(set(recorded), set(self.classes))
            for label in self.classes:
                self.assert_close(recorded[label], per_class[label][class_key])
        for key in (
            "source_group_balanced_accuracy",
            "ceramic_worst_group_recall",
            "robustness_score",
        ):
            self.assert_close(payload[key], source[key])
        for label in self.classes:
            self.assert_close(
                payload["per_class_group_recall"][label],
                source["per_class_group_recall"][label],
            )

        opishnyan, bubnivka = self.metrics["pair_classes"]
        ceramic_f1 = statistics.fmean(
            per_class[label]["f1"] for label in self.ceramics
        )
        components = [
            per_class[opishnyan]["recall"],
            per_class[bubnivka]["recall"],
            per_class[bubnivka]["precision"],
            ceramic_f1,
        ]
        self.assert_close(payload["opishnyan_recall"], components[0])
        self.assert_close(payload["bubnivka_recall"], components[1])
        self.assert_close(payload["bubnivka_precision"], components[2])
        self.assert_close(payload["ceramic_macro_f1"], components[3])
        self.assert_close(payload["boundary_score"], harmonic_mean(components))
        self.assertEqual(
            payload["opishnyan_to_bubnivka_errors"],
            sum(
                row["true_class"] == opishnyan
                and row["predicted_class"] == bubnivka
                for row in rows
            ),
        )
        self.assertEqual(
            payload["bubnivka_to_opishnyan_errors"],
            sum(
                row["true_class"] == bubnivka
                and row["predicted_class"] == opishnyan
                for row in rows
            ),
        )

    def test_required_artifacts_are_exact_and_canonical(self) -> None:
        required = set(self.contract["output_contract"]["required"])
        self.assertEqual({path.name for path in OUTPUT_ROOT.iterdir()}, required)
        self.assertEqual(set(self.metrics["required_outputs"]), required)
        for name in required:
            self.assertTrue((OUTPUT_ROOT / name).is_file(), name)
            self.assertGreater((OUTPUT_ROOT / name).stat().st_size, 0, name)
        self.assertEqual(
            sha256_file(OUTPUT_ROOT / "metrics.json"), EXPECTED_METRICS_SHA256
        )
        png = (OUTPUT_ROOT / "confusion_matrix.png").read_bytes()
        self.assertGreater(len(png), 1024)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_contract_inputs_code_and_embedding_provenance_are_pinned(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(
            self.metrics["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256
        )
        self.assertEqual(self.metrics["script_sha256"], "5e5afb6a952b9b8fc00cb7e7fa259e288661124394ba608e8415a44ffb98b479")
        for relative, expected in self.metrics["code_provenance"][
            "files_sha256"
        ].items():
            assert_recorded_file(self, relative, expected)
        inputs = self.contract["input_contract"]
        for path_key, hash_key in (
            ("development_csv", "development_csv_sha256"),
            ("split_audit", "split_audit_sha256"),
            ("phase_5_v1_contract", "phase_5_v1_contract_sha256"),
            ("phase_5_v1_metrics", "phase_5_v1_metrics_sha256"),
            ("phase_5_v1_full_selection", "phase_5_v1_full_selection_sha256"),
            ("phase_5_v1_nested_predictions", "phase_5_v1_nested_predictions_sha256"),
        ):
            assert_recorded_file(self, inputs[path_key], inputs[hash_key])
        allowlist = self.contract["cache_allowlist"]
        self.assertEqual(len(self.metrics["embedding_provenance"]), 6)
        for identifier, provenance in self.metrics["embedding_provenance"].items():
            cache_name, array_name = identifier.split(".", 1)
            frozen = allowlist[cache_name]
            self.assertEqual(provenance["cache_sha256"], frozen["file_sha256"])
            self.assertEqual(
                provenance["array_sha256"], frozen["arrays"][array_name]["sha256"]
            )
            self.assertEqual(
                provenance["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256
            )

    def test_predictions_cover_only_development_and_keep_sources_atomic(self) -> None:
        development_ids = {row["image_id"] for row in self.development}
        test_ids = {row["image_id"] for row in self.sealed_test}
        self.assertEqual(len(self.nested), len(development_ids), 1693)
        self.assertEqual({row["image_id"] for row in self.nested}, development_ids)
        self.assertEqual({row["image_id"] for row in self.selected}, development_ids)
        self.assertFalse(development_ids & test_ids)
        self.assertFalse({row["image_id"] for row in self.nested} & test_ids)
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertEqual(self.contract["scope"]["sealed_test_access"], "forbidden")
        source_folds: dict[str, set[str]] = defaultdict(set)
        for row in self.nested:
            source_folds[row["source_atomic_split_group_id"]].add(row["cv_fold"])
        self.assertTrue(source_folds)
        self.assertTrue(all(len(folds) == 1 for folds in source_folds.values()))

    def test_prediction_probabilities_labels_and_recipes_are_self_consistent(self) -> None:
        probability_columns = [f"probability_{label}" for label in self.classes]
        selected_by_fold = {
            row["outer_fold"]: (
                row["selected_base_configuration_id"],
                row["selected_correction_configuration_id"],
            )
            for row in self.outer
        }
        for row in self.nested + self.selected:
            probabilities = [float(row[column]) for column in probability_columns]
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in probabilities))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            order = sorted(range(len(self.classes)), key=lambda index: (-probabilities[index], index))
            self.assertEqual(row["predicted_class"], self.classes[order[0]])
            self.assertEqual(row["top_1_class"], self.classes[order[0]])
            self.assertEqual(row["top_2_class"], self.classes[order[1]])
            self.assertEqual(row["top_3_class"], self.classes[order[2]])
            self.assert_close(row["uncalibrated_max_probability"], probabilities[order[0]])
            self.assertEqual(
                parse_bool(row["is_correct"]),
                row["true_class"] == row["predicted_class"],
            )
        for row in self.nested:
            self.assertEqual(
                (
                    row["selected_base_configuration_id"],
                    row["selected_correction_configuration_id"],
                ),
                selected_by_fold[row["cv_fold"]],
            )
        full_recipe = self.metrics["full_development_selected_candidate"]
        for row in self.selected:
            self.assertEqual(
                row["selected_base_configuration_id"],
                full_recipe["base_configuration_id"],
            )
            self.assertEqual(
                row["selected_correction_configuration_id"],
                full_recipe["correction_configuration_id"],
            )

    def test_nested_and_full_metrics_are_independently_reconstructed(self) -> None:
        self.assert_metric_payload(
            self.nested, self.metrics["nested_aggregate_oof_metrics"]
        )
        self.assert_metric_payload(
            self.selected, self.metrics["full_development_selected_oof_metrics"]
        )
        recorded = {row["class_name"]: row for row in self.per_class}
        _, per_class = classification_metrics(self.nested, self.classes)
        self.assertEqual(set(recorded), set(self.classes))
        for label in self.classes:
            for key in ("precision", "recall", "f1", "support"):
                self.assert_close(recorded[label][key], per_class[label][key])

    def test_confusion_outputs_and_outer_fold_metrics_reconstruct_predictions(self) -> None:
        matrix = {
            true: Counter(
                row["predicted_class"]
                for row in self.nested
                if row["true_class"] == true
            )
            for true in self.classes
        }
        self.assertEqual(len(self.confusion), len(self.classes))
        for row in self.confusion:
            true = row["true_class"]
            for predicted in self.classes:
                self.assertEqual(int(row[predicted]), matrix[true][predicted])
        expected_pairs = {
            (true, predicted): matrix[true][predicted]
            for true in self.classes
            for predicted in self.classes
            if true != predicted
        }
        self.assertEqual(len(self.confusion_pairs), len(expected_pairs))
        for row in self.confusion_pairs:
            key = (row["true_class"], row["predicted_class"])
            count = expected_pairs[key]
            support = sum(matrix[key[0]].values())
            self.assertEqual(int(row["error_count"]), count)
            self.assert_close(row["fraction_of_true_class"], count / support)
        self.assertEqual(len(self.outer), 5)
        for outer in self.outer:
            rows = [row for row in self.nested if row["cv_fold"] == outer["outer_fold"]]
            aggregate, per_class = classification_metrics(rows, self.classes)
            source = source_metrics(rows, self.classes, self.ceramics)
            self.assertEqual(len(rows), int(outer["validation_examples"]))
            for key, value in aggregate.items():
                self.assert_close(outer[key], value)
            for key in (
                "source_group_balanced_accuracy",
                "ceramic_worst_group_recall",
                "robustness_score",
            ):
                self.assert_close(outer[key], source[key])
            ceramic_f1 = statistics.fmean(
                per_class[label]["f1"] for label in self.ceramics
            )
            self.assert_close(outer["ceramic_macro_f1"], ceramic_f1)

    def test_grid_cardinality_sequential_selection_and_eligibility_are_exact(self) -> None:
        self.assertEqual(len(self.base_grid), 3)
        self.assertEqual(len(self.correction_grid), 65)
        self.assertEqual(len(self.inner), 5 * (3 + 65))
        self.assertEqual(len(self.full), 3 + 65)
        self.assertEqual(
            {row["configuration_id"] for row in self.base_grid},
            {row["id"] for row in self.contract["base_procedure"]["configurations"]},
        )
        self.assertEqual(
            sum(row["correction"] == "none" for row in self.correction_grid), 1
        )
        for search_rows in (
            [row for row in self.inner if row["outer_fold"] == fold]
            for fold in self.contract["scope"]["outer_folds"]
        ):
            self.assertEqual(sum(parse_bool(row["selected"]) for row in search_rows if row["stage"] == "base"), 1)
            self.assertEqual(sum(parse_bool(row["selected"]) for row in search_rows if row["stage"] == "correction"), 1)
            selected_base = next(
                row["configuration_id"]
                for row in search_rows
                if row["stage"] == "base" and parse_bool(row["selected"])
            )
            self.assertTrue(
                all(
                    row["base_configuration_id"] == selected_base
                    for row in search_rows
                    if row["stage"] == "correction"
                )
            )

        limits = self.contract["correction_selection_rule"][
            "eligibility_relative_to_uncorrected_selected_base"
        ]
        opishnyan = self.metrics["pair_classes"][0]
        for rows in (
            [row for row in self.inner if row["outer_fold"] == fold and row["stage"] == "correction"]
            for fold in self.contract["scope"]["outer_folds"]
        ):
            reference = next(row for row in rows if row["configuration_id"] == "pair_none")
            reference_recalls = json.loads(reference["per_class_recall_json"])
            reference_groups = json.loads(reference["per_class_group_recall_json"])
            for row in rows:
                recalls = json.loads(row["per_class_recall_json"])
                groups = json.loads(row["per_class_group_recall_json"])
                failures: list[str] = []
                for key, failure, limit_key in (
                    ("accuracy", "accuracy_drop", "maximum_accuracy_drop"),
                    ("macro_f1", "macro_f1_drop", "maximum_macro_f1_drop"),
                    ("robustness_score", "source_robustness_drop", "maximum_source_robustness_drop"),
                ):
                    if float(row[key]) < float(reference[key]) - float(limits[limit_key]):
                        failures.append(failure)
                if groups[opishnyan] < reference_groups[opishnyan] - float(
                    limits["maximum_opishnyan_group_recall_drop"]
                ):
                    failures.append("opishnyan_group_recall_drop")
                for label, maximum in limits["maximum_other_class_recall_drop"].items():
                    if recalls[label] < reference_recalls[label] - float(maximum):
                        failures.append(f"recall_drop:{label}")
                self.assertEqual(parse_bool(row["eligible"]), not failures)
                self.assertEqual(row["eligibility_failures"], "|".join(failures))
            eligible = [row for row in rows if parse_bool(row["eligible"])]
            ranked = correction_rank(
                eligible,
                float(self.contract["correction_selection_rule"]["tie_window"]),
            )
            recorded = [
                row["configuration_id"]
                for row in sorted(eligible, key=lambda item: int(item["eligible_selection_rank"]))
            ]
            self.assertEqual(recorded, ranked)
            self.assertEqual(
                next(row["configuration_id"] for row in rows if parse_bool(row["selected"])),
                ranked[0],
            )

    def test_stability_and_v1_reproduction_claims_are_supported_by_search_rows(self) -> None:
        full_base = self.metrics["full_development_selected_candidate"][
            "base_configuration_id"
        ]
        full_correction = self.metrics["full_development_selected_candidate"][
            "correction_configuration_id"
        ]
        for stage, identifier, score, window, payload_key in (
            (
                "base",
                full_base,
                "robustness_score",
                float(self.contract["base_procedure"]["selection_rule"]["tie_window"]),
                "base_recipe_stability",
            ),
            (
                "correction",
                full_correction,
                "boundary_score",
                float(self.contract["correction_selection_rule"]["tie_window"]),
                "correction_recipe_stability",
            ),
        ):
            passing: list[str] = []
            for fold in self.contract["scope"]["outer_folds"]:
                rows = [
                    row
                    for row in self.inner
                    if row["outer_fold"] == fold and row["stage"] == stage
                ]
                target = next(row for row in rows if row["configuration_id"] == identifier)
                eligible = [row for row in rows if parse_bool(row["eligible"])]
                best = max(float(row[score]) for row in eligible)
                within = parse_bool(target["eligible"]) and float(target[score]) >= best - window
                if within:
                    passing.append(fold)
            payload = self.metrics[payload_key]
            self.assertEqual(payload["passing_outer_folds"], passing)
            self.assertEqual(payload["observed_outer_searches_within_tie_window"], len(passing))
            self.assertEqual(payload["required_outer_searches_within_tie_window"], 4)
            self.assertEqual(payload["passed"], len(passing) >= 4)
        self.assertTrue(self.metrics["base_recipe_stability"]["passed"])
        self.assertFalse(self.metrics["correction_recipe_stability"]["passed"])
        reproduction = self.metrics["v1_reproduction_validation"]
        self.assertTrue(reproduction["validated"])
        self.assertEqual(
            reproduction["expected_decision_fingerprint_sha256"],
            reproduction["observed_decision_fingerprint_sha256"],
        )

    def test_every_promotion_gate_is_recomputed_and_v2_is_rejected(self) -> None:
        gates = self.contract["promotion_gates"]
        definitions = self.contract["diagnostic_slice_definitions"]
        aggregate, per_class = classification_metrics(self.nested, self.classes)
        opishnyan, bubnivka = self.metrics["pair_classes"]

        def accuracy(rows: list[dict[str, str]]) -> float:
            self.assertTrue(rows)
            return sum(row["true_class"] == row["predicted_class"] for row in rows) / len(rows)

        hard_names = definitions["hard_three_opishnyan_cohorts"]
        hard = [
            row
            for row in self.nested
            if row["true_class"] == opishnyan
            and row["source_atomic_cohort_ids"] in hard_names
        ]
        challenging = [
            row
            for row in self.nested
            if row["true_class"] == opishnyan
            and (
                row["source_atomic_cohort_ids"] in hard_names
                or row["source_atomic_cohort_ids"] == ""
            )
        ]
        hard_macro = statistics.fmean(
            accuracy(
                [
                    row
                    for row in self.nested
                    if row["true_class"] == opishnyan
                    and row["source_atomic_cohort_ids"] == cohort
                ]
            )
            for cohort in hard_names
        )
        cohort_support = Counter(row["source_atomic_cohort_ids"] for row in self.nested)
        large = sorted(
            cohort
            for cohort, count in cohort_support.items()
            if cohort and count >= int(gates["large_named_source_minimum_size"])
        )
        large_macro = statistics.fmean(
            accuracy([row for row in self.nested if row["source_atomic_cohort_ids"] == cohort])
            for cohort in large
        )
        fold_accuracy = [float(row["accuracy"]) for row in self.outer]
        ceramic_macro_f1 = statistics.fmean(
            per_class[label]["f1"] for label in self.ceramics
        )
        actuals: list[tuple[str, str, object, object]] = [
            ("overall_accuracy", ">=", aggregate["accuracy"], gates["overall_accuracy_min"]),
            ("overall_macro_f1", ">=", aggregate["macro_f1"], gates["overall_macro_f1_min"]),
            ("opishnyan_recall", ">=", per_class[opishnyan]["recall"], gates["opishnyan_recall_min"]),
            ("bubnivka_precision", ">=", per_class[bubnivka]["precision"], gates["bubnivka_precision_min"]),
            ("ceramic_macro_f1", ">=", ceramic_macro_f1, gates["ceramic_macro_f1_min"]),
            ("opishnyan_to_bubnivka_errors", "<=", float(sum(row["true_class"] == opishnyan and row["predicted_class"] == bubnivka for row in self.nested)), gates["opishnyan_to_bubnivka_errors_max"]),
            ("challenging_opishnyan_accuracy", ">=", accuracy(challenging), gates["challenging_opishnyan_accuracy_min"]),
            ("hard_three_opishnyan_pooled_accuracy", ">=", accuracy(hard), gates["hard_three_opishnyan_pooled_accuracy_min"]),
            ("hard_three_opishnyan_cohort_macro_accuracy", ">=", hard_macro, gates["hard_three_opishnyan_cohort_macro_accuracy_min"]),
            ("large_named_source_macro_accuracy", ">=", large_macro, gates["large_named_source_macro_accuracy_min"]),
            ("worst_fold_accuracy", ">=", min(fold_accuracy), gates["worst_fold_accuracy_min"]),
            ("fold_accuracy_std", "<=", statistics.stdev(fold_accuracy), gates["fold_accuracy_std_max"]),
        ]
        for label in sorted(gates["other_class_recall_floors"]):
            actuals.append(
                (
                    f"recall_floor:{label}",
                    ">=",
                    per_class[label]["recall"],
                    gates["other_class_recall_floors"][label],
                )
            )
        actuals.extend(
            [
                ("base_recipe_stability", "required", self.metrics["base_recipe_stability"]["passed"], True),
                ("correction_recipe_stability", "required", self.metrics["correction_recipe_stability"]["passed"], True),
            ]
        )
        recorded = self.metrics["promotion_gates"]
        self.assertEqual(len(recorded["results"]), len(actuals), 18)
        failed: list[str] = []
        for result, (name, comparison, actual, threshold) in zip(recorded["results"], actuals):
            passed = actual >= threshold if comparison in (">=", "required") else actual <= threshold
            self.assertEqual(result["gate"], name)
            self.assertEqual(result["comparison"], comparison)
            self.assertEqual(result["threshold"], threshold)
            if isinstance(actual, bool):
                self.assertEqual(result["actual"], actual)
            else:
                self.assert_close(result["actual"], float(actual))
            self.assertEqual(result["passed"], passed)
            if not passed:
                failed.append(name)
        self.assertEqual(recorded["failed_gates"], failed)
        self.assertEqual(recorded["gate_count"], 18)
        self.assertEqual(recorded["passed_count"], 18 - len(failed))
        self.assertFalse(recorded["all_passed"])
        self.assertEqual(recorded["promotion_decision"], "reject")
        self.assertEqual(self.metrics["promotion_decision"], "reject")
        self.assertEqual(
            failed,
            [
                "opishnyan_recall",
                "bubnivka_precision",
                "ceramic_macro_f1",
                "opishnyan_to_bubnivka_errors",
                "recall_floor:03_bubnivka_ceramics",
                "correction_recipe_stability",
            ],
        )

    def test_v2_does_not_outperform_v1_and_cannot_advance(self) -> None:
        v1_rows = read_csv(V1_OUTPUT_ROOT / "nested_oof_predictions.csv")
        v1_aggregate, v1_per_class = classification_metrics(v1_rows, self.classes)
        v2_aggregate, v2_per_class = classification_metrics(self.nested, self.classes)
        self.assertLess(v2_aggregate["accuracy"], v1_aggregate["accuracy"])
        self.assertLess(v2_aggregate["macro_f1"], v1_aggregate["macro_f1"])
        self.assertLess(
            v2_per_class["03_bubnivka_ceramics"]["recall"],
            v1_per_class["03_bubnivka_ceramics"]["recall"],
        )
        self.assertEqual(
            sum(
                row["true_class"] == "01_opishnyan_ceramics"
                and row["predicted_class"] == "03_bubnivka_ceramics"
                for row in self.nested
            ),
            45,
        )
        self.assertFalse(self.metrics["correction_recipe_stability"]["passed"])
        self.assertEqual(self.metrics["promotion_decision"], "reject")


if __name__ == "__main__":
    unittest.main()

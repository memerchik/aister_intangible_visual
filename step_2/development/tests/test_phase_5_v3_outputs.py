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
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness_v3"
V1_OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness"
V2_OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness_v2"
CONTRACT_PATH = (
    STEP_ROOT / "phases" / "phase_05_source_robustness" / "experiment_contract_v3.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "828cc5aa45724518e10808929589722133ce336ee272cb8bda20b5089c0dc980"
)
EXPECTED_METRICS_SHA256 = (
    "c624e2c6a8254716f0fc62abe2d493e3d40ad67be447787ed86528aee14b89e0"
)
EXPECTED_RUNNER_SHA256 = (
    "a62266e0e3062103df24b43fb5ceb8ceafebfce2b3ef5a42ae3c728791a76260"
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
    raise AssertionError(f"Expected a serialized Boolean, observed {value!r}")


def harmonic(values: list[float]) -> float:
    if any(value == 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def classification(
    rows: list[dict[str, str]], classes: list[str]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    support = Counter(row["true_class"] for row in rows)
    predicted = Counter(row["predicted_class"] for row in rows)
    true_positive = Counter(
        row["true_class"]
        for row in rows
        if row["true_class"] == row["predicted_class"]
    )
    per_class = {}
    for label in classes:
        precision = true_positive[label] / predicted[label] if predicted[label] else 0.0
        recall = true_positive[label] / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support[label]),
        }
    return (
        {
            "accuracy": sum(true_positive.values()) / len(rows),
            "macro_f1": statistics.fmean(per_class[label]["f1"] for label in classes),
            "balanced_accuracy": statistics.fmean(
                per_class[label]["recall"] for label in classes
            ),
        },
        per_class,
    )


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
    balanced = statistics.fmean(per_class.values())
    worst = min(per_class[label] for label in ceramics)
    return {
        "per_class_group_recall": per_class,
        "source_group_balanced_accuracy": balanced,
        "ceramic_worst_group_recall": worst,
        "robustness_score": harmonic([balanced, worst]),
    }


def rank_fusions(rows: list[dict[str, str]], tie_window: float) -> list[str]:
    remaining = list(rows)
    ranked: list[str] = []
    while remaining:
        best = max(float(row["readiness_score"]) for row in remaining)
        tied = [
            row
            for row in remaining
            if float(row["readiness_score"]) >= best - tie_window
        ]
        tied.sort(
            key=lambda row: (
                -float(row["minimum_fold_readiness_score"]),
                -float(row["robustness_score"]),
                -float(row["macro_f1"]),
                int(row["intervention_rank"]),
                float(row["blend_weight"]),
                int(row["head_count"]),
                int(row["encoder_count"]),
                row["configuration_id"],
            )
        )
        ranked.extend(row["configuration_id"] for row in tied)
        tied_ids = {row["configuration_id"] for row in tied}
        remaining = [row for row in remaining if row["configuration_id"] not in tied_ids]
    return ranked


class PhaseFiveV3OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUTPUT_ROOT / "metrics.json").is_file():
            raise unittest.SkipTest("canonical Phase 5 v3 outputs have not been generated")
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.metrics = json.loads((OUTPUT_ROOT / "metrics.json").read_text(encoding="utf-8"))
        cls.development = read_csv(SPLIT_ROOT / "development.csv")
        cls.sealed_test = read_csv(SPLIT_ROOT / "test.csv")
        cls.base_grid = read_csv(OUTPUT_ROOT / "base_grid.csv")
        cls.fusion_grid = read_csv(OUTPUT_ROOT / "fusion_grid.csv")
        cls.inner = read_csv(OUTPUT_ROOT / "inner_search.csv")
        cls.outer = read_csv(OUTPUT_ROOT / "outer_fold_metrics.csv")
        cls.nested = read_csv(OUTPUT_ROOT / "nested_oof_predictions.csv")
        cls.full = read_csv(OUTPUT_ROOT / "full_development_selection.csv")
        cls.selected = read_csv(OUTPUT_ROOT / "selected_oof_predictions.csv")
        cls.per_class = read_csv(OUTPUT_ROOT / "per_class_metrics.csv")
        cls.confusion = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        cls.confusion_pairs = read_csv(OUTPUT_ROOT / "confusion_pairs.csv")
        cls.classes = list(cls.metrics["class_names"])
        cls.ceramics = list(cls.metrics["ceramic_classes"])

    def assert_close(self, observed: object, expected: float) -> None:
        self.assertIsNotNone(observed)
        self.assertAlmostEqual(float(observed), expected, places=12)

    def reconstructed_metrics(
        self, rows: list[dict[str, str]]
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, object]]:
        aggregate, per_class = classification(rows, self.classes)
        source = source_metrics(rows, self.classes, self.ceramics)
        return aggregate, per_class, source

    def assert_payload(
        self, rows: list[dict[str, str]], payload: dict[str, object]
    ) -> None:
        aggregate, per_class, source = self.reconstructed_metrics(rows)
        for key, value in aggregate.items():
            self.assert_close(payload[key], value)
        for payload_key, class_key in (
            ("per_class_precision", "precision"),
            ("per_class_recall", "recall"),
            ("per_class_f1", "f1"),
            ("per_class_support", "support"),
        ):
            for label in self.classes:
                self.assert_close(payload[payload_key][label], per_class[label][class_key])
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
        ceramic_f1 = statistics.fmean(per_class[label]["f1"] for label in self.ceramics)
        boundary = [
            per_class[opishnyan]["recall"],
            per_class[bubnivka]["recall"],
            per_class[bubnivka]["precision"],
            ceramic_f1,
        ]
        self.assert_close(payload["opishnyan_recall"], boundary[0])
        self.assert_close(payload["bubnivka_recall"], boundary[1])
        self.assert_close(payload["bubnivka_precision"], boundary[2])
        self.assert_close(payload["ceramic_macro_f1"], boundary[3])
        self.assert_close(payload["boundary_score"], harmonic(boundary))
        hard_names = set(
            self.contract["diagnostic_slice_definitions"]["hard_three_opishnyan_cohorts"]
        )
        hard = [
            row
            for row in rows
            if row["true_class"] == opishnyan
            and row["source_atomic_cohort_ids"] in hard_names
        ]
        hard_accuracy = sum(row["is_correct"] == "True" for row in hard) / len(hard)
        self.assert_close(payload["hard_three_accuracy"], hard_accuracy)
        readiness = harmonic(
            [
                aggregate["accuracy"],
                aggregate["macro_f1"],
                source["source_group_balanced_accuracy"],
                source["ceramic_worst_group_recall"],
                *boundary,
                hard_accuracy,
            ]
        )
        self.assert_close(payload["readiness_score"], readiness)
        self.assertEqual(
            payload["opishnyan_to_bubnivka_errors"],
            sum(
                row["true_class"] == opishnyan and row["predicted_class"] == bubnivka
                for row in rows
            ),
        )
        self.assertEqual(
            payload["bubnivka_to_opishnyan_errors"],
            sum(
                row["true_class"] == bubnivka and row["predicted_class"] == opishnyan
                for row in rows
            ),
        )

    def test_required_artifacts_hashes_and_png_are_exact(self) -> None:
        required = set(self.contract["output_contract"]["required"])
        self.assertEqual({path.name for path in OUTPUT_ROOT.iterdir()}, required)
        self.assertEqual(set(self.metrics["required_outputs"]), required)
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(sha256_file(OUTPUT_ROOT / "metrics.json"), EXPECTED_METRICS_SHA256)
        self.assertEqual(self.metrics["script_sha256"], EXPECTED_RUNNER_SHA256)
        png = (OUTPUT_ROOT / "confusion_matrix.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertGreater(width * height, 0)

    def test_inputs_code_and_nine_embedding_blocks_are_pinned(self) -> None:
        self.assertEqual(self.metrics["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)
        for relative, expected in self.metrics["code_provenance"]["files_sha256"].items():
            assert_recorded_file(self, relative, expected)
        inputs = self.contract["input_contract"]
        for prefix in ("phase_5_v1", "phase_5_v2"):
            for suffix in ("contract", "metrics", "full_selection", "nested_predictions"):
                path_key = f"{prefix}_{suffix}"
                hash_key = f"{path_key}_sha256"
                assert_recorded_file(self, inputs[path_key], inputs[hash_key])
        self.assertEqual(len(self.metrics["embedding_provenance"]), 9)
        for identifier, provenance in self.metrics["embedding_provenance"].items():
            cache_name, array_name = identifier.split(".", 1)
            frozen = self.contract["cache_allowlist"][cache_name]
            self.assertEqual(provenance["cache_sha256"], frozen["file_sha256"])
            self.assertEqual(provenance["array_sha256"], frozen["arrays"][array_name]["sha256"])
            self.assertEqual(provenance["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)

    def test_predictions_are_complete_development_only_and_source_atomic(self) -> None:
        development_ids = {row["image_id"] for row in self.development}
        test_ids = {row["image_id"] for row in self.sealed_test}
        for rows in (self.nested, self.selected):
            self.assertEqual(len(rows), 1693)
            self.assertEqual({row["image_id"] for row in rows}, development_ids)
            self.assertFalse({row["image_id"] for row in rows} & test_ids)
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertFalse(self.metrics["phase_6_transition_allowed"])
        source_folds: dict[str, set[str]] = defaultdict(set)
        for row in self.nested:
            source_folds[row["source_atomic_split_group_id"]].add(row["cv_fold"])
        self.assertTrue(all(len(values) == 1 for values in source_folds.values()))

    def test_probability_recipe_and_application_fields_are_consistent(self) -> None:
        probability_columns = [f"probability_{label}" for label in self.classes]
        by_fold = {
            row["outer_fold"]: (
                row["selected_base_configuration_id"],
                row["selected_fusion_configuration_id"],
                row["fusion_mode"],
            )
            for row in self.outer
        }
        for row in self.nested + self.selected:
            probabilities = [float(row[column]) for column in probability_columns]
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in probabilities))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            order = sorted(range(5), key=lambda index: (-probabilities[index], index))
            self.assertEqual(row["predicted_class"], self.classes[order[0]])
            self.assertEqual(row["top_1_class"], self.classes[order[0]])
            self.assertEqual(row["top_2_class"], self.classes[order[1]])
            self.assertEqual(row["top_3_class"], self.classes[order[2]])
            self.assert_close(row["uncalibrated_max_probability"], probabilities[order[0]])
            self.assertEqual(parse_bool(row["is_correct"]), row["true_class"] == row["predicted_class"])
            parse_bool(row["fusion_applied"])
        for row in self.nested:
            expected = by_fold[row["cv_fold"]]
            self.assertEqual(
                (
                    row["selected_base_configuration_id"],
                    row["selected_fusion_configuration_id"],
                    row["fusion_mode"],
                ),
                expected,
            )
        self.assertEqual(
            sum(parse_bool(row["fusion_applied"]) for row in self.nested),
            self.metrics["nested_aggregate_oof_metrics"]["fusion_applied_count"],
        )

    def test_nested_and_full_metrics_are_independently_reconstructed(self) -> None:
        self.assert_payload(self.nested, self.metrics["nested_aggregate_oof_metrics"])
        self.assert_payload(self.selected, self.metrics["full_development_selected_oof_metrics"])
        recorded = {row["class_name"]: row for row in self.per_class}
        _, per_class = classification(self.nested, self.classes)
        for label in self.classes:
            for key in ("precision", "recall", "f1", "support"):
                self.assert_close(recorded[label][key], per_class[label][key])

    def test_confusion_and_outer_metrics_reconstruct_predictions(self) -> None:
        matrix = {
            true: Counter(
                row["predicted_class"] for row in self.nested if row["true_class"] == true
            )
            for true in self.classes
        }
        for row in self.confusion:
            for predicted in self.classes:
                self.assertEqual(int(row[predicted]), matrix[row["true_class"]][predicted])
        self.assertEqual(len(self.confusion_pairs), 20)
        for row in self.confusion_pairs:
            true, predicted = row["true_class"], row["predicted_class"]
            count = matrix[true][predicted]
            self.assertEqual(int(row["error_count"]), count)
            self.assert_close(row["fraction_of_true_class"], count / sum(matrix[true].values()))
        for outer in self.outer:
            rows = [row for row in self.nested if row["cv_fold"] == outer["outer_fold"]]
            aggregate, _, source = self.reconstructed_metrics(rows)
            self.assertEqual(len(rows), int(outer["validation_examples"]))
            for key, value in aggregate.items():
                self.assert_close(outer[key], value)
            for key in (
                "source_group_balanced_accuracy",
                "ceramic_worst_group_recall",
                "robustness_score",
            ):
                self.assert_close(outer[key], source[key])

    def test_grid_sequential_selection_eligibility_and_ranking_are_exact(self) -> None:
        self.assertEqual(len(self.base_grid), 3)
        self.assertEqual(len(self.fusion_grid), 28)
        self.assertEqual(len(self.inner), 5 * (3 + 28))
        self.assertEqual(len(self.full), 3 + 28)
        self.assertEqual(sum(row["fusion_mode"] == "none" for row in self.fusion_grid), 1)
        limits = self.contract["fusion_selection_rule"][
            "eligibility_relative_to_uncorrected_selected_base"
        ]
        opishnyan = self.metrics["pair_classes"][0]
        for fold in self.contract["scope"]["outer_folds"]:
            search = [row for row in self.inner if row["outer_fold"] == fold]
            bases = [row for row in search if row["stage"] == "base"]
            fusions = [row for row in search if row["stage"] == "fusion"]
            self.assertEqual(sum(parse_bool(row["selected"]) for row in bases), 1)
            self.assertEqual(sum(parse_bool(row["selected"]) for row in fusions), 1)
            selected_base = next(row["configuration_id"] for row in bases if parse_bool(row["selected"]))
            self.assertTrue(all(row["base_configuration_id"] == selected_base for row in fusions))
            reference = next(row for row in fusions if row["configuration_id"] == "fusion_none")
            reference_recalls = json.loads(reference["per_class_recall_json"])
            reference_groups = json.loads(reference["per_class_group_recall_json"])
            for row in fusions:
                recalls = json.loads(row["per_class_recall_json"])
                groups = json.loads(row["per_class_group_recall_json"])
                failures = []
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
            eligible = [row for row in fusions if parse_bool(row["eligible"])]
            ranked = rank_fusions(eligible, self.contract["fusion_selection_rule"]["tie_window"])
            recorded = [
                row["configuration_id"]
                for row in sorted(eligible, key=lambda value: int(value["eligible_selection_rank"]))
            ]
            self.assertEqual(recorded, ranked)
            self.assertEqual(
                next(row["configuration_id"] for row in fusions if parse_bool(row["selected"])),
                ranked[0],
            )

    def test_base_and_fusion_stability_are_recomputed(self) -> None:
        full_candidate = self.metrics["full_development_selected_candidate"]
        for stage, identifier, score, window, payload_key in (
            (
                "base",
                full_candidate["base_configuration_id"],
                "robustness_score",
                self.contract["base_procedure"]["selection_rule"]["tie_window"],
                "base_recipe_stability",
            ),
            (
                "fusion",
                full_candidate["fusion_configuration_id"],
                "readiness_score",
                self.contract["fusion_selection_rule"]["tie_window"],
                "fusion_recipe_stability",
            ),
        ):
            passing = []
            for fold in self.contract["scope"]["outer_folds"]:
                rows = [
                    row
                    for row in self.inner
                    if row["outer_fold"] == fold and row["stage"] == stage
                ]
                target = next(row for row in rows if row["configuration_id"] == identifier)
                eligible = [row for row in rows if parse_bool(row["eligible"])]
                best = max(float(row[score]) for row in eligible)
                within = parse_bool(target["eligible"]) and float(target[score]) >= best - float(window)
                if within:
                    passing.append(fold)
            payload = self.metrics[payload_key]
            self.assertEqual(payload["passing_outer_folds"], passing)
            self.assertEqual(payload["observed_outer_searches_within_tie_window"], len(passing))
            self.assertEqual(payload["passed"], len(passing) >= 4)
        self.assertTrue(self.metrics["base_recipe_stability"]["passed"])
        self.assertFalse(self.metrics["fusion_recipe_stability"]["passed"])

    def test_every_promotion_gate_and_phase6_decision_are_recomputed(self) -> None:
        gates = self.contract["promotion_gates"]
        definitions = self.contract["diagnostic_slice_definitions"]
        aggregate, per_class, source = self.reconstructed_metrics(self.nested)
        opishnyan, bubnivka = self.metrics["pair_classes"]

        def accuracy(rows: list[dict[str, str]]) -> float:
            self.assertTrue(rows)
            return sum(row["is_correct"] == "True" for row in rows) / len(rows)

        hard_names = definitions["hard_three_opishnyan_cohorts"]
        hard = [
            row
            for row in self.nested
            if row["true_class"] == opishnyan and row["source_atomic_cohort_ids"] in hard_names
        ]
        challenging = [
            row
            for row in self.nested
            if row["true_class"] == opishnyan
            and (row["source_atomic_cohort_ids"] in hard_names or not row["source_atomic_cohort_ids"])
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
        supports = Counter(row["source_atomic_cohort_ids"] for row in self.nested)
        large = sorted(
            cohort
            for cohort, count in supports.items()
            if cohort and count >= gates["large_named_source_minimum_size"]
        )
        large_macro = statistics.fmean(
            accuracy([row for row in self.nested if row["source_atomic_cohort_ids"] == cohort])
            for cohort in large
        )
        ceramic_f1 = statistics.fmean(per_class[label]["f1"] for label in self.ceramics)
        fold_accuracy = [float(row["accuracy"]) for row in self.outer]
        actuals: list[tuple[str, str, object, object]] = [
            ("overall_accuracy", ">=", aggregate["accuracy"], gates["overall_accuracy_min"]),
            ("overall_macro_f1", ">=", aggregate["macro_f1"], gates["overall_macro_f1_min"]),
            ("opishnyan_recall", ">=", per_class[opishnyan]["recall"], gates["opishnyan_recall_min"]),
            ("bubnivka_precision", ">=", per_class[bubnivka]["precision"], gates["bubnivka_precision_min"]),
            ("ceramic_macro_f1", ">=", ceramic_f1, gates["ceramic_macro_f1_min"]),
            ("opishnyan_to_bubnivka_errors", "<=", float(sum(row["true_class"] == opishnyan and row["predicted_class"] == bubnivka for row in self.nested)), gates["opishnyan_to_bubnivka_errors_max"]),
            ("challenging_opishnyan_accuracy", ">=", accuracy(challenging), gates["challenging_opishnyan_accuracy_min"]),
            ("hard_three_opishnyan_pooled_accuracy", ">=", accuracy(hard), gates["hard_three_opishnyan_pooled_accuracy_min"]),
            ("hard_three_opishnyan_cohort_macro_accuracy", ">=", hard_macro, gates["hard_three_opishnyan_cohort_macro_accuracy_min"]),
            ("large_named_source_macro_accuracy", ">=", large_macro, gates["large_named_source_macro_accuracy_min"]),
            ("worst_fold_accuracy", ">=", min(fold_accuracy), gates["worst_fold_accuracy_min"]),
            ("fold_accuracy_std", "<=", statistics.stdev(fold_accuracy), gates["fold_accuracy_std_max"]),
        ]
        for label in sorted(gates["other_class_recall_floors"]):
            actuals.append((f"recall_floor:{label}", ">=", per_class[label]["recall"], gates["other_class_recall_floors"][label]))
        actuals.extend(
            [
                ("base_recipe_stability", "required", self.metrics["base_recipe_stability"]["passed"], True),
                ("fusion_recipe_stability", "required", self.metrics["fusion_recipe_stability"]["passed"], True),
                ("v1_accuracy_non_regression", ">=", aggregate["accuracy"], gates["v1_non_regression"]["accuracy_min"]),
                ("v1_macro_f1_non_regression", ">=", aggregate["macro_f1"], gates["v1_non_regression"]["macro_f1_min"]),
                ("v1_source_robustness_non_regression", ">=", source["robustness_score"], gates["v1_non_regression"]["source_robustness_min"]),
            ]
        )
        promotion = self.metrics["promotion_gates"]
        self.assertEqual(len(promotion["results"]), len(actuals), 21)
        failed = []
        for result, (name, comparison, actual, threshold) in zip(promotion["results"], actuals):
            passed = actual >= threshold if comparison in (">=", "required") else actual <= threshold
            self.assertEqual(result["gate"], name)
            self.assertEqual(result["comparison"], comparison)
            if isinstance(actual, bool):
                self.assertEqual(result["actual"], actual)
            else:
                self.assert_close(result["actual"], float(actual))
            self.assertEqual(result["passed"], passed)
            if not passed:
                failed.append(name)
        self.assertEqual(promotion["failed_gates"], failed)
        self.assertEqual(promotion["passed_count"], 21 - len(failed))
        self.assertEqual(promotion["promotion_decision"], "reject")
        self.assertFalse(self.metrics["phase_6_transition_allowed"])

    def test_v1_reproduction_and_v1_v2_v3_conclusions_are_exact(self) -> None:
        reproduction = self.metrics["v1_reproduction_validation"]
        self.assertTrue(reproduction["validated"])
        self.assertEqual(
            reproduction["expected_decision_fingerprint_sha256"],
            reproduction["observed_decision_fingerprint_sha256"],
        )
        v1_rows = read_csv(V1_OUTPUT_ROOT / "nested_oof_predictions.csv")
        v2_rows = read_csv(V2_OUTPUT_ROOT / "nested_oof_predictions.csv")
        v1_aggregate, v1_class = classification(v1_rows, self.classes)
        v2_aggregate, _ = classification(v2_rows, self.classes)
        v3_aggregate, v3_class = classification(self.nested, self.classes)
        self.assertGreater(v1_aggregate["accuracy"], v3_aggregate["accuracy"])
        self.assertGreater(v3_aggregate["accuracy"], v2_aggregate["accuracy"])
        self.assertGreater(
            v3_class["03_bubnivka_ceramics"]["recall"],
            v1_class["03_bubnivka_ceramics"]["recall"],
        )
        self.assertLess(
            v3_class["01_opishnyan_ceramics"]["recall"],
            v1_class["01_opishnyan_ceramics"]["recall"],
        )
        self.assertEqual(self.metrics["promotion_decision"], "reject")


if __name__ == "__main__":
    unittest.main()

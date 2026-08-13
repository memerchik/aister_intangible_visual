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

from relocation_support import assert_recorded_file, resolve_recorded_path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SPLIT_ROOT = STEP_ROOT / "splits"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness"
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "55d0dd0d3665b4b84b33a9c9461763c6858452bd0b10b679fd9b77fbe630a87f"
)
EXPECTED_PHASE4_DECISION_SHA256 = (
    "d6f5b90c2c6b8af39fb8e6bffbe3e44ba3e6d99b3f8f069b8f1febfe95bab307"
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


def harmonic_mean(first: float, second: float) -> float:
    if first == 0.0 or second == 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


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
    aggregates = {
        "accuracy": sum(true_positive.values()) / len(rows),
        "macro_f1": statistics.fmean(per_class[label]["f1"] for label in classes),
        "balanced_accuracy": statistics.fmean(
            per_class[label]["recall"] for label in classes
        ),
    }
    return aggregates, per_class


def source_group_metrics(
    rows: list[dict[str, str]], classes: list[str], ceramics: list[str]
) -> dict[str, object]:
    counts: dict[str, dict[str, list[int]]] = {
        label: defaultdict(lambda: [0, 0]) for label in classes
    }
    for row in rows:
        bucket = counts[row["true_class"]][row["source_atomic_split_group_id"]]
        bucket[0] += row["true_class"] == row["predicted_class"]
        bucket[1] += 1
    per_class = {
        label: statistics.fmean(correct / total for correct, total in counts[label].values())
        for label in classes
    }
    sgba = statistics.fmean(per_class.values())
    worst_ceramic = min(per_class[label] for label in ceramics)
    return {
        "per_class_group_recall": per_class,
        "source_group_balanced_accuracy": sgba,
        "ceramic_worst_group_recall": worst_ceramic,
        "robustness_score": harmonic_mean(sgba, worst_ceramic),
    }


def stable_rank(rows: list[dict[str, str]], tie_window: float) -> list[dict[str, str]]:
    remaining = list(rows)
    ranked: list[dict[str, str]] = []
    while remaining:
        best_score = max(float(row["robustness_score"]) for row in remaining)
        tied = [
            row
            for row in remaining
            if float(row["robustness_score"]) >= best_score - tie_window
        ]
        tied.sort(
            key=lambda row: (
                -float(row["minimum_fold_source_group_balanced_accuracy"]),
                -float(row["macro_f1"]),
                int(row["inference_cost_rank"]),
                0 if row["specialist"] == "none" else 1,
                0 if math.isclose(float(row["c_value"]), 10.0, abs_tol=1e-12) else 1,
                float(row["source_group_exponent"]),
                row["configuration_id"],
            )
        )
        ranked.extend(tied)
        tied_ids = {row["configuration_id"] for row in tied}
        remaining = [
            row for row in remaining if row["configuration_id"] not in tied_ids
        ]
    return ranked


def expected_eligibility_failures(
    row: dict[str, str],
    reference: dict[str, str],
    limits: dict[str, object],
) -> list[str]:
    recalls = json.loads(row["per_class_recall_json"])
    reference_recalls = json.loads(reference["per_class_recall_json"])
    failures: list[str] = []
    if float(row["macro_f1"]) < float(reference["macro_f1"]) - float(
        limits["maximum_macro_f1_drop"]
    ):
        failures.append("macro_f1_drop")
    if recalls["02_ornek"] < reference_recalls["02_ornek"] - float(
        limits["maximum_ornek_recall_drop"]
    ):
        failures.append("ornek_recall_drop")
    if recalls["04_petrykivka_painting"] < reference_recalls[
        "04_petrykivka_painting"
    ] - float(limits["maximum_petrykivka_recall_drop"]):
        failures.append("petrykivka_recall_drop")
    return failures


def slice_values(
    rows: list[dict[str, str]], selected: list[dict[str, str]]
) -> dict[str, object]:
    count = len(selected)
    correct = sum(row["true_class"] == row["predicted_class"] for row in selected)
    return {
        "image_count": count,
        "correct_count": correct,
        "error_count": count - correct,
        "accuracy": correct / count if count else None,
        "mean_uncalibrated_max_probability": (
            statistics.fmean(float(row["uncalibrated_max_probability"]) for row in selected)
            if count
            else None
        ),
    }


class PhaseFiveOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUTPUT_ROOT / "metrics.json").is_file():
            raise unittest.SkipTest("canonical Phase 5 outputs have not been generated")
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.metrics = json.loads((OUTPUT_ROOT / "metrics.json").read_text(encoding="utf-8"))
        cls.development = read_csv(SPLIT_ROOT / "development.csv")
        cls.sealed_test = read_csv(SPLIT_ROOT / "test.csv")
        cls.base_grid = read_csv(OUTPUT_ROOT / "base_grid.csv")
        cls.inner_search = read_csv(OUTPUT_ROOT / "inner_search.csv")
        cls.outer_folds = read_csv(OUTPUT_ROOT / "outer_fold_metrics.csv")
        cls.nested = read_csv(OUTPUT_ROOT / "nested_oof_predictions.csv")
        cls.full_selection = read_csv(OUTPUT_ROOT / "full_development_selection.csv")
        cls.selected = read_csv(OUTPUT_ROOT / "selected_oof_predictions.csv")
        cls.per_class = read_csv(OUTPUT_ROOT / "per_class_metrics.csv")
        cls.slices = read_csv(OUTPUT_ROOT / "diagnostic_slices.csv")
        cls.confusion_pairs = read_csv(OUTPUT_ROOT / "confusion_pairs.csv")
        cls.confusion = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        cls.classes = list(cls.metrics["class_names"])
        cls.ceramics = list(cls.metrics["ceramic_classes"])

    def assert_close(self, observed: object, expected: float, places: int = 12) -> None:
        self.assertIsNotNone(observed)
        self.assertAlmostEqual(float(observed), expected, places=places)

    def assert_metric_payload(
        self,
        observed: dict[str, object],
        aggregate: dict[str, float],
        per_class: dict[str, dict[str, float]],
        source: dict[str, object],
    ) -> None:
        for key, value in aggregate.items():
            self.assert_close(observed[key], value)
        for metric_key, row_key in (
            ("per_class_precision", "precision"),
            ("per_class_recall", "recall"),
            ("per_class_f1", "f1"),
            ("per_class_support", "support"),
        ):
            values = observed[metric_key]
            self.assertEqual(set(values), set(self.classes))
            for label in self.classes:
                self.assert_close(values[label], per_class[label][row_key])
        for key in (
            "source_group_balanced_accuracy",
            "ceramic_worst_group_recall",
            "robustness_score",
        ):
            self.assert_close(observed[key], source[key])
        recorded_group = observed["per_class_group_recall"]
        expected_group = source["per_class_group_recall"]
        self.assertEqual(set(recorded_group), set(self.classes))
        for label in self.classes:
            self.assert_close(recorded_group[label], expected_group[label])

    def test_required_output_contract_is_exact_and_png_is_nonempty(self) -> None:
        required = self.contract["output_contract"]["required"]
        self.assertEqual(self.metrics["required_outputs"], required)
        self.assertEqual({path.name for path in OUTPUT_ROOT.iterdir()}, set(required))
        for name in required:
            self.assertTrue((OUTPUT_ROOT / name).is_file(), name)
            self.assertGreater((OUTPUT_ROOT / name).stat().st_size, 0, name)

        png = (OUTPUT_ROOT / "confusion_matrix.png").read_bytes()
        self.assertGreater(len(png), 1024)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png[12:16], b"IHDR")
        width, height = struct.unpack(">II", png[16:24])
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_predictions_are_development_only_complete_ordered_and_well_formed(self) -> None:
        development_ids = [row["image_id"] for row in self.development]
        test_ids = {row["image_id"] for row in self.sealed_test}
        self.assertEqual([row["image_id"] for row in self.nested], development_ids)
        self.assertEqual([row["image_id"] for row in self.selected], development_ids)
        self.assertEqual(len(set(development_ids)), len(development_ids))
        self.assertFalse(set(development_ids) & test_ids)
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertEqual(
            self.metrics["evaluation_scope"],
            "development_only_selection_aware_nested_source_atomic_cv",
        )
        self.assertFalse(any("test_predictions" in path.name for path in OUTPUT_ROOT.iterdir()))
        self.assertFalse(self.metrics["performance_estimate"]["available"])
        self.assertTrue(
            self.metrics["performance_estimate"]["nested_development_estimate_available"]
        )
        self.assertFalse(self.metrics["probability_policy"]["calibrated"])
        self.assertFalse(
            self.metrics["probability_policy"]["threshold_selection_used"]
        )

        full_recipe = self.metrics["full_development_selected_candidate"]
        nested_recipes = self.metrics["nested_selected_recipes_by_outer_fold"]
        metadata_pairs = (
            ("relative_path", "relative_path"),
            ("true_class", "ornament_label"),
            ("cv_fold", "cv_fold"),
            ("production_split", "production_split"),
            ("source_atomic_split_group_id", "source_atomic_split_group_id"),
            ("source_atomic_cohort_ids", "source_atomic_cohort_ids"),
            ("object_type", "object_type"),
            ("motif_visibility", "motif_visibility"),
        )
        probability_fields = [f"probability_{label}" for label in self.classes]
        for development, nested, selected in zip(
            self.development, self.nested, self.selected
        ):
            for prediction in (nested, selected):
                for output_key, development_key in metadata_pairs:
                    self.assertEqual(prediction[output_key], development[development_key])
                probabilities = [float(prediction[field]) for field in probability_fields]
                self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities))
                self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
                ranked = sorted(
                    range(len(self.classes)), key=lambda index: (-probabilities[index], index)
                )
                self.assertEqual(
                    [prediction[f"top_{rank}_class"] for rank in (1, 2, 3)],
                    [self.classes[index] for index in ranked[:3]],
                )
                self.assertEqual(prediction["predicted_class"], self.classes[ranked[0]])
                self.assertEqual(
                    parse_bool(prediction["is_correct"]),
                    prediction["true_class"] == prediction["predicted_class"],
                )
                self.assert_close(
                    prediction["uncalibrated_max_probability"], max(probabilities)
                )

            self.assertEqual(
                nested["evaluation_status"],
                "nested_outer_oof_adaptive_phase5_procedure",
            )
            self.assertEqual(
                nested["selected_base_configuration_id"],
                nested_recipes[nested["cv_fold"]]["base_configuration_id"],
            )
            self.assertEqual(
                nested["selected_specialist"],
                nested_recipes[nested["cv_fold"]]["specialist"],
            )
            self.assertEqual(
                selected["evaluation_status"],
                "selection_conditional_full_development_oof",
            )
            self.assertEqual(
                selected["selected_base_configuration_id"],
                full_recipe["base_configuration_id"],
            )
            self.assertEqual(selected["selected_specialist"], full_recipe["specialist"])

    def test_source_groups_and_every_outer_boundary_are_disjoint(self) -> None:
        group_folds: dict[str, set[str]] = defaultdict(set)
        for row in self.development:
            self.assertEqual(
                row["source_atomic_split_group_id"], row["split_group_id"]
            )
            group_folds[row["source_atomic_split_group_id"]].add(row["cv_fold"])
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))

        by_outer = {row["outer_fold"]: row for row in self.outer_folds}
        self.assertEqual(set(by_outer), set(self.contract["scope"]["outer_folds"]))
        for outer_fold, output_row in by_outer.items():
            train = [row for row in self.development if row["cv_fold"] != outer_fold]
            validation = [row for row in self.development if row["cv_fold"] == outer_fold]
            train_groups = {row["source_atomic_split_group_id"] for row in train}
            validation_groups = {
                row["source_atomic_split_group_id"] for row in validation
            }
            self.assertFalse(train_groups & validation_groups)
            self.assertEqual(int(output_row["train_examples"]), len(train))
            self.assertEqual(int(output_row["validation_examples"]), len(validation))
            inner = [row for row in self.inner_search if row["outer_fold"] == outer_fold]
            self.assertTrue(inner)
            self.assertTrue(all(int(row["evaluated_examples"]) == len(train) for row in inner))

    def test_nested_and_full_selected_metrics_are_independently_recomputed(self) -> None:
        nested_aggregate, nested_per_class = classification_metrics(
            self.nested, self.classes
        )
        nested_source = source_group_metrics(self.nested, self.classes, self.ceramics)
        self.assert_metric_payload(
            self.metrics["nested_aggregate_oof_metrics"],
            nested_aggregate,
            nested_per_class,
            nested_source,
        )

        selected_aggregate, selected_per_class = classification_metrics(
            self.selected, self.classes
        )
        selected_source = source_group_metrics(self.selected, self.classes, self.ceramics)
        self.assert_metric_payload(
            self.metrics["full_development_selected_oof_metrics"],
            selected_aggregate,
            selected_per_class,
            selected_source,
        )

        self.assertEqual(len(self.per_class), len(self.classes))
        for index, row in enumerate(self.per_class):
            label = self.classes[index]
            self.assertEqual(row["evaluation_scope"], "nested_outer_oof")
            self.assertEqual(int(row["class_index"]), index)
            self.assertEqual(row["class_name"], label)
            for output_key, metric_key in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "f1"),
                ("support", "support"),
            ):
                self.assert_close(row[output_key], nested_per_class[label][metric_key])

    def test_outer_fold_metrics_and_recipe_provenance_recompute(self) -> None:
        metric_rows = {row["outer_fold"]: row for row in self.outer_folds}
        embedded_rows = {
            str(row["outer_fold"]): row
            for row in self.metrics["nested_outer_fold_metrics"]
        }
        recipes = self.metrics["nested_selected_recipes_by_outer_fold"]
        for fold in self.contract["scope"]["outer_folds"]:
            rows = [row for row in self.nested if row["cv_fold"] == fold]
            aggregate, per_class = classification_metrics(rows, self.classes)
            source = source_group_metrics(rows, self.classes, self.ceramics)
            output = metric_rows[fold]
            embedded = embedded_rows[fold]
            self.assertEqual(output["selected_base_configuration_id"], recipes[fold]["base_configuration_id"])
            self.assertEqual(output["selected_specialist"], recipes[fold]["specialist"])
            self.assertEqual(embedded["selected_base_configuration_id"], output["selected_base_configuration_id"])
            self.assertEqual(embedded["selected_specialist"], output["selected_specialist"])
            for key, value in aggregate.items():
                self.assert_close(output[key], value)
                self.assert_close(embedded[key], value)
            for key in (
                "source_group_balanced_accuracy",
                "ceramic_worst_group_recall",
                "robustness_score",
            ):
                self.assert_close(output[key], source[key])
                self.assert_close(embedded[key], source[key])
            recorded_recalls = json.loads(output["per_class_recall_json"])
            recorded_group_recalls = json.loads(output["per_class_group_recall_json"])
            for label in self.classes:
                self.assert_close(recorded_recalls[label], per_class[label]["recall"])
                self.assert_close(
                    recorded_group_recalls[label],
                    source["per_class_group_recall"][label],
                )

    def test_confusion_matrix_and_error_pairs_recompute(self) -> None:
        counts = Counter(
            (row["true_class"], row["predicted_class"]) for row in self.nested
        )
        self.assertEqual([row["true_class"] for row in self.confusion], self.classes)
        for row in self.confusion:
            for predicted in self.classes:
                self.assertEqual(
                    int(row[predicted]), counts[(row["true_class"], predicted)]
                )

        expected_pairs: list[tuple[str, str, int, float]] = []
        support = Counter(row["true_class"] for row in self.nested)
        for truth in self.classes:
            for predicted in self.classes:
                if truth != predicted:
                    count = counts[(truth, predicted)]
                    expected_pairs.append((truth, predicted, count, count / support[truth]))
        expected_pairs.sort(key=lambda value: (-value[2], value[0], value[1]))
        self.assertEqual(len(self.confusion_pairs), len(expected_pairs))
        for observed, expected in zip(self.confusion_pairs, expected_pairs):
            truth, predicted, count, fraction = expected
            self.assertEqual(observed["true_class"], truth)
            self.assertEqual(observed["predicted_class"], predicted)
            self.assertEqual(int(observed["error_count"]), count)
            self.assert_close(observed["fraction_of_true_class"], fraction)

    def test_all_diagnostic_slices_have_exact_supports_and_formulas(self) -> None:
        definitions = self.contract["diagnostic_slice_definitions"]
        hard = set(definitions["hard_three_opishnyan_cohorts"])
        opishnyan = definitions["ceramic_classes"][0]
        expected: dict[tuple[str, str], dict[str, object]] = {}

        def add(dimension: str, value: str, selected: list[dict[str, str]]) -> None:
            expected[(dimension, value)] = slice_values(self.nested, selected)

        for fold in sorted({row["cv_fold"] for row in self.nested}):
            add("cv_fold", fold, [row for row in self.nested if row["cv_fold"] == fold])
        cohorts = sorted({row["source_atomic_cohort_ids"] for row in self.nested if row["source_atomic_cohort_ids"]})
        for cohort in cohorts:
            add(
                "named_source_cohort",
                cohort,
                [row for row in self.nested if row["source_atomic_cohort_ids"] == cohort],
            )
        for value in sorted({row["object_type"] for row in self.nested}):
            add("object_type", value, [row for row in self.nested if row["object_type"] == value])
        for value in sorted({row["motif_visibility"] for row in self.nested}):
            add(
                "motif_visibility",
                value,
                [row for row in self.nested if row["motif_visibility"] == value],
            )
        add(
            "contract_slice",
            "hard_three_opishnyan_pooled",
            [
                row
                for row in self.nested
                if row["true_class"] == opishnyan
                and row["source_atomic_cohort_ids"] in hard
            ],
        )
        add(
            "contract_slice",
            "challenging_opishnyan",
            [
                row
                for row in self.nested
                if row["true_class"] == opishnyan
                and (
                    row["source_atomic_cohort_ids"] in hard
                    or row["source_atomic_cohort_ids"] == ""
                )
            ],
        )
        cohort_support = Counter(row["source_atomic_cohort_ids"] for row in self.nested)
        minimum = int(self.contract["promotion_gates"]["large_named_source_minimum_size"])
        large = {
            cohort for cohort, count in cohort_support.items() if cohort and count >= minimum
        }
        add(
            "contract_slice",
            "large_named_sources_pooled",
            [row for row in self.nested if row["source_atomic_cohort_ids"] in large],
        )

        observed = {(row["dimension"], row["value"]): row for row in self.slices}
        self.assertEqual(set(observed), set(expected))
        self.assertEqual(len(observed), len(self.slices))
        for key, values in expected.items():
            row = observed[key]
            for count_key in ("image_count", "correct_count", "error_count"):
                self.assertEqual(int(row[count_key]), values[count_key], key)
            if values["accuracy"] is None:
                self.assertEqual(row["accuracy"], "")
                self.assertEqual(row["mean_uncalibrated_max_probability"], "")
            else:
                self.assert_close(row["accuracy"], values["accuracy"])
                self.assert_close(
                    row["mean_uncalibrated_max_probability"],
                    values["mean_uncalibrated_max_probability"],
                )

    def _audit_selection_scope(self, rows: list[dict[str, str]]) -> None:
        base = [row for row in rows if row["stage"] == "base"]
        specialist = [row for row in rows if row["stage"] == "specialist"]
        self.assertEqual(len(base), 12)
        self.assertEqual(len(specialist), 2)
        reference_spec = self.contract["base_grid"]["exact_phase_4_reference"]
        reference_c = float(reference_spec["c_value"])
        reference_exponent = float(reference_spec["source_group_exponent"])
        reference_id = (
            f"{reference_spec['feature_family']}"
            f"__c{int(reference_c) if reference_c.is_integer() else str(reference_c).replace('.', 'p')}"
            f"__source{int(reference_exponent) if reference_exponent.is_integer() else str(reference_exponent).replace('.', 'p')}"
        )
        for row in rows:
            expected_reference_flag = row["configuration_id"] in {
                reference_id,
                reference_id + "__specialist_none",
            }
            self.assertEqual(
                parse_bool(row["is_exact_reference"]),
                expected_reference_flag,
                row["configuration_id"],
            )
        reference_rows = [row for row in base if parse_bool(row["is_exact_reference"])]
        self.assertEqual(len(reference_rows), 1)
        reference = reference_rows[0]
        limits = self.contract["selection_rule"][
            "eligibility_relative_to_exact_reference"
        ]
        tie_window = float(self.contract["selection_rule"]["tie_window"])

        for stage_rows in (base, specialist):
            for row in stage_rows:
                failures = expected_eligibility_failures(row, reference, limits)
                self.assertEqual(row["eligibility_failures"], "|".join(failures))
                self.assertEqual(parse_bool(row["eligible"]), not failures)
                recalls = json.loads(row["per_class_recall_json"])
                group_recalls = json.loads(row["per_class_group_recall_json"])
                self.assert_close(row["balanced_accuracy"], statistics.fmean(recalls.values()))
                self.assert_close(
                    row["source_group_balanced_accuracy"],
                    statistics.fmean(group_recalls.values()),
                )
                worst = min(group_recalls[label] for label in self.ceramics)
                self.assert_close(row["ceramic_worst_group_recall"], worst)
                self.assert_close(
                    row["robustness_score"],
                    harmonic_mean(statistics.fmean(group_recalls.values()), worst),
                )
                self.assertTrue(
                    0.0 <= float(row["minimum_fold_source_group_balanced_accuracy"]) <= 1.0
                )

            robustness_order = stable_rank(stage_rows, tie_window)
            expected_robustness_rank = {
                row["configuration_id"]: index
                for index, row in enumerate(robustness_order, start=1)
            }
            eligible = [row for row in stage_rows if parse_bool(row["eligible"])]
            eligible_order = stable_rank(eligible, tie_window)
            expected_eligible_rank = {
                row["configuration_id"]: index
                for index, row in enumerate(eligible_order, start=1)
            }
            for row in stage_rows:
                identifier = row["configuration_id"]
                self.assertEqual(
                    int(row["robustness_rank"]), expected_robustness_rank[identifier]
                )
                if identifier in expected_eligible_rank:
                    self.assertEqual(
                        int(row["eligible_selection_rank"]),
                        expected_eligible_rank[identifier],
                    )
                else:
                    self.assertEqual(row["eligible_selection_rank"], "")
                self.assertEqual(
                    parse_bool(row["selected"]),
                    expected_eligible_rank.get(identifier) == 1,
                )

        selected_base = next(row for row in base if parse_bool(row["selected"]))
        specialist_ids = {
            selected_base["configuration_id"] + "__specialist_none",
            selected_base["configuration_id"]
            + "__specialist_three_way_conditional",
        }
        self.assertEqual({row["configuration_id"] for row in specialist}, specialist_ids)
        no_specialist = next(row for row in specialist if row["specialist"] == "none")
        for key in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "source_group_balanced_accuracy",
            "ceramic_worst_group_recall",
            "robustness_score",
            "minimum_fold_source_group_balanced_accuracy",
            "per_class_recall_json",
            "per_class_group_recall_json",
            "fit_count",
            "maximum_iterations",
        ):
            self.assertEqual(no_specialist[key], selected_base[key])

    def test_grid_and_all_nested_and_full_selection_semantics(self) -> None:
        family_specs = self.contract["base_feature_families"]
        c_values = self.contract["base_grid"]["c_values"]
        exponents = self.contract["base_grid"]["source_group_exponents"]
        expected: list[dict[str, object]] = []
        for family in family_specs:
            dimensions = []
            for block in family["blocks"]:
                cache_name, array_name = block.split(".", 1)
                dimensions.append(
                    str(
                        self.contract["cache_allowlist"][cache_name]["arrays"][array_name][
                            "shape"
                        ][1]
                    )
                )
            for c_value in c_values:
                for exponent in exponents:
                    number_c = str(int(c_value)) if float(c_value).is_integer() else str(c_value).replace(".", "p")
                    number_exponent = str(int(exponent)) if float(exponent).is_integer() else str(exponent).replace(".", "p")
                    expected.append(
                        {
                            "configuration_id": f"{family['id']}__c{number_c}__source{number_exponent}",
                            "feature_family": family["id"],
                            "feature_kind": family["kind"],
                            "embedding_blocks": "|".join(family["blocks"]),
                            "embedding_dimensions": "|".join(dimensions),
                            "c_value": float(c_value),
                            "source_group_exponent": float(exponent),
                            "inference_cost_rank": int(family["inference_cost_rank"]),
                        }
                    )
        self.assertEqual(len(expected), self.contract["base_grid"]["configuration_count"])
        self.assertEqual(len(self.base_grid), len(expected))
        for observed, declared in zip(self.base_grid, expected):
            for key in (
                "configuration_id",
                "feature_family",
                "feature_kind",
                "embedding_blocks",
                "embedding_dimensions",
            ):
                self.assertEqual(observed[key], declared[key])
            self.assert_close(observed["c_value"], declared["c_value"])
            self.assert_close(
                observed["source_group_exponent"], declared["source_group_exponent"]
            )
            self.assertEqual(
                int(observed["inference_cost_rank"]), declared["inference_cost_rank"]
            )

        expected_base_ids = {row["configuration_id"] for row in self.base_grid}
        self.assertEqual(len(self.inner_search), 5 * 14)
        for fold in self.contract["scope"]["outer_folds"]:
            scope = [row for row in self.inner_search if row["outer_fold"] == fold]
            self.assertEqual({row["scope"] for row in scope}, {"outer_inner_oof"})
            self.assertEqual(
                {row["configuration_id"] for row in scope if row["stage"] == "base"},
                expected_base_ids,
            )
            self._audit_selection_scope(scope)
            selected_base = next(
                row for row in scope if row["stage"] == "base" and parse_bool(row["selected"])
            )
            selected_specialist = next(
                row
                for row in scope
                if row["stage"] == "specialist" and parse_bool(row["selected"])
            )
            recipe = self.metrics["nested_selected_recipes_by_outer_fold"][fold]
            self.assertEqual(recipe["base_configuration_id"], selected_base["configuration_id"])
            self.assertEqual(recipe["specialist"], selected_specialist["specialist"])

        self.assertEqual(len(self.full_selection), 14)
        self.assertEqual({row["scope"] for row in self.full_selection}, {"full_development_oof"})
        self.assertTrue(all(row["outer_fold"] == "" for row in self.full_selection))
        self.assertTrue(
            all(int(row["evaluated_examples"]) == len(self.development) for row in self.full_selection)
        )
        self._audit_selection_scope(self.full_selection)
        full_base = next(
            row
            for row in self.full_selection
            if row["stage"] == "base" and parse_bool(row["selected"])
        )
        full_specialist = next(
            row
            for row in self.full_selection
            if row["stage"] == "specialist" and parse_bool(row["selected"])
        )
        recipe = self.metrics["full_development_selected_candidate"]
        self.assertEqual(recipe["base_configuration_id"], full_base["configuration_id"])
        self.assertEqual(recipe["specialist"], full_specialist["specialist"])
        self.assertEqual(recipe["feature_family"], full_base["feature_family"])
        self.assert_close(recipe["c_value"], float(full_base["c_value"]))
        self.assert_close(
            recipe["source_group_exponent"],
            float(full_base["source_group_exponent"]),
        )
        selected_metrics = self.metrics["full_development_selected_oof_metrics"]
        for key in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "source_group_balanced_accuracy",
            "ceramic_worst_group_recall",
            "robustness_score",
        ):
            self.assert_close(full_specialist[key], selected_metrics[key])

    def test_full_recipe_stability_is_recomputed_from_inner_base_searches(self) -> None:
        stability = self.metrics["base_recipe_stability"]
        full_base = self.metrics["full_development_selected_candidate"][
            "base_configuration_id"
        ]
        tie_window = float(self.contract["selection_rule"]["tie_window"])
        passing: list[str] = []
        expected_details: list[dict[str, object]] = []
        for fold in self.contract["scope"]["outer_folds"]:
            base_rows = [
                row
                for row in self.inner_search
                if row["outer_fold"] == fold and row["stage"] == "base"
            ]
            eligible = [row for row in base_rows if parse_bool(row["eligible"])]
            best = max(float(row["robustness_score"]) for row in eligible)
            candidate = next(row for row in base_rows if row["configuration_id"] == full_base)
            candidate_eligible = parse_bool(candidate["eligible"])
            score = float(candidate["robustness_score"])
            within = candidate_eligible and score >= best - tie_window
            if within:
                passing.append(fold)
            expected_details.append(
                {
                    "outer_fold": fold,
                    "full_development_base_configuration_id": full_base,
                    "eligible": candidate_eligible,
                    "robustness_score": score,
                    "best_eligible_robustness_score": best,
                    "within_tie_window": within,
                }
            )
        self.assertEqual(stability["required_outer_searches_within_tie_window"], 4)
        self.assertEqual(stability["observed_outer_searches_within_tie_window"], len(passing))
        self.assertEqual(stability["passing_outer_folds"], passing)
        self.assertEqual(stability["passed"], len(passing) >= 4)
        self.assertEqual(len(stability["details"]), len(expected_details))
        for observed, expected in zip(stability["details"], expected_details):
            for key in (
                "outer_fold",
                "full_development_base_configuration_id",
                "eligible",
                "within_tie_window",
            ):
                self.assertEqual(observed[key], expected[key])
            self.assert_close(observed["robustness_score"], expected["robustness_score"])
            self.assert_close(
                observed["best_eligible_robustness_score"],
                expected["best_eligible_robustness_score"],
            )

    def test_every_promotion_gate_and_supporting_formula_is_recomputed(self) -> None:
        gates = self.contract["promotion_gates"]
        definitions = self.contract["diagnostic_slice_definitions"]
        aggregate, per_class = classification_metrics(self.nested, self.classes)
        opishnyan, bubnivka, _ = self.ceramics
        hard_names = definitions["hard_three_opishnyan_cohorts"]

        def accuracy(rows: list[dict[str, str]]) -> float:
            self.assertTrue(rows)
            return sum(row["true_class"] == row["predicted_class"] for row in rows) / len(rows)

        hard_rows = [
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
        hard_accuracies = {
            cohort: accuracy(
                [
                    row
                    for row in self.nested
                    if row["true_class"] == opishnyan
                    and row["source_atomic_cohort_ids"] == cohort
                ]
            )
            for cohort in hard_names
        }
        cohort_support = Counter(row["source_atomic_cohort_ids"] for row in self.nested)
        large_names = sorted(
            cohort
            for cohort, count in cohort_support.items()
            if cohort and count >= int(gates["large_named_source_minimum_size"])
        )
        large_accuracies = {
            cohort: accuracy(
                [row for row in self.nested if row["source_atomic_cohort_ids"] == cohort]
            )
            for cohort in large_names
        }
        fold_accuracies = [
            accuracy([row for row in self.nested if row["cv_fold"] == fold])
            for fold in self.contract["scope"]["outer_folds"]
        ]
        fold_sample_std = statistics.stdev(fold_accuracies)
        ceramic_macro_f1 = statistics.fmean(
            per_class[label]["f1"] for label in self.ceramics
        )
        opish_to_bub = sum(
            row["true_class"] == opishnyan and row["predicted_class"] == bubnivka
            for row in self.nested
        )
        actuals: list[tuple[str, str, object, float]] = [
            ("overall_accuracy", ">=", aggregate["accuracy"], float(gates["overall_accuracy_min"])),
            ("overall_macro_f1", ">=", aggregate["macro_f1"], float(gates["overall_macro_f1_min"])),
            ("opishnyan_recall", ">=", per_class[opishnyan]["recall"], float(gates["opishnyan_recall_min"])),
            ("bubnivka_precision", ">=", per_class[bubnivka]["precision"], float(gates["bubnivka_precision_min"])),
            ("ceramic_macro_f1", ">=", ceramic_macro_f1, float(gates["ceramic_macro_f1_min"])),
            ("opishnyan_to_bubnivka_errors", "<=", float(opish_to_bub), float(gates["opishnyan_to_bubnivka_errors_max"])),
            ("challenging_opishnyan_accuracy", ">=", accuracy(challenging), float(gates["challenging_opishnyan_accuracy_min"])),
            ("hard_three_opishnyan_pooled_accuracy", ">=", accuracy(hard_rows), float(gates["hard_three_opishnyan_pooled_accuracy_min"])),
            ("hard_three_opishnyan_cohort_macro_accuracy", ">=", statistics.fmean(hard_accuracies.values()), float(gates["hard_three_opishnyan_cohort_macro_accuracy_min"])),
            ("large_named_source_macro_accuracy", ">=", statistics.fmean(large_accuracies.values()), float(gates["large_named_source_macro_accuracy_min"])),
            ("worst_fold_accuracy", ">=", min(fold_accuracies), float(gates["worst_fold_accuracy_min"])),
            ("fold_accuracy_std", "<=", fold_sample_std, float(gates["fold_accuracy_std_max"])),
        ]
        for label in sorted(gates["other_class_recall_floors"]):
            actuals.append(
                (
                    f"recall_floor:{label}",
                    ">=",
                    per_class[label]["recall"],
                    float(gates["other_class_recall_floors"][label]),
                )
            )
        actuals.append(
            (
                "base_recipe_stability",
                "required",
                self.metrics["base_recipe_stability"]["passed"],
                True,
            )
        )

        promotion = self.metrics["promotion_gates"]
        self.assertEqual(promotion["all_required"], gates["all_required"])
        self.assertEqual(len(promotion["results"]), len(actuals))
        failed: list[str] = []
        for observed, expected in zip(promotion["results"], actuals):
            name, comparison, actual, threshold = expected
            passed = actual is not None and (
                actual >= threshold if comparison in (">=", "required") else actual <= threshold
            )
            self.assertEqual(observed["gate"], name)
            self.assertEqual(observed["comparison"], comparison)
            self.assertEqual(observed["threshold"], threshold)
            if isinstance(actual, bool):
                self.assertEqual(observed["actual"], actual)
            else:
                self.assert_close(observed["actual"], actual)
            self.assertEqual(observed["passed"], passed)
            if not passed:
                failed.append(name)
        self.assertEqual(promotion["gate_count"], len(actuals))
        self.assertEqual(promotion["passed_count"], len(actuals) - len(failed))
        self.assertEqual(promotion["failed_gates"], failed)
        self.assertEqual(promotion["all_passed"], not failed)
        decision = "promote" if not failed else "reject"
        self.assertEqual(promotion["promotion_decision"], decision)
        self.assertEqual(self.metrics["promotion_decision"], decision)
        self.assertEqual(decision, "reject")

        supporting = promotion["supporting_values"]
        self.assertEqual(supporting["large_named_source_cohorts"], large_names)
        for cohort, value in hard_accuracies.items():
            self.assert_close(supporting["hard_three_cohort_accuracies"][cohort], value)
        for cohort, value in large_accuracies.items():
            self.assert_close(
                supporting["large_named_source_cohort_accuracies"][cohort], value
            )
        recorded_std = next(
            row["actual"] for row in promotion["results"] if row["gate"] == "fold_accuracy_std"
        )
        self.assert_close(recorded_std, statistics.stdev(fold_accuracies))
        self.assertFalse(
            math.isclose(
                float(recorded_std),
                statistics.pstdev(fold_accuracies),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "The frozen gate requires sample standard deviation (ddof=1)",
        )

    def test_contract_inputs_code_cache_provenance_and_runtime_versions(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(self.metrics["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)
        self.assertEqual(
            resolve_recorded_path(self.metrics["experiment_contract_path"]),
            CONTRACT_PATH.resolve(),
        )
        self.assertEqual(self.contract["status"], "frozen_before_fit")
        self.assertEqual(self.contract["scope"]["sealed_test_access"], "forbidden")
        self.assertEqual(self.metrics["experiment_version"], self.contract["experiment_version"])

        inputs = self.contract["input_contract"]
        for path_key, hash_key in (
            ("development_csv", "development_csv_sha256"),
            ("split_audit", "split_audit_sha256"),
            ("phase_4_metrics", "phase_4_metrics_sha256"),
        ):
            assert_recorded_file(self, inputs[path_key], inputs[hash_key])
        split_audit = json.loads((SPLIT_ROOT / "split_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(
            split_audit["generated_csv_sha256"]["development.csv"],
            inputs["development_csv_sha256"],
        )
        self.assertEqual(self.metrics["split_version"], inputs["split_version"])
        self.assertEqual(self.metrics["split_seed"], inputs["split_seed"])
        self.assertEqual(
            self.metrics["split_assignment_fingerprint_sha256"],
            inputs["split_assignment_fingerprint_sha256"],
        )

        code = self.metrics["code_provenance"]
        provenance_order = (
            "step_02/scripts/run_phase5_source_robustness.py",
            "step_02/src/ornament_classifier/robustness.py",
            "step_02/src/ornament_classifier/embeddings.py",
            "step_02/src/ornament_classifier/contracts.py",
            "step_02/src/ornament_classifier/paths.py",
            "step_02/requirements-phase5.txt",
        )
        self.assertEqual(set(code["files_sha256"]), set(provenance_order))
        combined = hashlib.sha256()
        for relative in provenance_order:
            expected_hash = code["files_sha256"][relative]
            assert_recorded_file(self, relative, expected_hash)
            combined.update(f"{relative}\x1f{expected_hash}\n".encode("utf-8"))
        self.assertEqual(combined.hexdigest(), code["combined_sha256"])
        runner_relative = "step_02/scripts/run_phase5_source_robustness.py"
        self.assertEqual(self.metrics["script_sha256"], code["files_sha256"][runner_relative])

        expected_embedding_keys = {
            block
            for family in self.contract["base_feature_families"]
            for block in family["blocks"]
        }
        provenance = self.metrics["embedding_provenance"]
        self.assertEqual(set(provenance), expected_embedding_keys)
        verified_cache_hashes: dict[str, str] = {}
        for reference, recorded in provenance.items():
            cache_name, array_name = reference.split(".", 1)
            declared_cache = self.contract["cache_allowlist"][cache_name]
            declared_array = declared_cache["arrays"][array_name]
            if cache_name not in verified_cache_hashes:
                cache_path = resolve_recorded_path(declared_cache["path"])
                verified_cache_hashes[cache_name] = sha256_file(cache_path)
            self.assertEqual(verified_cache_hashes[cache_name], declared_cache["file_sha256"])
            self.assertEqual(recorded["cache_name"], cache_name)
            self.assertEqual(recorded["array_name"], array_name)
            self.assertEqual(recorded["cache_sha256"], declared_cache["file_sha256"])
            self.assertEqual(recorded["array_sha256"], declared_array["sha256"])
            self.assertEqual(recorded["embedding_fingerprint_sha256"], declared_cache["fingerprint"])
            self.assertEqual(recorded["data_fingerprint_sha256"], inputs["data_fingerprint_sha256"])
            self.assertEqual(recorded["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)
            self.assertEqual(recorded["experiment_version"], self.contract["experiment_version"])
            self.assertEqual(recorded["split_assignment_fingerprint_sha256"], inputs["split_assignment_fingerprint_sha256"])
            self.assertEqual(
                resolve_recorded_path(recorded["experiment_contract_path"]),
                CONTRACT_PATH.resolve(),
            )

        requirements = {}
        for line in (STEP_ROOT / "requirements-phase5.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                package, version = line.strip().split("==", 1)
                requirements[package] = version
        self.assertEqual(
            requirements,
            {
                "numpy": "2.0.2",
                "matplotlib": "3.9.4",
                "scikit-learn": "1.6.1",
                "threadpoolctl": "3.6.0",
            },
        )
        software = self.metrics["software"]
        self.assertEqual(software["python"], "3.9.6")
        runtime_keys = {
            "numpy": "numpy",
            "matplotlib": "matplotlib",
            "scikit-learn": "scikit_learn",
            "threadpoolctl": "threadpoolctl",
        }
        self.assertEqual(set(software), {"python", *runtime_keys.values()})
        for distribution, output_key in runtime_keys.items():
            self.assertEqual(software[output_key], requirements[distribution])

    def test_exact_phase4_reference_validation_is_independently_audited(self) -> None:
        validation = self.metrics["exact_phase4_reference_validation"]
        self.assertTrue(validation["validated"])
        self.assertFalse(validation["probability_identity_required"])
        self.assertEqual(validation["absolute_tolerance"], 1e-12)
        phase4_predictions = read_csv(
            STEP_ROOT / "outputs" / "phase_4_pretrained" / "oof_predictions.csv"
        )
        self.assertEqual(
            [row["image_id"] for row in phase4_predictions],
            [row["image_id"] for row in self.development],
        )
        digest = hashlib.sha256()
        for row in phase4_predictions:
            digest.update(
                f"{row['image_id']}\x1f{row['predicted_class']}\n".encode("utf-8")
            )
        observed_fingerprint = digest.hexdigest()
        self.assertEqual(observed_fingerprint, EXPECTED_PHASE4_DECISION_SHA256)
        self.assertEqual(
            validation["expected_prediction_fingerprint_sha256"], observed_fingerprint
        )
        self.assertEqual(
            validation["observed_prediction_fingerprint_sha256"], observed_fingerprint
        )

        phase4_metrics = json.loads(
            (STEP_ROOT / "outputs" / "phase_4_pretrained" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        reference = next(
            row
            for row in self.full_selection
            if row["stage"] == "base" and parse_bool(row["is_exact_reference"])
        )
        phase4_aggregate = phase4_metrics["aggregate_oof_metrics"]
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assert_close(reference[key], phase4_aggregate[key])
        phase4_per_class = {
            row["class_name"]: row for row in phase4_metrics["per_class_metrics"]
        }
        reference_recalls = json.loads(reference["per_class_recall_json"])
        for label in self.classes:
            self.assert_close(reference_recalls[label], phase4_per_class[label]["recall"])


if __name__ == "__main__":
    unittest.main()

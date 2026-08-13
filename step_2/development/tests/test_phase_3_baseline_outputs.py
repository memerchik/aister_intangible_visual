from __future__ import annotations

import csv
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SPLIT_ROOT = STEP_ROOT / "splits"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_3_classical_baseline"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PhaseThreeBaselineOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.development = read_csv(SPLIT_ROOT / "development.csv")
        cls.test = read_csv(SPLIT_ROOT / "test.csv")
        cls.predictions = read_csv(OUTPUT_ROOT / "oof_predictions.csv")
        cls.fold_metrics = read_csv(OUTPUT_ROOT / "cv_fold_metrics.csv")
        cls.summary = read_csv(OUTPUT_ROOT / "model_selection_summary.csv")
        cls.fixed_metrics = read_csv(OUTPUT_ROOT / "fixed_validation_metrics.csv")
        cls.confusion = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        with (OUTPUT_ROOT / "metrics.json").open(encoding="utf-8") as handle:
            cls.metrics = json.load(handle)
        with (SPLIT_ROOT / "split_audit.json").open(encoding="utf-8") as handle:
            cls.split_audit = json.load(handle)

    def test_evaluation_is_development_only(self) -> None:
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertEqual(self.metrics["evaluation_scope"], "development_only_out_of_fold")
        development_ids = {row["image_id"] for row in self.development}
        prediction_ids = {row["image_id"] for row in self.predictions}
        test_ids = {row["image_id"] for row in self.test}
        self.assertEqual(len(self.predictions), len(self.development))
        self.assertEqual(prediction_ids, development_ids)
        self.assertFalse(prediction_ids & test_ids)
        self.assertTrue(all(row["production_split"] in {"train", "validation"} for row in self.predictions))
        self.assertFalse((OUTPUT_ROOT / "test_predictions.csv").exists())

    def test_candidate_and_fold_coverage(self) -> None:
        self.assertEqual(len(self.summary), 5)
        self.assertEqual(len(self.fold_metrics), 25)
        self.assertEqual(len(self.fixed_metrics), 5)
        by_model: dict[str, set[str]] = defaultdict(set)
        for row in self.fold_metrics:
            by_model[row["model_name"]].add(row["cv_fold"])
        self.assertEqual(set(by_model), {row["model_name"] for row in self.summary})
        self.assertTrue(all(folds == {"0", "1", "2", "3", "4"} for folds in by_model.values()))

    def test_selected_model_matches_predeclared_ranking(self) -> None:
        expected = sorted(
            self.summary,
            key=lambda row: (
                -float(row["mean_accuracy"]),
                -float(row["mean_macro_f1"]),
                row["model_name"],
            ),
        )
        self.assertEqual([int(row["selection_rank"]) for row in expected], [1, 2, 3, 4, 5])
        self.assertEqual(self.metrics["selected_model"], expected[0]["model_name"])
        self.assertEqual(self.metrics["selected_feature_set"], expected[0]["feature_name"])

    def test_probabilities_and_aggregate_metrics_are_consistent(self) -> None:
        probability_fields = [field for field in self.predictions[0] if field.startswith("probability_")]
        self.assertEqual(len(probability_fields), 5)
        correct = 0
        top_three_correct = 0
        for row in self.predictions:
            probabilities = {field.removeprefix("probability_"): float(row[field]) for field in probability_fields}
            self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()))
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=8)
            self.assertEqual(row["predicted_class"], max(probabilities, key=probabilities.get))
            is_correct = row["true_class"] == row["predicted_class"]
            self.assertEqual(row["is_correct"], str(is_correct))
            correct += is_correct
            top_three_correct += row["true_class"] in {
                row["top_1_class"],
                row["top_2_class"],
                row["top_3_class"],
            }
        aggregate = self.metrics["aggregate_oof_metrics"]
        self.assertAlmostEqual(aggregate["accuracy"], correct / len(self.predictions), places=12)
        self.assertAlmostEqual(
            aggregate["top_3_accuracy"], top_three_correct / len(self.predictions), places=12
        )

    def test_confusion_matrix_matches_predictions(self) -> None:
        classes = self.metrics["class_names"]
        observed = defaultdict(int)
        for row in self.predictions:
            observed[(row["true_class"], row["predicted_class"])] += 1
        total = 0
        recalls = []
        f1_values = []
        for matrix_row in self.confusion:
            true_class = matrix_row["true_class"]
            support = sum(int(matrix_row[predicted_class]) for predicted_class in classes)
            true_positive = int(matrix_row[true_class])
            false_positive = sum(
                int(other[true_class])
                for other in self.confusion
                if other["true_class"] != true_class
            )
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
            recall = true_positive / support if support else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            recalls.append(recall)
            f1_values.append(f1)
            for predicted_class in classes:
                count = int(matrix_row[predicted_class])
                self.assertEqual(count, observed[(true_class, predicted_class)])
                total += count
        self.assertEqual(total, len(self.development))
        aggregate = self.metrics["aggregate_oof_metrics"]
        self.assertAlmostEqual(aggregate["balanced_accuracy"], sum(recalls) / 5, places=12)
        self.assertAlmostEqual(aggregate["macro_f1"], sum(f1_values) / 5, places=12)

    def test_group_members_are_in_one_held_out_fold(self) -> None:
        object_folds: dict[str, set[str]] = defaultdict(set)
        source_folds: dict[str, set[str]] = defaultdict(set)
        split_group_folds: dict[str, set[str]] = defaultdict(set)
        for row in self.predictions:
            object_folds[row["confirmed_object_group_id"]].add(row["cv_fold"])
            if row["confirmed_source_group_id"]:
                source_folds[row["confirmed_source_group_id"]].add(row["cv_fold"])
            split_group_folds[row["split_group_id"]].add(row["cv_fold"])
        self.assertTrue(all(len(folds) == 1 for folds in object_folds.values()))
        self.assertTrue(all(len(folds) == 1 for folds in source_folds.values()))
        self.assertTrue(all(len(folds) == 1 for folds in split_group_folds.values()))

    def test_split_fingerprint_is_pinned(self) -> None:
        self.assertEqual(
            self.metrics["split_assignment_fingerprint_sha256"],
            self.split_audit["assignment_fingerprint_sha256"],
        )


if __name__ == "__main__":
    unittest.main()

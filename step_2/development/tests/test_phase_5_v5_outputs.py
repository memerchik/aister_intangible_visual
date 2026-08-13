from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from relocation_support import assert_recorded_file


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness_v5"
CONTRACT_PATH = (
    STEP_ROOT / "phases" / "phase_05_source_robustness" / "experiment_contract_v5.json"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.contracts import load_development_contract  # noqa: E402
from ornament_classifier.pairwise import boundary_metrics  # noqa: E402
from ornament_classifier.robustness import ordinary_metrics, source_group_metrics  # noqa: E402


EXPECTED_CONTRACT_SHA256 = "c9473a7e20c57049fc4bc648f1133c5269434a955f955f390f10594adb80a70d"
EXPECTED_OUTPUT_HASHES = {
    "base_grid.csv": "1901de53ecb5ae08dde4d5c7525c38948acb51a00964b2e697fe942041bafbcd",
    "confusion_matrix.csv": "cbe59be0e3cbc46a2c27edef8eebf375be61f66300f1ef62bc22827b68c9d7d8",
    "confusion_matrix.png": "cb636f35df026a5086279c5b678d55464dc7d6810ecd896e48764c5a2f25a7bb",
    "confusion_pairs.csv": "b0ad2c69a65cd33cfaf83924be2d7b118e9dffd8781c42c79c96be701e113489",
    "consensus_recipe.csv": "04fa85d7daae055817c8baf6b989dc07581f067e7dd5070aa3c02a5b85707e28",
    "diagnostic_slices.csv": "bc20fe9c3cdb01921d3b3c3f46bef7bd78188cba80463b519133e25edd232587",
    "full_development_selection.csv": "c37d8201c767afc3c82c9e0b44ae0c9f374fd532a9f888769d832c2413ef52a2",
    "inner_search.csv": "6969bab205ed7135e527f62f10d87b59e8dccc3bf1bcd79d27b5325287db7b5e",
    "metrics.json": "8533b68e1903832642fae203d250c8c874b056929e4ee2cd56b1feaa36480732",
    "motif_proposals.csv": "183c63e087fffcea8dc847a64e697b664be944bc61dee2e87fdd75fef666ab0a",
    "nested_oof_predictions.csv": "dda0cc183c64c86ae5d81228093e1c32cf4efa46cd9c3e2f919dddb7251c363e",
    "outer_fold_metrics.csv": "3e40864db34d3c1c81d742ca355de990e692df38df1ca5676f7acc0b3720bce1",
    "per_class_metrics.csv": "78a3aef64a50a4bfe348196a3f6dc359d81c07e685ae8c0460184f0e404c41f1",
    "selected_oof_predictions.csv": "5951df4309098efb40efb87c99ea61f50572e4fe3d41c2152ddf75bf9419d750",
}


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
    raise AssertionError(f"Invalid Boolean serialization: {value!r}")


class PhaseFiveV5OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUTPUT_ROOT / "metrics.json").is_file():
            raise unittest.SkipTest("canonical Phase 5 v5 outputs have not been generated")
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.metrics = json.loads((OUTPUT_ROOT / "metrics.json").read_text(encoding="utf-8"))
        cls.development = load_development_contract()
        cls.records = cls.development.records
        cls.classes = tuple(cls.metrics["class_names"])
        cls.predictions = read_csv(OUTPUT_ROOT / "nested_oof_predictions.csv")
        cls.probabilities = np.asarray(
            [
                [float(row[f"probability_{label}"]) for label in cls.classes]
                for row in cls.predictions
            ],
            dtype=np.float64,
        )
        cls.labels = np.asarray([record.ornament_label for record in cls.records], dtype=object)
        cls.groups = np.asarray(
            [record.source_atomic_split_group_id for record in cls.records], dtype=object
        )

    def test_required_artifacts_contract_hashes_and_png_are_exact(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        required = set(self.contract["output_contract"]["required"])
        observed = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
        self.assertEqual(required, set(EXPECTED_OUTPUT_HASHES))
        self.assertEqual(observed, required)
        for name, expected in EXPECTED_OUTPUT_HASHES.items():
            self.assertEqual(sha256_file(OUTPUT_ROOT / name), expected, name)
        self.assertTrue((OUTPUT_ROOT / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG"))

    def test_predictions_are_complete_development_only_and_source_atomic(self) -> None:
        self.assertEqual(len(self.predictions), 1693)
        self.assertEqual(
            [row["image_id"] for row in self.predictions],
            [record.image_id for record in self.records],
        )
        sealed_ids = {row["image_id"] for row in read_csv(STEP_ROOT / "splits" / "test.csv")}
        self.assertFalse({row["image_id"] for row in self.predictions} & sealed_ids)
        group_folds: dict[str, set[str]] = {}
        for row in self.predictions:
            group_folds.setdefault(row["source_atomic_split_group_id"], set()).add(row["cv_fold"])
            self.assertTrue(parse_bool(row["motif_applied"]))
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))
        np.testing.assert_allclose(self.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        predicted = np.asarray(self.classes, dtype=object)[np.argmax(self.probabilities, axis=1)]
        self.assertEqual(predicted.tolist(), [row["predicted_class"] for row in self.predictions])

    def test_nested_metrics_and_boundary_are_independently_reconstructed(self) -> None:
        ordinary = ordinary_metrics(self.labels, self.probabilities, self.classes)
        predicted = np.asarray(self.classes, dtype=object)[np.argmax(self.probabilities, axis=1)]
        source = source_group_metrics(
            self.labels,
            predicted,
            self.groups,
            self.classes,
            tuple(self.metrics["ceramic_classes"]),
        )
        boundary = boundary_metrics(
            self.labels,
            self.probabilities,
            self.classes,
            pair_classes=tuple(self.metrics["pair_classes"]),
            ceramic_classes=tuple(self.metrics["ceramic_classes"]),
        )
        recorded = self.metrics["nested_aggregate_oof_metrics"]
        for name in ("accuracy", "macro_f1", "balanced_accuracy"):
            self.assertAlmostEqual(getattr(ordinary, name), recorded[name], places=14)
        for name in (
            "source_group_balanced_accuracy", "ceramic_worst_group_recall", "robustness_score"
        ):
            self.assertAlmostEqual(getattr(source, name), recorded[name], places=14)
        for name in (
            "opishnyan_recall", "bubnivka_recall", "bubnivka_precision",
            "ceramic_macro_f1", "opishnyan_to_bubnivka_errors", "bubnivka_to_opishnyan_errors",
        ):
            self.assertAlmostEqual(getattr(boundary, name), recorded[name], places=14)
        self.assertEqual(int((predicted == self.labels).sum()), 1600)
        self.assertEqual(int((predicted != self.labels).sum()), 93)

    def test_confusion_and_outer_metrics_reconstruct_predictions(self) -> None:
        matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        positions = {label: index for index, label in enumerate(self.classes)}
        for row in self.predictions:
            matrix[positions[row["true_class"]], positions[row["predicted_class"]]] += 1
        observed = np.asarray(
            [
                [int(row[label]) for label in self.classes]
                for row in read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(observed, matrix)
        outer = read_csv(OUTPUT_ROOT / "outer_fold_metrics.csv")
        self.assertEqual([row["outer_fold"] for row in outer], ["0", "1", "2", "3", "4"])
        self.assertEqual(sum(int(row["validation_examples"]) for row in outer), 1693)
        for row in outer:
            fold_rows = [value for value in self.predictions if value["cv_fold"] == row["outer_fold"]]
            accuracy = sum(parse_bool(value["is_correct"]) for value in fold_rows) / len(fold_rows)
            self.assertAlmostEqual(accuracy, float(row["accuracy"]), places=14)

    def test_fixed_recipe_and_sequential_base_selection_are_exact(self) -> None:
        recipe = read_csv(OUTPUT_ROOT / "consensus_recipe.csv")
        self.assertEqual(len(recipe), 1)
        self.assertEqual(recipe[0]["candidate_count"], "1")
        self.assertEqual(recipe[0]["head_weights"], "0.25|0.25|0.25|0.25")
        inner = read_csv(OUTPUT_ROOT / "inner_search.csv")
        for outer_fold in ("0", "1", "2", "3", "4"):
            rows = [row for row in inner if row["outer_fold"] == outer_fold]
            base = [row for row in rows if row["stage"] == "base"]
            fixed = [row for row in rows if row["stage"] == "motif"]
            self.assertEqual(len(base), 3)
            self.assertEqual(len(fixed), 1)
            self.assertEqual(sum(parse_bool(row["selected"]) for row in base), 1)
            self.assertTrue(parse_bool(fixed[0]["selected"]))
            self.assertTrue(parse_bool(fixed[0]["eligible"]))
        full = read_csv(OUTPUT_ROOT / "full_development_selection.csv")
        self.assertEqual(sum(row["stage"] == "base" for row in full), 3)
        self.assertEqual(sum(row["stage"] == "motif" for row in full), 1)

    def test_stability_and_all_21_gates_are_recomputed(self) -> None:
        base = self.metrics["base_recipe_stability"]
        fixed = self.metrics["fixed_consensus_stability"]
        self.assertEqual(base["observed_outer_searches"], 4)
        self.assertTrue(base["passed"])
        self.assertEqual(fixed["observed_outer_searches"], 5)
        self.assertTrue(fixed["passed"])
        self.assertTrue(all(row["eligible"] for row in fixed["details"]))
        gates = self.metrics["promotion_gates"]
        self.assertEqual(gates["gate_count"], 21)
        self.assertEqual(gates["passed_count"], 18)
        self.assertEqual(
            gates["failed_gates"],
            ["bubnivka_precision", "opishnyan_to_bubnivka_errors", "worst_fold_accuracy"],
        )
        for result in gates["results"]:
            if result["comparison"] == ">=":
                expected = result["actual"] >= result["threshold"]
            elif result["comparison"] == "<=":
                expected = result["actual"] <= result["threshold"]
            else:
                expected = result["actual"] is result["threshold"]
            self.assertEqual(result["passed"], expected, result["gate"])
        self.assertFalse(gates["all_passed"])
        self.assertEqual(self.metrics["promotion_decision"], "reject")
        self.assertFalse(self.metrics["phase_6_transition_allowed"])

    def test_provenance_and_previous_version_claims_are_pinned(self) -> None:
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertEqual(self.metrics["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)
        self.assertEqual(self.metrics["fixed_consensus_candidate_count"], 1)
        self.assertEqual(self.metrics["fixed_consensus_head_weights"], [0.25] * 4)
        self.assertEqual(set(self.metrics["previous_phase5_comparison"]), {"v1", "v2", "v3", "v4", "v5"})
        self.assertTrue(self.metrics["v1_reproduction_validation"]["validated"])
        cache = self.metrics["motif_embedding_provenance"]
        self.assertEqual(
            cache["cache_sha256"],
            "97ee78769b6b4349b58e9b2cb27c9f4b58748ecf7cd9fbcfcc554c2fa122db40",
        )
        for relative, expected in self.metrics["code_provenance"]["files_sha256"].items():
            assert_recorded_file(self, relative, expected)


if __name__ == "__main__":
    unittest.main()

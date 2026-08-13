from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from relocation_support import assert_recorded_file


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_5_source_robustness_v4"
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract_v4.json"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.contracts import load_development_contract  # noqa: E402
from ornament_classifier.embeddings_v4 import load_motif_embeddings_v4  # noqa: E402
from ornament_classifier.pairwise import boundary_metrics  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    ordinary_metrics,
    source_group_metrics,
)


EXPECTED_CONTRACT_SHA256 = (
    "7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da"
)
EXPECTED_OUTPUT_HASHES = {
    "base_grid.csv": "1901de53ecb5ae08dde4d5c7525c38948acb51a00964b2e697fe942041bafbcd",
    "confusion_matrix.csv": "801d193099e66c709e738138b179a8cae26ea00646c807f421c848cbcd3c6329",
    "confusion_matrix.png": "10f67938011c45fd03a47e1ec1e1c6a579d2e322a30dfa2319d462efd154523e",
    "confusion_pairs.csv": "70f95a4d8f6a34f546d25279d6f2b829451dd136cdf41bf61e62b487bc3b4cfe",
    "diagnostic_slices.csv": "b019ebd0dfebd2cc4be9a8dd78fe769b21b8dd313f613badc2902daabb2c2c7d",
    "full_development_selection.csv": "91f072d50709df1758e9fab49eade437b5066875269c3f551963e4db3a40bcf8",
    "inner_search.csv": "6c1ff1a74a1ef0c1074d2e98517d6d077fd2e28472f801f57184632afbdd0955",
    "metrics.json": "c53116230d3e558ab31538dee25c1c35131fb0e6176002bcc9214f70fbc727f3",
    "motif_grid.csv": "bfc1e0fea5d1e54b78c995ec61a45ede3027bd0b0ed521de2fb956e9b93dd0d5",
    "motif_proposals.csv": "183c63e087fffcea8dc847a64e697b664be944bc61dee2e87fdd75fef666ab0a",
    "nested_oof_predictions.csv": "2c4517ab3ce715535aa5fddaec856fcdb4dde980d4f53cef8bad061e32fc813f",
    "outer_fold_metrics.csv": "08d63478047b92ded4f68649bb668d88a78d16288c2067c73474ae50d88e3aa7",
    "per_class_metrics.csv": "51241a455490faf950985b231332bc644869e3fa3af5bdbd0dfdf9e95d3a23cb",
    "selected_oof_predictions.csv": "04e3e960eb39486b0381fbfda8e9e9a3d3ac3c955cf9e04fe953ca088a6de599",
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


class PhaseFiveV4OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUTPUT_ROOT / "metrics.json").is_file():
            raise unittest.SkipTest("canonical Phase 5 v4 outputs have not been generated")
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
        cls.labels = np.asarray(
            [record.ornament_label for record in cls.records], dtype=object
        )
        cls.groups = np.asarray(
            [record.source_atomic_split_group_id for record in cls.records],
            dtype=object,
        )

    def test_required_artifacts_contract_hashes_and_png_are_exact(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        required = set(self.contract["output_contract"]["required"])
        observed = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
        self.assertEqual(required, set(EXPECTED_OUTPUT_HASHES))
        self.assertEqual(observed, required)
        for name, expected in EXPECTED_OUTPUT_HASHES.items():
            self.assertEqual(sha256_file(OUTPUT_ROOT / name), expected, name)
        png = (OUTPUT_ROOT / "confusion_matrix.png").read_bytes()
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_predictions_are_complete_development_only_and_source_atomic(self) -> None:
        self.assertEqual(len(self.predictions), 1693)
        self.assertEqual(
            [row["image_id"] for row in self.predictions],
            [record.image_id for record in self.records],
        )
        self.assertEqual(len({row["image_id"] for row in self.predictions}), 1693)
        self.assertEqual(
            {row["production_split"] for row in self.predictions},
            {"train", "validation"},
        )
        sealed_ids = {
            row["image_id"] for row in read_csv(STEP_ROOT / "splits" / "test.csv")
        }
        self.assertFalse({row["image_id"] for row in self.predictions} & sealed_ids)
        group_folds: dict[str, set[str]] = {}
        for row in self.predictions:
            group_folds.setdefault(row["source_atomic_split_group_id"], set()).add(
                row["cv_fold"]
            )
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))
        np.testing.assert_allclose(
            self.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        )
        predicted = np.asarray(self.classes, dtype=object)[
            np.argmax(self.probabilities, axis=1)
        ]
        self.assertEqual(predicted.tolist(), [row["predicted_class"] for row in self.predictions])
        self.assertTrue(
            all(
                parse_bool(row["is_correct"])
                == (row["true_class"] == row["predicted_class"])
                for row in self.predictions
            )
        )

    def test_nested_metrics_and_boundary_are_independently_reconstructed(self) -> None:
        ordinary = ordinary_metrics(self.labels, self.probabilities, self.classes)
        predicted = np.asarray(self.classes, dtype=object)[
            np.argmax(self.probabilities, axis=1)
        ]
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
            "source_group_balanced_accuracy",
            "ceramic_worst_group_recall",
            "robustness_score",
        ):
            self.assertAlmostEqual(getattr(source, name), recorded[name], places=14)
        for name in (
            "opishnyan_recall",
            "bubnivka_recall",
            "bubnivka_precision",
            "ceramic_macro_f1",
            "opishnyan_to_bubnivka_errors",
            "bubnivka_to_opishnyan_errors",
        ):
            self.assertAlmostEqual(getattr(boundary, name), recorded[name], places=14)
        self.assertEqual(int((predicted == self.labels).sum()), 1603)
        self.assertEqual(int((predicted != self.labels).sum()), 90)

    def test_confusion_outer_metrics_and_slices_reconstruct_predictions(self) -> None:
        expected_matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        for row in self.predictions:
            expected_matrix[
                class_to_index[row["true_class"]],
                class_to_index[row["predicted_class"]],
            ] += 1
        matrix_rows = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        observed_matrix = np.asarray(
            [[int(row[label]) for label in self.classes] for row in matrix_rows],
            dtype=int,
        )
        np.testing.assert_array_equal(observed_matrix, expected_matrix)
        outer = read_csv(OUTPUT_ROOT / "outer_fold_metrics.csv")
        self.assertEqual([row["outer_fold"] for row in outer], ["0", "1", "2", "3", "4"])
        self.assertEqual(sum(int(row["validation_examples"]) for row in outer), 1693)
        for row in outer:
            selected = [item for item in self.predictions if item["cv_fold"] == row["outer_fold"]]
            accuracy = sum(parse_bool(item["is_correct"]) for item in selected) / len(selected)
            self.assertAlmostEqual(accuracy, float(row["accuracy"]), places=14)
        diagnostics = read_csv(OUTPUT_ROOT / "diagnostic_slices.csv")
        supports = Counter((row["dimension"], row["value"]) for row in diagnostics)
        self.assertTrue(diagnostics)
        self.assertTrue(all(value == 1 for value in supports.values()))

    def test_grid_sequential_selection_eligibility_and_ranking_are_exact(self) -> None:
        self.assertEqual(len(read_csv(OUTPUT_ROOT / "base_grid.csv")), 3)
        self.assertEqual(len(read_csv(OUTPUT_ROOT / "motif_grid.csv")), 9)
        inner = read_csv(OUTPUT_ROOT / "inner_search.csv")
        for outer_fold in ("0", "1", "2", "3", "4"):
            rows = [row for row in inner if row["outer_fold"] == outer_fold]
            base = [row for row in rows if row["stage"] == "base"]
            motif = [row for row in rows if row["stage"] == "motif"]
            self.assertEqual(len(base), 3)
            self.assertEqual(len(motif), 9)
            self.assertEqual(sum(parse_bool(row["selected"]) for row in base), 1)
            self.assertEqual(sum(parse_bool(row["selected"]) for row in motif), 1)
            selected_base = next(row for row in base if parse_bool(row["selected"]))
            selected_motif = next(row for row in motif if parse_bool(row["selected"]))
            self.assertTrue(parse_bool(selected_base["eligible"]))
            self.assertTrue(parse_bool(selected_motif["eligible"]))
            self.assertEqual(selected_base["eligible_selection_rank"], "1")
            self.assertEqual(selected_motif["eligible_selection_rank"], "1")
            self.assertEqual(
                {row["base_configuration_id"] for row in motif},
                {selected_base["configuration_id"]},
            )
        full = read_csv(OUTPUT_ROOT / "full_development_selection.csv")
        self.assertEqual(sum(row["stage"] == "base" for row in full), 3)
        self.assertEqual(sum(row["stage"] == "motif" for row in full), 9)
        self.assertEqual(sum(parse_bool(row["selected"]) for row in full if row["stage"] == "base"), 1)
        self.assertEqual(sum(parse_bool(row["selected"]) for row in full if row["stage"] == "motif"), 1)

    def test_base_and_motif_stability_are_recomputed(self) -> None:
        base = self.metrics["base_recipe_stability"]
        motif = self.metrics["motif_recipe_stability"]
        for payload in (base, motif):
            passing = [
                row["outer_fold"]
                for row in payload["details"]
                if row["eligible"] and row["within_tie_window"]
            ]
            self.assertEqual(passing, payload["passing_outer_folds"])
            self.assertEqual(len(passing), payload["observed_outer_searches_within_tie_window"])
            self.assertEqual(payload["passed"], len(passing) >= 4)
        self.assertTrue(base["passed"])
        self.assertEqual(base["observed_outer_searches_within_tie_window"], 4)
        self.assertFalse(motif["passed"])
        self.assertEqual(motif["observed_outer_searches_within_tie_window"], 2)

    def test_every_gate_and_phase6_decision_are_recomputed(self) -> None:
        gates = self.metrics["promotion_gates"]
        results = {row["gate"]: row for row in gates["results"]}
        self.assertEqual(len(results), 21)
        for row in results.values():
            if row["comparison"] == ">=":
                expected = row["actual"] >= row["threshold"]
            elif row["comparison"] == "<=":
                expected = row["actual"] <= row["threshold"]
            elif row["comparison"] == "required":
                expected = row["actual"] is row["threshold"] is True
            else:
                self.fail(f"Unknown gate comparison {row['comparison']}")
            self.assertEqual(row["passed"], expected, row["gate"])
        failed = [row["gate"] for row in gates["results"] if not row["passed"]]
        self.assertEqual(
            failed,
            [
                "bubnivka_precision",
                "opishnyan_to_bubnivka_errors",
                "recall_floor:03_bubnivka_ceramics",
                "motif_recipe_stability",
            ],
        )
        self.assertEqual(gates["passed_count"], 17)
        self.assertFalse(gates["all_passed"])
        self.assertEqual(self.metrics["promotion_decision"], "reject")
        self.assertFalse(self.metrics["phase_6_transition_allowed"])
        self.assertFalse(self.metrics["sealed_test_evaluated"])

    def test_motif_proposals_and_cache_provenance_are_exact(self) -> None:
        cache = load_motif_embeddings_v4()
        rows = read_csv(OUTPUT_ROOT / "motif_proposals.csv")
        self.assertEqual(len(rows), 1693 * 6)
        self.assertEqual(Counter(row["image_id"] for row in rows), Counter({record.image_id: 6 for record in self.records}))
        for image_index in (0, 846, 1692):
            selected = rows[image_index * 6 : image_index * 6 + 6]
            for proposal_index, row in enumerate(selected):
                self.assertEqual(row["proposal_id"], str(cache.proposal_ids[image_index, proposal_index]))
                self.assertEqual(
                    [int(row[name]) for name in ("left", "top", "right", "bottom")],
                    cache.proposal_boxes[image_index, proposal_index].tolist(),
                )
                self.assertAlmostEqual(
                    float(row["texture_score"]),
                    float(cache.proposal_scores[image_index, proposal_index]),
                    places=15,
                )
        provenance = self.metrics["motif_embedding_provenance"]
        self.assertEqual(provenance["cache_sha256"], self.contract["motif_cache"]["file_sha256"])
        self.assertEqual(provenance["cache_fingerprint_sha256"], self.contract["motif_cache"]["fingerprint"])
        self.assertFalse(provenance["metadata"]["sealed_test_evaluated"])

    def test_v1_reproduction_and_v1_v2_v3_v4_comparison_are_exact(self) -> None:
        reproduction = self.metrics["v1_reproduction_validation"]
        self.assertTrue(reproduction["validated"])
        self.assertTrue(reproduction["outer_base_recipes_match"])
        self.assertTrue(reproduction["full_base_recipe_matches"])
        self.assertTrue(reproduction["aggregate_metrics_match"])
        self.assertEqual(
            reproduction["observed_decision_fingerprint_sha256"],
            "4533decae2de7855c914758b622a1f43e1825ecfc63285b95714b60585947a15",
        )
        comparison = self.metrics["previous_phase5_comparison"]
        self.assertEqual(set(comparison), {"v1", "v2", "v3", "v4"})
        self.assertGreater(comparison["v4"]["accuracy"], max(comparison[name]["accuracy"] for name in ("v1", "v2", "v3")))
        self.assertGreater(comparison["v4"]["macro_f1"], max(comparison[name]["macro_f1"] for name in ("v1", "v2", "v3")))
        self.assertGreater(comparison["v4"]["source_robustness_score"], max(comparison[name]["source_robustness_score"] for name in ("v1", "v2", "v3")))

    def test_code_and_cache_inputs_are_pinned(self) -> None:
        self.assertEqual(self.metrics["experiment_contract_sha256"], EXPECTED_CONTRACT_SHA256)
        code = self.metrics["code_provenance"]
        combined = hashlib.sha256()
        provenance_order = (
            "step_02/scripts/run_phase5_v4_motif.py",
            "step_02/scripts/run_phase5_source_robustness.py",
            "step_02/scripts/run_phase5_v2_pairwise.py",
            "step_02/scripts/run_phase5_v3_consensus.py",
            "step_02/scripts/run_phase5_v4_motif_extraction.py",
            "step_02/src/ornament_classifier/motif.py",
            "step_02/src/ornament_classifier/embeddings_v4.py",
            "step_02/src/ornament_classifier/embeddings_v3.py",
            "step_02/src/ornament_classifier/embeddings.py",
            "step_02/src/ornament_classifier/consensus.py",
            "step_02/src/ornament_classifier/pairwise.py",
            "step_02/src/ornament_classifier/robustness.py",
            "step_02/src/ornament_classifier/contracts.py",
            "step_02/src/ornament_classifier/paths.py",
            "step_02/requirements-phase5.txt",
        )
        self.assertEqual(set(provenance_order), set(code["files_sha256"]))
        for relative in provenance_order:
            expected = code["files_sha256"][relative]
            assert_recorded_file(self, relative, expected)
            combined.update(f"{relative}\x1f{expected}\n".encode("utf-8"))
        self.assertEqual(combined.hexdigest(), code["combined_sha256"])
        base_contract = self.contract["base_embedding_contract"]
        for relative, expected in base_contract["cache_files_sha256"].items():
            assert_recorded_file(self, relative, expected)


if __name__ == "__main__":
    unittest.main()

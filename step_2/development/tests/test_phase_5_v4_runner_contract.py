from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract_v4.json"
)
RUNNER_PATH = STEP_ROOT / "scripts" / "run_phase5_v4_motif.py"
EXPECTED_CONTRACT_SHA256 = (
    "7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase5V4FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_contract_is_frozen_development_only_and_exact(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(self.contract["iteration"], "v4")
        self.assertEqual(self.contract["status"], "frozen_before_fit")
        self.assertEqual(self.contract["scope"]["development_image_count"], 1693)
        self.assertEqual(self.contract["scope"]["sealed_test_access"], "forbidden")
        self.assertTrue(self.contract["scope"]["source_atomic_isolation_required"])

    def test_grid_is_bounded_and_representation_level(self) -> None:
        base = self.contract["base_procedure"]["configurations"]
        self.assertEqual(len(base), 3)
        self.assertEqual(
            {item["id"] for item in base},
            {
                "global_cls__c10__source0",
                "global_cls__c10__source0p5",
                "three_view_cls_logit_ensemble__c100__source0p5",
            },
        )
        grid = self.contract["motif_grid"]
        self.assertEqual(len(grid["descriptors"]), 4)
        self.assertEqual(grid["c_values"], [1.0, 10.0])
        self.assertEqual(grid["source_group_exponent"], 0.5)
        self.assertEqual(grid["configuration_count_including_none"], 9)
        self.assertIn("concatenate", grid["application"])
        self.assertIn("probability correction", grid["forbidden"])

    def test_v1_gates_non_regression_and_motif_stability_are_required(self) -> None:
        gates = self.contract["promotion_gates"]
        self.assertTrue(gates["all_required"])
        self.assertTrue(gates["base_recipe_stability_required"])
        self.assertTrue(gates["motif_recipe_stability_required"])
        self.assertEqual(
            gates["phase_6_transition"],
            "allowed only when every promotion result passes and promotion_decision is promote",
        )
        self.assertEqual(
            gates["v1_non_regression"],
            {
                "accuracy_min": 0.9377519196692262,
                "macro_f1_min": 0.9341346585193265,
                "source_robustness_min": 0.7058830417592783,
            },
        )

    def test_cache_and_required_outputs_are_exact(self) -> None:
        cache = self.contract["motif_cache"]
        self.assertEqual(cache["file_sha256"], "97ee78769b6b4349b58e9b2cb27c9f4b58748ecf7cd9fbcfcc554c2fa122db40")
        self.assertEqual(len(cache["required_arrays"]), 14)
        required = self.contract["output_contract"]["required"]
        self.assertEqual(len(required), 14)
        self.assertEqual(len(required), len(set(required)))
        self.assertIn("motif_proposals.csv", required)
        self.assertNotIn("test_predictions.csv", required)


class Phase5V4RunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")

    def test_runner_pins_contract_and_uses_strict_loaders(self) -> None:
        match = re.search(
            r'EXPECTED_CONTRACT_SHA256\s*=\s*\(\s*"([0-9a-f]{64})"',
            self.source,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), EXPECTED_CONTRACT_SHA256)
        self.assertIn("load_motif_embeddings_v4", self.source)
        self.assertIn("load_allowlisted_embeddings_v3", self.source)
        self.assertNotIn("np.load(", self.source)

    def test_runner_contains_nested_sequential_lifecycle(self) -> None:
        for required in (
            "evaluate_base_oof",
            "select_search_candidates",
            "evaluate_motif_oof_grid",
            "select_motif_candidates",
            "fit_outer_procedure",
            "validate_v1_reproduction",
            "motif_recipe_stability",
            "phase_6_transition_allowed",
        ):
            self.assertIn(required, self.source)

    def test_fit_boundaries_are_source_disjoint_and_thread_limited(self) -> None:
        self.assertGreaterEqual(
            self.source.count("_assert_source_disjoint_partitions("), 2
        )
        self.assertIn("threadpool_limits(limits=1)", self.source)
        self.assertIn("compute_source_group_weights(", self.source)
        self.assertIn("max_iter=MAX_ITERATIONS", self.source)

    def test_runner_has_no_sealed_prediction_or_calibration_path(self) -> None:
        self.assertNotIn('"test.csv"', self.source)
        self.assertNotIn("'test.csv'", self.source)
        self.assertIn('"sealed_test_evaluated": False', self.source)
        self.assertIn('"calibrated": False', self.source)
        self.assertIn('"threshold_selection_used": False', self.source)


if __name__ == "__main__":
    unittest.main()

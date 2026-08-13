import hashlib
import json
import unittest
from pathlib import Path


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    DEVELOPMENT_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract_v5.json"
)
RUNNER_PATH = DEVELOPMENT_ROOT / "scripts" / "run_phase5_v5_fixed_consensus.py"
EXPECTED_SHA256 = "c9473a7e20c57049fc4bc648f1133c5269434a955f955f390f10594adb80a70d"


class Phase5V5FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_contract_is_exactly_frozen_and_final(self):
        observed = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
        self.assertEqual(observed, EXPECTED_SHA256)
        self.assertEqual(self.payload["iteration"], "v5")
        self.assertEqual(self.payload["status"], "frozen_before_fit")
        self.assertTrue(self.payload["final_phase5_attempt"])

    def test_scope_is_development_only_and_sealed(self):
        scope = self.payload["scope"]
        self.assertTrue(scope["development_only"])
        self.assertEqual(scope["development_image_count"], 1693)
        self.assertEqual(scope["outer_folds"], ["0", "1", "2", "3", "4"])
        self.assertEqual(scope["sealed_test_access"], "forbidden")
        self.assertEqual(scope["sealed_test_embeddings"], "forbidden")
        self.assertEqual(scope["sealed_test_predictions"], "forbidden")
        self.assertEqual(scope["calibration"], "forbidden")
        self.assertEqual(scope["threshold_selection"], "forbidden")

    def test_consensus_is_one_fixed_class_symmetric_recipe(self):
        fixed = self.payload["fixed_consensus"]
        self.assertEqual(fixed["candidate_count"], 1)
        self.assertFalse(fixed["selection_after_base"])
        self.assertEqual(fixed["head_count"], 4)
        self.assertEqual(fixed["head_weights"], [0.25, 0.25, 0.25, 0.25])
        self.assertTrue(fixed["class_symmetric"])
        self.assertEqual(
            [head["id"] for head in fixed["heads"]],
            ["base_anchor", "uniform_mean", "texture_weighted_mean", "texture_top2_mean"],
        )

    def test_all_21_gates_and_stability_are_required(self):
        gates = self.payload["promotion_gates"]
        self.assertTrue(gates["all_required"])
        self.assertTrue(gates["base_recipe_stability_required"])
        self.assertTrue(gates["fixed_consensus_stability_required"])
        self.assertIn("every one of the 21", gates["phase_6_transition"])
        self.assertEqual(self.payload["stability_rule"]["required_outer_searches"], 4)

    def test_required_outputs_are_exact(self):
        required = self.payload["output_contract"]["required"]
        self.assertEqual(len(required), 14)
        self.assertEqual(len(required), len(set(required)))
        self.assertIn("metrics.json", required)
        self.assertIn("nested_oof_predictions.csv", required)
        self.assertIn("consensus_recipe.csv", required)


class Phase5V5RunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")

    def test_runner_pins_contract_and_fixed_weights(self):
        self.assertIn(EXPECTED_SHA256, self.source)
        self.assertIn("HEAD_WEIGHTS = (0.25, 0.25, 0.25, 0.25)", self.source)
        self.assertIn("fixed_probability_consensus", self.source)

    def test_runner_contains_nested_source_disjoint_lifecycle(self):
        self.assertIn("v1._assert_source_disjoint_partitions", self.source)
        self.assertIn("v1.evaluate_base_oof", self.source)
        self.assertIn("v1.select_search_candidates", self.source)
        self.assertIn("fit_outer_fixed", self.source)
        self.assertIn("threadpool_limits(limits=1)", self.source)

    def test_runner_has_no_calibration_or_sealed_prediction_path(self):
        self.assertNotIn("load_test", self.source)
        self.assertNotIn("test.csv", self.source)
        self.assertNotIn("CalibratedClassifier", self.source)
        self.assertIn('"sealed_test_evaluated": False', self.source)
        self.assertIn('"threshold_selection_used": False', self.source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract_v2.json"
)
RUNNER_PATH = STEP_ROOT / "scripts" / "run_phase5_v2_pairwise.py"
EXPECTED_CONTRACT_SHA256 = (
    "481fcab8c19b412a59354f53735a3c6299da5c16ec06ed1c6e4ffbe49c1a5a68"
)
EXPECTED_OUTPUTS = {
    "metrics.json",
    "base_grid.csv",
    "correction_grid.csv",
    "inner_search.csv",
    "outer_fold_metrics.csv",
    "nested_oof_predictions.csv",
    "full_development_selection.csv",
    "selected_oof_predictions.csv",
    "per_class_metrics.csv",
    "diagnostic_slices.csv",
    "confusion_pairs.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
}


class Phase5V2FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = CONTRACT_PATH.read_bytes()
        cls.contract = json.loads(cls.raw)

    def test_contract_is_frozen_and_development_only(self) -> None:
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(self.contract["iteration"], "v2")
        self.assertEqual(self.contract["status"], "frozen_before_fit")
        self.assertEqual(self.contract["scope"]["development_image_count"], 1693)
        self.assertEqual(self.contract["scope"]["sealed_test_access"], "forbidden")
        self.assertEqual(self.contract["scope"]["outer_folds"], list("01234"))

    def test_grid_and_pair_application_are_exact(self) -> None:
        pair = self.contract["pairwise_correction"]
        expected = (
            len(pair["feature_families"])
            * len(pair["c_values"])
            * len(pair["source_group_exponents"])
            * len(pair["blend_weights"])
        )
        self.assertEqual(expected, 64)
        self.assertEqual(pair["grid_configuration_count_excluding_none"], 64)
        self.assertEqual(pair["grid_configuration_count_including_none"], 65)
        self.assertEqual(
            pair["classes"],
            ["01_opishnyan_ceramics", "03_bubnivka_ceramics"],
        )
        self.assertIn("preserve the base total probability mass", pair["application"])
        self.assertIn("leave all other class columns exactly unchanged", pair["application"])

    def test_v1_gates_are_retained_and_correction_stability_added(self) -> None:
        v1_contract = json.loads(
            (
                STEP_ROOT
                / "phases"
                / "phase_05_source_robustness"
                / "experiment_contract.json"
            ).read_text(encoding="utf-8")
        )
        for key, value in v1_contract["promotion_gates"].items():
            self.assertEqual(self.contract["promotion_gates"][key], value)
        self.assertTrue(
            self.contract["promotion_gates"][
                "correction_recipe_stability_required"
            ]
        )

    def test_required_outputs_are_exact(self) -> None:
        self.assertEqual(
            set(self.contract["output_contract"]["required"]), EXPECTED_OUTPUTS
        )


class Phase5V2RunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.function_names = {
            node.name
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
        }

    def test_runner_pins_contract_and_uses_only_v2_loader(self) -> None:
        self.assertIn(EXPECTED_CONTRACT_SHA256, self.source)
        self.assertIn("load_allowlisted_embeddings_v2", self.source)
        self.assertNotIn("np.load", self.source)
        self.assertNotIn("test.csv", self.source.lower())
        attributes = {
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(attributes & {"glob", "rglob", "iglob"})

    def test_runner_contains_sequential_nested_lifecycle(self) -> None:
        required = {
            "evaluate_correction_oof_grid",
            "select_correction_candidates",
            "_fit_outer_procedure",
            "validate_v1_reproduction",
            "_stability",
            "evaluate_v2_promotion_gates",
            "main",
        }
        self.assertFalse(required - self.function_names)
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        source = ast.get_source_segment(self.source, main) or ""
        self.assertLess(source.index("base_selection ="), source.index("correction_results ="))
        self.assertIn("uncorrected_nested_probabilities", source)
        self.assertIn("full_correction_selection", source)

    def test_every_fit_boundary_is_thread_limited_and_source_disjoint(self) -> None:
        self.assertIn("_assert_source_disjoint_partitions", self.source)
        self.assertGreaterEqual(self.source.count("threadpool_limits(limits=1)"), 2)
        self.assertIn('"blas_threads_per_fit": 1', self.source)

    def test_pair_head_cache_shares_fits_without_changing_candidate_count(self) -> None:
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_correction_oof_grid"
        )
        source = ast.get_source_segment(self.source, function) or ""
        self.assertIn("head_cache", source)
        self.assertIn("configuration.blend_weight", source)
        self.assertIn("apply_pairwise_logit_blend", source)

    def test_outputs_disclaim_calibration_and_sealed_evaluation(self) -> None:
        self.assertIn('"sealed_test_evaluated": False', self.source)
        self.assertIn('"calibrated": False', self.source)
        self.assertIn('"threshold_selection_used": False', self.source)
        strings = {
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertFalse(EXPECTED_OUTPUTS - strings)


if __name__ == "__main__":
    unittest.main()

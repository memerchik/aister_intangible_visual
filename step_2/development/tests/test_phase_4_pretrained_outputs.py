from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
import unittest
from collections import Counter, defaultdict
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SPLIT_ROOT = STEP_ROOT / "splits"
OUTPUT_ROOT = STEP_ROOT / "outputs" / "phase_4_pretrained"
RUNNER = STEP_ROOT / "scripts" / "run_pretrained_embeddings.py"
REQUIREMENTS = STEP_ROOT / "requirements-phase4.txt"
EXPECTED_EXPERIMENT_VERSION = "frozen_pretrained_views_v2"
EXPECTED_FOLDS = {"0", "1", "2", "3", "4"}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "top_3_accuracy",
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


def audit_development_sha256(audit: dict[str, object]) -> str:
    for audit_key in ("output_file_sha256", "generated_csv_sha256"):
        recorded = audit.get(audit_key)
        if isinstance(recorded, str):
            return recorded
        if isinstance(recorded, dict):
            for key in ("development.csv", "development"):
                value = recorded.get(key)
                if isinstance(value, str):
                    return value
    raise AssertionError("split_audit.json has no development.csv output SHA-256")


def classification_metrics(rows: list[dict[str, str]], class_names: list[str]) -> dict[str, float]:
    support = Counter(row["true_class"] for row in rows)
    predicted_count = Counter(row["predicted_class"] for row in rows)
    true_positive = Counter(
        row["true_class"] for row in rows if row["true_class"] == row["predicted_class"]
    )
    precision = [
        true_positive[name] / predicted_count[name] if predicted_count[name] else 0.0
        for name in class_names
    ]
    recall = [true_positive[name] / support[name] for name in class_names]
    f1 = [
        2 * class_precision * class_recall / (class_precision + class_recall)
        if class_precision + class_recall
        else 0.0
        for class_precision, class_recall in zip(precision, recall)
    ]
    correct = sum(true_positive.values())
    top_three_correct = sum(
        row["true_class"] in {row["top_1_class"], row["top_2_class"], row["top_3_class"]}
        for row in rows
    )
    return {
        "accuracy": correct / len(rows),
        "balanced_accuracy": statistics.mean(recall),
        "macro_precision": statistics.mean(precision),
        "macro_recall": statistics.mean(recall),
        "macro_f1": statistics.mean(f1),
        "top_3_accuracy": top_three_correct / len(rows),
    }


class PhaseFourRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_outer_c_and_experiment_version_are_explicit(self) -> None:
        assignments: dict[str, object] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertEqual(assignments["OUTER_C"], 10.0)
        self.assertEqual(assignments["EXPERIMENT_VERSION"], EXPECTED_EXPERIMENT_VERSION)
        self.assertIn("fit_probe(features, labels, train_indices, OUTER_C)", self.source)
        self.assertIn("frozen_after_prior_prototype_exploration", self.source)

    def test_exploratory_oof_is_not_overclaimed(self) -> None:
        self.assertIn("not an unbiased performance estimate", self.source)
        self.assertIn('"performance_estimate"', self.source)
        self.assertIn('"available": False', self.source)
        self.assertIn("same OOF results select the representation", self.source)

    def test_image_bytes_are_verified_before_cache_access(self) -> None:
        main = self.source[self.source.index("def main()") :]
        verification = main.index("validate_development(")
        cache_access = main.index("get_or_extract_dinov3(")
        self.assertLess(verification, cache_access)
        self.assertIn("observed_hash != row[\"content_sha256\"]", self.source)

    def test_cache_identity_and_atomic_save_are_hardened(self) -> None:
        for token in (
            '"requested_batch_size"',
            '"effective_batch_size_by_input"',
            '"threads"',
            '"platform"',
            '"determinism"',
            '"revision"',
            '"model_config_sha256"',
            '"preprocessor_config_sha256"',
        ):
            self.assertIn(token, self.source)
        self.assertIn("tempfile.NamedTemporaryFile", self.source)
        self.assertIn("os.fsync", self.source)
        self.assertIn("os.replace(temporary_path, path)", self.source)

    def test_cpu_determinism_and_probe_convergence_are_asserted(self) -> None:
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=False)", self.source)
        self.assertIn("torch.are_deterministic_algorithms_enabled()", self.source)
        self.assertIn('warnings.simplefilter("error", ConvergenceWarning)', self.source)
        self.assertIn("threadpool_limits(limits=1)", self.source)
        self.assertNotIn('warnings.simplefilter("ignore")', self.source)

    def test_timm_uses_native_preprocessing_and_cautious_license_metadata(self) -> None:
        self.assertIn("timm.get_pretrained_cfg", self.source)
        self.assertIn("int(input_size / crop_pct)", self.source)
        self.assertIn("Upstream model card/license review required before deployment", self.source)
        self.assertNotIn('license_name="Apache-2.0"', self.source)
        self.assertIn("not a controlled backbone-only comparison", self.source)

    def test_quick_mode_cannot_overwrite_canonical_outputs(self) -> None:
        self.assertIn("DEFAULT_QUICK_OUTPUT_DIR", self.source)
        self.assertIn(
            'parser.error("--quick cannot write to the canonical Phase 4 output directory")',
            self.source,
        )

    def test_phase_four_environment_is_exactly_pinned(self) -> None:
        observed = {
            line.strip()
            for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        expected = {
            "numpy==2.0.2",
            "Pillow==11.3.0",
            "matplotlib==3.9.4",
            "scikit-learn==1.6.1",
            "torch==2.8.0",
            "torchvision==0.23.0",
            "timm==1.0.26",
            "transformers==4.57.6",
            "huggingface-hub==0.36.2",
            "safetensors==0.7.0",
            "threadpoolctl==3.6.0",
        }
        self.assertEqual(observed, expected)


class PhaseFourPretrainedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        metrics_path = OUTPUT_ROOT / "metrics.json"
        if not metrics_path.is_file():
            raise unittest.SkipTest("canonical Phase 4 outputs have not been generated")
        with metrics_path.open(encoding="utf-8") as handle:
            cls.metrics = json.load(handle)
        observed_version = cls.metrics.get("experiment_version")
        if observed_version != EXPECTED_EXPERIMENT_VERSION:
            raise AssertionError(
                "Stale canonical Phase 4 outputs: expected experiment version "
                f"{EXPECTED_EXPERIMENT_VERSION}, found {observed_version!r}. Rerun the runner."
            )
        required_files = (
            "oof_predictions.csv",
            "cv_fold_metrics.csv",
            "representation_summary.csv",
            "probe_screen.csv",
            "diagnostic_slices.csv",
            "representation_metadata.csv",
            "per_class_metrics.csv",
            "confusion_matrix.csv",
            "confusion_pairs.csv",
            "confusion_matrix.png",
        )
        missing = [name for name in required_files if not (OUTPUT_ROOT / name).is_file()]
        if missing:
            raise AssertionError(f"Incomplete canonical Phase 4 outputs: {missing}")
        cls.development = read_csv(SPLIT_ROOT / "development.csv")
        cls.test = read_csv(SPLIT_ROOT / "test.csv")
        cls.predictions = read_csv(OUTPUT_ROOT / "oof_predictions.csv")
        cls.fold_metrics = read_csv(OUTPUT_ROOT / "cv_fold_metrics.csv")
        cls.summary = read_csv(OUTPUT_ROOT / "representation_summary.csv")
        cls.screen = read_csv(OUTPUT_ROOT / "probe_screen.csv")
        cls.slices = read_csv(OUTPUT_ROOT / "diagnostic_slices.csv")
        cls.representation_metadata = read_csv(OUTPUT_ROOT / "representation_metadata.csv")
        cls.per_class = read_csv(OUTPUT_ROOT / "per_class_metrics.csv")
        cls.confusion = read_csv(OUTPUT_ROOT / "confusion_matrix.csv")
        cls.confusion_pairs = read_csv(OUTPUT_ROOT / "confusion_pairs.csv")
        with (SPLIT_ROOT / "split_audit.json").open(encoding="utf-8") as handle:
            cls.split_audit = json.load(handle)
        cls.class_names = list(cls.metrics["class_names"])

    def assert_metric_dict_close(
        self, observed: dict[str, object], expected: dict[str, float], prefix: str = ""
    ) -> None:
        for metric, value in expected.items():
            self.assertAlmostEqual(float(observed[f"{prefix}{metric}"]), value, places=12)

    def test_evaluation_contains_only_canonical_development_ids(self) -> None:
        self.assertFalse(self.metrics["sealed_test_evaluated"])
        self.assertEqual(
            self.metrics["evaluation_scope"],
            "development_only_exploratory_model_selection",
        )
        development_ids = [row["image_id"] for row in self.development]
        prediction_ids = [row["image_id"] for row in self.predictions]
        test_ids = {row["image_id"] for row in self.test}
        self.assertEqual(prediction_ids, development_ids)
        self.assertFalse(set(prediction_ids) & test_ids)
        self.assertFalse((OUTPUT_ROOT / "test_predictions.csv").exists())

    def test_oof_status_explicitly_disclaims_performance_estimate(self) -> None:
        self.assertFalse(self.metrics["performance_estimate"]["available"])
        self.assertIn("selection-conditional", self.metrics["performance_estimate"]["reason"])
        self.assertIn(
            "not_an_unbiased_performance_estimate",
            self.metrics["aggregate_oof_metric_status"],
        )
        self.assertEqual(
            self.metrics["outer_c_provenance"],
            "frozen_after_prior_prototype_exploration",
        )
        self.assertTrue(
            all(
                row["evaluation_status"]
                == "exploratory_selected_representation_oof"
                for row in self.predictions
            )
        )
        self.assertTrue(
            all(row["metric_status"] == "exploratory_candidate_cv" for row in self.fold_metrics)
        )
        self.assertTrue(
            all(
                row["metric_status"] == "exploratory_selection_conditional"
                for row in self.summary
            )
        )

    def test_outer_cv_uses_only_frozen_c_and_all_probes_converged(self) -> None:
        self.assertEqual(float(self.metrics["selected_c"]), 10.0)
        self.assertIn("C=10", self.metrics["outer_cv_probe_rule"])
        self.assertTrue(self.fold_metrics)
        self.assertTrue(all(float(row["c_value"]) == 10.0 for row in self.fold_metrics))
        self.assertTrue(all(float(row["c_value"]) == 10.0 for row in self.summary))
        self.assertTrue(self.screen)
        self.assertTrue(all(row["diagnostic_only"] == "True" for row in self.screen))
        for row in (*self.screen, *self.fold_metrics):
            self.assertEqual(row["converged"], "True")
            self.assertLess(int(row["iterations"]), int(row["max_iterations"]))
        convergence = self.metrics["probe_convergence"]
        self.assertTrue(convergence["warnings_treated_as_errors"])
        self.assertTrue(convergence["all_fits_converged"])
        self.assertLess(
            convergence["maximum_observed_iterations"], convergence["max_iterations"]
        )

    def test_development_sha_split_and_image_integrity_are_pinned(self) -> None:
        expected_sha = audit_development_sha256(self.split_audit)
        observed_sha = sha256_file(SPLIT_ROOT / "development.csv")
        self.assertEqual(observed_sha, expected_sha)
        integrity = self.metrics["development_integrity"]
        self.assertEqual(integrity["csv_sha256"], expected_sha)
        self.assertEqual(integrity["audit_csv_sha256"], expected_sha)
        self.assertTrue(integrity["class_coverage_verified"])
        self.assertTrue(integrity["production_group_isolation_verified"])
        self.assertTrue(integrity["cv_group_isolation_verified"])
        self.assertTrue(integrity["all_image_bytes_sha256_verified"])
        self.assertEqual(integrity["verified_image_count"], len(self.development))
        self.assertEqual(
            integrity["verified_image_fingerprint_sha256"],
            self.metrics["data_fingerprint_sha256"],
        )
        self.assertEqual(
            self.metrics["split_assignment_fingerprint_sha256"],
            self.split_audit["assignment_fingerprint_sha256"],
        )
        self.assertEqual(self.metrics["split_version"], self.split_audit["split_version"])

    def test_groups_and_content_do_not_cross_boundaries(self) -> None:
        policies = [
            ("split_group_id", False),
            ("confirmed_object_group_id", False),
            ("confirmed_source_group_id", True),
            ("content_sha256", False),
        ]
        for optional, allow_blank in (
            ("global_source_cohort_id", True),
            ("source_atomic_cohort_ids", True),
            ("pre_source_cohort_split_group_id", False),
            ("source_atomic_split_group_id", False),
        ):
            if optional in self.development[0]:
                policies.append((optional, allow_blank))
                self.assertEqual(
                    [row[optional] for row in self.predictions],
                    [row[optional] for row in self.development],
                )
        for identifier, allow_blank in policies:
            for boundary in ("production_split", "cv_fold"):
                assignments: dict[str, set[str]] = defaultdict(set)
                for row in self.development:
                    value = row[identifier]
                    if not value and allow_blank:
                        continue
                    self.assertTrue(value)
                    assignments[value].add(row[boundary])
                self.assertTrue(all(len(values) == 1 for values in assignments.values()))

    def test_probabilities_top_three_and_aggregate_metrics_are_exact(self) -> None:
        policy = self.metrics["probability_policy"]
        self.assertFalse(policy["calibrated"])
        self.assertFalse(policy["threshold_selection_used"])
        probability_fields = [
            f"probability_{class_name}" for class_name in self.class_names
        ]
        self.assertTrue(all(field in self.predictions[0] for field in probability_fields))
        for row in self.predictions:
            probabilities = {
                class_name: float(row[f"probability_{class_name}"])
                for class_name in self.class_names
            }
            self.assertTrue(all(math.isfinite(value) for value in probabilities.values()))
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=8)
            ordered = sorted(self.class_names, key=probabilities.get, reverse=True)
            self.assertEqual(row["predicted_class"], ordered[0])
            self.assertEqual(
                [row["top_1_class"], row["top_2_class"], row["top_3_class"]],
                ordered[:3],
            )
            self.assertAlmostEqual(
                float(row["uncalibrated_max_probability"]),
                probabilities[ordered[0]],
                places=12,
            )
            self.assertEqual(
                row["is_correct"], str(row["true_class"] == row["predicted_class"])
            )
        expected = classification_metrics(self.predictions, self.class_names)
        self.assert_metric_dict_close(self.metrics["aggregate_oof_metrics"], expected)
        selected_summary = next(
            row for row in self.summary if row["representation"] == self.metrics["selected_representation"]
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "top_3_accuracy",
        ):
            self.assertAlmostEqual(
                float(selected_summary[f"aggregate_{metric}"]), expected[metric], places=12
            )

    def test_selected_fold_metrics_are_recomputed_from_predictions(self) -> None:
        selected = self.metrics["selected_representation"]
        selected_rows = [row for row in self.fold_metrics if row["representation"] == selected]
        self.assertEqual({row["cv_fold"] for row in selected_rows}, EXPECTED_FOLDS)
        payload_rows = {
            str(row["cv_fold"]): row for row in self.metrics["selected_fold_metrics"]
        }
        for row in selected_rows:
            fold_predictions = [
                prediction
                for prediction in self.predictions
                if prediction["cv_fold"] == row["cv_fold"]
            ]
            expected = classification_metrics(fold_predictions, self.class_names)
            self.assert_metric_dict_close(row, expected)
            self.assert_metric_dict_close(payload_rows[row["cv_fold"]], expected)

    def test_representation_selection_and_summary_statistics_are_exact(self) -> None:
        self.assertEqual(len(self.summary), self.metrics["representation_count"])
        by_representation: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.fold_metrics:
            by_representation[row["representation"]].append(row)
        self.assertEqual(set(by_representation), {row["representation"] for row in self.summary})
        self.assertTrue(all(len(rows) == 5 for rows in by_representation.values()))
        for summary in self.summary:
            folds = by_representation[summary["representation"]]
            self.assertEqual(int(summary["fold_count"]), 5)
            for metric in METRICS:
                values = [float(row[metric]) for row in folds]
                self.assertAlmostEqual(float(summary[f"mean_{metric}"]), statistics.mean(values), places=12)
                self.assertAlmostEqual(float(summary[f"std_{metric}"]), statistics.stdev(values), places=12)
            self.assertEqual(summary["all_folds_converged"], "True")
            self.assertEqual(
                int(summary["max_fold_iterations"]),
                max(int(row["iterations"]) for row in folds),
            )
        best_accuracy = max(float(row["mean_accuracy"]) for row in self.summary)
        near_ties = [
            row for row in self.summary if float(row["mean_accuracy"]) >= best_accuracy - 0.005
        ]
        expected_selected = sorted(
            near_ties,
            key=lambda row: (
                -float(row["mean_macro_f1"]),
                -float(row["worst_class_recall"]),
                row["representation"],
            ),
        )[0]["representation"]
        self.assertEqual(self.metrics["selected_representation"], expected_selected)
        selected_flags = [row for row in self.summary if row["selected"] == "True"]
        self.assertEqual([row["representation"] for row in selected_flags], [expected_selected])
        payload_summary = self.metrics["selected_summary"]
        self.assertEqual(payload_summary["representation"], expected_selected)
        for field in (
            "aggregate_accuracy",
            "aggregate_macro_f1",
            "aggregate_balanced_accuracy",
            "aggregate_top_3_accuracy",
            "mean_accuracy",
            "mean_macro_f1",
            "worst_class_recall",
        ):
            self.assertAlmostEqual(
                float(payload_summary[field]),
                float(selected_flags[0][field]),
                places=12,
            )
        ranked = sorted(
            self.summary,
            key=lambda row: (
                -float(row["mean_accuracy"]),
                -float(row["mean_macro_f1"]),
                row["representation"],
            ),
        )
        self.assertEqual([int(row["accuracy_rank"]) for row in ranked], list(range(1, len(ranked) + 1)))

    def test_per_class_metrics_and_confusion_outputs_are_exact(self) -> None:
        expected_counts = Counter(
            (row["true_class"], row["predicted_class"]) for row in self.predictions
        )
        confusion_by_true = {row["true_class"]: row for row in self.confusion}
        self.assertEqual(set(confusion_by_true), set(self.class_names))
        for true_class in self.class_names:
            for predicted_class in self.class_names:
                self.assertEqual(
                    int(confusion_by_true[true_class][predicted_class]),
                    expected_counts[(true_class, predicted_class)],
                )
        per_class_by_name = {row["class_name"]: row for row in self.per_class}
        payload_by_name = {
            row["class_name"]: row for row in self.metrics["per_class_metrics"]
        }
        for class_index, class_name in enumerate(self.class_names):
            support = sum(expected_counts[(class_name, predicted)] for predicted in self.class_names)
            predicted_count = sum(expected_counts[(truth, class_name)] for truth in self.class_names)
            true_positive = expected_counts[(class_name, class_name)]
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            for observed in (per_class_by_name[class_name], payload_by_name[class_name]):
                self.assertEqual(int(observed["class_index"]), class_index)
                self.assertEqual(int(observed["support"]), support)
                self.assertAlmostEqual(float(observed["precision"]), precision, places=12)
                self.assertAlmostEqual(float(observed["recall"]), recall, places=12)
                self.assertAlmostEqual(float(observed["f1"]), f1, places=12)
        expected_pairs = {
            (truth, predicted): count
            for (truth, predicted), count in expected_counts.items()
            if truth != predicted and count
        }
        observed_pairs = {
            (row["true_class"], row["predicted_class"]): int(row["error_count"])
            for row in self.confusion_pairs
        }
        self.assertEqual(observed_pairs, expected_pairs)
        support = Counter(row["true_class"] for row in self.predictions)
        for row in self.confusion_pairs:
            self.assertAlmostEqual(
                float(row["fraction_of_true_class"]),
                int(row["error_count"]) / support[row["true_class"]],
                places=12,
            )

    def test_representation_cache_provenance_and_native_input_are_recorded(self) -> None:
        self.assertEqual(len(self.representation_metadata), self.metrics["representation_count"])
        for row in self.representation_metadata:
            self.assertTrue(row["model_source"])
            self.assertTrue(row["model_revision"])
            self.assertEqual(row["experiment_version"], EXPECTED_EXPERIMENT_VERSION)
            for hash_field in (
                "weight_sha256",
                "model_config_sha256",
                "preprocessor_config_sha256",
                "script_sha256",
                "embedding_fingerprint_sha256",
            ):
                self.assertEqual(len(row[hash_field]), 64)
            runtime = json.loads(row["extraction_runtime_json"])
            self.assertTrue(
                {"python", "numpy", "Pillow", "torch", "timm", "transformers"}
                <= set(runtime)
            )
            input_policy = json.loads(row["input_policy_json"])
            self.assertGreaterEqual(input_policy["resize_short_side"], input_policy["input_size"])
            self.assertEqual(input_policy["interpolation"], "bicubic")
            execution = json.loads(row["execution_identity_json"])
            self.assertEqual(execution, self.metrics["execution_identity"])
            self.assertEqual(
                int(row["effective_batch_size"]),
                execution["effective_batch_size_by_input"][str(input_policy["input_size"])],
            )
            cache_identity = json.loads(row["cache_identity_json"])
            self.assertEqual(cache_identity["execution"], execution)
            self.assertEqual(cache_identity["input_policy"], input_policy)
            self.assertNotIn("seconds", row["cache_identity_json"])
            self.assertTrue(row["nondeterministic_extraction_seconds"])
            self.assertTrue(row["nondeterministic_images_per_second"])
            if row["encoder_alias"] != "dinov3_vits16":
                self.assertEqual(
                    row["license"],
                    "Upstream model card/license review required before deployment",
                )
                self.assertEqual(input_policy["source"], "timm.get_pretrained_cfg(model_name)")
                self.assertEqual(input_policy["crop_mode"], "center")
            if row["encoder_alias"] == "dinov2_small_reg4":
                self.assertEqual(input_policy["input_size"], 518)
        self.assertTrue(self.metrics["determinism"]["torch_deterministic_algorithms"])
        self.assertEqual(len(self.metrics["script_sha256"]), 64)
        if not self.metrics["cache_enabled"]:
            self.assertEqual(self.metrics["cache_files"], [])


if __name__ == "__main__":
    unittest.main()

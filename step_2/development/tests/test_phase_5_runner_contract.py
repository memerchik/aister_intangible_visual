from __future__ import annotations

import ast
import csv
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set, Tuple

from relocation_support import assert_recorded_file, resolve_recorded_path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SOURCE_ROOT = STEP_ROOT / "src" / "ornament_classifier"
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "experiment_contract.json"
)
RUNNER_PATH = STEP_ROOT / "scripts" / "run_phase5_source_robustness.py"

EXPECTED_CONTRACT_SHA256 = (
    "55d0dd0d3665b4b84b33a9c9461763c6858452bd0b10b679fd9b77fbe630a87f"
)
EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256 = (
    "d6f5b90c2c6b8af39fb8e6bffbe3e44ba3e6d99b3f8f069b8f1febfe95bab307"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "phase",
    "experiment_version",
    "status",
    "frozen_on",
    "seed",
    "scope",
    "input_contract",
    "candidate_set_provenance",
    "cache_allowlist",
    "base_feature_families",
    "base_grid",
    "source_weighting",
    "ceramic_specialist",
    "nested_evaluation",
    "selection_rule",
    "stability_rule",
    "promotion_gates",
    "diagnostic_slice_definitions",
    "output_contract",
}
EXPECTED_OUTPUTS = {
    "metrics.json",
    "base_grid.csv",
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
EXPECTED_CACHE_SPECS = {
    "dinov3_center": {
        "path": (
            ".cache/step_02/pretrained_embeddings/"
            "dinov3_vits16__center_crop__a6de7aa2fa2137dd.npz"
        ),
        "file_sha256": (
            "61f46b287d5fbf7ce3fe72585d054d7c9c44b5122bf8c8cfc18d95fd33445f30"
        ),
        "fingerprint": (
            "a6de7aa2fa2137dd71e36d004adfff0ddf2abd85aae7b19478efb3bd5cdb080a"
        ),
        "arrays": {
            "embedding__cls": {
                "shape": [1693, 384],
                "dtype": "float32",
                "sha256": (
                    "afe1e2b5d898f257df5e51b44fb5e6cef627ece658d16ed87a0a9c1bc4d5a7d4"
                ),
            }
        },
    },
    "dinov3_letterbox": {
        "path": (
            ".cache/step_02/pretrained_embeddings/"
            "dinov3_vits16__letterbox__443acf6c3b85afd5.npz"
        ),
        "file_sha256": (
            "a4551b0e94d2fc3838ae7cc036af5b748f335bd498f24a737840994477052822"
        ),
        "fingerprint": (
            "443acf6c3b85afd5042f78daa35c426f7f4d36058ab3080de8d7ea55bd1d3faa"
        ),
        "arrays": {
            "embedding__cls": {
                "shape": [1693, 384],
                "dtype": "float32",
                "sha256": (
                    "d0691de4b7d16d8fbe56bff810470f7166b7bb0760ed398f770c18ea6fd4bc79"
                ),
            },
            "embedding__cls_patch_concat": {
                "shape": [1693, 768],
                "dtype": "float32",
                "sha256": (
                    "67162b1a0e4668bfa2f20bfdb94686b2521488a4490701088c71f73fd4eeccc3"
                ),
            },
        },
    },
    "dinov3_global": {
        "path": (
            ".cache/step_02/pretrained_embeddings/"
            "dinov3_vits16__global_fivecrop__2783a020edd6f819.npz"
        ),
        "file_sha256": (
            "cf04c78f40db527306f7775e1c9277f99ec2ab97c8aaf5180e09f87e6af35bfa"
        ),
        "fingerprint": (
            "2783a020edd6f819e2d9530ae635e786fde7618e3c96244b1ef78fc49f927722"
        ),
        "arrays": {
            "embedding__cls": {
                "shape": [1693, 384],
                "dtype": "float32",
                "sha256": (
                    "20cb4b8e5b2c7ca8e72e8e01fe465a252d5cb96687c2178dc256a8cb3b2ac1ae"
                ),
            }
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def literal_strings(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def called_attribute_names(tree: ast.AST) -> Set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


class PhaseFiveFrozenExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_bytes = CONTRACT_PATH.read_bytes()
        cls.contract = json.loads(cls.contract_bytes)

    def test_contract_is_the_exact_frozen_before_fit_document(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.contract_bytes).hexdigest(),
            EXPECTED_CONTRACT_SHA256,
        )
        self.assertEqual(set(self.contract), EXPECTED_TOP_LEVEL_KEYS)
        self.assertEqual(self.contract["schema_version"], 1)
        self.assertEqual(self.contract["phase"], "phase_05_source_robustness")
        self.assertEqual(
            self.contract["experiment_version"], "nested_source_robust_heads_v1"
        )
        self.assertEqual(self.contract["status"], "frozen_before_fit")
        self.assertEqual(self.contract["seed"], 20260719)

    def test_scope_is_development_only_and_has_five_complete_outer_folds(self) -> None:
        scope = self.contract["scope"]
        self.assertEqual(
            scope["evaluation"],
            "development_only_selection_aware_nested_source_atomic_cv",
        )
        self.assertEqual(scope["development_image_count"], 1693)
        self.assertEqual(scope["outer_folds"], ["0", "1", "2", "3", "4"])
        self.assertEqual(scope["sealed_test_access"], "forbidden")
        self.assertEqual(scope["probabilities"], "uncalibrated_ranking_scores_only")

        rows = read_csv(STEP_ROOT / "splits" / "development.csv")
        self.assertEqual(len(rows), 1693)
        self.assertEqual(
            Counter(row["cv_fold"] for row in rows),
            Counter({"0": 500, "1": 244, "2": 270, "3": 342, "4": 337}),
        )
        self.assertEqual(
            {row["production_split"] for row in rows}, {"train", "validation"}
        )
        group_folds: Dict[str, Set[str]] = defaultdict(set)
        for row in rows:
            group_folds[row["source_atomic_split_group_id"]].add(row["cv_fold"])
        self.assertTrue(group_folds)
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))

    def test_every_canonical_input_path_and_hash_is_pinned(self) -> None:
        inputs = self.contract["input_contract"]
        expected_paths = {
            "development_csv": "step_02/splits/development.csv",
            "split_audit": "step_02/splits/split_audit.json",
            "phase_4_metrics": "step_02/outputs/phase_4_pretrained/metrics.json",
        }
        for key, relative_path in expected_paths.items():
            self.assertEqual(inputs[key], relative_path)
            resolved = resolve_recorded_path(relative_path)
            self.assertTrue(resolved.is_file())
            self.assertEqual(sha256_file(resolved), inputs[f"{key}_sha256"])

        audit = json.loads((STEP_ROOT / "splits" / "split_audit.json").read_text())
        self.assertEqual(inputs["split_version"], "step02_source_atomic_v3")
        self.assertEqual(inputs["split_seed"], 20260719)
        self.assertEqual(
            inputs["split_assignment_fingerprint_sha256"],
            audit["assignment_fingerprint_sha256"],
        )
        self.assertEqual(
            inputs["data_fingerprint_sha256"],
            "39c195336191e36a9a9a2c1cd8edec5160a68fe99366910b4d09c47e1c20024d",
        )
        data_digest = hashlib.sha256()
        for row in read_csv(STEP_ROOT / "splits" / "development.csv"):
            data_digest.update(
                f"{row['image_id']}\x1f{row['content_sha256']}\n".encode("utf-8")
            )
        self.assertEqual(data_digest.hexdigest(), inputs["data_fingerprint_sha256"])

    def test_cache_and_array_allowlist_is_exact_and_content_addressed(self) -> None:
        self.assertEqual(self.contract["cache_allowlist"], EXPECTED_CACHE_SPECS)
        for cache in self.contract["cache_allowlist"].values():
            self.assertNotIn("*", cache["path"])
            self.assertNotIn("?", cache["path"])
            self.assertEqual(Path(cache["path"]).suffix, ".npz")
            cache_path = REPO_ROOT / cache["path"]
            self.assertTrue(cache_path.is_file())
            self.assertEqual(sha256_file(cache_path), cache["file_sha256"])
            for array in cache["arrays"].values():
                self.assertEqual(array["shape"][0], 1693)
                self.assertEqual(array["dtype"], "float32")
                self.assertRegex(array["sha256"], r"^[0-9a-f]{64}$")

    def test_candidate_count_references_and_selection_lifecycle_are_declared(self) -> None:
        families = self.contract["base_feature_families"]
        grid = self.contract["base_grid"]
        expected_count = (
            len(families)
            * len(grid["c_values"])
            * len(grid["source_group_exponents"])
        )
        self.assertEqual(expected_count, 12)
        self.assertEqual(grid["configuration_count"], expected_count)
        self.assertEqual(grid["c_values"], [10.0, 100.0])
        self.assertEqual(grid["source_group_exponents"], [0.0, 0.5])
        self.assertEqual(
            grid["exact_phase_4_reference"],
            {
                "feature_family": "global_cls",
                "c_value": 10.0,
                "source_group_exponent": 0.0,
            },
        )
        self.assertEqual(
            self.contract["source_weighting"]["group_field"],
            "source_atomic_split_group_id",
        )
        self.assertEqual(
            self.contract["ceramic_specialist"][
                "versions_compared_after_base_selection"
            ],
            ["none", "three_way_conditional"],
        )

        allowed_blocks = {
            f"{cache_name}.{array_name}"
            for cache_name, cache in self.contract["cache_allowlist"].items()
            for array_name in cache["arrays"]
        }
        referenced_blocks = {
            block for family in families for block in family["blocks"]
        }
        self.assertTrue(referenced_blocks)
        self.assertFalse(referenced_blocks - allowed_blocks)

        nested = self.contract["nested_evaluation"]
        for key in (
            "outer_rule",
            "inner_rule",
            "outer_prediction_rule",
            "full_development_selection",
            "interpretation",
        ):
            self.assertIsInstance(nested[key], str)
            self.assertTrue(nested[key].strip())
        self.assertIn("never use it for candidate", nested["outer_rule"])
        self.assertIn("predict the outer fold exactly once", nested["outer_prediction_rule"])
        self.assertIn("freeze the Phase 6 candidate", nested["full_development_selection"])

    def test_output_contract_is_complete_but_does_not_require_outputs_to_exist(self) -> None:
        outputs = self.contract["output_contract"]
        self.assertEqual(
            outputs["directory"], "step_02/outputs/phase_5_source_robustness"
        )
        self.assertEqual(set(outputs["required"]), EXPECTED_OUTPUTS)
        self.assertEqual(len(outputs["required"]), len(EXPECTED_OUTPUTS))
        self.assertFalse(any("test" in name.lower() for name in outputs["required"]))


class PhaseFiveSourceGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources: Dict[Path, str] = {}
        cls.trees: Dict[Path, ast.AST] = {}
        for path in sorted(SOURCE_ROOT.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            cls.sources[path] = source
            cls.trees[path] = ast.parse(source, filename=str(path))

    def test_development_modules_expose_no_sealed_test_path_or_loader(self) -> None:
        forbidden_path_literals = {
            "test.csv",
            "splits/test.csv",
            "step_02/splits/test.csv",
            "test_predictions.csv",
        }
        forbidden_api_names = {
            "load_test",
            "load_test_contract",
            "load_test_manifest",
            "load_sealed_test",
            "test_manifest_path",
            "sealed_test_path",
        }
        for path, tree in self.trees.items():
            with self.subTest(path=path.name):
                strings = {value.replace("\\", "/").lower() for value in literal_strings(tree)}
                self.assertFalse(strings & forbidden_path_literals)
                names = {
                    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                }
                attributes = {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                }
                definitions = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self.assertFalse((names | attributes | definitions) & forbidden_api_names)

    def test_no_source_module_discovers_embedding_caches(self) -> None:
        for path, tree in self.trees.items():
            with self.subTest(path=path.name):
                self.assertFalse(
                    called_attribute_names(tree) & {"glob", "rglob", "iglob"}
                )
                imported_modules = {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                self.assertNotIn("glob", imported_modules)

    def test_reusable_modules_do_not_import_historical_runners(self) -> None:
        for path, tree in self.trees.items():
            with self.subTest(path=path.name):
                imported = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                self.assertFalse(
                    any("run_pretrained_embeddings" in name for name in imported)
                )

    def test_robustness_module_is_array_only_and_thread_limited(self) -> None:
        path = SOURCE_ROOT / "robustness.py"
        tree = self.trees[path]
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertFalse(imported_roots & {"csv", "glob", "json", "os", "pathlib", "pickle"})
        self.assertIn("threadpoolctl", imported_roots)
        self.assertIn("threadpool_limits", self.sources[path])
        self.assertIn("limits=1", self.sources[path])


class PhaseFiveRunnerStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RUNNER_PATH.is_file():
            raise AssertionError(f"Phase 5 runner is missing: {RUNNER_PATH}")
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(RUNNER_PATH))
        cls.strings = set(literal_strings(cls.tree))
        cls.function_names = {
            node.name
            for node in ast.walk(cls.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_runner_pins_and_actively_checks_the_exact_frozen_contract(self) -> None:
        assignments: Dict[str, Any] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (TypeError, ValueError):
                        pass
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                try:
                    assignments[node.target.id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    pass

        self.assertEqual(
            assignments["EXPECTED_CONTRACT_SHA256"], EXPECTED_CONTRACT_SHA256
        )
        self.assertIn(
            "phases/phase_05_source_robustness/experiment_contract.json",
            self.strings,
        )
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        loader = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_frozen_contract"
        )
        loader_source = ast.get_source_segment(self.source, loader) or ""
        self.assertIn("actual_sha256 != EXPECTED_CONTRACT_SHA256", loader_source)
        self.assertIn('scope.get("sealed_test_access") != "forbidden"', loader_source)

    def test_runner_uses_only_development_loaders_and_has_no_sealed_path_api(self) -> None:
        self.assertIn("load_development_contract", self.source)
        self.assertIn("load_allowlisted_embeddings", self.source)
        forbidden_api_names = {
            "load_test",
            "load_test_contract",
            "load_test_manifest",
            "load_sealed_test",
            "test_manifest_path",
            "sealed_test_path",
        }
        used_names = {
            node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
        }
        used_names.update(
            node.attr for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)
        )
        used_names.update(self.function_names)
        self.assertFalse(used_names & forbidden_api_names)

        normalized_strings = {
            value.replace("\\", "/").lower() for value in self.strings
        }
        self.assertFalse(
            normalized_strings
            & {
                "test.csv",
                "splits/test.csv",
                "step_02/splits/test.csv",
                "test_predictions.csv",
            }
        )
        self.assertIn('"sealed_test_evaluated": False', self.source)

    def test_runner_contains_the_nested_and_full_selection_lifecycle(self) -> None:
        required_functions = {
            "build_base_configurations",
            "evaluate_base_oof",
            "evaluate_specialist_oof",
            "select_search_candidates",
            "evaluate_promotion_gates",
            "main",
        }
        self.assertFalse(required_functions - self.function_names)
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(self.source, main) or ""
        for concept in (
            "outer_fold",
            "inner_folds",
            "nested_probabilities",
            "full_base_results",
            "full_specialist_results",
            "stability_passed",
            "evaluate_promotion_gates",
        ):
            self.assertIn(concept, main_source)
        self.assertIn("Nested OOF predictions are incomplete", self.strings)
        self.assertIn("selection_conditional_full_development_oof", self.strings)

    def test_runner_names_every_required_output_without_needing_outputs_present(self) -> None:
        self.assertFalse(EXPECTED_OUTPUTS - self.strings)
        output_literals = {
            value
            for value in self.strings
            if value.endswith((".csv", ".json", ".png"))
        }
        self.assertFalse(
            {
                "test_predictions.csv",
                "test_metrics.json",
                "sealed_test_predictions.csv",
            }
            & output_literals
        )

    def test_runner_limits_each_fit_to_one_thread(self) -> None:
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "threadpool_limits"
        ]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            limits = [keyword.value for keyword in call.keywords if keyword.arg == "limits"]
            self.assertEqual(len(limits), 1)
            self.assertEqual(ast.literal_eval(limits[0]), 1)
        self.assertIn("blas_threads_per_fit", self.strings)

    def test_promotion_fold_variability_uses_sample_standard_deviation(self) -> None:
        promotion = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_promotion_gates"
        )
        standard_deviations = [
            node
            for node in ast.walk(promotion)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "std"
        ]
        self.assertEqual(len(standard_deviations), 1)
        call = standard_deviations[0]
        receiver = ast.get_source_segment(self.source, call.func.value) or ""
        self.assertEqual(receiver, "fold_accuracies")
        ddof = [keyword.value for keyword in call.keywords if keyword.arg == "ddof"]
        self.assertEqual(len(ddof), 1)
        self.assertEqual(ast.literal_eval(ddof[0]), 1)

    def test_metrics_fingerprint_every_local_phase5_implementation_input(self) -> None:
        provenance = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "code_provenance"
        )
        provenance_source = ast.get_source_segment(self.source, provenance) or ""
        for relative_path in (
            "step_02/scripts/run_phase5_source_robustness.py",
            "step_02/src/ornament_classifier/robustness.py",
            "step_02/src/ornament_classifier/embeddings.py",
            "step_02/src/ornament_classifier/contracts.py",
            "step_02/requirements-phase5.txt",
        ):
            assert_recorded_file(self, relative_path)
            if relative_path.endswith("run_phase5_source_robustness.py"):
                self.assertIn("SCRIPT_PATH", provenance_source)
            else:
                self.assertIn(Path(relative_path).name, provenance_source)
        self.assertIn("digest = sha256_file(path)", provenance_source)
        self.assertIn("files[relative] = digest", provenance_source)
        self.assertIn("combined.update", provenance_source)
        self.assertIn('"files_sha256"', provenance_source)
        self.assertIn('"combined_sha256"', provenance_source)

        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(self.source, main) or ""
        self.assertIn('"script_sha256": sha256_file(SCRIPT_PATH)', main_source)
        self.assertIn('"code_provenance": code_provenance()', main_source)
        self.assertIn(
            '"experiment_contract_sha256": EXPECTED_CONTRACT_SHA256', main_source
        )

    def test_json_writer_is_deterministic_and_rejects_nonfinite_numbers(self) -> None:
        writer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_write_json"
        )
        dump_calls = [
            node
            for node in ast.walk(writer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr == "dump"
        ]
        self.assertEqual(len(dump_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in dump_calls[0].keywords}
        self.assertIn("sort_keys", keywords)
        self.assertIn("allow_nan", keywords)
        self.assertIs(ast.literal_eval(keywords["sort_keys"]), True)
        self.assertIs(ast.literal_eval(keywords["allow_nan"]), False)

    def test_every_oof_and_outer_fit_asserts_source_group_disjointness(self) -> None:
        helper = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_assert_source_disjoint_partitions"
        )
        helper_source = ast.get_source_segment(self.source, helper) or ""
        self.assertIn("np.intersect1d(train_indices, validation_indices)", helper_source)
        self.assertIn("train_groups & validation_groups", helper_source)
        self.assertIn("Source-atomic groups cross", helper_source)

        required_boundaries = {
            "evaluate_base_oof": "_fit_base_model(",
            "evaluate_specialist_oof": "fit_ceramic_specialist(",
            "_fit_outer_and_predict": "_fit_base_model(",
        }
        for function_name, first_fit in required_boundaries.items():
            function = next(
                node
                for node in self.tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            function_source = ast.get_source_segment(self.source, function) or ""
            with self.subTest(function=function_name):
                assertion = function_source.index("_assert_source_disjoint_partitions(")
                fitting = function_source.index(first_fit)
                self.assertLess(assertion, fitting)
                calls = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_assert_source_disjoint_partitions"
                ]
                self.assertEqual(len(calls), 1)
                argument_names = [
                    argument.id
                    for argument in calls[0].args[:3]
                    if isinstance(argument, ast.Name)
                ]
                self.assertEqual(
                    argument_names,
                    ["train_indices", "validation_indices", "groups"],
                )

    def test_full_reference_checks_phase4_metrics_and_argmax_predictions(self) -> None:
        assignments: Dict[str, Any] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (TypeError, ValueError):
                        pass
        self.assertEqual(
            assignments["EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256"],
            EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256,
        )

        phase4_prediction_digest = hashlib.sha256()
        for row in read_csv(
            STEP_ROOT / "outputs" / "phase_4_pretrained" / "oof_predictions.csv"
        ):
            phase4_prediction_digest.update(
                f"{row['image_id']}\x1f{row['predicted_class']}\n".encode("utf-8")
            )
        self.assertEqual(
            phase4_prediction_digest.hexdigest(),
            EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256,
        )

        validator = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_phase4_reference_metrics"
        )
        validator_source = ast.get_source_segment(self.source, validator) or ""
        for required in (
            '"phase_4_metrics"',
            '"phase_4_metrics_sha256"',
            "sha256_file(path)",
            'phase4.get("aggregate_oof_metrics")',
            'phase4.get("per_class_metrics")',
            'phase4.get("selected_fold_metrics")',
            "_predict_labels(reference_result.probabilities, classes)",
            "records[int(index)].image_id",
            "observed_prediction_sha256 != EXPECTED_PHASE4_REFERENCE_PREDICTION_SHA256",
            '"probability_identity_required": False',
        ):
            self.assertIn(required, validator_source)
        self.assertNotIn("oof_predictions.csv", validator_source)
        self.assertNotIn(".tobytes(", validator_source)
        self.assertNotIn("np.array_equal", validator_source)

        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(self.source, main) or ""
        reference_assignment = main_source.index(
            "full_reference = full_base_by_id[reference_configuration.configuration_id]"
        )
        validation = main_source.index("validate_phase4_reference_metrics(")
        selection = main_source.index("full_base_selection = select_search_candidates(")
        self.assertLess(reference_assignment, validation)
        self.assertLess(validation, selection)
        self.assertIn(
            '"exact_phase4_reference_validation": phase4_reference_validation',
            main_source,
        )

    def test_selection_rows_marks_only_the_exact_phase4_reference(self) -> None:
        selection_rows = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_selection_rows"
        )
        selection_source = ast.get_source_segment(self.source, selection_rows) or ""
        self.assertNotIn("startswith(reference_id)", selection_source)
        prefix_reference_calls = [
            node
            for node in ast.walk(selection_rows)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
            and any(
                isinstance(argument, ast.Name) and argument.id == "reference_id"
                for argument in node.args
            )
        ]
        self.assertFalse(prefix_reference_calls)

        exact_reference_values = []
        for node in ast.walk(selection_rows):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "is_exact_reference"
                ):
                    exact_reference_values.append(value)
        self.assertEqual(len(exact_reference_values), 1)
        value = exact_reference_values[0]
        self.assertIsInstance(value, ast.Compare)
        self.assertEqual(len(value.ops), 1)
        self.assertIsInstance(value.ops[0], ast.In)
        self.assertEqual(len(value.comparators), 1)
        allowed = value.comparators[0]
        self.assertIsInstance(allowed, ast.Tuple)
        self.assertEqual(len(allowed.elts), 2)
        self.assertIsInstance(allowed.elts[0], ast.Name)
        self.assertEqual(allowed.elts[0].id, "reference_id")
        specialist_none = allowed.elts[1]
        self.assertIsInstance(specialist_none, ast.BinOp)
        self.assertIsInstance(specialist_none.op, ast.Add)
        self.assertIsInstance(specialist_none.left, ast.Name)
        self.assertEqual(specialist_none.left.id, "reference_id")
        self.assertIsInstance(specialist_none.right, ast.Constant)
        self.assertEqual(specialist_none.right.value, "__specialist_none")

    def test_runner_never_scans_for_embedding_caches(self) -> None:
        self.assertFalse(
            called_attribute_names(self.tree) & {"glob", "rglob", "iglob"}
        )
        imported_modules = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        self.assertNotIn("glob", imported_modules)
        self.assertNotIn("run_pretrained_embeddings", "\n".join(imported_modules))
        self.assertNotIn("np.load", self.source)


if __name__ == "__main__":
    unittest.main()

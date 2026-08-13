from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"
PHASE_ROOT = STEP_ROOT / "phases"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import ornament_classifier  # noqa: E402
from ornament_classifier import ProjectPaths, load_development_contract  # noqa: E402


EXPECTED_PHASES = (
    "phase_01_data_truth",
    "phase_02_evaluation_protocol",
    "phase_03_classical_baseline",
    "phase_04_pretrained_screen",
    "phase_05_source_robustness",
)
CANONICAL_SHARED_DIRECTORIES = (
    "data",
    "metadata",
    "review",
    "splits",
    "scripts",
    "tests",
    "outputs",
)
LIGHTWEIGHT_PHASE_SUFFIXES = frozenset((".md", ".json", ".yaml", ".yml", ".toml"))
BULK_ARTIFACT_SUFFIXES = frozenset(
    (
        ".csv",
        ".tsv",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
        ".xlsx",
        ".ipynb",
    )
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class PhaseStructureTests(unittest.TestCase):
    def test_expected_phase_indexes_exist(self) -> None:
        self.assertTrue(PHASE_ROOT.is_dir())
        for name in EXPECTED_PHASES:
            phase = PHASE_ROOT / name
            self.assertTrue(phase.is_dir(), name)
            self.assertFalse(phase.is_symlink(), name)
            readme = phase / "README.md"
            self.assertTrue(readme.is_file(), str(readme))
            self.assertGreater(len(readme.read_text(encoding="utf-8").strip()), 0, name)

    def test_phase_registry_matches_navigation_and_sealed_status(self) -> None:
        registry_path = PHASE_ROOT / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = registry["phases"]

        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(
            [entry["navigation"].split("/", 1)[0] for entry in entries],
            list(EXPECTED_PHASES),
        )
        self.assertTrue(all(entry["status"] == "complete" for entry in entries))
        self.assertEqual(entries[-1]["outcome"], "v1_v2_v3_v4_v5_candidates_rejected")

        for entry in entries:
            self.assertTrue((PHASE_ROOT / entry["navigation"]).is_file())
            for key in ("findings", "entrypoint"):
                target = entry[key]
                if target is not None:
                    self.assertTrue((PHASE_ROOT / target).resolve().is_file())

        phase_five = entries[-1]
        phase_five_outputs = (PHASE_ROOT / phase_five["outputs"]).resolve()
        self.assertTrue(phase_five_outputs.is_dir())
        self.assertEqual(
            {path.name for path in phase_five_outputs.iterdir() if path.is_file()},
            set(
                json.loads(
                    (PHASE_ROOT / phase_five["experiment_contract"]).read_text(
                        encoding="utf-8"
                    )
                )["output_contract"]["required"]
            ),
        )

        previous_iterations = phase_five["previous_iterations"]
        self.assertEqual(
            [iteration["version"] for iteration in previous_iterations],
            ["v4", "v3", "v2", "v1"],
        )
        for previous in previous_iterations:
            self.assertEqual(previous["outcome"], "candidate_rejected")
            for key in ("findings", "entrypoint", "experiment_contract"):
                self.assertTrue((PHASE_ROOT / previous[key]).resolve().is_file())
            previous_outputs = (PHASE_ROOT / previous["outputs"]).resolve()
            self.assertTrue(previous_outputs.is_dir())

        sealed = registry["sealed_test"]
        sealed_manifest = (PHASE_ROOT / sealed["manifest"]).resolve()
        self.assertEqual(sealed["status"], "sealed_unevaluated")
        self.assertEqual(len(read_csv(sealed_manifest)), sealed["image_count"])

    def test_canonical_shared_directories_remain_at_step_root(self) -> None:
        for name in CANONICAL_SHARED_DIRECTORIES:
            canonical = STEP_ROOT / name
            self.assertTrue(canonical.is_dir(), str(canonical))
            self.assertFalse(canonical.is_symlink(), str(canonical))

    def test_phase_indexes_do_not_copy_data_models_or_results(self) -> None:
        canonical_artifact_hashes = set()
        for directory_name in ("metadata", "review", "splits", "outputs"):
            for candidate in (STEP_ROOT / directory_name).rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() != ".md":
                    canonical_artifact_hashes.add(sha256_file(candidate))

        for phase_name in EXPECTED_PHASES:
            phase = PHASE_ROOT / phase_name
            for child in phase.iterdir():
                self.assertTrue(
                    child.is_file(),
                    f"Phase index directories must stay flat; found {child}",
                )
                self.assertFalse(child.is_symlink(), str(child))
                suffix = child.suffix.lower()
                self.assertIn(suffix, LIGHTWEIGHT_PHASE_SUFFIXES, str(child))
                self.assertNotIn(suffix, BULK_ARTIFACT_SUFFIXES, str(child))
                self.assertLessEqual(
                    child.stat().st_size,
                    256 * 1024,
                    f"Phase index file is unexpectedly large: {child}",
                )
                if suffix != ".md":
                    self.assertNotIn(
                        sha256_file(child),
                        canonical_artifact_hashes,
                        f"Phase file duplicates a canonical artifact: {child}",
                    )

    def test_reusable_package_has_no_public_sealed_evaluation_api(self) -> None:
        public_names = set(ornament_classifier.__all__)
        forbidden = {
            name
            for name in public_names
            if "sealed" in name.lower() or "test" in name.lower()
        }
        self.assertEqual(forbidden, set())

        paths = ProjectPaths.discover()
        path_api = {name for name in dir(paths) if not name.startswith("_")}
        forbidden_paths = {
            name
            for name in path_api
            if "sealed" in name.lower() or "test" in name.lower()
        }
        self.assertEqual(forbidden_paths, set())

        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SRC_ROOT / "ornament_classifier").glob("*.py")
        )
        self.assertNotIn('"test.csv"', source)
        self.assertNotIn("'test.csv'", source)

    def test_development_contract_excludes_production_test_rows(self) -> None:
        contract = load_development_contract()
        development_ids = {record.image_id for record in contract.records}
        sealed_ids = {
            row["image_id"] for row in read_csv(STEP_ROOT / "splits" / "test.csv")
        }
        train_ids = {
            row["image_id"] for row in read_csv(STEP_ROOT / "splits" / "train.csv")
        }
        validation_ids = {
            row["image_id"]
            for row in read_csv(STEP_ROOT / "splits" / "validation.csv")
        }

        self.assertFalse(development_ids & sealed_ids)
        self.assertEqual(development_ids, train_ids | validation_ids)
        self.assertFalse(train_ids & validation_ids)
        self.assertTrue(
            all(
                record.production_split in {"train", "validation"}
                for record in contract.records
            )
        )
        self.assertTrue(
            all(
                record.source_atomic_split_group_id == record.split_group_id
                for record in contract.records
            )
        )
        self.assertTrue(all(record.object_type for record in contract.records))
        self.assertTrue(all(record.motif_visibility for record in contract.records))
        expected_count = sum(
            contract.audit["production_split_counts"][destination]
            for destination in ("train", "validation")
        )
        self.assertEqual(len(contract.records), expected_count)


if __name__ == "__main__":
    unittest.main()

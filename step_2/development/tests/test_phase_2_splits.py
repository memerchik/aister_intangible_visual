from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SCRIPTS_ROOT = STEP_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_splits


SPLIT_ROOT = STEP_ROOT / "splits"
SIMILARITY_REVIEW_ROOT = STEP_ROOT / "review" / "pretrained_similarity"
ADJUDICATION_PATH = SIMILARITY_REVIEW_ROOT / "adjudication.json"
GLOBAL_SOURCE_ADJUDICATION_PATH = (
    STEP_ROOT / "review" / "global_source_cohorts" / "adjudication.json"
)
SOURCE_ATOMIC_SPLIT_VERSION = "step02_source_atomic_v3"
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_LABELS = {
    "01_opishnyan_ceramics",
    "02_ornek",
    "03_bubnivka_ceramics",
    "04_petrykivka_painting",
    "05_kosiv_ceramics",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def crossings(rows: list[dict[str, str]], key: str, destination: str) -> list[str]:
    destinations: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        destinations[row[key]].add(row[destination])
    return sorted(value for value, assigned in destinations.items() if len(assigned) > 1)


class PhaseTwoSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_csv(SPLIT_ROOT / "split_manifest.csv")
        cls.included = [row for row in cls.manifest if row["inclusion_status"] == "included"]
        cls.excluded = [row for row in cls.manifest if row["inclusion_status"] == "excluded"]
        cls.development = [row for row in cls.included if row["production_split"] != "test"]
        with (SPLIT_ROOT / "split_audit.json").open(encoding="utf-8") as handle:
            cls.audit = json.load(handle)

    def require_resealed_source_atomic_v3(self) -> None:
        if not GLOBAL_SOURCE_ADJUDICATION_PATH.is_file():
            self.skipTest("global source-cohort adjudication is still pending")
        if self.audit.get("split_version") != SOURCE_ATOMIC_SPLIT_VERSION:
            self.skipTest("canonical split has not yet been resealed as source-atomic v3")

    def test_inventory_and_exclusion_decisions_are_complete(self) -> None:
        self.assertEqual(len(self.manifest), 2055)
        self.assertEqual(len(self.included), 2023)
        self.assertEqual(len(self.excluded), 32)
        self.assertEqual(len({row["image_id"] for row in self.manifest}), 2055)
        self.assertEqual(
            Counter(row["exclusion_reason"] for row in self.excluded),
            {
                "exact_or_reencoded_duplicate": 28,
                "no_usable_visible_motif": 4,
            },
        )
        self.assertTrue(all(not row["production_split"] and not row["cv_fold"] for row in self.excluded))

    def test_split_files_exactly_partition_included_images(self) -> None:
        observed: dict[str, set[str]] = {}
        for split in sorted(EXPECTED_SPLITS):
            rows = read_csv(SPLIT_ROOT / f"{split}.csv")
            observed[split] = {row["image_id"] for row in rows}
            self.assertTrue(all(row["production_split"] == split for row in rows))
        self.assertFalse(observed["train"] & observed["validation"])
        self.assertFalse(observed["train"] & observed["test"])
        self.assertFalse(observed["validation"] & observed["test"])
        self.assertEqual(set().union(*observed.values()), {row["image_id"] for row in self.included})

    def test_production_split_has_no_group_or_hash_leakage(self) -> None:
        keys = ["split_group_id", "confirmed_object_group_id", "content_sha256"]
        if self.audit.get("split_version") == SOURCE_ATOMIC_SPLIT_VERSION:
            keys.extend(("semantic_split_group_id", "source_atomic_split_group_id"))
        for key in keys:
            self.assertEqual(crossings(self.included, key, "production_split"), [])

    def test_cv_folds_have_no_group_or_hash_leakage(self) -> None:
        self.assertTrue(all(row["cv_fold"] == "" for row in self.included if row["production_split"] == "test"))
        self.assertEqual({row["cv_fold"] for row in self.development}, {"0", "1", "2", "3", "4"})
        keys = ["split_group_id", "confirmed_object_group_id", "content_sha256"]
        if self.audit.get("split_version") == SOURCE_ATOMIC_SPLIT_VERSION:
            keys.extend(("semantic_split_group_id", "source_atomic_split_group_id"))
        for key in keys:
            self.assertEqual(crossings(self.development, key, "cv_fold"), [])

    def test_every_split_and_cv_fold_contains_every_class(self) -> None:
        for split in EXPECTED_SPLITS:
            labels = {row["ornament_label"] for row in self.included if row["production_split"] == split}
            self.assertEqual(labels, EXPECTED_LABELS)
        for fold in {"0", "1", "2", "3", "4"}:
            labels = {row["ornament_label"] for row in self.development if row["cv_fold"] == fold}
            self.assertEqual(labels, EXPECTED_LABELS)

    def test_class_targets_and_audit_match_rows(self) -> None:
        class_counts = Counter(row["ornament_label"] for row in self.included)
        split_counts = Counter(row["production_split"] for row in self.included)
        self.assertEqual(dict(sorted(class_counts.items())), self.audit["class_counts"])
        self.assertEqual(
            {split: split_counts[split] for split in ("train", "validation", "test")},
            self.audit["production_split_counts"],
        )
        if self.audit.get("split_version") == SOURCE_ATOMIC_SPLIT_VERSION:
            allocation = self.audit["production_allocation"]
            self.assertEqual(allocation["observed_by_class"], self.audit["class_split_counts"])
            self.assertEqual(
                allocation["observed_total"], self.audit["production_split_counts"]
            )
            self.assertTrue(allocation["every_class_in_every_destination"])
            for label in class_counts:
                for split in EXPECTED_SPLITS:
                    self.assertEqual(
                        allocation["deviation_by_class"][label][split],
                        allocation["observed_by_class"][label][split]
                        - allocation["target_by_class"][label][split],
                    )
        else:
            for label, total in class_counts.items():
                observed = Counter(
                    row["production_split"]
                    for row in self.included
                    if row["ornament_label"] == label
                )
                self.assertLessEqual(abs(observed["train"] / total - 0.70), 1 / total)
                self.assertLessEqual(abs(observed["validation"] / total - 0.15), 1 / total)
                self.assertLessEqual(abs(observed["test"] / total - 0.15), 1 / total)

    def test_build_is_byte_reproducible(self) -> None:
        self.require_resealed_source_atomic_v3()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            subprocess.run(
                [
                    "python3",
                    str(STEP_ROOT / "scripts" / "build_splits.py"),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "20260719",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            for name in (
                "split_manifest.csv",
                "train.csv",
                "validation.csv",
                "test.csv",
                "development.csv",
                "split_summary.csv",
                "split_audit.json",
            ):
                expected = hashlib.sha256((SPLIT_ROOT / name).read_bytes()).hexdigest()
                rebuilt = hashlib.sha256((output / name).read_bytes()).hexdigest()
                self.assertEqual(rebuilt, expected, name)

    def test_builder_refuses_to_reseal_without_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing_adjudication.json"
            result = subprocess.run(
                [
                    "python3",
                    str(STEP_ROOT / "scripts" / "build_splits.py"),
                    "--output-dir",
                    str(Path(temporary) / "output"),
                    "--adjudication-json",
                    str(missing),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot be built before similarity adjudication", result.stderr)

    def test_builder_refuses_to_reseal_without_global_source_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing_source_adjudication.json"
            result = subprocess.run(
                [
                    "python3",
                    str(STEP_ROOT / "scripts" / "build_splits.py"),
                    "--output-dir",
                    str(Path(temporary) / "output"),
                    "--global-source-adjudication",
                    str(missing),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "cannot be built before global source-cohort adjudication",
                result.stderr,
            )

    def test_inventory_projection_fingerprint_has_documented_byte_contract(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "image_id": "img_b",
                "ornament_label": "label_2",
                "inclusion_status": "excluded",
                "semantic_split_group_id": "sem_b",
            },
            {
                "image_id": "img_a",
                "ornament_label": "label_1",
                "inclusion_status": "included",
                "semantic_split_group_id": "sem_a",
            },
        ]
        expected_bytes = (
            b"img_a\x1flabel_1\x1fincluded\x1fsem_a\n"
            b"img_b\x1flabel_2\x1fexcluded\x1fsem_b\n"
        )
        self.assertEqual(
            build_splits.source_inventory_projection_fingerprint(rows),
            hashlib.sha256(expected_bytes).hexdigest(),
        )

    def test_global_source_union_propagates_complete_semantic_groups(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "image_id": "image_a",
                "ornament_label": "label",
                "inclusion_status": "included",
                "semantic_split_group_id": "semantic_a",
            },
            {
                "image_id": "image_a_related",
                "ornament_label": "label",
                "inclusion_status": "included",
                "semantic_split_group_id": "semantic_a",
            },
            {
                "image_id": "image_b",
                "ornament_label": "label",
                "inclusion_status": "included",
                "semantic_split_group_id": "semantic_b",
            },
            {
                "image_id": "image_c",
                "ornament_label": "label",
                "inclusion_status": "included",
                "semantic_split_group_id": "semantic_c",
            },
        ]
        fingerprint = build_splits.source_inventory_projection_fingerprint(rows)
        payload = {
            "review_version": "global_source_cohorts_v1",
            "review_status": "complete",
            "inventory_projection_fingerprint_sha256": fingerprint,
            "cohorts": [
                {
                    "cohort_id": "cohort_001",
                    "ornament_label": "label",
                    "relationship": "same_acquisition_source",
                    "image_ids": ["image_a", "image_b"],
                    "evidence": "Matching acquisition setup.",
                    "notes": "",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adjudication.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            source = build_splits.load_global_source_adjudication(rows, path)
        build_splits.apply_global_source_adjudication(rows, source)
        by_id = {str(row["image_id"]): row for row in rows}
        merged_id = by_id["image_a"]["source_atomic_split_group_id"]
        self.assertEqual(merged_id, by_id["image_b"]["source_atomic_split_group_id"])
        self.assertEqual(
            merged_id, by_id["image_a_related"]["source_atomic_split_group_id"]
        )
        self.assertNotEqual(merged_id, by_id["image_c"]["source_atomic_split_group_id"])
        self.assertEqual(by_id["image_a"]["global_source_cohort_id"], "cohort_001")
        self.assertEqual(by_id["image_b"]["global_source_cohort_id"], "cohort_001")
        self.assertEqual(by_id["image_a_related"]["global_source_cohort_id"], "")
        self.assertEqual(
            by_id["image_a_related"]["source_atomic_cohort_ids"], "cohort_001"
        )
        self.assertEqual(
            by_id["image_a_related"]["pre_source_cohort_split_group_id"],
            "semantic_a",
        )

    def test_approximate_allocator_handles_group_larger_than_fold_quota(self) -> None:
        sizes = (23, 17, 11, 7, 5, 4, 3, 2, 1, 1)
        groups = [
            build_splits.AllocationGroup(
                group_id=f"group_{index}",
                label="label",
                image_ids=tuple(f"image_{index}_{member}" for member in range(size)),
            )
            for index, size in enumerate(sizes)
        ]
        destinations = tuple(str(index) for index in range(5))
        ratios = {destination: 0.2 for destination in destinations}
        assignment = build_splits.allocate_groups_approximately(
            groups,
            ratios,
            destinations,
            seed=20260719,
            namespace="unit_test",
        )
        rebuilt = build_splits.allocate_groups_approximately(
            list(reversed(groups)),
            ratios,
            destinations,
            seed=20260719,
            namespace="unit_test",
        )
        self.assertEqual(assignment, rebuilt)
        self.assertEqual(set(assignment.values()), set(destinations))
        self.assertEqual(
            {assignment[f"group_{index}"] for index in range(5)},
            set(destinations),
        )
        self.assertGreater(sizes[0], round(sum(sizes) / len(destinations)))

    def test_semantic_union_propagates_existing_related_views(self) -> None:
        rows: list[dict[str, object]] = [
            {"image_id": "image_a", "split_group_id": "base_a"},
            {"image_id": "image_a_related", "split_group_id": "base_a"},
            {"image_id": "image_b", "split_group_id": "base_b"},
            {"image_id": "image_c", "split_group_id": "base_c"},
        ]
        semantic = build_splits.SemanticAdjudication(
            audit_fingerprint_sha256="a" * 64,
            adjudication_sha256="b" * 64,
            component_members={
                "component_merge": ("image_a", "image_b"),
                "component_distinct": ("image_c",),
            },
            component_decisions={
                "component_merge": "same_physical_object",
                "component_distinct": "distinct_unrelated",
            },
            merge_groups=(
                build_splits.SemanticMergeGroup(
                    component_id="component_merge",
                    group_id="merge_group",
                    relationship="same_physical_object",
                    image_ids=("image_a", "image_b"),
                ),
            ),
        )
        build_splits.apply_semantic_adjudication(rows, semantic)
        by_id = {str(row["image_id"]): row for row in rows}
        merged_id = by_id["image_a"]["semantic_split_group_id"]
        self.assertEqual(merged_id, by_id["image_b"]["semantic_split_group_id"])
        self.assertEqual(merged_id, by_id["image_a_related"]["semantic_split_group_id"])
        self.assertNotEqual(merged_id, by_id["image_c"]["semantic_split_group_id"])
        self.assertEqual(by_id["image_a_related"]["semantic_merge_group_ids"], "merge_group")
        self.assertEqual(by_id["image_a_related"]["semantic_review_component_ids"], "")

    def test_resealed_outputs_bind_source_and_semantic_adjudications_and_csv_hashes(self) -> None:
        self.require_resealed_source_atomic_v3()
        self.assertTrue(
            all(row["split_version"] == SOURCE_ATOMIC_SPLIT_VERSION for row in self.manifest)
        )
        self.assertTrue(
            all(
                row["split_group_id"] == row["source_atomic_split_group_id"]
                and row["pre_source_cohort_split_group_id"] == row["semantic_split_group_id"]
                and row["presemantic_split_group_id"]
                and row["semantic_similarity_audit_fingerprint_sha256"]
                and row["semantic_adjudication_file_sha256"]
                and row["global_source_inventory_projection_fingerprint_sha256"]
                and row["global_source_cohort_file_sha256"]
                for row in self.manifest
            )
        )
        self.assertEqual(self.audit["semantic_group_leakage_violations"], [])
        self.assertEqual(self.audit["cv_semantic_group_leakage_violations"], [])
        self.assertEqual(self.audit["source_atomic_group_leakage_violations"], [])
        self.assertEqual(self.audit["cv_source_atomic_group_leakage_violations"], [])
        self.assertEqual(
            self.audit["semantic_adjudication"]["reviewed_component_count"], 46
        )
        expected_hashes = self.audit["generated_csv_sha256"]
        self.assertIn("development.csv", expected_hashes)
        self.assertEqual(
            expected_hashes,
            {
                name: hashlib.sha256((SPLIT_ROOT / name).read_bytes()).hexdigest()
                for name in expected_hashes
            },
        )

    def test_global_source_cohorts_are_directly_and_propagatively_atomic(self) -> None:
        self.require_resealed_source_atomic_v3()
        payload = json.loads(
            GLOBAL_SOURCE_ADJUDICATION_PATH.read_text(encoding="utf-8")
        )
        by_id = {row["image_id"]: row for row in self.manifest}
        self.assertEqual(payload["review_status"], "complete")
        self.assertEqual(len(payload["cohorts"]), 28)
        self.assertEqual(
            self.audit["global_source_adjudication"]["adjudication_file_sha256"],
            hashlib.sha256(GLOBAL_SOURCE_ADJUDICATION_PATH.read_bytes()).hexdigest(),
        )
        for cohort in payload["cohorts"]:
            cohort_id = cohort["cohort_id"]
            direct_rows = [by_id[image_id] for image_id in cohort["image_ids"]]
            source_ids = {
                row["source_atomic_split_group_id"] for row in direct_rows
            }
            self.assertEqual(len(source_ids), 1, cohort_id)
            self.assertEqual(
                {row["production_split"] for row in direct_rows}, {direct_rows[0]["production_split"]}
            )
            self.assertTrue(
                all(row["global_source_cohort_id"] == cohort_id for row in direct_rows)
            )
            propagated = [
                row
                for row in self.manifest
                if row["source_atomic_split_group_id"] in source_ids
            ]
            self.assertTrue(
                all(cohort_id in row["source_atomic_cohort_ids"].split("|") for row in propagated),
                cohort_id,
            )

    def test_cv_approximate_imbalance_audit_matches_rows(self) -> None:
        self.require_resealed_source_atomic_v3()
        allocation = self.audit["development_cv_allocation"]
        observed_total = Counter(row["cv_fold"] for row in self.development)
        self.assertEqual(
            allocation["observed_total"],
            {str(index): observed_total[str(index)] for index in range(5)},
        )
        self.assertTrue(allocation["every_class_in_every_destination"])
        for label in EXPECTED_LABELS:
            for fold in map(str, range(5)):
                observed = sum(
                    row["ornament_label"] == label and row["cv_fold"] == fold
                    for row in self.development
                )
                self.assertEqual(allocation["observed_by_class"][label][fold], observed)
                self.assertEqual(
                    allocation["deviation_by_class"][label][fold],
                    observed - allocation["target_by_class"][label][fold],
                )

    def test_adjudicated_merge_groups_are_atomic_after_reseal(self) -> None:
        self.require_resealed_source_atomic_v3()
        adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
        by_id = {row["image_id"]: row for row in self.manifest}
        component_members = read_csv(SIMILARITY_REVIEW_ROOT / "component_members.csv")
        expected_components = {row["component_id"] for row in component_members}
        observed_components = {
            component["component_id"] for component in adjudication["components"]
        }
        self.assertEqual(observed_components, expected_components)
        for component in adjudication["components"]:
            for group in component["groups"]:
                semantic_ids = {
                    by_id[image_id]["semantic_split_group_id"]
                    for image_id in group["image_ids"]
                }
                production_splits = {
                    by_id[image_id]["production_split"]
                    for image_id in group["image_ids"]
                }
                self.assertEqual(len(semantic_ids), 1, group["group_id"])
                self.assertEqual(len(production_splits), 1, group["group_id"])
                member_base_groups = {
                    by_id[image_id]["presemantic_split_group_id"]
                    for image_id in group["image_ids"]
                }
                propagated_rows = [
                    row
                    for row in self.manifest
                    if row["presemantic_split_group_id"] in member_base_groups
                ]
                self.assertEqual(
                    {row["semantic_split_group_id"] for row in propagated_rows},
                    semantic_ids,
                    group["group_id"],
                )
                self.assertTrue(
                    all(
                        group["group_id"] in row["semantic_merge_group_ids"].split("|")
                        for row in propagated_rows
                    ),
                    group["group_id"],
                )


if __name__ == "__main__":
    unittest.main()

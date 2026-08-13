"""Integrity checks for the deterministic all-data similarity review queue."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
MANIFEST_PATH = STEP_ROOT / "splits" / "split_manifest.csv"
SPLIT_AUDIT_PATH = STEP_ROOT / "splits" / "split_audit.json"
OUTPUT_DIR = STEP_ROOT / "review" / "pretrained_similarity"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
PAIR_PATH = OUTPUT_DIR / "pair_candidates.csv"
COMPONENT_PATH = OUTPUT_DIR / "component_candidates.csv"
MEMBER_PATH = OUTPUT_DIR / "component_members.csv"
ADJUDICATION_PATH = OUTPUT_DIR / "adjudication.json"
ADJUDICATION_CSV_PATH = OUTPUT_DIR / "adjudication.csv"

# The similarity audit was run against the now-invalidated semantic-v1 manifest.
# Keep that historical input identity immutable even though the live manifest has
# subsequently been resealed first as semantic v2 and then as source-atomic v3.
HISTORICAL_V1_MANIFEST_SHA256 = (
    "2d7c32558e6b33d972aaf49f5a1911f956267676b7a0c3c8cf1ea2db1d9e22ad"
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


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PretrainedSimilarityAuditOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.pairs = read_csv(PAIR_PATH)
        cls.components = read_csv(COMPONENT_PATH)
        cls.members = read_csv(MEMBER_PATH)
        cls.manifest = read_csv(MANIFEST_PATH)
        cls.split_audit = json.loads(SPLIT_AUDIT_PATH.read_text(encoding="utf-8"))
        cls.adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
        cls.adjudication_csv = read_csv(ADJUDICATION_CSV_PATH)
        cls.included = {
            row["image_id"]: row
            for row in cls.manifest
            if row["inclusion_status"] == "included"
        }

    def test_expected_scope_and_review_only_status(self) -> None:
        self.assertEqual(self.audit["audit_version"], "step02_dinov3_similarity_v1")
        self.assertFalse(self.audit["auto_merge_performed"])
        self.assertEqual(
            self.audit["current_split_status"],
            "invalidated_pending_similarity_adjudication_and_reseal",
        )
        self.assertEqual(len(self.included), 2023)
        self.assertEqual(self.audit["input"]["included_image_count"], 2023)
        self.assertEqual(self.audit["summary"]["candidate_pair_count"], 82)
        self.assertEqual(self.audit["summary"]["candidate_component_count"], 46)
        self.assertEqual(self.audit["summary"]["candidate_image_count"], 118)
        self.assertEqual(self.audit["summary"]["touches_current_test_pair_count"], 12)

    def test_file_hashes_and_audit_fingerprint(self) -> None:
        paths = {
            "pair_candidates.csv": PAIR_PATH,
            "component_candidates.csv": COMPONENT_PATH,
            "component_members.csv": MEMBER_PATH,
        }
        actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
        self.assertEqual(actual_hashes, self.audit["output_sha256"])
        # audit.json records its historical semantic-v1 input. The current live
        # manifest is source-atomic v3 and is authenticated by split_audit.json instead.
        self.assertEqual(
            self.audit["input"]["manifest_sha256"],
            HISTORICAL_V1_MANIFEST_SHA256,
        )
        live_manifest_sha256 = sha256_file(MANIFEST_PATH)
        self.assertNotEqual(live_manifest_sha256, HISTORICAL_V1_MANIFEST_SHA256)
        self.assertEqual(
            live_manifest_sha256,
            self.split_audit["generated_csv_sha256"]["split_manifest.csv"],
        )
        payload = {
            "audit_version": self.audit["audit_version"],
            "manifest_sha256": self.audit["input"]["manifest_sha256"],
            "embedding_fingerprint_sha256": self.audit["embedding"][
                "embedding_fingerprint_sha256"
            ],
            "thresholds": self.audit["thresholds"],
            "output_sha256": actual_hashes,
        }
        fingerprint = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        self.assertEqual(fingerprint, self.audit["audit_fingerprint_sha256"])

    def test_historical_audit_to_resealed_split_provenance_chain(self) -> None:
        audit_fingerprint = self.audit["audit_fingerprint_sha256"]
        adjudication_sha256 = sha256_file(ADJUDICATION_PATH)
        semantic_audit = self.split_audit["semantic_adjudication"]

        self.assertEqual(self.split_audit["split_version"], "step02_source_atomic_v3")
        self.assertEqual(
            self.adjudication["audit_fingerprint_sha256"], audit_fingerprint
        )
        self.assertEqual(
            semantic_audit["similarity_audit_fingerprint_sha256"], audit_fingerprint
        )
        self.assertEqual(
            semantic_audit["adjudication_file_sha256"], adjudication_sha256
        )
        self.assertEqual(
            {
                row["semantic_similarity_audit_fingerprint_sha256"]
                for row in self.manifest
            },
            {audit_fingerprint},
        )
        self.assertEqual(
            {row["semantic_adjudication_file_sha256"] for row in self.manifest},
            {adjudication_sha256},
        )

    def test_pairs_obey_thresholds_and_exclusion_rule(self) -> None:
        thresholds = self.audit["thresholds"]
        seen: set[tuple[str, str]] = set()
        candidate_images: set[str] = set()
        for expected_rank, pair in enumerate(self.pairs, start=1):
            self.assertEqual(int(pair["rank"]), expected_rank)
            image_a = pair["image_id_a"]
            image_b = pair["image_id_b"]
            self.assertLess(image_a, image_b)
            self.assertIn(image_a, self.included)
            self.assertIn(image_b, self.included)
            self.assertNotIn((image_a, image_b), seen)
            seen.add((image_a, image_b))
            candidate_images.update((image_a, image_b))
            self.assertTrue(
                float(pair["cosine_cls_patch_concat"])
                >= thresholds["cls_patch_concat"]
                or float(pair["cosine_patch_mean"]) >= thresholds["patch_mean"]
            )
            row_a = self.included[image_a]
            row_b = self.included[image_b]
            self.assertNotEqual(
                row_a["confirmed_object_group_id"], row_b["confirmed_object_group_id"]
            )
            same_source = (
                bool(row_a["confirmed_source_group_id"])
                and row_a["confirmed_source_group_id"]
                == row_b["confirmed_source_group_id"]
            )
            self.assertFalse(same_source)
        self.assertEqual(len(candidate_images), 118)

    def test_components_and_members_are_consistent(self) -> None:
        members_by_component: dict[str, list[dict[str, str]]] = {}
        for member in self.members:
            members_by_component.setdefault(member["component_id"], []).append(member)
        self.assertEqual(set(members_by_component), {row["component_id"] for row in self.components})
        for component in self.components:
            members = members_by_component[component["component_id"]]
            self.assertEqual(len(members), int(component["member_count"]))
            self.assertEqual(
                sorted(member["image_id"] for member in members),
                sorted(component["image_ids"].split("|")),
            )
            component_pair_count = sum(
                pair["component_id"] == component["component_id"] for pair in self.pairs
            )
            self.assertEqual(component_pair_count, int(component["pair_count"]))

    def test_adjudication_is_complete_well_formed_and_bound_to_components(self) -> None:
        self.assertEqual(self.adjudication["review_status"], "complete")
        decisions = self.adjudication["components"]
        decision_by_component = {
            decision["component_id"]: decision for decision in decisions
        }
        expected_component_ids = {row["component_id"] for row in self.components}
        self.assertEqual(len(decisions), 46)
        self.assertEqual(len(decision_by_component), len(decisions))
        self.assertEqual(set(decision_by_component), expected_component_ids)

        expected_decision_counts = {
            "distinct_unrelated": 1,
            "partitioned": 1,
            "same_physical_object": 5,
            "same_source_distinct_objects": 39,
        }
        self.assertEqual(
            dict(sorted(Counter(row["decision"] for row in decisions).items())),
            expected_decision_counts,
        )
        self.assertEqual(
            self.split_audit["semantic_adjudication"]["decision_counts"],
            expected_decision_counts,
        )

        members_by_component: dict[str, set[str]] = defaultdict(set)
        labels_by_component: dict[str, set[str]] = defaultdict(set)
        for member in self.members:
            members_by_component[member["component_id"]].add(member["image_id"])
            labels_by_component[member["component_id"]].add(
                member["ornament_label"]
            )

        seen_group_ids: set[str] = set()
        observed_merge_group_count = 0
        for component_id, decision in decision_by_component.items():
            self.assertEqual(decision["review_status"], "complete")
            self.assertTrue(decision["notes"].strip())
            expected_members = members_by_component[component_id]
            self.assertEqual(len(labels_by_component[component_id]), 1)
            groups = decision["groups"]
            listed_members: set[str] = set()

            if decision["decision"] == "distinct_unrelated":
                self.assertEqual(groups, [])
            elif decision["decision"] == "partitioned":
                self.assertGreaterEqual(len(groups), 1)
            else:
                self.assertEqual(len(groups), 1)

            for group in groups:
                observed_merge_group_count += 1
                group_id = group["group_id"]
                self.assertNotIn(group_id, seen_group_ids)
                seen_group_ids.add(group_id)
                image_ids = group["image_ids"]
                self.assertGreaterEqual(len(image_ids), 2)
                self.assertEqual(len(image_ids), len(set(image_ids)))
                self.assertTrue(set(image_ids) <= expected_members)
                self.assertFalse(listed_members & set(image_ids))
                listed_members.update(image_ids)
                self.assertIn(
                    group["relationship"], {"same_physical_object", "same_source"}
                )

            if decision["decision"] == "same_physical_object":
                self.assertEqual(listed_members, expected_members)
                self.assertEqual(groups[0]["relationship"], "same_physical_object")
            elif decision["decision"] == "same_source_distinct_objects":
                self.assertEqual(listed_members, expected_members)
                self.assertEqual(groups[0]["relationship"], "same_source")

        self.assertEqual(observed_merge_group_count, 45)
        self.assertEqual(
            observed_merge_group_count,
            self.split_audit["semantic_adjudication"]["merge_group_count"],
        )

    def test_adjudication_csv_is_an_exact_human_readable_projection(self) -> None:
        decisions = self.adjudication["components"]
        self.assertEqual(
            [row["component_id"] for row in self.adjudication_csv],
            [row["component_id"] for row in decisions],
        )
        members_by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
        for member in self.members:
            members_by_component[member["component_id"]].append(member)

        for csv_row, decision in zip(self.adjudication_csv, decisions):
            members = members_by_component[decision["component_id"]]
            labels = {member["ornament_label"] for member in members}
            self.assertEqual(len(labels), 1)
            self.assertEqual(csv_row["review_status"], decision["review_status"])
            self.assertEqual(csv_row["ornament_label"], next(iter(labels)))
            self.assertEqual(int(csv_row["image_count"]), len(members))
            self.assertEqual(
                csv_row["relative_paths"],
                "|".join(member["relative_path"] for member in members),
            )
            self.assertEqual(csv_row["decision"], decision["decision"])
            self.assertEqual(json.loads(csv_row["groups"]), decision["groups"])
            self.assertEqual(csv_row["notes"], decision["notes"])

    def test_known_semantic_leaks_remain_in_review_queue(self) -> None:
        path_pairs = {
            frozenset((pair["relative_path_a"], pair["relative_path_b"]))
            for pair in self.pairs
        }
        expected = (
            (
                "dataset_test/05_kosiv_ceramics/k160.jpg",
                "dataset_test/05_kosiv_ceramics/k168.jpg",
            ),
            (
                "dataset_dev/04_petrykivka_painting/201.png",
                "dataset_dev/04_petrykivka_painting/207.jpg",
            ),
            (
                "dataset_dev/05_kosiv_ceramics/k017.jpg",
                "dataset_dev/05_kosiv_ceramics/k074.jpg",
            ),
            (
                "dataset_dev/03_bubnivka_ceramics/b091.jpg",
                "dataset_dev/03_bubnivka_ceramics/b092.jpg",
            ),
            (
                "dataset_test/04_petrykivka_painting/198.png",
                "dataset_dev/04_petrykivka_painting/208.jpg",
            ),
        )
        for pair in expected:
            self.assertIn(frozenset(pair), path_pairs)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
DATA_ROOT = STEP_ROOT / "data"
METADATA_ROOT = STEP_ROOT / "metadata"


def read_csv(name: str) -> list[dict[str, str]]:
    with (METADATA_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PhaseOneOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_csv("manifest.csv")
        cls.duplicates = read_csv("duplicate_candidates.csv")
        cls.review_queue = read_csv("review_queue.csv")
        with (METADATA_ROOT / "data_audit.json").open(encoding="utf-8") as handle:
            cls.audit = json.load(handle)

    def test_manifest_count_and_ids_are_unique(self) -> None:
        self.assertEqual(len(self.manifest), self.audit["image_count"])
        self.assertEqual(len({row["image_id"] for row in self.manifest}), len(self.manifest))
        self.assertEqual(len({row["relative_path"] for row in self.manifest}), len(self.manifest))

    def test_every_manifest_path_exists(self) -> None:
        missing = [row["relative_path"] for row in self.manifest if not (DATA_ROOT / row["relative_path"]).is_file()]
        self.assertEqual(missing, [])

    def test_every_image_has_a_candidate_group(self) -> None:
        missing = [row["image_id"] for row in self.manifest if not row["candidate_object_group_id"]]
        self.assertEqual(missing, [])

    def test_review_queue_covers_every_image_once(self) -> None:
        self.assertEqual(len(self.review_queue), len(self.manifest))
        self.assertEqual(len({row["image_id"] for row in self.review_queue}), len(self.manifest))

    def test_manual_fields_are_explicit(self) -> None:
        for row in self.manifest:
            self.assertEqual(row["source_group_id"], "unreviewed")
            self.assertEqual(row["object_group_id"], "unreviewed")
            self.assertEqual(row["object_type"], "unreviewed")
            self.assertEqual(row["motif_visibility"], "unreviewed")
            self.assertEqual(row["provenance"], "unknown")
            self.assertEqual(row["license"], "unknown")

    def test_audit_duplicate_counts_match_candidates(self) -> None:
        perceptual_pairs = sum(
            "perceptual_review" in row["candidate_reasons"] for row in self.duplicates
        )
        perceptual_cross_split = sum(
            "perceptual_review" in row["candidate_reasons"]
            and row["same_split"] == "False"
            for row in self.duplicates
        )
        self.assertEqual(perceptual_pairs, self.audit["perceptual_review_pair_count"])
        self.assertEqual(
            perceptual_cross_split,
            self.audit["perceptual_cross_split_pair_count"],
        )


if __name__ == "__main__":
    unittest.main()

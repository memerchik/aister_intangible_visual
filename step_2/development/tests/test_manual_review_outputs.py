from __future__ import annotations

import collections
import json
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
DATA_ROOT = STEP_ROOT / "data"
REVIEW_OUTPUT = (
    STEP_ROOT
    / "review"
    / "outputs"
    / "019f6af8-e0cf-7513-85b2-f6eb3920df46"
)


class ManualReviewOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (REVIEW_OUTPUT / "phase_1_manual_review.json").open(encoding="utf-8") as handle:
            cls.review = json.load(handle)
        cls.groups = cls.review["group_decisions"]
        cls.images = cls.review["image_decisions"]

    def test_review_is_complete_and_unique(self) -> None:
        self.assertEqual(len(self.groups), 143)
        self.assertEqual(len(self.images), 429)
        self.assertEqual(len({row["candidate_group_id"] for row in self.groups}), 143)
        self.assertEqual(len({row["image_id"] for row in self.images}), 429)
        self.assertTrue(all(row["review_status"] == "complete" for row in self.groups))
        self.assertTrue(all(row["review_status"] == "complete" for row in self.images))

    def test_every_reviewed_image_exists(self) -> None:
        missing = [
            row["relative_path"]
            for row in self.images
            if not (DATA_ROOT / row["relative_path"]).is_file()
        ]
        self.assertEqual(missing, [])

    def test_summary_counts_match_rows(self) -> None:
        summary = self.review["summary"]
        self.assertEqual(
            summary["decision_counts"],
            dict(collections.Counter(row["group_decision"] for row in self.groups)),
        )
        self.assertEqual(
            summary["object_type_counts"],
            dict(collections.Counter(row["object_type"] for row in self.images)),
        )
        self.assertEqual(
            summary["motif_visibility_counts"],
            dict(collections.Counter(row["motif_visibility"] for row in self.images)),
        )
        self.assertEqual(
            summary["training_use_counts"],
            dict(collections.Counter(row["training_use"] for row in self.images)),
        )

    def test_required_review_fields_are_filled(self) -> None:
        required = (
            "confirmed_object_group_id",
            "object_type",
            "motif_visibility",
            "label_status",
            "training_use",
            "reviewer_notes",
        )
        incomplete = [row["image_id"] for row in self.images if any(not row[field] for field in required)]
        self.assertEqual(incomplete, [])
        self.assertTrue((REVIEW_OUTPUT / "phase_1_image_review.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()

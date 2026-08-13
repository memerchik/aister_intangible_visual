from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEVELOPMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.consensus import (  # noqa: E402
    FusionCandidateSummary,
    FusionEligibilityReference,
    aggregate_head_scores,
    apply_multiclass_logit_fusion,
    apply_pair_consensus,
    fusion_eligibility,
    rank_fusion_candidates,
)


CLASSES = ("opish", "ornek", "bub", "petryk", "kosiv")
PAIR = ("opish", "bub")


class ConsensusApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray(
            [
                [0.45, 0.10, 0.35, 0.05, 0.05],
                [0.10, 0.55, 0.20, 0.10, 0.05],
                [0.20, 0.10, 0.55, 0.10, 0.05],
            ],
            dtype=np.float64,
        )
        self.heads = (
            np.asarray(
                [
                    [0.20, 0.10, 0.60, 0.05, 0.05],
                    [0.40, 0.30, 0.15, 0.10, 0.05],
                    [0.70, 0.05, 0.15, 0.05, 0.05],
                ]
            ),
            np.asarray(
                [
                    [0.25, 0.10, 0.55, 0.05, 0.05],
                    [0.35, 0.35, 0.15, 0.10, 0.05],
                    [0.65, 0.05, 0.20, 0.05, 0.05],
                ]
            ),
            np.asarray(
                [
                    [0.30, 0.10, 0.50, 0.05, 0.05],
                    [0.20, 0.50, 0.15, 0.10, 0.05],
                    [0.60, 0.05, 0.25, 0.05, 0.05],
                ]
            ),
        )

    def test_multiclass_fusion_is_normalized_and_has_declared_endpoints(self) -> None:
        zero = apply_multiclass_logit_fusion(self.base, self.heads, 0.0)
        full = apply_multiclass_logit_fusion(self.base, self.heads, 1.0)
        np.testing.assert_allclose(zero, self.base, atol=1e-12)
        np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(full.shape, self.base.shape)
        self.assertTrue(np.isfinite(aggregate_head_scores(self.heads)).all())

    def test_pair_consensus_changes_only_base_pair_predictions_and_pair_columns(self) -> None:
        combined, applied = apply_pair_consensus(
            self.base,
            self.heads,
            CLASSES,
            PAIR,
            0.5,
            majority_gated=False,
        )
        np.testing.assert_array_equal(applied, [True, False, True])
        np.testing.assert_allclose(combined[1], self.base[1], atol=0.0)
        np.testing.assert_allclose(combined[:, [1, 3, 4]], self.base[:, [1, 3, 4]], atol=0.0)
        np.testing.assert_allclose(
            combined[:, 0] + combined[:, 2],
            self.base[:, 0] + self.base[:, 2],
            atol=1e-12,
        )
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-12)

    def test_majority_gate_requires_two_thirds_pair_agreement(self) -> None:
        disagreeing = list(self.heads)
        disagreeing[2] = disagreeing[2].copy()
        disagreeing[2][0, [0, 2]] = disagreeing[2][0, [2, 0]]
        _, applied = apply_pair_consensus(
            self.base,
            disagreeing,
            CLASSES,
            PAIR,
            0.5,
            majority_gated=True,
        )
        self.assertTrue(applied[0])
        self.assertFalse(applied[1])

    def test_invalid_shapes_or_blends_fail(self) -> None:
        with self.assertRaises(ValueError):
            apply_multiclass_logit_fusion(self.base, (), 0.5)
        with self.assertRaises(ValueError):
            apply_pair_consensus(
                self.base, self.heads, CLASSES, PAIR, 1.1, majority_gated=False
            )


class ConsensusSelectionTests(unittest.TestCase):
    def summary(self, identifier: str, **changes: object) -> FusionCandidateSummary:
        values = dict(
            configuration_id=identifier,
            readiness_score=0.90,
            minimum_fold_readiness_score=0.80,
            accuracy=0.94,
            macro_f1=0.935,
            source_robustness_score=0.71,
            opishnyan_group_recall=0.59,
            intervention_rank=1,
            blend_weight=0.3,
            head_count=4,
            encoder_count=3,
            per_class_recall={label: 0.95 for label in CLASSES},
        )
        values.update(changes)
        return FusionCandidateSummary(**values)

    def test_eligibility_reports_all_non_regression_failures(self) -> None:
        reference = FusionEligibilityReference(
            accuracy=0.94,
            macro_f1=0.935,
            source_robustness_score=0.71,
            opishnyan_group_recall=0.59,
            per_class_recall={label: 0.95 for label in CLASSES},
        )
        candidate = self.summary(
            "bad",
            accuracy=0.93,
            macro_f1=0.92,
            source_robustness_score=0.70,
            opishnyan_group_recall=0.58,
            per_class_recall={**reference.per_class_recall, "ornek": 0.94},
        )
        result = fusion_eligibility(
            candidate,
            reference,
            other_class_recall_drops={"ornek": 0.005},
        )
        self.assertFalse(result.eligible)
        self.assertEqual(
            set(result.failures),
            {
                "accuracy_drop",
                "macro_f1_drop",
                "source_robustness_drop",
                "opishnyan_group_recall_drop",
                "recall_drop:ornek",
            },
        )

    def test_readiness_tie_uses_frozen_tie_break_order(self) -> None:
        best_score = self.summary(
            "best_score", readiness_score=0.901, minimum_fold_readiness_score=0.79
        )
        stable = self.summary(
            "stable", readiness_score=0.900, minimum_fold_readiness_score=0.81
        )
        ranked = rank_fusion_candidates((best_score, stable), tie_window=0.0015)
        self.assertEqual(ranked[0].configuration_id, "stable")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEVELOPMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.pairwise import (  # noqa: E402
    CorrectionCandidateSummary,
    CorrectionEligibilityReference,
    apply_pairwise_logit_blend,
    boundary_metrics,
    correction_eligibility,
    fit_pairwise_correction,
    harmonic_mean_many,
    rank_correction_candidates,
)


CLASSES = ("opish", "ornek", "bub", "petryk", "kosiv")
PAIR = ("opish", "bub")
CERAMICS = ("opish", "bub", "kosiv")


class PairwiseBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray(
            [
                [0.30, 0.15, 0.25, 0.20, 0.10],
                [0.05, 0.10, 0.55, 0.20, 0.10],
            ],
            dtype=np.float64,
        )
        self.head = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)

    def test_pair_mass_and_non_pair_columns_are_preserved(self) -> None:
        combined = apply_pairwise_logit_blend(
            self.base, CLASSES, self.head, PAIR, PAIR, 0.5
        )
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            combined[:, [1, 3, 4]], self.base[:, [1, 3, 4]], atol=0.0
        )
        np.testing.assert_allclose(
            combined[:, 0] + combined[:, 2],
            self.base[:, 0] + self.base[:, 2],
            atol=1e-12,
        )

    def test_zero_and_full_blend_have_declared_endpoints(self) -> None:
        zero = apply_pairwise_logit_blend(
            self.base, CLASSES, self.head, PAIR, PAIR, 0.0
        )
        full = apply_pairwise_logit_blend(
            self.base, CLASSES, self.head, PAIR, PAIR, 1.0
        )
        np.testing.assert_allclose(zero, self.base, atol=1e-12)
        mass = self.base[:, 0] + self.base[:, 2]
        np.testing.assert_allclose(full[:, 0], mass * self.head[:, 0], atol=1e-12)
        np.testing.assert_allclose(full[:, 2], mass * self.head[:, 1], atol=1e-12)

    def test_invalid_blend_or_probability_shapes_fail(self) -> None:
        with self.assertRaises(ValueError):
            apply_pairwise_logit_blend(
                self.base, CLASSES, self.head, PAIR, PAIR, 1.1
            )
        with self.assertRaises(ValueError):
            apply_pairwise_logit_blend(
                self.base, CLASSES, self.head[:, :1], PAIR, PAIR, 0.5
            )


class PairwiseFitAndMetricTests(unittest.TestCase):
    def test_fit_uses_only_pair_rows_and_predicts_all_rows(self) -> None:
        features = np.asarray(
            [
                [-3.0, 0.0],
                [-2.0, 0.2],
                [2.0, 0.1],
                [3.0, -0.2],
                [0.0, 4.0],
                [0.0, -4.0],
            ],
            dtype=np.float64,
        )
        labels = ("opish", "opish", "bub", "bub", "ornek", "petryk")
        groups = tuple(f"g{index}" for index in range(len(labels)))
        correction = fit_pairwise_correction(
            features,
            labels,
            groups,
            pair_classes=PAIR,
            c_value=10.0,
            source_group_exponent=0.5,
            seed=7,
        )
        self.assertTrue(correction.converged)
        probabilities = correction.predict_proba(features)
        self.assertEqual(probabilities.shape, (6, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_boundary_metrics_match_declared_components(self) -> None:
        truth = ("opish", "opish", "bub", "bub", "ornek", "petryk", "kosiv")
        predicted = ("opish", "bub", "bub", "opish", "ornek", "petryk", "kosiv")
        probabilities = np.full((len(truth), len(CLASSES)), 0.01, dtype=np.float64)
        for row, label in enumerate(predicted):
            probabilities[row, CLASSES.index(label)] = 0.96
        metrics = boundary_metrics(
            truth,
            probabilities,
            CLASSES,
            pair_classes=PAIR,
            ceramic_classes=CERAMICS,
        )
        self.assertEqual(metrics.opishnyan_recall, 0.5)
        self.assertEqual(metrics.bubnivka_recall, 0.5)
        self.assertEqual(metrics.bubnivka_precision, 0.5)
        self.assertEqual(metrics.opishnyan_to_bubnivka_errors, 1)
        self.assertEqual(metrics.bubnivka_to_opishnyan_errors, 1)
        self.assertAlmostEqual(
            metrics.boundary_score,
            harmonic_mean_many(
                (
                    metrics.opishnyan_recall,
                    metrics.bubnivka_recall,
                    metrics.bubnivka_precision,
                    metrics.ceramic_macro_f1,
                )
            ),
        )


class PairwiseSelectionTests(unittest.TestCase):
    def _summary(self, identifier: str, **changes: object) -> CorrectionCandidateSummary:
        values = dict(
            configuration_id=identifier,
            boundary_score=0.90,
            minimum_fold_boundary_score=0.80,
            accuracy=0.94,
            macro_f1=0.93,
            source_robustness_score=0.71,
            opishnyan_group_recall=0.59,
            combined_view_count=6,
            correction="pairwise",
            blend_weight=0.5,
            c_value=10.0,
            source_group_exponent=0.5,
            per_class_recall={label: 0.95 for label in CLASSES},
        )
        values.update(changes)
        return CorrectionCandidateSummary(**values)

    def test_eligibility_reports_every_non_regression_failure(self) -> None:
        reference = CorrectionEligibilityReference(
            accuracy=0.94,
            macro_f1=0.93,
            source_robustness_score=0.71,
            opishnyan_group_recall=0.59,
            per_class_recall={label: 0.95 for label in CLASSES},
        )
        candidate = self._summary(
            "bad",
            accuracy=0.93,
            macro_f1=0.92,
            source_robustness_score=0.70,
            opishnyan_group_recall=0.58,
            per_class_recall={**reference.per_class_recall, "ornek": 0.94},
        )
        result = correction_eligibility(
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

    def test_tie_window_uses_frozen_tie_break_order(self) -> None:
        higher_score = self._summary(
            "higher_score", boundary_score=0.901, minimum_fold_boundary_score=0.79
        )
        stable = self._summary(
            "stable", boundary_score=0.900, minimum_fold_boundary_score=0.81
        )
        ranked = rank_correction_candidates((higher_score, stable), tie_window=0.002)
        self.assertEqual(ranked[0].configuration_id, "stable")


if __name__ == "__main__":
    unittest.main()

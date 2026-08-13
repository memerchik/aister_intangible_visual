from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STEP_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STEP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.robustness import (  # noqa: E402
    CandidateSummary,
    EligibilityReference,
    ModelFitError,
    apply_ceramic_specialist,
    candidate_eligibility,
    compute_source_group_weights,
    fit_centered_logit_ensemble,
    fit_ceramic_specialist,
    fit_logistic_probe,
    ordinary_metrics,
    rank_candidates,
    rank_eligible_candidates,
    source_group_metrics,
)


CLASSES = ("opish", "ornek", "bub", "petrykivka", "kosiv")
CERAMICS = ("opish", "bub", "kosiv")


class SourceWeightTests(unittest.TestCase):
    def test_weights_balance_classes_and_downweight_large_groups(self) -> None:
        labels = ["a", "a", "a", "b"]
        groups = ["large", "large", "small", "only"]
        weights = compute_source_group_weights(labels, groups, 1.0)

        np.testing.assert_allclose(weights, [0.5, 0.5, 1.0, 2.0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3:].sum()))

    def test_group_sizes_are_recomputed_from_supplied_training_rows(self) -> None:
        full = compute_source_group_weights(
            ["a", "a", "a", "b", "b"], ["g", "g", "h", "j", "k"], 1.0
        )
        subset = compute_source_group_weights(
            ["a", "a", "b", "b"], ["g", "h", "j", "k"], 1.0
        )
        self.assertNotAlmostEqual(float(full[0]), float(subset[0]))
        np.testing.assert_allclose(subset, np.ones(4))


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.asarray(["a"] * 5 + ["b"] * 5 + ["c"] * 5, dtype=object)
        self.features = np.vstack(
            (
                np.column_stack((np.linspace(-3, -2, 5), np.zeros(5))),
                np.column_stack((np.linspace(2, 3, 5), np.zeros(5))),
                np.column_stack((np.zeros(5), np.linspace(2, 3, 5))),
            )
        )

    def test_probe_exposes_order_iterations_logits_and_probabilities(self) -> None:
        order = ("c", "a", "b")
        probe = fit_logistic_probe(
            self.features,
            self.labels,
            c_value=10.0,
            classes=order,
            seed=7,
        )
        self.assertEqual(probe.classes, order)
        self.assertTrue(probe.converged)
        self.assertTrue(all(iteration > 0 for iteration in probe.iterations))
        self.assertEqual(probe.decision_logits(self.features).shape, (15, 3))
        np.testing.assert_allclose(
            probe.decision_logits(self.features).mean(axis=1), 0.0, atol=1e-12
        )
        probabilities = probe.predict_proba(self.features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertGreater(float(np.mean(probe.predict(self.features) == self.labels)), 0.9)

    def test_probe_raises_when_iteration_budget_is_exhausted(self) -> None:
        with self.assertRaises(ModelFitError):
            fit_logistic_probe(
                self.features,
                self.labels,
                c_value=100.0,
                max_iter=1,
                tolerance=1e-15,
            )

    def test_centered_logit_ensemble_is_deterministic_and_aligned(self) -> None:
        blocks = (self.features, self.features[:, ::-1])
        first = fit_centered_logit_ensemble(
            blocks, self.labels, c_value=10.0, classes=("b", "c", "a"), seed=11
        )
        second = fit_centered_logit_ensemble(
            blocks, self.labels, c_value=10.0, classes=("b", "c", "a"), seed=11
        )
        self.assertEqual(first.classes, ("b", "c", "a"))
        self.assertTrue(first.converged)
        np.testing.assert_array_equal(
            first.predict_proba(blocks), second.predict_proba(blocks)
        )
        np.testing.assert_allclose(first.predict_proba(blocks).sum(axis=1), 1.0)


class SpecialistAndMetricTests(unittest.TestCase):
    def test_specialist_preserves_total_ceramic_mass_and_nonceramics(self) -> None:
        base = np.asarray(
            [[0.20, 0.10, 0.30, 0.10, 0.30], [0.05, 0.50, 0.10, 0.25, 0.10]]
        )
        specialist = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        combined = apply_ceramic_specialist(
            base, CLASSES, specialist, CERAMICS, CERAMICS
        )
        ceramic_indices = [CLASSES.index(label) for label in CERAMICS]
        np.testing.assert_allclose(
            combined[:, ceramic_indices].sum(axis=1),
            base[:, ceramic_indices].sum(axis=1),
        )
        np.testing.assert_array_equal(combined[:, 1], base[:, 1])
        np.testing.assert_array_equal(combined[:, 3], base[:, 3])
        np.testing.assert_allclose(combined.sum(axis=1), 1.0)

    def test_fit_specialist_uses_only_three_ceramic_classes(self) -> None:
        labels = np.asarray(
            ["opish"] * 4 + ["bub"] * 4 + ["kosiv"] * 4 + ["ornek"] * 4,
            dtype=object,
        )
        features = np.vstack(
            [
                np.tile([index * 2.0, 1.0], (4, 1))
                + np.column_stack((np.linspace(0, 0.2, 4), np.zeros(4)))
                for index in range(4)
            ]
        )
        groups = [f"g{index // 2}" for index in range(16)]
        specialist = fit_ceramic_specialist(
            features,
            labels,
            groups,
            ceramic_classes=CERAMICS,
            c_value=10.0,
            source_group_exponent=0.5,
        )
        self.assertEqual(specialist.ceramic_classes, CERAMICS)
        self.assertEqual(specialist.model.classes, CERAMICS)
        self.assertTrue(specialist.converged)

    def test_ordinary_and_source_group_metrics_have_distinct_weighting(self) -> None:
        truth = np.asarray(
            ["a", "a", "a", "a", "b", "b", "c", "c"], dtype=object
        )
        predicted = np.asarray(
            ["a", "a", "a", "b", "b", "b", "c", "c"], dtype=object
        )
        groups = ["large", "large", "large", "small", "b", "b", "c", "c"]
        probabilities = np.zeros((8, 3), dtype=float)
        for row, label in enumerate(predicted):
            probabilities[row, ("a", "b", "c").index(label)] = 1.0
        ordinary = ordinary_metrics(truth, probabilities, ("a", "b", "c"))
        source = source_group_metrics(
            truth, predicted, groups, ("a", "b", "c"), ("a", "b", "c")
        )

        self.assertAlmostEqual(ordinary.accuracy, 7 / 8)
        self.assertAlmostEqual(ordinary.balanced_accuracy, (0.75 + 1.0 + 1.0) / 3)
        self.assertAlmostEqual(source.per_class_group_recall["a"], 0.5)
        self.assertAlmostEqual(source.per_class_group_recall["b"], 1.0)
        self.assertAlmostEqual(source.source_group_balanced_accuracy, (0.5 + 1 + 1) / 3)


class RankingTests(unittest.TestCase):
    def summary(self, name: str, **changes: object) -> CandidateSummary:
        values = dict(
            configuration_id=name,
            robustness_score=0.90,
            minimum_inner_fold_source_group_balanced_accuracy=0.80,
            macro_f1=0.93,
            inference_cost_rank=5,
            specialist="three_way_conditional",
            c_value=100.0,
            source_group_exponent=0.5,
            per_class_recall={"ornek": 0.97, "pet": 0.96},
        )
        values.update(changes)
        return CandidateSummary(**values)

    def test_eligibility_reports_each_non_regression_failure(self) -> None:
        candidate = self.summary(
            "bad", macro_f1=0.90, per_class_recall={"ornek": 0.90, "pet": 0.89}
        )
        reference = EligibilityReference(
            macro_f1=0.93, per_class_recall={"ornek": 0.98, "pet": 0.97}
        )
        result = candidate_eligibility(
            candidate, reference, ornek_class="ornek", petrykivka_class="pet"
        )
        self.assertFalse(result.eligible)
        self.assertEqual(
            result.failures,
            ("macro_f1_drop", "ornek_recall_drop", "petrykivka_recall_drop"),
        )

    def test_ranking_uses_tie_window_then_declared_tie_breaks(self) -> None:
        higher_score = self.summary("higher_score", robustness_score=0.904)
        stable = self.summary(
            "stable",
            robustness_score=0.900,
            minimum_inner_fold_source_group_balanced_accuracy=0.85,
            specialist="none",
            c_value=10.0,
            source_group_exponent=0.0,
        )
        outside_window = self.summary(
            "outside", robustness_score=0.894, minimum_inner_fold_source_group_balanced_accuracy=0.99
        )
        ranked = rank_candidates([outside_window, higher_score, stable])
        self.assertEqual(
            [candidate.configuration_id for candidate in ranked],
            ["stable", "higher_score", "outside"],
        )

    def test_eligible_ranking_filters_candidates_deterministically(self) -> None:
        good = self.summary("good")
        bad = self.summary("bad", macro_f1=0.80)
        reference = EligibilityReference(
            macro_f1=0.93, per_class_recall={"ornek": 0.98, "pet": 0.97}
        )
        ranked = rank_eligible_candidates(
            [bad, good],
            reference,
            ornek_class="ornek",
            petrykivka_class="pet",
        )
        self.assertEqual([candidate.configuration_id for candidate in ranked], ["good"])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np


STEP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = STEP_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ornament_classifier.fixed_consensus import (  # noqa: E402
    centered_log_probabilities,
    fixed_probability_consensus,
)


class FixedConsensusTests(unittest.TestCase):
    def setUp(self):
        self.first = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
        self.second = np.asarray([[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]])

    def test_centered_log_probabilities_are_shift_invariant(self):
        logits = centered_log_probabilities(self.first)
        self.assertTrue(np.allclose(logits.mean(axis=1), 0.0, atol=1e-12))
        recovered = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        self.assertTrue(np.allclose(recovered, self.first, atol=1e-12))

    def test_equal_consensus_is_normalized_and_deterministic(self):
        observed = fixed_probability_consensus((self.first, self.second))
        repeated = fixed_probability_consensus((self.first, self.second))
        self.assertTrue(np.array_equal(observed, repeated))
        self.assertTrue(np.allclose(observed.sum(axis=1), 1.0, atol=1e-12))
        expected = np.sqrt(self.first * self.second)
        expected /= expected.sum(axis=1, keepdims=True)
        self.assertTrue(np.allclose(observed, expected, atol=1e-12))

    def test_identical_heads_are_identity(self):
        observed = fixed_probability_consensus(
            (self.first, self.first, self.first, self.first),
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        self.assertTrue(np.allclose(observed, self.first, atol=1e-12))

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            fixed_probability_consensus((self.first,))
        with self.assertRaises(ValueError):
            fixed_probability_consensus((self.first, self.second[:, :2]))
        with self.assertRaises(ValueError):
            fixed_probability_consensus((self.first, self.second), weights=(0.4, 0.4))
        with self.assertRaises(ValueError):
            centered_log_probabilities(np.asarray([[0.8, 0.3]]))
        broken = self.first.copy()
        broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fixed_probability_consensus((broken, self.second))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


STEP_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STEP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.motif import (  # noqa: E402
    aggregate_tile_embeddings,
    augment_feature_blocks,
    intersection_over_union,
    proposal_pool,
    select_motif_proposals,
)


class MotifProposalTests(unittest.TestCase):
    def test_pool_is_bounded_stable_and_inside_image(self) -> None:
        first = proposal_pool(320, 180)
        second = proposal_pool(320, 180)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len({item[1] for item in first}), 18)
        for _, (left, top, right, bottom), scale, x_position, y_position in first:
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, 320)
            self.assertLessEqual(bottom, 180)
            self.assertEqual(right - left, bottom - top)
            self.assertIn(scale, (0.45, 0.65))
            self.assertIn(x_position, (0.0, 0.5, 1.0))
            self.assertIn(y_position, (0.0, 0.5, 1.0))

    def test_selection_is_label_free_deterministic_and_texture_sensitive(self) -> None:
        pixels = np.zeros((180, 320, 3), dtype=np.uint8)
        pixels[:, :] = (120, 120, 120)
        pixels[:90, :160:2] = (255, 0, 0)
        pixels[:90, 1:160:2] = (0, 0, 255)
        image = Image.fromarray(pixels, mode="RGB")
        first = select_motif_proposals(image)
        second = select_motif_proposals(image)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({item.proposal_id for item in first}), 6)
        self.assertGreater(first[0].texture_score, first[-1].texture_score)
        self.assertTrue(any(item.left < 160 and item.top < 90 for item in first[:2]))

    def test_iou_validation_and_values(self) -> None:
        self.assertEqual(intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(intersection_over_union((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)
        self.assertAlmostEqual(
            intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)),
            1.0 / 3.0,
        )
        with self.assertRaises(ValueError):
            intersection_over_union((0, 0, 0, 10), (0, 0, 10, 10))


class MotifAggregationTests(unittest.TestCase):
    def test_descriptors_are_deterministic_finite_and_normalized(self) -> None:
        tiles = np.eye(6, dtype=np.float64)
        scores = np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64)
        first = aggregate_tile_embeddings(tiles, scores)
        second = aggregate_tile_embeddings(tiles, scores)
        self.assertEqual(
            set(first),
            {
                "uniform_mean",
                "texture_weighted_mean",
                "texture_top2_mean",
                "texture_dispersion",
            },
        )
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertTrue(np.isfinite(first[name]).all())
            self.assertAlmostEqual(float(np.linalg.norm(first[name])), 1.0, places=6)
        self.assertGreater(first["texture_weighted_mean"][0], first["texture_weighted_mean"][-1])
        self.assertGreater(first["texture_top2_mean"][0], 0)
        self.assertEqual(first["texture_top2_mean"][-1], 0)

    def test_augmentation_preserves_rows_and_base_order(self) -> None:
        first = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        second = np.asarray([[5.0], [6.0]])
        motif = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        augmented = augment_feature_blocks((first, second), motif)
        self.assertEqual(tuple(value.shape for value in augmented), ((2, 4), (2, 3)))
        np.testing.assert_array_equal(augmented[0][:, :2], first)
        np.testing.assert_array_equal(augmented[1][:, :1], second)
        np.testing.assert_array_equal(augmented[0][:, 2:], motif)
        np.testing.assert_array_equal(augmented[1][:, 1:], motif)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_tile_embeddings(np.ones((1, 3)), [1.0])
        with self.assertRaises(ValueError):
            aggregate_tile_embeddings(np.ones((2, 3)), [1.0, np.nan])
        with self.assertRaises(ValueError):
            augment_feature_blocks((np.ones((2, 3)),), np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()

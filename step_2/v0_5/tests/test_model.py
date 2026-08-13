import json
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = APP_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aister_runtime.inference import (  # noqa: E402
    CLASS_ORDER,
    MAX_DECODED_PIXELS,
    MODEL_WEIGHT_SHA256,
    AssistedOrnamentPredictor,
    Dinov3FeatureExtractor,
    LinearModelArtifact,
)


ARTIFACT_ROOT = APP_ROOT / "artifacts"
WEIGHT_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/blobs"
    / MODEL_WEIGHT_SHA256
)


class AssistedModelArtifactTests(unittest.TestCase):
    def test_low_memory_defaults_are_frozen_for_showcase_hosting(self):
        self.assertEqual(MAX_DECODED_PIXELS, 2_000_000)
        parameters = inspect.signature(Dinov3FeatureExtractor).parameters
        self.assertEqual(parameters["inference_batch_size"].default, 1)

    def test_inference_batch_size_must_be_positive(self):
        extractor = Dinov3FeatureExtractor.__new__(Dinov3FeatureExtractor)
        with self.assertRaisesRegex(ValueError, "inference_batch_size"):
            Dinov3FeatureExtractor.__init__(
                extractor, WEIGHT_PATH, threads=1, inference_batch_size=0
            )

    def test_packaged_model_contract_is_provisional_and_sealed(self):
        manifest = json.loads((ARTIFACT_ROOT / "model_manifest.json").read_text())
        self.assertEqual(manifest["status"], "provisional_human_assisted_not_promoted")
        self.assertEqual(manifest["development_evidence"]["promotion_decision"], "reject")
        self.assertFalse(manifest["calibrated"])
        self.assertFalse(manifest["sealed_test_evaluated"])
        self.assertFalse(manifest["public_deployment_ready"])
        self.assertEqual(manifest["training"]["scope"], "development_only")
        self.assertEqual(manifest["training"]["image_count"], 1693)

    def test_artifact_loads_and_returns_normalized_ranking_scores(self):
        artifact = LinearModelArtifact.load(ARTIFACT_ROOT)
        self.assertEqual(artifact.classes, CLASS_ORDER)
        self.assertEqual(artifact.coefficients.shape, (5, 768))
        self.assertEqual(artifact.intercept.shape, (5,))
        scores = artifact.predict_scores(np.zeros((2, 768), dtype=np.float32))
        self.assertEqual(scores.shape, (2, 5))
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        self.assertTrue(np.isfinite(scores).all())

    def test_product_contract_forbids_phase_6_and_default_storage(self):
        contract = json.loads(
            (APP_ROOT / "product_contract.json").read_text()
        )
        self.assertFalse(contract["evaluation_boundary"]["formal_phase_6_started"])
        self.assertEqual(contract["evaluation_boundary"]["sealed_test_access"], "forbidden")
        self.assertEqual(contract["privacy_and_learning"]["default_prediction_storage"], "none")
        self.assertFalse(contract["privacy_and_learning"]["automatic_training_from_predictions"])
        self.assertFalse(contract["privacy_and_learning"]["automatic_training_from_contributions"])


@unittest.skipUnless(WEIGHT_PATH.is_file(), "Pinned local DINOv3 weight is unavailable")
class AssistedExtractionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = AssistedOrnamentPredictor(
            application_root=APP_ROOT,
            weight_path=WEIGHT_PATH,
            threads=2,
        )
        cls.predictor.warmup()

    def test_runtime_extracts_both_feature_blocks_without_development_tree(self):
        values = np.linspace(0, 255, 320 * 240 * 3, dtype=np.uint8).reshape(240, 320, 3)
        image = Image.fromarray(values, mode="RGB")
        features, regions = self.predictor.extractor.extract(image)
        self.assertEqual(features.shape, (1, 768))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(regions), 6)

    def test_synthetic_image_returns_five_ranking_scores(self):
        values = np.linspace(0, 255, 256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        image = Image.fromarray(values, mode="RGB")
        result = self.predictor.predict_image(image)
        self.assertEqual(len(result["ranking"]), 5)
        self.assertAlmostEqual(sum(row["score"] for row in result["ranking"]), 1.0, places=12)
        self.assertFalse(result["calibrated"])
        self.assertFalse(result["sealed_test_evaluated"])


if __name__ == "__main__":
    unittest.main()

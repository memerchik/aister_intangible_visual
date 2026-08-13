"""Cross-check the deployable v0.5 runtime against frozen development caches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DEVELOPMENT_ROOT.parents[1]
DEVELOPMENT_SOURCE = DEVELOPMENT_ROOT / "src"
APPLICATION_ROOT = DEVELOPMENT_ROOT.parent / "v0_5"
APPLICATION_SOURCE = APPLICATION_ROOT / "src"
for source in (str(APPLICATION_SOURCE), str(DEVELOPMENT_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

from aister_runtime.inference import (  # noqa: E402
    MODEL_WEIGHT_SHA256,
    AssistedOrnamentPredictor,
)
from ornament_classifier.contracts import load_development_contract  # noqa: E402
from ornament_classifier.paths import ProjectPaths  # noqa: E402


WEIGHT_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/blobs"
    / MODEL_WEIGHT_SHA256
)


@unittest.skipUnless(WEIGHT_PATH.is_file(), "Pinned local DINOv3 weight is unavailable")
class V05FrozenFeatureReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = ProjectPaths.from_step_root(DEVELOPMENT_ROOT)
        cls.development = load_development_contract(cls.paths)
        cls.predictor = AssistedOrnamentPredictor(
            application_root=APPLICATION_ROOT,
            weight_path=WEIGHT_PATH,
            threads=4,
        )
        cls.predictor.warmup()

    def test_first_development_image_reproduces_frozen_feature_blocks(self):
        record = self.development.records[0]
        with Image.open(self.development.image_path(record)) as opened:
            image = opened.convert("RGB")
        features, regions = self.predictor.extractor.extract(image)
        cache_root = WORKSPACE_ROOT / ".cache/step_02/pretrained_embeddings"
        with np.load(
            cache_root / "dinov3_vits16__global_fivecrop__2783a020edd6f819.npz",
            allow_pickle=False,
        ) as stored:
            global_reference = stored["embedding__cls"][0]
        with np.load(
            cache_root / "dinov3_vits16__motif_tiles_v4.npz",
            allow_pickle=False,
        ) as stored:
            motif_reference = stored["embedding__texture_weighted_mean"][0]
        np.testing.assert_allclose(features[0, :384], global_reference, rtol=0, atol=3e-7)
        np.testing.assert_allclose(features[0, 384:], motif_reference, rtol=0, atol=3e-7)
        self.assertEqual(len(regions), 6)


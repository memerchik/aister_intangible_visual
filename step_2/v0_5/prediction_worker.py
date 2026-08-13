#!/usr/bin/env python3
"""One-shot DINOv3 prediction worker for low-memory hosted deployments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APPLICATION_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = APPLICATION_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aister_runtime.inference import AssistedOrnamentPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--inference-batch-size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = AssistedOrnamentPredictor(
        application_root=args.application_root,
        weight_path=args.weight_path,
        threads=args.threads,
        inference_batch_size=args.inference_batch_size,
    )
    predictor.warmup()
    result, detected_format = predictor.predict_bytes(args.input.read_bytes())
    args.output.write_text(
        json.dumps(
            {"result": result, "detected_format": detected_format},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

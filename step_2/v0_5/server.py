#!/usr/bin/env python3
"""Dependency-light local server for the AISTER assisted v0.5 application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse


APPLICATION_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = APPLICATION_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aister_runtime.inference import (  # noqa: E402
    APP_VERSION,
    CLASS_CATALOG,
    CLASS_ORDER,
    MAX_UPLOAD_BYTES,
    AssistedInferenceError,
    AssistedOrnamentPredictor,
    InvalidImageError,
    decode_data_url,
)
from aister_runtime.model_bootstrap import ensure_model_weight  # noqa: E402


MAX_REQUEST_BYTES = 17 * 1024 * 1024
MAX_NOTE_LENGTH = 500
PREDICTION_CACHE_SIZE = 32
STATIC_FILES = {
    "/": ("static/index.html", "text/html; charset=utf-8"),
    "/index.html": ("static/index.html", "text/html; charset=utf-8"),
    "/app.css": ("static/app.css", "text/css; charset=utf-8"),
    "/app.js": ("static/app.js", "text/javascript; charset=utf-8"),
    "/aister-logo.png": ("static/aister-logo.png", "image/png"),
    "/favicon.svg": ("static/favicon.svg", "image/svg+xml"),
}


@dataclass(frozen=True)
class PendingPrediction:
    prediction_id: str
    created_at: float
    image_bytes: bytes
    detected_format: str
    file_name: str
    result: Mapping[str, object]


class PredictionRegistry:
    """Bounded memory-only cache used to support explicit contributions."""

    def __init__(self, maximum_size: int = PREDICTION_CACHE_SIZE):
        self.maximum_size = maximum_size
        self._items: "OrderedDict[str, PendingPrediction]" = OrderedDict()
        self._lock = threading.Lock()

    def add(
        self,
        image_bytes: bytes,
        detected_format: str,
        file_name: str,
        result: Mapping[str, object],
    ) -> PendingPrediction:
        prediction_id = "pred_" + uuid.uuid4().hex
        item = PendingPrediction(
            prediction_id=prediction_id,
            created_at=time.time(),
            image_bytes=image_bytes,
            detected_format=detected_format,
            file_name=Path(file_name).name[:120] or "uploaded-image",
            result=result,
        )
        with self._lock:
            self._items[prediction_id] = item
            self._items.move_to_end(prediction_id)
            while len(self._items) > self.maximum_size:
                self._items.popitem(last=False)
        return item

    def get(self, prediction_id: str) -> Optional[PendingPrediction]:
        with self._lock:
            item = self._items.get(prediction_id)
            if item is not None:
                self._items.move_to_end(prediction_id)
            return item

    def remove(self, prediction_id: str) -> None:
        with self._lock:
            self._items.pop(prediction_id, None)


class ContributionStore:
    """Quarantine explicitly consented examples for later expert review."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.Lock()

    @staticmethod
    def _safe_note(value: object) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("The contribution note must be text")
        normalized = " ".join(value.strip().split())
        if len(normalized) > MAX_NOTE_LENGTH:
            raise ValueError("The contribution note is longer than 500 characters")
        return normalized

    def save(
        self,
        pending: PendingPrediction,
        final_label: str,
        note: object,
    ) -> Mapping[str, object]:
        if final_label not in CLASS_ORDER:
            raise ValueError("Choose one of the five supported ornament traditions")
        normalized_note = self._safe_note(note)
        contribution_id = "contrib_" + uuid.uuid4().hex
        extension = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(
            pending.detected_format
        )
        if extension is None:
            raise ValueError("The contributed image format is unsupported")
        self.root.mkdir(parents=True, exist_ok=True)
        image_name = contribution_id + extension
        metadata_name = contribution_id + ".json"
        image_path = self.root / image_name
        metadata_path = self.root / metadata_name
        top_match = pending.result.get("top_match", {})
        suggested_label = top_match.get("label") if isinstance(top_match, dict) else None
        payload = {
            "schema_version": 1,
            "contribution_id": contribution_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "application_version": APP_VERSION,
            "prediction_id": pending.prediction_id,
            "original_file_name": pending.file_name,
            "image_file": image_name,
            "image_byte_count": len(pending.image_bytes),
            "image_content_sha256": hashlib.sha256(pending.image_bytes).hexdigest(),
            "suggested_label": suggested_label,
            "contributor_selected_label": final_label,
            "contributor_selected_name": CLASS_CATALOG[final_label]["name"],
            "changed_from_suggestion": final_label != suggested_label,
            "note": normalized_note,
            "consent": {
                "affirmed": True,
                "text": "I have the right to share this image and agree that it may be stored for research and model improvement.",
            },
            "review_status": "pending_expert_review",
            "rights_review_status": "pending",
            "deduplication_status": "pending",
            "source_object_grouping_status": "pending",
            "expert_label_status": "pending",
            "training_eligible": False,
            "automatic_training_permitted": False,
            "prediction_snapshot": pending.result,
        }
        image_temporary = self.root / ("." + image_name + ".tmp")
        metadata_temporary = self.root / ("." + metadata_name + ".tmp")
        with self._lock:
            try:
                image_temporary.write_bytes(pending.image_bytes)
                metadata_temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(image_temporary, image_path)
                os.replace(metadata_temporary, metadata_path)
            finally:
                if image_temporary.exists():
                    image_temporary.unlink()
                if metadata_temporary.exists():
                    metadata_temporary.unlink()
        return {
            "contribution_id": contribution_id,
            "status": "pending_expert_review",
            "message": "Thank you. The example is stored separately and will not be used for training until an expert reviews it.",
        }


class ApplicationService:
    def __init__(
        self,
        predictor: AssistedOrnamentPredictor,
        contribution_root: Path,
        contributions_enabled: bool = True,
    ):
        self.predictor = predictor
        self.registry = PredictionRegistry()
        self.contributions = ContributionStore(contribution_root)
        self.contributions_enabled = contributions_enabled
        self.started_at = time.time()

    def health(self) -> Mapping[str, object]:
        return {
            "status": "ready",
            "application": "AISTER assisted ornament classifier",
            "version": APP_VERSION,
            "model_status": "provisional_human_assisted",
            "calibrated": False,
            "sealed_test_evaluated": False,
            "contributions_enabled": self.contributions_enabled,
        }

    def public_config(self) -> Mapping[str, object]:
        return {"contributions_enabled": self.contributions_enabled}

    def predict(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        image_value = payload.get("image")
        file_name = payload.get("file_name", "uploaded-image")
        if not isinstance(image_value, str):
            raise InvalidImageError("Select an image before analyzing")
        if not isinstance(file_name, str):
            raise InvalidImageError("The uploaded file name is invalid")
        image_bytes, _declared_mime = decode_data_url(image_value)
        started = time.perf_counter()
        result, detected_format = self.predictor.predict_bytes(image_bytes)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        response = dict(result)
        if self.contributions_enabled:
            pending = self.registry.add(
                image_bytes=image_bytes,
                detected_format=detected_format,
                file_name=file_name,
                result=result,
            )
            response["prediction_id"] = pending.prediction_id
            response["storage"] = "memory_only_unless_explicitly_contributed"
        else:
            response["prediction_id"] = None
            response["storage"] = "none"
        response["processing_ms"] = elapsed_ms
        return response

    def contribute(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        if not self.contributions_enabled:
            raise ValueError("Contribution collection is disabled for this deployment")
        if payload.get("consent") is not True:
            raise ValueError("Explicit contribution consent is required")
        prediction_id = payload.get("prediction_id")
        final_label = payload.get("final_label")
        if not isinstance(prediction_id, str) or not re.fullmatch(r"pred_[0-9a-f]{32}", prediction_id):
            raise ValueError("The prediction is no longer available; analyze the image again")
        if not isinstance(final_label, str):
            raise ValueError("Choose a final ornament label")
        pending = self.registry.get(prediction_id)
        if pending is None:
            raise ValueError("The prediction is no longer in memory; analyze the image again")
        saved = self.contributions.save(pending, final_label, payload.get("note"))
        self.registry.remove(prediction_id)
        return saved


def build_handler(service: ApplicationService):
    class AisterRequestHandler(BaseHTTPRequestHandler):
        server_version = "AisterLocal/0.5"

        def log_message(self, format_string: str, *args: object) -> None:
            sys.stdout.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), format_string % args)
            )

        def _security_headers(self, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )

        def _send_json(self, payload: Mapping[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._security_headers("application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error_json(self, message: str, status: HTTPStatus) -> None:
            self._send_json({"error": message, "status": status.value}, status=status)

        def _read_json(self) -> Mapping[str, object]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                raise ValueError("Requests must use application/json")
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "0")
            except ValueError as error:
                raise ValueError("The request length is invalid") from error
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("The request is empty or larger than the 17 MB limit")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("The request body is not valid JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._send_json(service.health())
                return
            if path == "/api/config":
                self._send_json(service.public_config())
                return
            static = STATIC_FILES.get(path)
            if static is None:
                self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
                return
            relative_path, content_type = static
            file_path = APPLICATION_ROOT / relative_path
            try:
                data = file_path.read_bytes()
            except OSError:
                self._send_error_json("Application asset unavailable", HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self._security_headers(content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/predict":
                    response = service.predict(payload)
                elif path == "/api/contribute":
                    response = service.contribute(payload)
                else:
                    self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
                    return
                self._send_json(response)
            except (InvalidImageError, ValueError) as error:
                self._send_error_json(str(error), HTTPStatus.BAD_REQUEST)
            except AssistedInferenceError as error:
                self._send_error_json(str(error), HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as error:
                self.log_error("Unhandled application error: %s", error)
                self._send_error_json(
                    "The application could not complete this request. Check the server log for details.",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return AisterRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")))
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("AISTER_THREADS", "4"))
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=int(os.environ.get("AISTER_INFERENCE_BATCH_SIZE", "1")),
        help="Number of image views encoded together; keep at 1 on low-memory hosts",
    )
    parser.add_argument("--open", action="store_true", help="Open the local app in the default browser")
    parser.add_argument(
        "--contribution-directory",
        type=Path,
        default=APPLICATION_ROOT / "contributions" / "pending",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    contributions_enabled = os.environ.get("AISTER_ENABLE_CONTRIBUTIONS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    print("Preparing the provisional v0.5 model...", flush=True)
    weight_path = ensure_model_weight()
    predictor = AssistedOrnamentPredictor(
        application_root=APPLICATION_ROOT,
        weight_path=weight_path,
        threads=args.threads,
        inference_batch_size=args.inference_batch_size,
    )
    predictor.warmup()
    service = ApplicationService(
        predictor,
        args.contribution_directory,
        contributions_enabled=contributions_enabled,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"AISTER v0.5 is ready at {url}", flush=True)
    if contributions_enabled:
        print("Predictions are not written to disk unless contribution consent is given.", flush=True)
    else:
        print("Contribution collection is disabled; prediction images are not stored.", flush=True)
    if args.open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AISTER v0.5...", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

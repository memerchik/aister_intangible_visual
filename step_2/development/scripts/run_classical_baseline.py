#!/usr/bin/env python3
"""Run the frozen Step 1 classical baseline on Phase 2 development folds.

The sealed Phase 2 test manifest is deliberately never read by this script.
Model selection and all reported predictions are development-only and out-of-fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
DEFAULT_DEVELOPMENT = STEP_ROOT / "splits" / "development.csv"
DEFAULT_SPLIT_AUDIT = STEP_ROOT / "splits" / "split_audit.json"
DEFAULT_OUTPUT_DIR = STEP_ROOT / "outputs" / "phase_3_classical_baseline"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "step_02"
DATA_ROOT = STEP_ROOT / "data"

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import PIL  # noqa: E402
import skimage  # noqa: E402
import sklearn  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.feature import hog, local_binary_pattern  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402


PHASE = "step02_phase_3_classical_baseline"
BASELINE_VERSION = "step01_frozen_features_grouped_cv_v1"
EXPECTED_FOLDS = ("0", "1", "2", "3", "4")
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "top_3_accuracy",
)


@dataclass(frozen=True)
class RunConfig:
    image_size: tuple[int, int] = (96, 96)
    hsv_bins: tuple[int, int, int] = (10, 4, 4)
    lbp_points: int = 8
    lbp_radius: int = 1
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (16, 16)
    hog_cells_per_block: tuple[int, int] = (2, 2)


@dataclass(frozen=True)
class Candidate:
    model_name: str
    feature_name: str
    factory: Callable[[], Pipeline]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_development(
    rows: list[dict[str, str]], split_audit: dict[str, object]
) -> None:
    production_counts = split_audit.get("production_split_counts")
    if not isinstance(production_counts, dict):
        raise ValueError("Split audit is missing production split counts")
    expected_count = sum(int(production_counts[name]) for name in ("train", "validation"))
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count:,} development rows from the split audit, "
            f"found {len(rows):,}"
        )
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Development image IDs are not unique")
    if {row["production_split"] for row in rows} != {"train", "validation"}:
        raise ValueError("Development input must contain only train and validation records")
    if {row["cv_fold"] for row in rows} != set(EXPECTED_FOLDS):
        raise ValueError("Development input does not contain the expected five folds")
    if any(row["inclusion_status"] != "included" for row in rows):
        raise ValueError("Development input contains an excluded record")
    if {row["split_version"] for row in rows} != {str(split_audit["split_version"])}:
        raise ValueError("Development split version does not match the split audit")
    if {row["split_seed"] for row in rows} != {str(split_audit["split_seed"])}:
        raise ValueError("Development split seed does not match the split audit")

    identifier_policies = (
        ("split_group_id", False),
        ("confirmed_object_group_id", False),
        ("confirmed_source_group_id", True),
        ("content_sha256", False),
    )
    for identifier, allow_blank in identifier_policies:
        for boundary in ("production_split", "cv_fold"):
            assignments: dict[str, set[str]] = {}
            for row in rows:
                value = row[identifier]
                if not value and allow_blank:
                    continue
                if not value:
                    raise ValueError(f"Blank {identifier} in development input")
                assignments.setdefault(value, set()).add(row[boundary])
            crossing = sorted(value for value, observed in assignments.items() if len(observed) > 1)
            if crossing:
                raise ValueError(f"{identifier} crosses {boundary}: {crossing[:5]}")
    missing = [row["relative_path"] for row in rows if not (DATA_ROOT / row["relative_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing development images: {missing[:5]}")


def load_image_array(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path).convert("RGB") as image:
        resized = image.resize(image_size)
        return np.asarray(resized, dtype=np.float32) / 255.0


def extract_feature_groups(image_path: Path, config: RunConfig) -> dict[str, np.ndarray]:
    """Exact extraction logic used by the main Step 1 notebook."""

    rgb = load_image_array(image_path, config.image_size)
    gray = rgb.mean(axis=2)
    hsv = np.asarray(
        Image.fromarray((rgb * 255.0).astype(np.uint8)).convert("HSV"),
        dtype=np.float32,
    ) / 255.0

    color_hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=config.hsv_bins,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    color_hist = color_hist.astype(np.float32).ravel()
    color_hist /= color_hist.sum() + 1e-8

    hog_vector = hog(
        gray,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    lbp = local_binary_pattern(
        (gray * 255.0).astype(np.uint8),
        P=config.lbp_points,
        R=config.lbp_radius,
        method="uniform",
    )
    lbp_hist, _ = np.histogram(
        lbp,
        bins=np.arange(0, config.lbp_points + 3),
        range=(0, config.lbp_points + 2),
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= lbp_hist.sum() + 1e-8

    return {
        "color": color_hist,
        "hog": hog_vector,
        "lbp": lbp_hist,
        "all": np.concatenate([color_hist, hog_vector, lbp_hist]).astype(np.float32),
    }


def feature_fingerprint(rows: list[dict[str, str]], config: RunConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    digest.update(b"\n")
    for row in rows:
        digest.update(f'{row["image_id"]}\x1f{row["content_sha256"]}\n'.encode("utf-8"))
    return digest.hexdigest()


def build_feature_tables(
    rows: list[dict[str, str]], config: RunConfig, cache_dir: Path, use_cache: bool
) -> tuple[dict[str, np.ndarray], str, Path, bool]:
    fingerprint = feature_fingerprint(rows, config)
    cache_path = cache_dir / f"classical_features_{fingerprint[:20]}.npz"
    if use_cache and cache_path.is_file():
        with np.load(cache_path) as cached:
            cached_ids = [str(value) for value in cached["image_ids"]]
            if cached_ids == [row["image_id"] for row in rows]:
                tables = {name: cached[name] for name in ("color", "hog", "lbp", "all")}
                print(f"Loaded feature cache: {cache_path}", flush=True)
                return tables, fingerprint, cache_path, True

    store: dict[str, list[np.ndarray]] = {name: [] for name in ("color", "hog", "lbp", "all")}
    for index, row in enumerate(rows, start=1):
        groups = extract_feature_groups(DATA_ROOT / row["relative_path"], config)
        for name, vector in groups.items():
            store[name].append(vector)
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"Extracted features: {index}/{len(rows)}", flush=True)

    tables = {name: np.vstack(vectors) for name, vectors in store.items()}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            image_ids=np.asarray([row["image_id"] for row in rows]),
            **tables,
        )
        print(f"Saved feature cache: {cache_path}", flush=True)
    return tables, fingerprint, cache_path, False


def build_candidates() -> list[Candidate]:
    """Frozen candidate set and hyperparameters from the Step 1 notebook."""

    return [
        Candidate(
            "linear_svm",
            "color",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000),
                    ),
                ]
            ),
        ),
        Candidate(
            "logistic_regression",
            "all",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
        ),
        *[
            Candidate(
                f"knn_cosine_k{neighbors}",
                "all",
                lambda neighbors=neighbors: Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            KNeighborsClassifier(
                                n_neighbors=neighbors,
                                weights="distance",
                                metric="cosine",
                            ),
                        ),
                    ]
                ),
            )
            for neighbors in (3, 5, 7)
        ],
    ]


def probabilities_for(pipeline: Pipeline, features: np.ndarray, class_count: int) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        raw = pipeline.predict_proba(features)
    else:
        decision = pipeline.decision_function(features)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        stabilized = decision - decision.max(axis=1, keepdims=True)
        exp_scores = np.exp(stabilized)
        raw = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    result = np.zeros((len(features), class_count), dtype=np.float64)
    for source_index, class_index in enumerate(pipeline.classes_):
        result[:, int(class_index)] = raw[:, source_index]
    return result


def calculate_metrics(
    truth: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray, class_count: int
) -> dict[str, float]:
    labels = np.arange(class_count)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    top_three = np.argsort(probabilities, axis=1)[:, -3:]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top_3_accuracy": float(np.mean(np.any(top_three == truth[:, None], axis=1))),
    }


def evaluate_candidates(
    rows: list[dict[str, str]],
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    class_count: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    folds = np.asarray([row["cv_fold"] for row in rows])
    production = np.asarray([row["production_split"] for row in rows])
    fold_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    candidate_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for candidate in build_candidates():
        print(f"Evaluating {candidate.model_name} ({candidate.feature_name})", flush=True)
        X = features[candidate.feature_name]
        oof_predicted = np.full(len(rows), -1, dtype=np.int64)
        oof_probabilities = np.zeros((len(rows), class_count), dtype=np.float64)

        for fold in EXPECTED_FOLDS:
            validation_indices = np.flatnonzero(folds == fold)
            training_indices = np.flatnonzero(folds != fold)
            pipeline = candidate.factory()
            pipeline.fit(X[training_indices], labels[training_indices])
            predicted = pipeline.predict(X[validation_indices]).astype(np.int64)
            probabilities = probabilities_for(pipeline, X[validation_indices], class_count)
            oof_predicted[validation_indices] = predicted
            oof_probabilities[validation_indices] = probabilities
            metrics = calculate_metrics(labels[validation_indices], predicted, probabilities, class_count)
            fold_rows.append(
                {
                    "model_name": candidate.model_name,
                    "feature_name": candidate.feature_name,
                    "cv_fold": fold,
                    "train_examples": len(training_indices),
                    "validation_examples": len(validation_indices),
                    **metrics,
                }
            )
            print(f"  fold {fold}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}", flush=True)

        if np.any(oof_predicted < 0):
            raise RuntimeError(f"Missing out-of-fold prediction for {candidate.model_name}")
        candidate_predictions[candidate.model_name] = (oof_predicted, oof_probabilities)

        train_indices = np.flatnonzero(production == "train")
        validation_indices = np.flatnonzero(production == "validation")
        pipeline = candidate.factory()
        pipeline.fit(X[train_indices], labels[train_indices])
        fixed_predicted = pipeline.predict(X[validation_indices]).astype(np.int64)
        fixed_probabilities = probabilities_for(pipeline, X[validation_indices], class_count)
        fixed_metrics = calculate_metrics(
            labels[validation_indices], fixed_predicted, fixed_probabilities, class_count
        )
        fixed_rows.append(
            {
                "model_name": candidate.model_name,
                "feature_name": candidate.feature_name,
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                **fixed_metrics,
            }
        )

    summary_rows: list[dict[str, object]] = []
    for candidate in build_candidates():
        candidate_folds = [row for row in fold_rows if row["model_name"] == candidate.model_name]
        summary: dict[str, object] = {
            "model_name": candidate.model_name,
            "feature_name": candidate.feature_name,
            "fold_count": len(candidate_folds),
        }
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in candidate_folds])
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1))
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            -float(row["mean_accuracy"]),
            -float(row["mean_macro_f1"]),
            str(row["model_name"]),
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["selection_rank"] = rank
    return fold_rows, fixed_rows, summary_rows, candidate_predictions


def per_class_rows(
    truth: np.ndarray, predicted: np.ndarray, class_names: list[str]
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return [
        {
            "class_index": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]


def diagnostic_slice_rows(
    rows: list[dict[str, str]], predicted: np.ndarray, probabilities: np.ndarray, labels: np.ndarray
) -> list[dict[str, object]]:
    dimensions: dict[str, list[str]] = {
        "old_split": [row["old_split"] for row in rows],
        "review_cohort": [
            "manually_reviewed" if row["manual_review_status"] == "complete" else "singleton"
            for row in rows
        ],
        "motif_visibility": [row["motif_visibility"] for row in rows],
        "object_type": [row["object_type"] for row in rows],
        "source_cohort": [
            row.get("source_atomic_cohort_ids", "") or "unassigned" for row in rows
        ],
        "cv_fold": [row["cv_fold"] for row in rows],
    }
    confidence = probabilities.max(axis=1)
    result: list[dict[str, object]] = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            indices = np.asarray([index for index, observed in enumerate(values) if observed == value])
            if len(indices) < 5:
                continue
            correct = predicted[indices] == labels[indices]
            result.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "image_count": len(indices),
                    "correct_count": int(correct.sum()),
                    "error_count": int((~correct).sum()),
                    "accuracy": float(correct.mean()),
                    "mean_confidence": float(confidence[indices].mean()),
                }
            )
    return result


def confusion_pair_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_index, true_class in enumerate(class_names):
        support = int(confusion[true_index].sum())
        for predicted_index, predicted_class in enumerate(class_names):
            if true_index == predicted_index or confusion[true_index, predicted_index] == 0:
                continue
            count = int(confusion[true_index, predicted_index])
            rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "error_count": count,
                    "fraction_of_true_class": count / support,
                }
            )
    rows.sort(key=lambda row: (-int(row["error_count"]), str(row["true_class"]), str(row["predicted_class"])))
    return rows


def save_confusion_plot(confusion: np.ndarray, class_names: list[str], path: Path) -> None:
    short_names = [name.split("_", 1)[1].replace("_", " ") for name in class_names]
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=short_names, rotation=35, ha="right")
    axis.set_yticks(range(len(class_names)), labels=short_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Classical baseline: development out-of-fold confusion matrix")
    threshold = confusion.max() / 2 if confusion.size else 0
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            value = int(confusion[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.split_audit.open(encoding="utf-8") as handle:
        split_audit = json.load(handle)
    rows = read_csv(args.development)
    validate_development(rows, split_audit)

    config = RunConfig()
    class_names = sorted({row["ornament_label"] for row in rows})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_index[row["ornament_label"]] for row in rows], dtype=np.int64)

    features, feature_hash, cache_path, _ = build_feature_tables(
        rows,
        config,
        args.cache_dir,
        use_cache=not args.no_cache,
    )
    fold_rows, fixed_rows, summary_rows, predictions_by_candidate = evaluate_candidates(
        rows,
        features,
        labels,
        len(class_names),
    )

    selected_name = str(summary_rows[0]["model_name"])
    selected_feature = str(summary_rows[0]["feature_name"])
    predicted, probabilities = predictions_by_candidate[selected_name]
    aggregate = calculate_metrics(labels, predicted, probabilities, len(class_names))
    per_class = per_class_rows(labels, predicted, class_names)
    confusion = confusion_matrix(labels, predicted, labels=np.arange(len(class_names)))
    top_order = np.argsort(probabilities, axis=1)[:, ::-1]

    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        output: dict[str, object] = {
            "image_id": row["image_id"],
            "relative_path": row["relative_path"],
            "true_class": row["ornament_label"],
            "predicted_class": class_names[int(predicted[index])],
            "is_correct": str(int(predicted[index]) == int(labels[index])),
            "confidence": float(probabilities[index].max()),
            "top_1_class": class_names[int(top_order[index, 0])],
            "top_2_class": class_names[int(top_order[index, 1])],
            "top_3_class": class_names[int(top_order[index, 2])],
            "cv_fold": row["cv_fold"],
            "production_split": row["production_split"],
            "old_split": row["old_split"],
            "manual_review_status": row["manual_review_status"],
            "confirmed_object_group_id": row["confirmed_object_group_id"],
            "confirmed_source_group_id": row["confirmed_source_group_id"],
            "split_group_id": row["split_group_id"],
            "global_source_cohort_id": row.get("global_source_cohort_id", ""),
            "source_atomic_cohort_ids": row.get("source_atomic_cohort_ids", ""),
            "object_type": row["object_type"],
            "motif_visibility": row["motif_visibility"],
        }
        for class_index, class_name in enumerate(class_names):
            output[f"probability_{class_name}"] = float(probabilities[index, class_index])
        prediction_rows.append(output)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_fields = (
        "model_name",
        "feature_name",
        "cv_fold",
        "train_examples",
        "validation_examples",
        *METRIC_NAMES,
    )
    fixed_fields = (
        "model_name",
        "feature_name",
        "train_examples",
        "validation_examples",
        *METRIC_NAMES,
    )
    summary_fields = (
        "selection_rank",
        "model_name",
        "feature_name",
        "fold_count",
        *(field for metric in METRIC_NAMES for field in (f"mean_{metric}", f"std_{metric}")),
    )
    prediction_fields = tuple(prediction_rows[0].keys())
    write_csv(output_dir / "cv_fold_metrics.csv", fold_rows, fold_fields)
    write_csv(output_dir / "fixed_validation_metrics.csv", fixed_rows, fixed_fields)
    write_csv(output_dir / "model_selection_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "oof_predictions.csv", prediction_rows, prediction_fields)
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ("class_index", "class_name", "precision", "recall", "f1", "support"),
    )
    write_csv(
        output_dir / "diagnostic_slices.csv",
        diagnostic_slice_rows(rows, predicted, probabilities, labels),
        ("dimension", "value", "image_count", "correct_count", "error_count", "accuracy", "mean_confidence"),
    )
    write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pair_rows(confusion, class_names),
        ("true_class", "predicted_class", "error_count", "fraction_of_true_class"),
    )
    confusion_rows = [
        {"true_class": class_name, **{name: int(confusion[index, column]) for column, name in enumerate(class_names)}}
        for index, class_name in enumerate(class_names)
    ]
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows, ("true_class", *class_names))
    save_confusion_plot(confusion, class_names, output_dir / "confusion_matrix.png")

    selected_fold_rows = [row for row in fold_rows if row["model_name"] == selected_name]
    fixed_selected = next(row for row in fixed_rows if row["model_name"] == selected_name)
    metrics_payload = {
        "phase": PHASE,
        "baseline_version": BASELINE_VERSION,
        "evaluation_scope": "development_only_out_of_fold",
        "sealed_test_evaluated": False,
        "selection_rule": "highest mean five-fold accuracy; ties use mean macro F1 then model name",
        "split_version": split_audit["split_version"],
        "split_seed": split_audit["split_seed"],
        "split_assignment_fingerprint_sha256": split_audit["assignment_fingerprint_sha256"],
        "development_image_count": len(rows),
        "train_image_count": sum(row["production_split"] == "train" for row in rows),
        "validation_image_count": sum(row["production_split"] == "validation" for row in rows),
        "class_names": class_names,
        "selected_model": selected_name,
        "selected_feature_set": selected_feature,
        "aggregate_oof_metrics": aggregate,
        "selected_model_fold_metrics": selected_fold_rows,
        "selected_model_fixed_validation_metrics": fixed_selected,
        "per_class_metrics": per_class,
        "feature_config": asdict(config),
        "feature_dimensions": {name: int(values.shape[1]) for name, values in features.items()},
        "feature_fingerprint_sha256": feature_hash,
        "feature_cache_path": str(cache_path.relative_to(REPO_ROOT)),
        "candidate_count": len(summary_rows),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "selected_model": selected_name,
                "selected_feature_set": selected_feature,
                "aggregate_oof_metrics": aggregate,
                "sealed_test_evaluated": False,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

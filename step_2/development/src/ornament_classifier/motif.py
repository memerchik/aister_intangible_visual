"""Deterministic, label-free motif proposals and tile aggregation for Phase 5 v4.

This module accepts pixels or arrays supplied by its caller. It has no project
path, split-manifest, model, or sealed-evaluation access. Proposal selection is
based only on within-image texture and colour statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MotifProposal:
    proposal_id: str
    left: int
    top: int
    right: int
    bottom: int
    scale: float
    x_position: float
    y_position: float
    texture_score: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def proposal_pool(
    width: int,
    height: int,
    *,
    scales: Sequence[float] = (0.45, 0.65),
    positions: Sequence[float] = (0.0, 0.5, 1.0),
) -> Tuple[Tuple[str, Tuple[int, int, int, int], float, float, float], ...]:
    """Return a stable multiscale square-window pool in original coordinates."""

    width = _positive_dimension(width, "width")
    height = _positive_dimension(height, "height")
    if not scales or not positions:
        raise ValueError("scales and positions must be non-empty")
    if any(
        isinstance(value, bool) or not np.isfinite(value) or value <= 0 or value > 1
        for value in scales
    ):
        raise ValueError("scales must contain finite values in (0, 1]")
    if any(
        isinstance(value, bool) or not np.isfinite(value) or value < 0 or value > 1
        for value in positions
    ):
        raise ValueError("positions must contain finite values in [0, 1]")
    if len(set(float(value) for value in scales)) != len(scales):
        raise ValueError("scales must be unique")
    if len(set(float(value) for value in positions)) != len(positions):
        raise ValueError("positions must be unique")

    shortest = min(width, height)
    result = []
    observed_boxes = set()
    for scale_index, scale_value in enumerate(scales):
        scale = float(scale_value)
        side = max(1, min(shortest, int(round(shortest * scale))))
        for y_index, y_value in enumerate(positions):
            y_position = float(y_value)
            top = int(round(y_position * (height - side)))
            for x_index, x_value in enumerate(positions):
                x_position = float(x_value)
                left = int(round(x_position * (width - side)))
                box = (left, top, left + side, top + side)
                if box in observed_boxes:
                    continue
                observed_boxes.add(box)
                result.append(
                    (
                        f"s{scale_index}_y{y_index}_x{x_index}",
                        box,
                        scale,
                        x_position,
                        y_position,
                    )
                )
    if not result:
        raise ValueError("Proposal policy produced no boxes")
    return tuple(result)


def texture_score(image: Image.Image, box: Sequence[int], score_size: int = 64) -> float:
    """Score a crop with fixed edge, luminance-variation, and chroma terms."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    _positive_dimension(score_size, "score_size")
    if len(box) != 4:
        raise ValueError("box must have four coordinates")
    left, top, right, bottom = (int(value) for value in box)
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("box is invalid")
    if right > image.width or bottom > image.height:
        raise ValueError("box exceeds the image")
    crop = image.convert("RGB").crop((left, top, right, bottom)).resize(
        (score_size, score_size), Image.Resampling.BILINEAR
    )
    rgb = np.asarray(crop, dtype=np.float64) / 255.0
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    edge_energy = float((horizontal + vertical) / 2.0)
    luminance_variation = float(gray.std())
    chroma = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    score = 0.50 * edge_energy + 0.30 * luminance_variation + 0.20 * chroma
    if not np.isfinite(score):
        raise ValueError("texture score is not finite")
    return score


def intersection_over_union(first: Sequence[int], second: Sequence[int]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("boxes must have four coordinates")
    a_left, a_top, a_right, a_bottom = (int(value) for value in first)
    b_left, b_top, b_right, b_bottom = (int(value) for value in second)
    if a_right <= a_left or a_bottom <= a_top or b_right <= b_left or b_bottom <= b_top:
        raise ValueError("boxes must have positive area")
    intersection_width = max(0, min(a_right, b_right) - max(a_left, b_left))
    intersection_height = max(0, min(a_bottom, b_bottom) - max(a_top, b_top))
    intersection = intersection_width * intersection_height
    union = (
        (a_right - a_left) * (a_bottom - a_top)
        + (b_right - b_left) * (b_bottom - b_top)
        - intersection
    )
    return float(intersection / union)


def select_motif_proposals(
    image: Image.Image,
    *,
    scales: Sequence[float] = (0.45, 0.65),
    positions: Sequence[float] = (0.0, 0.5, 1.0),
    proposal_count: int = 6,
    maximum_iou: float = 0.60,
    score_size: int = 64,
) -> Tuple[MotifProposal, ...]:
    """Select a fixed number of diverse, high-texture proposals without labels."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    _positive_dimension(proposal_count, "proposal_count")
    if (
        isinstance(maximum_iou, bool)
        or not np.isfinite(maximum_iou)
        or maximum_iou < 0
        or maximum_iou > 1
    ):
        raise ValueError("maximum_iou must be in [0, 1]")
    candidates = []
    for identifier, box, scale, x_position, y_position in proposal_pool(
        image.width, image.height, scales=scales, positions=positions
    ):
        candidates.append(
            MotifProposal(
                identifier,
                *box,
                scale,
                x_position,
                y_position,
                texture_score(image, box, score_size),
            )
        )
    candidates.sort(key=lambda item: (-item.texture_score, item.proposal_id))
    selected = []
    for candidate in candidates:
        if all(
            intersection_over_union(candidate.box, previous.box) <= maximum_iou
            for previous in selected
        ):
            selected.append(candidate)
            if len(selected) == proposal_count:
                break
    if len(selected) < proposal_count:
        observed = {item.proposal_id for item in selected}
        selected.extend(
            item
            for item in candidates
            if item.proposal_id not in observed
        )
        selected = selected[:proposal_count]
    if len(selected) != proposal_count:
        raise ValueError(
            f"Proposal pool cannot supply {proposal_count} unique crops"
        )
    return tuple(selected)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("values must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("values must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("values must have nonzero row norms")
    return matrix / norms


def aggregate_tile_embeddings(
    tile_embeddings: np.ndarray, texture_scores: Sequence[float]
) -> Mapping[str, np.ndarray]:
    """Create the four frozen v4 motif descriptors for one image."""

    tiles = _normalize_rows(np.asarray(tile_embeddings))
    scores = np.asarray(texture_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(tiles):
        raise ValueError("texture_scores must have one value per tile")
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("at least two finite texture scores are required")
    centered = scores - float(scores.mean())
    scale = float(scores.std())
    standardized = centered / max(scale, 1e-12)
    exponentials = np.exp(standardized - standardized.max())
    weights = exponentials / exponentials.sum()
    uniform_mean = tiles.mean(axis=0)
    texture_weighted_mean = np.sum(tiles * weights[:, None], axis=0)
    top_two = np.argsort(-scores, kind="stable")[:2]
    top2_mean = tiles[top_two].mean(axis=0)
    centered_tiles = tiles - texture_weighted_mean[None, :]
    dispersion = np.sqrt(
        np.sum(weights[:, None] * centered_tiles * centered_tiles, axis=0)
    )
    descriptors = {
        "uniform_mean": uniform_mean,
        "texture_weighted_mean": texture_weighted_mean,
        "texture_top2_mean": top2_mean,
        "texture_dispersion": dispersion,
    }
    result = {}
    for name, vector in descriptors.items():
        normalized = _normalize_rows(np.asarray(vector)[None, :])[0]
        result[name] = normalized.astype(np.float32)
    return result


def augment_feature_blocks(
    base_blocks: Sequence[np.ndarray], motif_descriptor: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Concatenate one image-level motif descriptor to every selected base block."""

    blocks = tuple(np.asarray(block) for block in base_blocks)
    descriptor = np.asarray(motif_descriptor)
    if not blocks or descriptor.ndim != 2 or not descriptor.shape[1]:
        raise ValueError("base blocks and motif_descriptor must be non-empty matrices")
    if not np.isfinite(descriptor).all():
        raise ValueError("motif_descriptor must be finite")
    row_count = descriptor.shape[0]
    result = []
    for block in blocks:
        if block.ndim != 2 or block.shape[0] != row_count or not block.shape[1]:
            raise ValueError("all feature blocks must align with motif_descriptor")
        if not np.isfinite(block).all():
            raise ValueError("base feature blocks must be finite")
        result.append(np.concatenate((block, descriptor), axis=1))
    return tuple(result)

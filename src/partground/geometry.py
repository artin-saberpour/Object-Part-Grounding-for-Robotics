"""Intersect 2D intervals from explicitly calibrated orthographic views.

This is an approximation based on bounding boxes, not calibrated reconstruction
from arbitrary camera images. See the README for the coordinate contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import math


@dataclass(frozen=True)
class ViewBox:
    box: tuple[float, float, float, float]
    image_size: tuple[int, int]
    horizontal_axis: str
    vertical_axis: str
    flip_horizontal: bool = False
    flip_vertical: bool = False


def _interval(start: float, end: float, length: int, flip: bool) -> tuple[float, float]:
    if length <= 0 or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Image dimensions must be positive and box coordinates finite")
    if start >= end:
        raise ValueError("Each box must have positive width and height")
    lo, hi = max(0.0, start / length), min(1.0, end / length)
    if lo >= hi:
        raise ValueError("Box does not overlap the image")
    return (1.0 - hi, 1.0 - lo) if flip else (lo, hi)


def intersect_views(
    views: Iterable[ViewBox],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> Mapping[str, list[float]] | None:
    """Intersect normalized view boxes and return their 3D extent and center.

    Each view must describe which 3D axis increases along each image axis.
    The flip flags explicitly reverse an image axis before intersection.
    Every 3D axis must be constrained by at least one view.
    """
    if len(bounds_min) != 3 or len(bounds_max) != 3:
        raise ValueError("Bounds must each contain three values")
    if any(not math.isfinite(v) for v in (*bounds_min, *bounds_max)):
        raise ValueError("Bounds must be finite")
    if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max)):
        raise ValueError("Every maximum bound must exceed its minimum")

    axes = {name: [] for name in "xyz"}
    for view in views:
        ha, va = view.horizontal_axis.lower(), view.vertical_axis.lower()
        if ha not in axes or va not in axes or ha == va:
            raise ValueError("Each view must name two different axes from x, y, z")
        x0, y0, x1, y1 = view.box
        width, height = view.image_size
        axes[ha].append(_interval(x0, x1, width, view.flip_horizontal))
        axes[va].append(_interval(y0, y1, height, view.flip_vertical))
    if any(not intervals for intervals in axes.values()):
        raise ValueError("Views must constrain all three axes")

    minimum, maximum = [], []
    for index, axis in enumerate("xyz"):
        lower = max(interval[0] for interval in axes[axis])
        upper = min(interval[1] for interval in axes[axis])
        if lower >= upper:
            return None
        extent = bounds_max[index] - bounds_min[index]
        minimum.append(bounds_min[index] + lower * extent)
        maximum.append(bounds_min[index] + upper * extent)
    return {
        "bbox_min": minimum,
        "bbox_max": maximum,
        "center": [(a + b) / 2 for a, b in zip(minimum, maximum)],
    }

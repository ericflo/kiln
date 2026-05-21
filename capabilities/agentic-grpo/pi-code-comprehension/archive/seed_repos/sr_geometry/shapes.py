"""Basic 2D geometry primitives and area/perimeter routines."""

from __future__ import annotations

import math
from typing import List, Tuple


def rect_area(width: float, height: float) -> float:
    """Return the area of a rectangle.

    Args:
        width:  must be non-negative.
        height: must be non-negative.

    Raises ValueError if either input is negative.
    """
    if width < 0 or height < 0:
        raise ValueError("dimensions must be non-negative")
    return width * height


def rect_perimeter(width: float, height: float) -> float:
    """Return the perimeter of a rectangle.

    Requires width and height to be non-negative.
    """
    if width < 0 or height < 0:
        raise ValueError("dimensions must be non-negative")
    return 2.0 * (width + height)


def triangle_area(a: float, b: float, c: float) -> float:
    """Heron's formula. All sides must be positive and satisfy
    the triangle inequality."""
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("sides must be positive")
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("triangle inequality not satisfied")
    s = (a + b + c) / 2.0
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def polygon_perimeter(points: List[Tuple[float, float]]) -> float:
    """Closed-polygon perimeter: sum of euclidean edge lengths.

    Requires at least 3 points; assumes points are in order.
    """
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def scale_points(points: List[Tuple[float, float]], factor: float) -> None:
    """In-place uniform scaling of points by `factor`.

    Mutates the input list; raises if factor is zero (degenerate).
    """
    if factor == 0:
        raise ValueError("factor must be non-zero")
    for i, (x, y) in enumerate(points):
        points[i] = (x * factor, y * factor)

"""Example consumer that imports shapes.* and assembles a small report."""

from __future__ import annotations

from shapes import rect_area, rect_perimeter, triangle_area, polygon_perimeter, scale_points


def report_rectangle(w: float, h: float) -> dict:
    return {
        "area": rect_area(w, h),
        "perimeter": rect_perimeter(w, h),
    }


def report_triangle(a: float, b: float, c: float) -> dict:
    return {"area": triangle_area(a, b, c)}


def main() -> None:
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    p = polygon_perimeter(points)
    scale_points(points, 2.0)
    p2 = polygon_perimeter(points)
    print(f"perimeter before scale: {p}; after: {p2}")

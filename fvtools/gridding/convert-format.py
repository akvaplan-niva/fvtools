"""
This script converts (in place) islands.txt or boundary.txt file from old
format to current format.

Usage:
    $ python convert-format.py islands.txt
    $ python convert-format.py boundary.txt
"""


import sys
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float
    r: float
    fixed: bool


def first_matches_last(points):
    first = points[0]
    last = points[-1]
    small = sys.float_info.epsilon
    if abs(first.x - last.x) > small:
        return False
    if abs(first.y - last.y) > small:
        return False
    return True


def unclose_polygon(points):
    if first_matches_last(points):
        return points[:-1]
    else:
        return points


def read_polygons(file_name):
    polygons = []
    with open(file_name, "r") as f:
        while True:
            try:
                line = next(f)
            except StopIteration:
                break
            num_points = int(line)
            points = []
            for _ in range(num_points):
                line = next(f)
                numbers = list(map(float, line.split()))
                if len(numbers) == 3:
                    x, y, r = numbers
                    points.append(Point(x, y, r, False))
                else:
                    x, y, r, m = numbers
                    is_fixed = abs(m) < sys.float_info.epsilon
                    points.append(Point(x, y, r, is_fixed))
            polygons.append(points)
    return polygons


def get_fixed_edges(polygon):
    edges = []
    for i, point in enumerate(polygon):
        j = (i + 1) % len(polygon)
        next_point = polygon[j]
        if point.fixed and next_point.fixed:
            edges.append((i, j))
    return edges


def write_polygons(polygons, is_boundary_file, file_name):
    if is_boundary_file:
        tag = "boundary"
    else:
        tag = "island"
    with open(file_name, "w") as f:
        for polygon in polygons:
            edges = get_fixed_edges(polygon)
            f.write(f"{tag} {len(polygon)} {len(edges)}\n")
            for point in polygon:
                f.write(f"{point.x} {point.y} {point.r}\n")
            for (i, j) in edges:
                f.write(f"{i} {j}\n")


if __name__ == "__main__":
    file_name = sys.argv[-1]
    polygons = read_polygons(file_name)
    is_boundary_file = "boundary" in file_name
    polygons = list(map(unclose_polygon, polygons))
    write_polygons(polygons, is_boundary_file, file_name)

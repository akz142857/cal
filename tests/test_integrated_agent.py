"""Tests for the V2-I1 connected-component detection front end."""

import numpy as np

from cal.model.integrated_agent import connected_component_centroids


def _grid(*rows: str) -> np.ndarray:
    return np.asarray(
        [[1 if char == "#" else 0 for char in row] for row in rows],
        dtype=np.uint8,
    )


def test_empty_patch_returns_no_detections() -> None:
    patch = _grid("...", "...", "...")

    result = connected_component_centroids(patch, 0, 0)

    assert result.shape == (0, 2)


def test_isolated_cells_are_separate_detections() -> None:
    patch = _grid(
        "#...",
        "....",
        "..#.",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 2
    points = {tuple(row) for row in result}
    assert points == {(0.0, 0.0), (2.0, 2.0)}


def test_touching_cells_merge_into_one_centroid() -> None:
    # A 1x3 horizontal blob: an isolation filter would drop all three
    # cells (each has a neighbor); connected components must report one
    # detection at the blob's centroid instead.
    patch = _grid(
        "....",
        ".###",
        "....",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 1
    assert tuple(result[0]) == (2.0, 1.0)


def test_diagonal_cells_merge_under_eight_connectivity() -> None:
    patch = _grid(
        "#..",
        ".#.",
        "..#",
    )

    merged = connected_component_centroids(patch, 0, 0, connectivity=8)
    separate = connected_component_centroids(patch, 0, 0, connectivity=4)

    assert len(merged) == 1
    assert len(separate) == 3


def test_offset_shifts_absolute_coordinates() -> None:
    patch = _grid("#.", "..")

    result = connected_component_centroids(patch, 5, 9)

    assert tuple(result[0]) == (5.0, 9.0)


def test_dense_wall_is_a_single_stable_blob() -> None:
    # A vertical wall of six adjacent cells (as in the V2-I1 arena) must
    # collapse to one centroid, not six flickering single-cell detections.
    patch = _grid(
        ".#.",
        ".#.",
        ".#.",
        ".#.",
        ".#.",
        ".#.",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 1
    assert tuple(result[0]) == (1.0, 2.5)

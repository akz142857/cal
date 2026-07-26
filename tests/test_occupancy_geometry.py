"""Tests for the shared world-simulator line-of-sight helper."""

import numpy as np

from cal.model.occupancy import VIEW_RADIUS, sense_via_line_of_sight


def _truth_grid(size: int, occupied: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    for x, y in occupied:
        grid[y, x] = 1
    return grid


def test_empty_truth_is_fully_visible_and_empty() -> None:
    truth = _truth_grid(25, set())

    sensed, visibility = sense_via_line_of_sight((12, 12), truth)

    assert not sensed.any()
    assert visibility.all()


def test_camera_cell_itself_is_always_sensed_if_occupied() -> None:
    truth = _truth_grid(25, {(12, 12)})

    sensed, visibility = sense_via_line_of_sight((12, 12), truth)

    size = 2 * VIEW_RADIUS + 1
    center = size // 2
    assert sensed[center, center] == 1
    assert visibility[center, center] == 1


def test_cell_directly_behind_an_occupier_is_hidden() -> None:
    # Occupier at (14, 12) sits between the camera (12, 12) and (16, 12).
    truth = _truth_grid(25, {(14, 12), (16, 12)})

    sensed, visibility = sense_via_line_of_sight((12, 12), truth)

    size = 2 * VIEW_RADIUS + 1
    center = size // 2
    occluder_local = (center + 2, center)
    hidden_local = (center + 4, center)
    assert sensed[occluder_local[1], occluder_local[0]] == 1
    assert visibility[hidden_local[1], hidden_local[0]] == 0
    assert sensed[hidden_local[1], hidden_local[0]] == 0


def test_view_radius_controls_patch_size() -> None:
    truth = _truth_grid(25, set())

    sensed, visibility = sense_via_line_of_sight((12, 12), truth, view_radius=3)

    assert sensed.shape == visibility.shape == (7, 7)

"""Belief-free smearing baseline for the permanence battery.

This is the adversarial control the 2026-08-08 review built to show that the
confirmatory gates did not test what they were meant to test: it passed every
gate while performing no belief filtering whatsoever.  Keeping it as a first
class reference predictor turns that attack into a permanent regression guard,
so a candidate must show what it adds *over* calibrated smearing rather than
merely beating a point-mass extrapolation.

What it does:

1.  extrapolate the last seen position and velocity forward with the same
    reflecting advance the ``geometric`` baseline uses;
2.  spread probability around that cell using a table fitted on training
    seeds, keyed only by geometry -- occlusion length, offset along and across
    the heading, movement parity, and reachability.

What it deliberately does NOT do: propagate a posterior step by step, condition
on the object having stayed unobserved, or represent the occluded region's
topology.  Any advantage a candidate holds over this baseline is attributable
to those, which is exactly the claim the permanence experiment makes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cal.evaluation.permanence_forward_benchmark import _Sample


MAXIMUM_ALONG_OFFSET = 6
MAXIMUM_ACROSS_OFFSET = 6
MAXIMUM_OCCLUSION_KEY = 12
_PRIOR_STRENGTH = 1.0


def _anchor(sample: "_Sample", track: tuple) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reflecting constant-velocity extrapolation: the shortcut's best guess."""

    from cal.evaluation.permanence_forward_benchmark import _bounce_advance

    last_seen, velocity, hidden_steps = track
    position, heading = last_seen, velocity
    for _ in range(hidden_steps):
        position, heading = _bounce_advance(position, heading, sample.static)
    return position, heading


def _key(
    cell: tuple[int, int],
    *,
    anchor: tuple[int, int],
    heading: tuple[int, int],
    last_seen: tuple[int, int],
    hidden_steps: int,
) -> tuple[int, int, int, int, int]:
    dx = cell[0] - anchor[0]
    dy = cell[1] - anchor[1]
    hx, hy = heading
    along = dx * hx + dy * hy
    across = abs(dx * hy - dy * hx)
    walked = abs(cell[0] - last_seen[0]) + abs(cell[1] - last_seen[1])
    parity = (walked - hidden_steps) % 2
    reachable = int(walked <= hidden_steps)
    return (
        min(hidden_steps, MAXIMUM_OCCLUSION_KEY),
        max(-MAXIMUM_ALONG_OFFSET, min(MAXIMUM_ALONG_OFFSET, int(along))),
        min(MAXIMUM_ACROSS_OFFSET, int(across)),
        parity,
        reachable,
    )


def _sample_keys(
    sample: "_Sample",
) -> tuple[dict[tuple[int, int], tuple], dict[tuple[int, int], int]]:
    """Key每个场内格子，按锚点最近的轨道归属。

    With several hidden objects a cell sits at a different offset from each
    track's anchor.  Attributing it to the *nearest* anchor is the strongest
    choice available to a belief-free model, and the control has to be given
    its strongest form or the gate it anchors is argued against a strawman.
    """

    anchors = [_anchor(sample, track) for track in sample.hidden_tracks]
    keys: dict[tuple[int, int], tuple] = {}
    distances: dict[tuple[int, int], int] = {}
    for cell in sample.candidate_cells:
        best_key = None
        best_distance = None
        for (anchor, heading), track in zip(anchors, sample.hidden_tracks):
            distance = abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
            if best_distance is None or distance < best_distance:
                last_seen, _velocity, hidden_steps = track
                best_distance = distance
                best_key = _key(
                    cell,
                    anchor=anchor,
                    heading=heading,
                    last_seen=last_seen,
                    hidden_steps=hidden_steps,
                )
        if best_key is not None:
            keys[cell] = best_key
            distances[cell] = int(best_distance)
    return keys, distances


def fit_belief_free_table(train: list["_Sample"]) -> dict[tuple, float]:
    """Fit P(cell hides an object | geometry key) on training samples only."""

    if not train:
        raise ValueError("belief-free training samples must not be empty")
    positive = defaultdict(float)
    total = defaultdict(float)
    for sample in train:
        positives = set(sample.positives)
        keys, _distances = _sample_keys(sample)
        for cell, key in keys.items():
            total[key] += 1.0
            if cell in positives:
                positive[key] += 1.0
    return {
        key: (positive[key] + 0.5 * _PRIOR_STRENGTH)
        / (total[key] + _PRIOR_STRENGTH)
        for key in total
    }


def belief_free_predictor_maps(
    train: list["_Sample"], evaluation: list["_Sample"]
) -> np.ndarray:
    """Fit on train samples and return per-eval-sample occupancy maps."""

    from cal.evaluation.permanence_forward_benchmark import _SIDE, _cell_index

    if not evaluation:
        raise ValueError("belief-free evaluation samples must not be empty")
    table = fit_belief_free_table(train)
    maps = np.zeros((len(evaluation), _SIDE * _SIDE), dtype=np.float64)
    for row, sample in enumerate(evaluation):
        keys, distances = _sample_keys(sample)
        default = 1.0 / max(len(sample.candidate_cells), 1)
        for cell, key in keys.items():
            # A tiny anchor-ward tilt breaks ties between cells that share a
            # key; without it the argmax is decided by field order, which
            # understates the shortcut and would make the gate look stronger
            # than it is.
            maps[row, _cell_index(cell)] = table.get(key, default) + 1e-6 / (
                1.0 + distances[cell]
            )
    return maps

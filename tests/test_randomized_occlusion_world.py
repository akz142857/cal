"""Tests for the randomized-occlusion permanence-retest world.

These assert the two properties that make the world a valid permanence test:
drop-in interface parity + determinism with ``_IntegratedWorld``, and -- the
whole point -- that the fixed-geometry / constant-velocity shortcut that
defeated the L0 V8 permanence control is broken here.
"""

from __future__ import annotations

import numpy as np
import pytest

from cal.evaluation.permanence_geometry_diagnostic import run_diagnostic
from cal.evaluation.randomized_occlusion_world import RandomizedOcclusionWorld
from cal.evaluation.v2_i1_integration import (
    ARENA_HIGH,
    ARENA_LOW,
    _global_visibility,
)


def _run_episode(world: RandomizedOcclusionWorld, seed: int, steps: int):
    action_rng = np.random.default_rng(seed + 50_000)
    trace = []
    for _ in range(steps):
        action = int(action_rng.integers(0, 5))
        sensed, visibility = world.step(action)
        trace.append(
            (
                tuple(map(int, world.self_position)),
                tuple(map(int, world.distractor_a)),
                tuple(map(int, world.distractor_b)),
                sensed.copy(),
            )
        )
    return trace


def test_interface_parity_and_shapes():
    world = RandomizedOcclusionWorld(60000)
    assert world.grid_size == 25
    assert isinstance(world.static, frozenset)
    assert (12, 12) not in world.static  # camera cell stays clear
    for attr in ("self_position", "distractor_a", "distractor_b"):
        value = getattr(world, attr)
        assert value.shape == (2,)
        assert ARENA_LOW <= int(value[0]) <= ARENA_HIGH
        assert ARENA_LOW <= int(value[1]) <= ARENA_HIGH
    sensed, visibility = world.observe()
    assert sensed.shape == (11, 11)
    assert visibility.shape == (11, 11)
    assert world.truth().shape == (25, 25)


def test_deterministic_given_seed():
    first = _run_episode(RandomizedOcclusionWorld(60001), 60001, 60)
    second = _run_episode(RandomizedOcclusionWorld(60001), 60001, 60)
    for (a_pos, a_a, a_b, a_sensed), (b_pos, b_a, b_b, b_sensed) in zip(
        first, second, strict=True
    ):
        assert a_pos == b_pos
        assert a_a == b_a
        assert a_b == b_b
        assert np.array_equal(a_sensed, b_sensed)


def test_invalid_world_configuration_is_rejected():
    with pytest.raises(ValueError, match="hidden_turn_probability"):
        RandomizedOcclusionWorld(60001, hidden_turn_probability=1.01)
    with pytest.raises(ValueError, match="grid_size"):
        RandomizedOcclusionWorld(60001, grid_size=ARENA_HIGH)


def test_layout_varies_across_seeds():
    layouts = {
        frozenset(RandomizedOcclusionWorld(60000 + i).static) for i in range(8)
    }
    # Per-episode randomization must not collapse to a single fixed screen.
    assert len(layouts) >= 4


def test_layouts_are_connected_reachable_and_movable():
    for seed in range(60000, 60032):
        world = RandomizedOcclusionWorld(seed)
        diagnostics = world.layout_diagnostics()
        assert diagnostics["valid"] is True
        assert (
            diagnostics["reachable_free_cell_count"]
            == diagnostics["free_cell_count"]
        )
        assert diagnostics["hidden_free_cell_count"] >= 3
        assert diagnostics["visible_free_cell_count"] >= 3
        assert diagnostics["movable_free_cell_count"] >= 3
        assert diagnostics["visibility_boundary_edge_count"] >= 1
        assert all(
            any(
                ARENA_LOW <= int(point[0]) + dx <= ARENA_HIGH
                and ARENA_LOW <= int(point[1]) + dy <= ARENA_HIGH
                and (int(point[0]) + dx, int(point[1]) + dy) not in world.static
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            for point in (
                world.self_position,
                world.distractor_a,
                world.distractor_b,
            )
        )


def test_occlusion_events_occur():
    world = RandomizedOcclusionWorld(60002)
    action_rng = np.random.default_rng(60002 + 50_000)
    hidden_runs = {"a": 0, "b": 0}
    max_hidden = 0
    for _ in range(200):
        _, visibility = world.step(int(action_rng.integers(0, 5)))
        visible = _global_visibility(visibility, world.grid_size)
        for name in ("a", "b"):
            point = getattr(world, f"distractor_{name}")
            if visible[int(point[1]), int(point[0])]:
                hidden_runs[name] = 0
            else:
                hidden_runs[name] += 1
                max_hidden = max(max_hidden, hidden_runs[name])
    # Permanence samples require at least one >= 2-step occlusion.
    assert max_hidden >= 2


def test_hidden_distractor_trajectories_do_not_depend_on_actions():
    still = RandomizedOcclusionWorld(60003)
    moving = RandomizedOcclusionWorld(60003)
    for step in range(120):
        still.step(0)
        moving.step(1 + step % 4)
        assert np.array_equal(still.distractor_a, moving.distractor_a)
        assert np.array_equal(still.distractor_b, moving.distractor_b)
        assert np.array_equal(still.velocity_a, moving.velocity_a)
        assert np.array_equal(still.velocity_b, moving.velocity_b)


def test_shadow_occupancy_is_absent_from_agent_observation():
    world = RandomizedOcclusionWorld(60004)
    hidden = next(iter(world.shadow_cells))
    world.distractor_a = np.asarray(hidden, dtype=np.int64)
    sensed, visibility = world.observe()
    local_x = hidden[0] - (12 - visibility.shape[1] // 2)
    local_y = hidden[1] - (12 - visibility.shape[0] // 2)
    assert world.truth()[hidden[1], hidden[0]] == 1
    assert visibility[local_y, local_x] == 0
    assert sensed[local_y, local_x] == 0


def test_geometry_shortcut_is_broken_vs_fixed_world():
    """The fixed world lets constant-velocity extrapolation reconstruct the
    hidden cell almost perfectly; the randomized world must not."""

    train = list(range(60000, 60006))
    evaluation = list(range(60100, 60104))
    fixed = run_diagnostic(train, evaluation, steps=160, warmup=12, world="fixed")
    randomized = run_diagnostic(
        train, evaluation, steps=160, warmup=12, world="randomized"
    )

    fixed_recon = fixed["hidden_position_reconstruction_accuracy"]
    randomized_recon = randomized["hidden_position_reconstruction_accuracy"]
    assert fixed_recon > 0.9
    assert randomized_recon < 0.6
    # And the belief-free analytic extrapolator must lose most of its edge.
    assert (
        randomized["analytic_extrapolation_permanence_balanced_accuracy"]
        < fixed["analytic_extrapolation_permanence_balanced_accuracy"] - 0.25
    )

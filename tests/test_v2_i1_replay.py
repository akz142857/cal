from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cal.evaluation import v2_i1_replay
from cal.evaluation.v2_i1_integration_v2 import _episode
from cal.model.entity_belief_graph import IntegratedBeliefAgentV2


@pytest.fixture(scope="module")
def replay_payload() -> dict:
    return v2_i1_replay.build_replay_payload(seed=30_000)


@pytest.fixture(scope="module")
def replay_html(replay_payload: dict) -> str:
    from cal.evaluation.v2_i1_replay_template import render_replay_html

    return render_replay_html(replay_payload)


def test_replay_seed_allowlist_is_calibration_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _ = v2_i1_replay._load_protocol()
    assert v2_i1_replay.allowed_calibration_seeds(protocol) == tuple(
        range(30_000, 30_016)
    )

    def unexpected_record(**_: object) -> dict:
        raise AssertionError("a rejected seed must not start simulation")

    monkeypatch.setattr(v2_i1_replay, "record_condition", unexpected_record)
    for protected_seed in (32_000, 31_000, 99_999):
        with pytest.raises(ValueError):
            v2_i1_replay.build_replay_payload(seed=protected_seed)


@pytest.mark.parametrize(
    ("condition", "runner_options"),
    [
        ("formal", {}),
        ("no_action", {"use_action": False}),
        ("time_shuffled", {"shuffle_lag": 5}),
        ("assume_all_visible", {"infer_occlusion": False}),
    ],
)
def test_replay_metrics_exactly_match_frozen_runner(
    replay_payload: dict,
    condition: str,
    runner_options: dict,
) -> None:
    recorded = replay_payload["conditions"][condition]
    expected = _episode(30_000, steps=200, **runner_options)
    assert recorded["metrics"] == expected
    assert len(recorded["frames"]) == 201


def test_all_conditions_share_the_exact_action_schedule(
    replay_payload: dict,
) -> None:
    conditions = replay_payload["conditions"].values()
    schedules = {
        tuple(frame["action"] for frame in condition["frames"])
        for condition in conditions
    }
    hashes = {
        condition["actionScheduleSha256"]
        for condition in replay_payload["conditions"].values()
    }
    assert len(schedules) == 1
    assert hashes == {replay_payload["actionScheduleSha256"]}


def test_agent_update_receives_only_sensed_grid_and_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    class SpyAgent(IntegratedBeliefAgentV2):
        def update(self, sensed_occupancy: np.ndarray, action: int) -> None:
            calls.append((sensed_occupancy.shape, action))
            super().update(sensed_occupancy, action)

    monkeypatch.setattr(v2_i1_replay, "IntegratedBeliefAgentV2", SpyAgent)
    v2_i1_replay.record_condition(
        seed=30_000,
        steps=8,
        condition_name="formal",
    )
    assert len(calls) == 9
    assert calls[0] == ((11, 11), 0)
    assert all(shape == (11, 11) for shape, _ in calls)
    assert all(action in range(5) for _, action in calls)


def test_replay_includes_explanatory_events(replay_payload: dict) -> None:
    event_types = {
        event["type"]
        for event in replay_payload["conditions"]["formal"]["events"]
    }
    assert {
        "start",
        "self_acquired",
        "a_hidden",
        "a_reappeared",
        "merge",
        "end",
    } <= event_types


def test_render_is_byte_deterministic_and_checkable(
    replay_html: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "replay.html"
    target.write_text(replay_html, encoding="utf-8")
    monkeypatch.setattr(v2_i1_replay, "render_replay", lambda **_: replay_html)

    matches, expected_sha, actual_sha = v2_i1_replay.check_replay(
        seed=30_000,
        replay_path=target,
    )
    assert matches
    assert expected_sha == actual_sha

    target.write_text(replay_html + "\nchanged", encoding="utf-8")
    matches, expected_sha, actual_sha = v2_i1_replay.check_replay(
        seed=30_000,
        replay_path=target,
    )
    assert not matches
    assert expected_sha != actual_sha


def test_html_is_standalone_small_and_contains_all_controls(
    replay_html: str,
) -> None:
    assert len(replay_html.encode("utf-8")) < 2_000_000
    assert "http://" not in replay_html
    assert "https://" not in replay_html
    assert "<!doctype html>" in replay_html
    for element_id in (
        "play",
        "prev",
        "step",
        "condition",
        "event",
        "truth",
        "observation",
        "belief",
    ):
        assert f'id="{element_id}"' in replay_html
    for condition_name in v2_i1_replay.CONDITION_CONFIGS:
        assert condition_name in replay_html


def test_embedded_javascript_parses(replay_html: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for JavaScript syntax checking")
    scripts = re.findall(r"<script>(.*?)</script>", replay_html, flags=re.DOTALL)
    assert len(scripts) == 1
    source = tmp_path / "replay.js"
    source.write_text(scripts[0], encoding="utf-8")
    subprocess.run(
        [node, "--check", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_tracked_reference_replay_matches_generator(replay_html: str) -> None:
    reference = v2_i1_replay.default_output(30_000)
    assert reference.read_text(encoding="utf-8") == replay_html

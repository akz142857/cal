from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cal.evaluation import v2_l0_language_replay
from cal.evaluation import v2_l0_language_readout


@pytest.fixture(scope="module")
def replay_payload() -> dict:
    return v2_l0_language_replay.build_replay_payload(seed=33_100)


@pytest.fixture(scope="module")
def replay_html(replay_payload: dict) -> str:
    from cal.evaluation.v2_l0_language_replay_template import (
        render_replay_html,
    )

    return render_replay_html(replay_payload)


def test_replay_seed_allowlist_is_development_validation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _ = v2_l0_language_readout._load_protocol(
        v2_l0_language_replay.DEFAULT_PROTOCOL
    )
    assert v2_l0_language_replay.allowed_development_seeds(protocol) == (
        33_100,
        33_101,
        33_102,
        33_103,
    )

    def unexpected_collection(*_: object, **__: object) -> object:
        raise AssertionError("a rejected seed must not start simulation")

    monkeypatch.setattr(
        v2_l0_language_readout,
        "collect_language_data",
        unexpected_collection,
    )
    for protected_seed in (
        33_000,
        33_600,
        33_601,
        33_602,
        33_603,
        30_000,
        99_999,
    ):
        with pytest.raises(ValueError):
            v2_l0_language_replay.build_replay_payload(seed=protected_seed)


def test_payload_aligns_language_samples_with_visual_frames(
    replay_payload: dict,
) -> None:
    steps = replay_payload["steps"]
    warmup = replay_payload["warmup"]
    assert len(replay_payload["visualFrames"]) == steps + 1
    for condition in replay_payload["languageConditions"].values():
        frames = condition["frames"]
        assert len(frames) == steps + 1
        assert all(not frame["ready"] for frame in frames[:warmup])
        assert frames[warmup]["ready"]
        assert len(frames[warmup]["items"]) == 10
        assert [frame["step"] for frame in frames] == list(
            range(steps + 1)
        )


def test_language_rows_have_truth_mask_and_probability(
    replay_payload: dict,
) -> None:
    formal = replay_payload["languageConditions"][
        "formal_entity_graph"
    ]["frames"]
    ready = next(frame for frame in formal if frame["activeCount"] > 0)
    assert len(ready["items"]) == 10
    assert ready["correctCount"] <= ready["activeCount"]
    for item in ready["items"]:
        assert 0.0 <= item["probabilityTrue"] <= 1.0
        assert isinstance(item["truthTrue"], bool)
        assert isinstance(item["active"], bool)
        assert item["correct"] is None or isinstance(item["correct"], bool)
        assert item["queryPosition"] is None or len(
            item["queryPosition"]
        ) == 2


def test_formal_and_raw_are_distinct_readouts(replay_payload: dict) -> None:
    formal = replay_payload["languageConditions"][
        "formal_entity_graph"
    ]["frames"]
    raw = replay_payload["languageConditions"]["raw_sensor"]["frames"]
    formal_probabilities = [
        item["probabilityTrue"]
        for frame in formal
        for item in frame["items"]
    ]
    raw_probabilities = [
        item["probabilityTrue"]
        for frame in raw
        for item in frame["items"]
    ]
    assert formal_probabilities != raw_probabilities
    assert (
        replay_payload["aggregateMetrics"]["formal_entity_graph"][
            "macro_balanced_accuracy"
        ]
        > replay_payload["aggregateMetrics"]["raw_sensor"][
            "macro_balanced_accuracy"
        ]
    )


def test_payload_states_evidence_boundary(replay_payload: dict) -> None:
    assert replay_payload["presentationOnly"] is True
    assert replay_payload["holdoutSeedsAccessed"] is False
    assert replay_payload["evaluatorTruthUsedForI1"] is False
    assert replay_payload["evaluatorTruthUsedForReadoutTraining"] is True
    assert replay_payload["languageGradientsReachI1"] is False
    assert replay_payload["formalEvidence"]["allGatesPassed"] is True
    assert replay_payload["formalEvidence"]["gateCount"] == 24


def test_render_is_byte_deterministic_and_checkable(
    replay_html: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "l0-replay.html"
    target.write_text(replay_html, encoding="utf-8")
    monkeypatch.setattr(
        v2_l0_language_replay,
        "render_replay",
        lambda **_: replay_html,
    )
    matches, expected_sha, actual_sha = (
        v2_l0_language_replay.check_replay(
            seed=33_100,
            replay_path=target,
        )
    )
    assert matches
    assert expected_sha == actual_sha

    target.write_text(replay_html + "\nchanged", encoding="utf-8")
    matches, expected_sha, actual_sha = (
        v2_l0_language_replay.check_replay(
            seed=33_100,
            replay_path=target,
        )
    )
    assert not matches
    assert expected_sha != actual_sha


def test_html_is_standalone_small_and_contains_controls(
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
        "representation",
        "event",
        "activeOnly",
        "truth",
        "observation",
        "belief",
        "languageList",
    ):
        assert f'id="{element_id}"' in replay_html
    for representation in v2_l0_language_replay.REPRESENTATIONS:
        assert representation in replay_html


def test_embedded_javascript_parses(
    replay_html: str,
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for JavaScript syntax checking")
    scripts = re.findall(
        r"<script(?: [^>]*)?>(.*?)</script>",
        replay_html,
        flags=re.DOTALL,
    )
    executable = [script for script in scripts if "(() => {" in script]
    assert len(executable) == 1
    source = tmp_path / "l0-replay.js"
    source.write_text(executable[0], encoding="utf-8")
    subprocess.run(
        [node, "--check", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_tracked_reference_replay_matches_generator(
    replay_html: str,
) -> None:
    reference = v2_l0_language_replay.default_output(33_100)
    assert reference.read_text(encoding="utf-8") == replay_html

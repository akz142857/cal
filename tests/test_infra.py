"""Tests for experiment provenance and result indexing."""

import json

from cal.infra.provenance import capture_provenance
from cal.infra.results import build_result_index


def test_provenance_contains_stable_source_digest() -> None:
    first = capture_provenance()
    second = capture_provenance()

    assert first["schema_version"] == 1
    assert len(first["source_sha256"]) == 64
    assert first["source_sha256"] == second["source_sha256"]
    assert first["source_file_count"] > 0


def test_result_index_discovers_known_summaries(tmp_path: object) -> None:
    root = tmp_path / "results"  # type: ignore[operator]
    run = root / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"name": "test"}),
        encoding="utf-8",
    )

    index = build_result_index(root)

    assert index["entry_count"] == 1
    assert index["entries"][0]["kind"] == "prediction"
    assert (root / "INDEX.json").exists()


def test_result_index_discovers_preregistered_mechanism_screens(
    tmp_path: object,
) -> None:
    root = tmp_path / "results"  # type: ignore[operator]
    root.mkdir()
    (root / "M1v-screen-summary.json").write_text(
        json.dumps({"candidate": "m1v_action_basis"}),
        encoding="utf-8",
    )

    index = build_result_index(root)

    assert index["entry_count"] == 1
    assert index["entries"][0]["kind"] == "m1_mechanism_screen"
    assert index["entries"][0]["name"] == "m1v_action_basis"

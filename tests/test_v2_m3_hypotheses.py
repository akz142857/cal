from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from calmodel.evaluation.v2_m3_hypotheses import (
    _load_protocol,
    _shared_observation,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_PROTOCOL.json"
HOLDOUT = ROOT / "results/V2-M3-body-graph-holdout-summary.json"


def test_protocol_lock_matches_frozen_bytes() -> None:
    protocol, digest = _load_protocol(PROTOCOL)

    assert digest == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert protocol["fixed_gates"][
        "exactly_two_complete_symmetric_hypotheses"
    ]
    assert protocol["scenario_generator"][
        "shared_base_is_single_visual_detection"
    ]


def test_shared_observation_contains_only_one_base_detection() -> None:
    base = np.asarray((16.0, 16.0))
    detections, _, _ = _shared_observation(
        base,
        [-0.7, 0.9],
        [3.141592653589793 + 0.7, -0.9],
        rng=np.random.default_rng(42),
        noise_std=0.0,
        hide_endpoint=None,
    )

    base_matches = np.all(np.isclose(detections, base), axis=1)
    assert len(detections) == 5
    assert int(base_matches.sum()) == 1


def test_frozen_holdout_records_exactly_one_run() -> None:
    result = json.loads(HOLDOUT.read_text(encoding="utf-8"))

    assert result["review_split"] == "holdout"
    assert result["holdout_run_count"] == 1
    assert result["protocol_sha256"] == hashlib.sha256(
        PROTOCOL.read_bytes()
    ).hexdigest()

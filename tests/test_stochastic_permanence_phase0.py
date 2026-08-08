from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import cal.evaluation.stochastic_permanence_phase0 as phase0
import pytest

from cal.evaluation.stochastic_permanence_benchmark import REFERENCE_PREDICTORS


ROOT = Path(__file__).resolve().parents[1]


def _metrics(top1: float, nll: float, brier: float, error: float) -> dict[str, float]:
    return {
        "top1_accuracy": top1,
        "categorical_nll": nll,
        "brier": brier,
        "argmax_position_error": error,
    }


def _predictor(seed_ids: list[int], values: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "seed_bin_scores": {
            str(seed): {scope: dict(metrics) for scope, metrics in values.items()}
            for seed in seed_ids
        }
    }


def test_phase0_runner_wires_only_reference_predictors(
    monkeypatch,
) -> None:
    registry_path = ROOT / phase0.DEFAULT_REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in registry["evaluation_seeds"]]
    oracle = {
        scope: _metrics(0.8, 1.0, 0.02, 1.0)
        for scope in ("2-3", "4-5", "6+")
    }
    geometric = {
        scope: _metrics(0.3, 5.0, 0.05, 7.0)
        for scope in ("2-3", "4-5", "6+")
    }
    belief_free = {
        scope: _metrics(0.45, 4.0, 0.04, 5.0)
        for scope in ("2-3", "4-5", "6+")
    }
    uniform = {
        scope: _metrics(0.1, 3.0, 0.04, 6.0)
        for scope in ("2-3", "4-5", "6+")
    }
    old_i1 = {
        scope: _metrics(0.4, 4.0, 0.07, 3.0)
        for scope in ("2-3", "4-5", "6+")
    }
    report = {
        "episode_binned_predictors": {
            "belief": _predictor(seeds, oracle),
            "geometric": _predictor(seeds, geometric),
            "belief_free": _predictor(seeds, belief_free),
            "entity_graph": _predictor(seeds, old_i1),
        },
        "leakage_audit": {
            "baselines": {
                "uniform_field": {
                    "episode_binned_primary": _predictor(seeds, uniform)
                }
            }
        },
    }
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(phase0, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(
        phase0, "reproduce_registry_artifact", lambda payload: deepcopy(payload)
    )

    artifact = phase0.run_phase0(
        registry_path=registry_path,
        workspace_root=ROOT,
        power_simulation_trials=4,
        power_bootstrap_samples=16,
    )

    assert artifact["candidate_maps_read"] is False
    assert artifact["provenance"]["candidate_source_imported"] is False
    assert calls[0][1]["include_gru"] is False
    assert calls[0][1]["include_slot"] is False
    assert calls[0][1]["include_entity_graph"] is True
    assert calls[0][1]["paired_bootstrap_samples"] == 0
    assert artifact["complete_seed_ids"] == sorted(seeds)
    assert artifact["provenance"]["coverage"]["complete_seed_ids"] == sorted(
        seeds
    )
    assert set(artifact["per_seed_per_bin"]) == set(REFERENCE_PREDICTORS)


def test_phase0_rejects_a_common_but_incomplete_seed_population(
    monkeypatch,
) -> None:
    registry_path = ROOT / phase0.DEFAULT_REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in registry["evaluation_seeds"]]
    incomplete = seeds[:-1]
    values = {
        scope: _metrics(0.8, 1.0, 0.02, 1.0)
        for scope in ("2-3", "4-5", "6+")
    }
    report = {
        "episode_binned_predictors": {
            "belief": _predictor(incomplete, values),
            "geometric": _predictor(incomplete, values),
            "belief_free": _predictor(incomplete, values),
            "entity_graph": _predictor(incomplete, values),
        },
        "leakage_audit": {
            "baselines": {
                "uniform_field": {
                    "episode_binned_primary": _predictor(incomplete, values)
                }
            }
        },
    }
    monkeypatch.setattr(phase0, "run_benchmark", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        phase0, "reproduce_registry_artifact", lambda payload: deepcopy(payload)
    )

    with pytest.raises(RuntimeError, match="precommitted seed population"):
        phase0.run_phase0(
            registry_path=registry_path,
            workspace_root=ROOT,
            power_simulation_trials=4,
            power_bootstrap_samples=16,
        )


def test_phase0_rejects_registry_seed_or_digest_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    registry = json.loads(
        (ROOT / phase0.DEFAULT_REGISTRY).read_text(encoding="utf-8")
    )
    tampered = deepcopy(registry)
    tampered["selection_digest_sha256"] = "0" * 64
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        phase0,
        "reproduce_registry_artifact",
        lambda payload: deepcopy(registry),
    )

    with pytest.raises(RuntimeError, match="selection digest or reproduction"):
        phase0._load_registry(path)

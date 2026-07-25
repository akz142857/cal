"""Strict validation helpers for the staged V2 result chain."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


_GROUND_TRUTH_PARAMETER_SUBSTRINGS = (
    "mask",
    "label",
    "truth",
    "privileged",
    "oracle",
)


def constructor_apis_reject_ground_truth(*constructors: type) -> bool:
    """True if no constructor's public parameters look like a ground-truth input.

    The formal M1-M3 learners are online estimators whose constructors take
    only tuning/config values, never body masks or other privileged truth.
    This checks that structural fact at gate-evaluation time instead of
    asserting it as a fixed literal, so a future change that actually added
    such a parameter would flip this gate rather than leave it silently
    reporting success.
    """
    for constructor in constructors:
        for name in inspect.signature(constructor).parameters:
            lowered = name.lower()
            if any(
                token in lowered
                for token in _GROUND_TRUTH_PARAMETER_SUBSTRINGS
            ):
                return False
    return True


def require_authorization(
    path: str | Path,
    *,
    expected_name: str,
    expected_decision: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    name = payload.get("experiment", payload.get("audit"))
    if payload.get("result_schema_version") != 1:
        raise RuntimeError(
            f"invalid {expected_name} V2 artifact schema: {path}"
        )
    if name != expected_name:
        raise RuntimeError(f"expected {expected_name} prerequisite: {path}")
    if payload.get("passed") is not True:
        raise RuntimeError(f"{expected_name} prerequisite did not pass")
    if payload.get("decision") != expected_decision:
        raise RuntimeError(f"{expected_name} did not authorize this stage")
    gates = payload.get("gates", payload.get("chain_validation"))
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise RuntimeError(f"{expected_name} has incomplete gates")
    if not isinstance(payload.get("provenance"), dict):
        raise RuntimeError(f"{expected_name} has no provenance")
    return payload

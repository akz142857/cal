"""Strict validation helpers for the staged V2 result chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

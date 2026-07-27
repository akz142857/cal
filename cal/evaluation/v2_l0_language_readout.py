"""V2-L0: language readability from the frozen I1 entity belief graph.

This evaluator never sends language labels or evaluator truth into I1. It
records detached, ID-invariant spatial features after each normal I1 update,
then fits a separate linear readout for controlled Chinese propositions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from cal.evaluation.v2_i1_integration import (
    ACTION_DELTAS,
    ARENA_HIGH,
    ARENA_LOW,
    PERMANENCE_WARMUP,
    WARMUP,
    _IntegratedWorld,
    _global_visibility,
)
from cal.evaluation.v2_i1_integration_v2 import (
    _git_command,
    _load_result_evidence,
    _publish_annotated_tag,
    _publish_result_evidence,
    _remote_tag_exists,
    _reserve_one_shot,
    _reserve_shared_one_shot,
)
from cal.infra.provenance import capture_provenance
from cal.model.entity_belief_graph import (
    STATIC_THRESHOLD,
    IntegratedBeliefAgentV2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_V1 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL.json"
)
PROTOCOL_V2 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V2.json"
)
PROTOCOL_V3 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V3.json"
)
PROTOCOL_V4 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V4.json"
)
PROTOCOL_V5 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V5.json"
)
PROTOCOL_V6 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V6.json"
)
PROTOCOL_V7 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V7.json"
)
PROTOCOL_V8 = (
    PROJECT_ROOT / "experiments" / "V2_L0_LANGUAGE_READOUT_PROTOCOL_V8.json"
)
DEFAULT_PROTOCOL = PROTOCOL_V8
KNOWN_PROTOCOL_DIGESTS = {
    1: "39026a8ef6c1253ea40e830356504741636532fa7fcecbefadff4fabd8199493",
    2: "bb70d7983621a4756bf8e1030eb729ccca776888f3ba0fad062ba319022c32e3",
    3: "35aa6458ce5b40bf788dd12394ab5736647f140dbbfc3f556916cfd63cd5daa9",
    4: "85aee950e9ff1c5ea5838112912350d90002683cce2d80b026a94dd2e748f6a9",
    5: "51a4f561bceb23de2c9c483895b82e2f5b1cd4168736b22b166e236be6ce1aae",
    6: "6742a36cc6b6b572ed89642fc154a2104c814ec1e348a9c7af5bfa691f3776aa",
}
PROPOSITION_NAMES = (
    "first_pointed_entity_is_self",
    "second_pointed_entity_is_self",
    "horizontal_mover_left_of_self",
    "horizontal_mover_above_self",
    "vertical_mover_left_of_self",
    "vertical_mover_above_self",
    "first_hidden_cell_contains_entity",
    "second_hidden_cell_contains_entity",
    "first_reappeared_candidate_matches_reference",
    "second_reappeared_candidate_matches_reference",
)
GROUP_NAMES = ("self", "spatial", "permanence", "identity")
GRAPH_MAP_NAMES = (
    "occupancy_probability",
    "inferred_visibility",
    "sensed_occupancy",
    "learned_static",
    "entity_mass",
    "entity_existence",
    "entity_age",
    "entity_missed",
    "horizontal_motion",
    "vertical_motion",
    "velocity_x",
    "velocity_y",
    "self_evidence",
    "self_posterior",
    "selected_self",
)


@dataclass(frozen=True, slots=True)
class CollectedLanguageData:
    """Detached learner representations with evaluator-only propositions."""

    graph_features: Tensor
    raw_features: Tensor
    labels: Tensor
    training_mask: Tensor
    group_masks: Mapping[str, Tensor]
    episode_ids: Tensor
    graph_base_feature_count: int
    raw_base_feature_count: int
    query_count: int = 6
    graph_query_block_size: int = 0
    raw_query_block_size: int = 0
    identity_reference_roles: Tensor | None = None
    identity_opposite_control_references: Tensor | None = None

    def __post_init__(self) -> None:
        sample_count = self.labels.shape[0]
        if self.identity_reference_roles is None:
            object.__setattr__(
                self,
                "identity_reference_roles",
                torch.full((sample_count,), -1, dtype=torch.int64),
            )
        if self.identity_reference_roles.shape != (sample_count,):
            raise ValueError("identity reference role shape mismatch")
        if self.identity_opposite_control_references is None:
            object.__setattr__(
                self,
                "identity_opposite_control_references",
                torch.zeros(
                    (sample_count, len(GRAPH_MAP_NAMES)),
                    dtype=torch.float32,
                ),
            )
        if self.identity_opposite_control_references.shape != (
            sample_count,
            len(GRAPH_MAP_NAMES),
        ):
            raise ValueError("identity control reference shape mismatch")
        if self.labels.ndim != 2 or self.labels.shape[1] != len(
            PROPOSITION_NAMES
        ):
            raise ValueError("unexpected proposition label shape")
        if self.training_mask.shape != self.labels.shape:
            raise ValueError("training mask and labels must have equal shape")
        if self.graph_features.ndim != 2 or self.raw_features.ndim != 2:
            raise ValueError("language features must be rank-two tensors")
        if not 0 < self.graph_base_feature_count <= self.graph_features.shape[1]:
            raise ValueError("invalid graph base feature count")
        if not 0 < self.raw_base_feature_count <= self.raw_features.shape[1]:
            raise ValueError("invalid raw base feature count")
        if self.query_count < 0:
            raise ValueError("query_count must be non-negative")
        if self.query_count:
            if self.graph_query_block_size < 1 or self.raw_query_block_size < 1:
                raise ValueError("query block sizes must be positive")
            if (
                self.graph_base_feature_count
                + self.query_count * self.graph_query_block_size
                != self.graph_features.shape[1]
                or self.raw_base_feature_count
                + self.query_count * self.raw_query_block_size
                != self.raw_features.shape[1]
            ):
                raise ValueError("query feature layout mismatch")
        if {
            self.graph_features.shape[0],
            self.raw_features.shape[0],
            self.episode_ids.shape[0],
            sample_count,
        } != {sample_count}:
            raise ValueError("language data sample counts disagree")
        if set(self.group_masks) != set(GROUP_NAMES):
            raise ValueError("language concept masks are incomplete")
        if any(mask.shape != self.labels.shape for mask in self.group_masks.values()):
            raise ValueError("language concept mask shape mismatch")
        if any(
            tensor.requires_grad
            for tensor in (
                self.graph_features,
                self.raw_features,
                self.labels,
                self.training_mask,
            )
        ):
            raise ValueError("I1 language readout data must be detached")

    def features(self, representation: str) -> Tensor:
        if representation == "formal_entity_graph":
            return self.graph_features
        if representation == "raw_sensor":
            return self.raw_features
        raise ValueError(f"unknown representation {representation!r}")

    def with_graph_features(self, features: Tensor) -> "CollectedLanguageData":
        return CollectedLanguageData(
            graph_features=features,
            raw_features=self.raw_features,
            labels=self.labels,
            training_mask=self.training_mask,
            group_masks=self.group_masks,
            episode_ids=self.episode_ids,
            graph_base_feature_count=self.graph_base_feature_count,
            raw_base_feature_count=self.raw_base_feature_count,
            query_count=self.query_count,
            graph_query_block_size=self.graph_query_block_size,
            raw_query_block_size=self.raw_query_block_size,
            identity_reference_roles=self.identity_reference_roles,
            identity_opposite_control_references=(
                self.identity_opposite_control_references
            ),
        )

    def with_labels(
        self,
        labels: Tensor,
        training_mask: Tensor | None = None,
    ) -> "CollectedLanguageData":
        return CollectedLanguageData(
            graph_features=self.graph_features,
            raw_features=self.raw_features,
            labels=labels,
            training_mask=(
                self.training_mask
                if training_mask is None
                else training_mask
            ),
            group_masks=self.group_masks,
            episode_ids=self.episode_ids,
            graph_base_feature_count=self.graph_base_feature_count,
            raw_base_feature_count=self.raw_base_feature_count,
            query_count=self.query_count,
            graph_query_block_size=self.graph_query_block_size,
            raw_query_block_size=self.raw_query_block_size,
            identity_reference_roles=self.identity_reference_roles,
            identity_opposite_control_references=(
                self.identity_opposite_control_references
            ),
        )

    def base_feature_count(self, representation: str) -> int:
        if representation == "formal_entity_graph":
            return self.graph_base_feature_count
        if representation == "raw_sensor":
            return self.raw_base_feature_count
        raise ValueError(f"unknown representation {representation!r}")


@dataclass(frozen=True, slots=True)
class ReadoutConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ReadoutModel:
    probe: "LinearLanguageReadout"
    mean: Tensor
    scale: Tensor
    final_loss: float


class LinearLanguageReadout(nn.Module):
    """Four affine heads with evaluator-query isolation between concepts."""

    def __init__(
        self,
        feature_count: int,
        base_feature_count: int,
        *,
        query_count: int = 6,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if not 0 < base_feature_count <= feature_count:
            raise ValueError("invalid base_feature_count")
        self.feature_count = feature_count
        self.base_feature_count = base_feature_count
        self.query_count = query_count
        remaining = feature_count - base_feature_count
        if query_count:
            if remaining % query_count:
                raise ValueError("feature count is incompatible with query layout")
            self.query_block_size = remaining // query_count
        else:
            self.query_block_size = 0
        query_pair_size = 2 * self.query_block_size
        self.self_linear = nn.Linear(
            query_pair_size or base_feature_count, 2
        )
        self.spatial_linear = nn.Linear(base_feature_count, 4)
        self.permanence_linear = nn.Linear(
            base_feature_count + 2 * self.query_block_size, 2
        )
        self.identity_linear = nn.Linear(
            query_pair_size or base_feature_count, 2
        )

    def _group_features(
        self,
        features: Tensor,
        first_query: int,
        *,
        hide_absolute_query_positions: bool = False,
        include_base: bool = True,
        self_channels_only: bool = False,
    ) -> Tensor:
        start = self.base_feature_count + first_query * self.query_block_size
        end = start + 2 * self.query_block_size
        queries = features[:, start:end]
        if hide_absolute_query_positions and self.query_block_size:
            queries = queries.clone()
            arena_cells = (ARENA_HIGH - ARENA_LOW + 1) ** 2
            queries[:, :arena_cells] = 0.0
            queries[
                :, self.query_block_size : self.query_block_size + arena_cells
            ] = 0.0
        if self_channels_only and self.query_block_size:
            source = queries
            queries = torch.zeros_like(source)
            arena_cells = (ARENA_HIGH - ARENA_LOW + 1) ** 2
            for query_offset in (0, self.query_block_size):
                current_start = query_offset + arena_cells
                queries[:, current_start + 12 : current_start + 15] = (
                    source[:, current_start + 12 : current_start + 15]
                )
        if include_base or not self.query_block_size:
            return torch.cat(
                (features[:, : self.base_feature_count], queries), dim=1
            )
        return queries

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("unexpected language readout feature shape")
        return torch.cat(
            (
                self.self_linear(
                    self._group_features(
                        features,
                        0,
                        hide_absolute_query_positions=True,
                        include_base=False,
                        self_channels_only=True,
                    )
                ),
                self.spatial_linear(features[:, : self.base_feature_count]),
                self.permanence_linear(self._group_features(features, 2)),
                self.identity_linear(
                    self._group_features(
                        features,
                        4,
                        hide_absolute_query_positions=True,
                        include_base=False,
                    )
                ),
            ),
            dim=1,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_protocol_path(version: int) -> Path:
    paths = {
        1: PROTOCOL_V1,
        2: PROTOCOL_V2,
        3: PROTOCOL_V3,
        4: PROTOCOL_V4,
        5: PROTOCOL_V5,
        6: PROTOCOL_V6,
        7: PROTOCOL_V7,
        8: PROTOCOL_V8,
    }
    try:
        return paths[version].resolve()
    except KeyError as exc:
        raise RuntimeError("unsupported V2-L0 protocol version") from exc


def _read_protocol_document(path: Path) -> tuple[dict[str, Any], str]:
    source = path.resolve()
    raw = source.read_bytes()
    document = json.loads(raw)
    version = int(document.get("protocol_version", 0))
    if source != _canonical_protocol_path(version):
        raise RuntimeError("V2-L0 protocol path is not canonical")
    digest = hashlib.sha256(raw).hexdigest()
    if version not in {*KNOWN_PROTOCOL_DIGESTS, 7, 8}:
        raise RuntimeError("unsupported V2-L0 protocol version")
    if (
        version in KNOWN_PROTOCOL_DIGESTS
        and digest != KNOWN_PROTOCOL_DIGESTS[version]
    ):
        raise RuntimeError("V2-L0 protocol digest is not recognized")
    sidecar = source.with_suffix(".sha256")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if recorded != digest:
        raise RuntimeError("V2-L0 protocol SHA-256 sidecar mismatch")
    return document, digest


def _merge_v2(document: Mapping[str, Any]) -> dict[str, Any]:
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V1.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != KNOWN_PROTOCOL_DIGESTS[1]
    ):
        raise RuntimeError("V2-L0 V2 amendment base mismatch")
    base, _ = _read_protocol_document(PROTOCOL_V1)
    prior_result = (
        PROJECT_ROOT
        / document["amendment_record"]["prior_development_result_path"]
    )
    if (
        _sha256(prior_result)
        != document["amendment_record"]["prior_development_result_sha256"]
    ):
        raise RuntimeError("V2-L0 V1 development result changed")
    payload = copy.deepcopy(base)
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 2,
            "status": document["status"],
            "amendment_record": document["amendment_record"],
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "result_paths": document["result_paths"],
        }
    )
    payload["learner_boundary"].update(
        document["learner_boundary_amendment"]
    )
    schema_amendment = document["semantic_schema_amendment"]
    replacements = schema_amendment["replace_propositions"]
    payload["semantic_schema"]["propositions"] = [
        replacements.get(name, name)
        for name in payload["semantic_schema"]["propositions"]
    ]
    payload["semantic_schema"]["concept_groups"]["self"] = (
        schema_amendment["replace_concept_group"]["self"]
    )
    for split, replacement in schema_amendment[
        "replace_templates"
    ].items():
        payload["semantic_schema"]["templates"][split][:2] = replacement
    payload["deictic_query_protocol"] = document[
        "deictic_query_protocol"
    ]
    return payload


def _merge_v3(document: Mapping[str, Any]) -> dict[str, Any]:
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V2.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != KNOWN_PROTOCOL_DIGESTS[2]
    ):
        raise RuntimeError("V2-L0 V3 amendment base mismatch")
    base_document, _ = _read_protocol_document(PROTOCOL_V2)
    payload = _merge_v2(base_document)
    prior_result = (
        PROJECT_ROOT
        / document["amendment_record"]["prior_development_result_path"]
    )
    if (
        _sha256(prior_result)
        != document["amendment_record"]["prior_development_result_sha256"]
    ):
        raise RuntimeError("V2-L0 V2 development result changed")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 3,
            "status": document["status"],
            "amendment_record_v2": payload["amendment_record"],
            "amendment_record": document["amendment_record"],
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "review_record": document["review_record"],
            "holdout_contamination_record": document[
                "holdout_contamination_record"
            ],
            "paired_query_protocol": document["paired_query_protocol"],
            "coverage_and_evidence": document["coverage_and_evidence"],
            "result_paths": document["result_paths"],
            "required_final_source_locks": document[
                "required_final_source_locks"
            ],
        }
    )
    payload["semantic_schema"].update(
        document["semantic_schema_replacement"]
    )
    payload["controls"].update(document["control_amendment"])
    payload["fixed_gates"].update(document["additional_fixed_gates"])
    payload["splits"].update(document["split_amendment"])
    return payload


def _merge_v4(document: Mapping[str, Any]) -> dict[str, Any]:
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V3.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != KNOWN_PROTOCOL_DIGESTS[3]
    ):
        raise RuntimeError("V2-L0 V4 amendment base mismatch")
    base_document, _ = _read_protocol_document(PROTOCOL_V3)
    payload = _merge_v3(base_document)
    prior_result = (
        PROJECT_ROOT
        / document["amendment_record"]["prior_development_result_path"]
    )
    prior_raw = prior_result.read_bytes()
    if (
        hashlib.sha256(prior_raw).hexdigest()
        != document["amendment_record"]["prior_development_result_sha256"]
    ):
        raise RuntimeError("V2-L0 V3 development result changed")
    prior_payload = json.loads(prior_raw)
    if (
        prior_payload.get("passed")
        is not document["amendment_record"]["prior_development_passed"]
        or prior_payload.get("decision")
        != document["amendment_record"]["prior_development_decision"]
    ):
        raise RuntimeError("V2-L0 V3 negative result record mismatch")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 4,
            "status": document["status"],
            "amendment_record_v3": payload["amendment_record"],
            "amendment_record": document["amendment_record"],
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "query_and_head_amendment": document[
                "query_and_head_amendment"
            ],
            "control_amendment_v4": document["control_amendment"],
            "authorization": document["authorization"],
            "result_paths": document["result_paths"],
        }
    )
    payload["controls"]["identity_scrambled_at_occlusion"] = document[
        "control_amendment"
    ]["identity_scrambled_at_occlusion"]
    return payload


def _merge_v5(document: Mapping[str, Any]) -> dict[str, Any]:
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V4.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != KNOWN_PROTOCOL_DIGESTS[4]
    ):
        raise RuntimeError("V2-L0 V5 amendment base mismatch")
    base_document, _ = _read_protocol_document(PROTOCOL_V4)
    payload = _merge_v4(base_document)
    prior_result = (
        PROJECT_ROOT
        / document["amendment_record"]["prior_development_result_path"]
    )
    prior_raw = prior_result.read_bytes()
    if (
        hashlib.sha256(prior_raw).hexdigest()
        != document["amendment_record"]["prior_development_result_sha256"]
    ):
        raise RuntimeError("V2-L0 passing development result changed")
    prior_payload = json.loads(prior_raw)
    if (
        prior_payload.get("passed") is not True
        or prior_payload.get("decision") != "authorize_review_and_source_lock"
        or not all(prior_payload.get("gates", {}).values())
    ):
        raise RuntimeError("V2-L0 development result does not authorize locking")
    for relative, expected in document["exact_source_locks"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"V2-L0 exact source changed: {relative}")
    provenance = capture_provenance(PROJECT_ROOT)
    if provenance["source_sha256"] != document["exact_source_sha256"]:
        raise RuntimeError("V2-L0 full source digest changed")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 5,
            "status": document["status"],
            "amendment_record_v4": payload["amendment_record"],
            "amendment_record": document["amendment_record"],
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "exact_source_locks": document["exact_source_locks"],
            "exact_source_sha256": document["exact_source_sha256"],
            "review_attestations": document["review_attestations"],
            "shared_git_registry": document["shared_git_registry"],
            "authorization": document["authorization"],
            "result_paths": document["result_paths"],
        }
    )
    return payload


def _merge_v6(document: Mapping[str, Any]) -> dict[str, Any]:
    _, base_digest = _read_protocol_document(PROTOCOL_V5)
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V5.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != base_digest
    ):
        raise RuntimeError("V2-L0 V6 amendment base mismatch")
    base_document, _ = _read_protocol_document(PROTOCOL_V4)
    payload = _merge_v4(base_document)
    amendment = document["amendment_record"]
    prior_result = PROJECT_ROOT / amendment["prior_development_result_path"]
    prior_raw = prior_result.read_bytes()
    if (
        hashlib.sha256(prior_raw).hexdigest()
        != amendment["prior_development_result_sha256"]
    ):
        raise RuntimeError("V2-L0 V6 prior development result changed")
    prior_payload = json.loads(prior_raw)
    if (
        prior_payload.get("passed") is not True
        or prior_payload.get("decision")
        != amendment["prior_development_decision"]
        or not all(prior_payload.get("gates", {}).values())
    ):
        raise RuntimeError("V2-L0 V6 prior development result is invalid")
    evidence_payloads: dict[str, dict[str, Any]] = {}
    for path_key, sha_key in (
        ("consumed_v5_failure_path", "consumed_v5_failure_sha256"),
        ("consumed_v5_reservation_path", "consumed_v5_reservation_sha256"),
    ):
        raw = (PROJECT_ROOT / amendment[path_key]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != amendment[sha_key]:
            raise RuntimeError(f"V2-L0 V5 evidence changed: {path_key}")
        evidence_payloads[path_key] = json.loads(raw)
    failure = evidence_payloads["consumed_v5_failure_path"]
    reservation = evidence_payloads["consumed_v5_reservation_path"]
    if (
        failure.get("outcome") != "consumed_failed_before_result"
        or failure.get("retry_allowed") is not False
        or failure.get("result_created") is not False
        or failure.get("evidence_tag_created") is not False
        or reservation.get("status") != "consumed_before_first_episode"
    ):
        raise RuntimeError("V2-L0 V5 consumed failure evidence is invalid")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 6,
            "status": document["status"],
            "amendment_record_v4": payload["amendment_record"],
            "amendment_record": amendment,
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "control_amendment_v6": document["control_amendment"],
            "control_integrity_gates": document["control_integrity_gates"],
            "holdout_contamination_record_v6": document[
                "holdout_contamination_record"
            ],
            "authorization": document["authorization"],
            "result_paths": document["result_paths"],
        }
    )
    payload["controls"]["identity_scrambled_at_occlusion"] = document[
        "control_amendment"
    ]["identity_scrambled_at_occlusion"]
    payload["splits"].update(document["split_amendment"])
    return payload


def _merge_v7(
    document: Mapping[str, Any],
    *,
    verify_exact_source: bool = True,
) -> dict[str, Any]:
    base_document, base_digest = _read_protocol_document(PROTOCOL_V6)
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V6.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != base_digest
    ):
        raise RuntimeError("V2-L0 V7 amendment base mismatch")
    payload = _merge_v6(base_document)
    amendment = document["amendment_record"]
    locked_payloads: dict[str, dict[str, Any]] = {}
    for path_key, sha_key in (
        ("prior_development_result_path", "prior_development_result_sha256"),
        ("prior_review_record_path", "prior_review_record_sha256"),
    ):
        raw = (PROJECT_ROOT / amendment[path_key]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != amendment[sha_key]:
            raise RuntimeError(f"V2-L0 V7 locked evidence changed: {path_key}")
        locked_payloads[path_key] = json.loads(raw)
    development = locked_payloads["prior_development_result_path"]
    review = locked_payloads["prior_review_record_path"]
    if (
        development.get("passed") is not True
        or development.get("decision") != "authorize_review_and_source_lock"
        or not all(development.get("gates", {}).values())
        or review.get("decision") != "authorize_v7_exact_source_lock"
        or any(
            int(review["final_reviews"][name][severity]) != 0
            for name in review["final_reviews"]
            for severity in ("p0", "p1", "p2")
        )
    ):
        raise RuntimeError("V2-L0 V7 review evidence does not authorize locking")
    if verify_exact_source:
        for relative, expected in document["exact_source_locks"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"V2-L0 exact source changed: {relative}")
        provenance = capture_provenance(PROJECT_ROOT)
        if provenance["source_sha256"] != document["exact_source_sha256"]:
            raise RuntimeError("V2-L0 full source digest changed")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 7,
            "status": document["status"],
            "amendment_record_v6": payload["amendment_record"],
            "amendment_record": amendment,
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "exact_source_locks": document["exact_source_locks"],
            "exact_source_sha256": document["exact_source_sha256"],
            "historical_v5_failure_evidence": document[
                "historical_v5_failure_evidence"
            ],
            "review_attestations": document["review_attestations"],
            "shared_git_registry": document["shared_git_registry"],
            "authorization": document["authorization"],
            "result_paths": document["result_paths"],
        }
    )
    return payload


def _merge_v8(document: Mapping[str, Any]) -> dict[str, Any]:
    base_document, base_digest = _read_protocol_document(PROTOCOL_V7)
    if (
        document["base_protocol_path"]
        != str(PROTOCOL_V7.relative_to(PROJECT_ROOT))
        or document["base_protocol_sha256"] != base_digest
    ):
        raise RuntimeError("V2-L0 V8 amendment base mismatch")
    payload = _merge_v7(base_document, verify_exact_source=False)
    amendment = document["amendment_record"]
    review_raw = (
        PROJECT_ROOT / amendment["post_fix_review_record_path"]
    ).read_bytes()
    if (
        hashlib.sha256(review_raw).hexdigest()
        != amendment["post_fix_review_record_sha256"]
    ):
        raise RuntimeError("V2-L0 V8 post-fix review evidence changed")
    review = json.loads(review_raw)
    if (
        review.get("decision") != "authorize_v8_exact_source_lock"
        or any(
            int(review["final_reviews"][name][severity]) != 0
            for name in review["final_reviews"]
            for severity in ("p0", "p1", "p2")
        )
    ):
        raise RuntimeError("V2-L0 V8 review does not authorize locking")
    predecessor = document["superseded_v7_source_lock"]
    if (
        predecessor.get("tag") != "calmodel-l0-v7-source-locked"
        or predecessor.get("tag_object_sha")
        != "b8b391abc5b54aa7acbf58bef6a6cdf2c7d32664"
        or predecessor.get("target_commit")
        != "db524a3d1b65a232c2159541a79d7098227848f5"
        or predecessor.get("authorization_published") is not False
        or predecessor.get("consumption_published") is not False
        or predecessor.get("post_lock_review_decision")
        != "reject_authorization_and_supersede_with_v8"
    ):
        raise RuntimeError("V2-L0 V7 supersession evidence mismatch")
    for relative, expected in document["exact_source_locks"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"V2-L0 V8 exact source changed: {relative}")
    provenance = capture_provenance(PROJECT_ROOT)
    if provenance["source_sha256"] != document["exact_source_sha256"]:
        raise RuntimeError("V2-L0 V8 full source digest changed")
    payload.update(
        {
            "protocol_name": document["protocol_name"],
            "protocol_version": 8,
            "status": document["status"],
            "amendment_record_v7": payload["amendment_record"],
            "amendment_record": amendment,
            "base_protocol_path": document["base_protocol_path"],
            "base_protocol_sha256": document["base_protocol_sha256"],
            "exact_source_locks": document["exact_source_locks"],
            "exact_source_sha256": document["exact_source_sha256"],
            "superseded_v7_source_lock": predecessor,
            "review_attestations": document["review_attestations"],
            "shared_git_registry": document["shared_git_registry"],
            "authorization": document["authorization"],
            "result_paths": document["result_paths"],
        }
    )
    return payload


def _load_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    document, digest = _read_protocol_document(path)
    version = int(document["protocol_version"])
    if version == 1:
        payload = document
    elif version == 2:
        payload = _merge_v2(document)
    elif version == 3:
        payload = _merge_v3(document)
    elif version == 4:
        payload = _merge_v4(document)
    elif version == 5:
        payload = _merge_v5(document)
    elif version == 6:
        payload = _merge_v6(document)
    elif version == 7:
        payload = _merge_v7(document)
    elif version == 8:
        payload = _merge_v8(document)
    else:
        raise RuntimeError("unsupported V2-L0 protocol version")
    if payload.get("status") not in {
        "frozen_before_language_readout_implementation",
        "frozen_before_deictic_self_correction_implementation",
        "frozen_after_initial_subagent_review_before_paired_query_implementation",
        "frozen_before_cross_episode_identity_control_implementation",
        "source_locked_awaiting_explicit_holdout_authorization",
        "frozen_after_language_readout_review_before_holdout",
        "frozen_after_v5_consumed_failure_before_row_local_control_implementation",
        "source_locked_awaiting_explicit_holdout_authorization_v7",
        "source_locked_awaiting_explicit_holdout_authorization_v8",
    }:
        raise RuntimeError("V2-L0 protocol is not frozen")

    prerequisite = payload["prerequisite"]
    artifact = PROJECT_ROOT / prerequisite["artifact"]
    artifact_raw = artifact.read_bytes()
    if (
        hashlib.sha256(artifact_raw).hexdigest()
        != prerequisite["artifact_sha256"]
    ):
        raise RuntimeError("V2-L0 I1 prerequisite artifact changed")
    result = json.loads(artifact_raw)
    if (
        result.get("decision") != prerequisite["expected_decision"]
        or result.get("passed") is not prerequisite["expected_passed"]
        or result.get("run_start", {}).get("git_commit")
        != prerequisite["reviewed_implementation_commit"]
        or result.get("run_start", {}).get("source_sha256")
        != prerequisite["formal_source_sha256"]
    ):
        raise RuntimeError("V2-L0 I1 prerequisite does not authorize language readout")
    for relative, expected in payload["frozen_i1_sources"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen I1 source changed: {relative}")
    if tuple(payload["semantic_schema"]["propositions"]) != PROPOSITION_NAMES:
        raise RuntimeError("V2-L0 proposition schema mismatch")
    return payload, digest


def _crop(array: np.ndarray) -> np.ndarray:
    return array[
        ARENA_LOW : ARENA_HIGH + 1,
        ARENA_LOW : ARENA_HIGH + 1,
    ]


def _deterministic_order_bit(seed: int, step: int, domain: str) -> int:
    raw = f"{domain}:{seed}:{step}".encode("utf-8")
    return hashlib.sha256(raw).digest()[0] & 1


def _accumulate(
    target: np.ndarray,
    weights: np.ndarray,
    *,
    x: int,
    y: int,
    mass: float,
    value: float,
) -> None:
    if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
        local_y, local_x = y - ARENA_LOW, x - ARENA_LOW
        target[local_y, local_x] += mass * value
        weights[local_y, local_x] += mass


def _graph_maps(
    agent: IntegratedBeliefAgentV2,
    *,
    steps: int,
) -> tuple[np.ndarray, ...]:
    """Render ID-invariant maps from the frozen graph without truth access."""

    arena_size = ARENA_HIGH - ARENA_LOW + 1
    shape = (arena_size, arena_size)
    entity_mass = np.zeros(shape, dtype=np.float64)
    accumulators = {
        name: np.zeros(shape, dtype=np.float64)
        for name in (
            "existence",
            "age",
            "missed",
            "horizontal",
            "vertical",
            "velocity_x",
            "velocity_y",
            "self_evidence",
        )
    }
    accumulator_weights = {
        name: np.zeros(shape, dtype=np.float64)
        for name in accumulators
    }

    for hypothesis in agent.graph._hypotheses:
        for entity in hypothesis.entities:
            x, y = int(entity.position[0]), int(entity.position[1])
            mass = float(hypothesis.weight * entity.existence)
            if mass <= 0.0:
                continue
            if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
                local_y, local_x = y - ARENA_LOW, x - ARENA_LOW
                entity_mass[local_y, local_x] += mass
            motion = entity.motion_delta_counts
            motion_total = max(float(motion.sum()), 1e-9)
            values = {
                "existence": float(entity.existence),
                "age": min(float(entity.age) / max(steps, 1), 1.0),
                "missed": min(float(entity.missed) / 40.0, 1.0),
                "horizontal": float(motion[1] + motion[2]) / motion_total,
                "vertical": float(motion[3] + motion[4]) / motion_total,
                "velocity_x": float(np.clip(entity.velocity[0], -2, 2)) / 2.0,
                "velocity_y": float(np.clip(entity.velocity[1], -2, 2)) / 2.0,
                "self_evidence": 1.0 / (
                    1.0 + np.exp(-float(np.clip(entity.self_logit, -20.0, 20.0)))
                ),
            }
            for name, value in values.items():
                _accumulate(
                    accumulators[name],
                    accumulator_weights[name],
                    x=x,
                    y=y,
                    mass=mass,
                    value=value,
                )

    averaged: dict[str, np.ndarray] = {}
    for name, values in accumulators.items():
        weights = accumulator_weights[name]
        averaged[name] = np.divide(
            values,
            weights,
            out=np.zeros_like(values),
            where=weights > 1e-12,
        )

    self_posterior = np.zeros(shape, dtype=np.float64)
    posterior = agent.graph.self_posterior()
    positions = agent.track_positions()
    for identity, probability in posterior.items():
        position = positions.get(identity)
        if position is None:
            continue
        x, y = position
        if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
            self_posterior[y - ARENA_LOW, x - ARENA_LOW] = max(
                self_posterior[y - ARENA_LOW, x - ARENA_LOW],
                float(probability),
            )

    selected_self = np.zeros(shape, dtype=np.float64)
    self_identity = agent.self_track_identity()
    self_position = positions.get(self_identity) if self_identity is not None else None
    if self_position is not None:
        x, y = self_position
        if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
            selected_self[y - ARENA_LOW, x - ARENA_LOW] = 1.0

    learned_static = (
        agent.front_end.static_score >= STATIC_THRESHOLD
    ).astype(np.float64)
    maps = (
        _crop(agent.probability()),
        _crop(agent.front_end.last_visibility).astype(np.float64),
        _crop(agent.front_end.last_sensed).astype(np.float64),
        _crop(learned_static),
        np.clip(entity_mass, 0.0, 1.0),
        averaged["existence"],
        averaged["age"],
        averaged["missed"],
        averaged["horizontal"],
        averaged["vertical"],
        averaged["velocity_x"],
        averaged["velocity_y"],
        averaged["self_evidence"],
        self_posterior,
        selected_self,
    )
    if len(maps) != len(GRAPH_MAP_NAMES):
        raise RuntimeError("graph language feature schema mismatch")
    return tuple(np.asarray(array, dtype=np.float32) for array in maps)


def _sample_maps(
    maps: Sequence[np.ndarray],
    position: tuple[int, int] | None,
) -> np.ndarray:
    sampled = np.zeros(len(maps), dtype=np.float32)
    if position is None:
        return sampled
    x, y = position
    if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
        local_y, local_x = y - ARENA_LOW, x - ARENA_LOW
        sampled = np.asarray(
            [array[local_y, local_x] for array in maps],
            dtype=np.float32,
        )
    return sampled


def _query_block(
    maps: Sequence[np.ndarray],
    position: tuple[int, int] | None,
    reference: np.ndarray | None,
) -> np.ndarray:
    arena_size = ARENA_HIGH - ARENA_LOW + 1
    query_mask = np.zeros((arena_size, arena_size), dtype=np.float32)
    current = _sample_maps(maps, position)
    if position is not None:
        x, y = position
        if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
            query_mask[y - ARENA_LOW, x - ARENA_LOW] = 1.0
    saved = (
        np.zeros(len(maps), dtype=np.float32)
        if reference is None
        else np.asarray(reference, dtype=np.float32)
    )
    if saved.shape != current.shape:
        raise ValueError("graph query reference descriptor shape mismatch")
    return np.concatenate(
        (query_mask.ravel(), current, saved, current * saved, np.abs(current - saved))
    ).astype(np.float32)


def _graph_features(
    agent: IntegratedBeliefAgentV2,
    action: int,
    *,
    steps: int,
    query_positions: Sequence[tuple[int, int] | None] = (None,) * 6,
    reference_descriptors: Sequence[np.ndarray | None] = (None,) * 6,
) -> np.ndarray:
    if len(query_positions) != 6 or len(reference_descriptors) != 6:
        raise ValueError("V3 requires exactly six query blocks")
    maps = _graph_maps(agent, steps=steps)
    action_one_hot = np.zeros(len(ACTION_DELTAS), dtype=np.float32)
    action_one_hot[action] = 1.0
    base = np.concatenate(
        [*(array.ravel() for array in maps), action_one_hot]
    ).astype(np.float32)
    query_features: list[np.ndarray] = []
    for position, reference in zip(
        query_positions, reference_descriptors, strict=True
    ):
        query_features.append(_query_block(maps, position, reference))
    result = np.concatenate((base, *query_features)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite graph language feature")
    return result


def _raw_features(
    sensed: np.ndarray,
    action: int,
    *,
    query_positions: Sequence[tuple[int, int] | None] = (None,) * 6,
    reference_descriptors: Sequence[np.ndarray | None] = (None,) * 6,
) -> np.ndarray:
    if len(query_positions) != 6 or len(reference_descriptors) != 6:
        raise ValueError("V3 requires exactly six raw query blocks")
    action_one_hot = np.zeros(len(ACTION_DELTAS), dtype=np.float32)
    action_one_hot[action] = 1.0
    base = np.concatenate(
        (sensed.astype(np.float32).ravel(), action_one_hot)
    )
    query_features: list[np.ndarray] = []
    for position, reference in zip(
        query_positions, reference_descriptors, strict=True
    ):
        query_mask = np.zeros_like(sensed, dtype=np.float32)
        current = np.zeros(1, dtype=np.float32)
        if position is not None:
            x, y = position
            local_x, local_y = x - ARENA_LOW, y - ARENA_LOW
            if (
                0 <= local_x < sensed.shape[1]
                and 0 <= local_y < sensed.shape[0]
            ):
                query_mask[local_y, local_x] = 1.0
                current[0] = float(sensed[local_y, local_x])
        saved = (
            np.zeros(1, dtype=np.float32)
            if reference is None
            else np.asarray(reference, dtype=np.float32)
        )
        if saved.shape != current.shape:
            raise ValueError("raw query reference descriptor shape mismatch")
        query_features.append(
            np.concatenate(
                (
                    query_mask.ravel(),
                    current,
                    saved,
                    current * saved,
                    np.abs(current - saved),
                )
            )
        )
    return np.concatenate((base, *query_features)).astype(np.float32)


def _labels_and_masks(
    world: _IntegratedWorld,
    visible: np.ndarray,
    *,
    query_actor_indices: Sequence[int],
    query_validity: Sequence[bool],
    permanence_labels: Sequence[float] | None,
    identity_labels: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    self_x, self_y = map(int, world.self_position)
    a_x, a_y = map(int, world.distractor_a)
    b_x, b_y = map(int, world.distractor_b)
    a_visible = bool(visible[a_y, a_x])
    b_visible = bool(visible[b_y, b_x])
    labels = np.zeros(len(PROPOSITION_NAMES), dtype=np.float32)
    labels[:6] = np.asarray(
        (
            query_actor_indices[0] == 0,
            query_actor_indices[1] == 0,
            a_x < self_x,
            a_y < self_y,
            b_x < self_x,
            b_y < self_y,
        ),
        dtype=np.float32,
    )
    if permanence_labels is not None:
        labels[6:8] = np.asarray(permanence_labels, dtype=np.float32)
    if identity_labels is not None:
        labels[8:10] = np.asarray(identity_labels, dtype=np.float32)
    self_mask = np.zeros(len(PROPOSITION_NAMES), dtype=bool)
    self_mask[0] = bool(query_validity[0])
    self_mask[1] = bool(query_validity[1])

    spatial = np.zeros(len(PROPOSITION_NAMES), dtype=bool)
    for offset, position, is_visible in (
        (2, (a_x, a_y), a_visible),
        (4, (b_x, b_y), b_visible),
    ):
        comparable = (
            position[0] != self_x,
            position[1] != self_y,
        )
        for axis in range(2):
            index = offset + axis
            spatial[index] = bool(is_visible and comparable[axis])
    permanence = np.zeros(len(PROPOSITION_NAMES), dtype=bool)
    identity = np.zeros(len(PROPOSITION_NAMES), dtype=bool)
    if permanence_labels is not None:
        permanence[6:8] = True
    if identity_labels is not None:
        identity[8:10] = True
    training = self_mask | spatial | permanence | identity
    return labels, training, {
        "self": self_mask,
        "spatial": spatial,
        "permanence": permanence,
        "identity": identity,
    }


def _audited_agent_update(
    agent: IntegratedBeliefAgentV2,
    sensed: np.ndarray,
    action: int,
    *,
    received_hasher: Any,
) -> None:
    """Capture the exact bytes at the learner call boundary."""

    received_hasher.update(np.ascontiguousarray(sensed).tobytes())
    received_hasher.update(bytes((int(action),)))
    agent.update(sensed, action)


def collect_language_data(
    seeds: Sequence[int],
    *,
    steps: int,
    warmup: int,
    reappearance_window: int,
    use_action: bool = True,
    infer_occlusion: bool = True,
    audit_log: dict[str, Any] | None = None,
) -> CollectedLanguageData:
    if not seeds:
        raise ValueError("at least one language-readout seed is required")
    graph_features: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    training_masks: list[np.ndarray] = []
    group_masks: dict[str, list[np.ndarray]] = {
        name: [] for name in GROUP_NAMES
    }
    episode_ids: list[int] = []
    identity_reference_roles: list[int] = []
    identity_opposite_control_references: list[np.ndarray] = []
    expected_input_hasher = hashlib.sha256()
    received_input_hasher = hashlib.sha256()
    update_count = 0
    query_after_update_count = 0
    permanence_pair_count = 0
    permanence_positive_visible_count = 0
    permanence_min_hidden_steps: int | None = None
    identity_pair_count = 0
    identity_invalid_boundary_count = 0
    identity_min_hidden_steps: int | None = None

    for episode_id, seed in enumerate(seeds):
        world = _IntegratedWorld(int(seed))
        agent = IntegratedBeliefAgentV2(
            grid_size=world.grid_size,
            infer_occlusion=infer_occlusion,
            use_action=use_action,
            seed=int(seed) + 40_000,
        )
        action_rng = np.random.default_rng(int(seed) + 50_000)
        sensed, local_visibility = world.observe()
        expected_input_hasher.update(np.ascontiguousarray(sensed).tobytes())
        expected_input_hasher.update(bytes((0,)))
        _audited_agent_update(
            agent, sensed, 0, received_hasher=received_input_hasher
        )
        update_count += 1
        visible = _global_visibility(local_visibility, world.grid_size)
        actor_positions = {
            "self": tuple(map(int, world.self_position)),
            "a": tuple(map(int, world.distractor_a)),
            "b": tuple(map(int, world.distractor_b)),
        }
        position_counts = {
            position: tuple(actor_positions.values()).count(position)
            for position in actor_positions.values()
        }
        previous_sensor_visible = {
            name: bool(visible[position[1], position[0]])
            for name, position in actor_positions.items()
            if name != "self"
        }
        previous_unique_visible = {
            name: bool(
                visible[position[1], position[0]]
                and position_counts[position] == 1
            )
            for name, position in actor_positions.items()
            if name != "self"
        }
        hidden_steps = {"a": 0, "b": 0}
        identity_window = {"a": 0, "b": 0}
        identity_occlusion_steps = {"a": 0, "b": 0}
        identity_reference_graph: dict[str, np.ndarray | None] = {
            "a": None,
            "b": None,
        }
        identity_reference_raw: dict[str, np.ndarray | None] = {
            "a": None,
            "b": None,
        }
        identity_event_counter = 0
        initial_maps = _graph_maps(agent, steps=steps)
        for name in ("a", "b"):
            if previous_unique_visible[name]:
                position = actor_positions[name]
                identity_reference_graph[name] = _sample_maps(
                    initial_maps, position
                )
                identity_reference_raw[name] = np.asarray(
                    [float(sensed[position[1] - ARENA_LOW, position[0] - ARENA_LOW])],
                    dtype=np.float32,
                )

        for step in range(1, steps + 1):
            action = int(action_rng.integers(0, len(ACTION_DELTAS)))
            sensed, local_visibility = world.step(action)
            expected_input_hasher.update(
                np.ascontiguousarray(sensed).tobytes()
            )
            expected_input_hasher.update(bytes((action,)))
            _audited_agent_update(
                agent,
                sensed,
                action,
                received_hasher=received_input_hasher,
            )
            update_count += 1
            visible = _global_visibility(local_visibility, world.grid_size)
            actor_positions_tuple = (
                tuple(map(int, world.self_position)),
                tuple(map(int, world.distractor_a)),
                tuple(map(int, world.distractor_b)),
            )
            position_counts = {
                position: actor_positions_tuple.count(position)
                for position in actor_positions_tuple
            }
            current_sensor_visible = {
                name: bool(visible[position[1], position[0]])
                for name, position in zip(
                    ("a", "b"), actor_positions_tuple[1:], strict=True
                )
            }
            current_unique_visible = {
                name: bool(
                    current_sensor_visible[name]
                    and position_counts[position] == 1
                )
                for name, position in zip(
                    ("a", "b"), actor_positions_tuple[1:], strict=True
                )
            }
            prior_hidden_steps = hidden_steps.copy()
            for name in ("a", "b"):
                if (
                    previous_sensor_visible[name]
                    and not current_sensor_visible[name]
                ):
                    if not previous_unique_visible[name]:
                        identity_reference_graph[name] = None
                        identity_reference_raw[name] = None
                    hidden_steps[name] = 1
                elif not current_sensor_visible[name]:
                    hidden_steps[name] += 1
                else:
                    if (
                        not previous_sensor_visible[name]
                        and prior_hidden_steps[name] >= 2
                        and current_unique_visible[name]
                        and identity_reference_graph[name] is not None
                    ):
                        identity_window[name] = reappearance_window
                        identity_occlusion_steps[name] = prior_hidden_steps[name]
                    elif identity_window[name] > 0:
                        identity_window[name] -= 1
                    hidden_steps[name] = 0
                previous_sensor_visible[name] = current_sensor_visible[name]
                previous_unique_visible[name] = current_unique_visible[name]

            if step < warmup:
                current_maps = _graph_maps(agent, steps=steps)
                for name, position in zip(
                    ("a", "b"), actor_positions_tuple[1:], strict=True
                ):
                    if current_unique_visible[name]:
                        identity_reference_graph[name] = _sample_maps(
                            current_maps, position
                        )
                        identity_reference_raw[name] = np.asarray(
                            [
                                float(
                                    sensed[
                                        position[1] - ARENA_LOW,
                                        position[0] - ARENA_LOW,
                                    ]
                                )
                            ],
                            dtype=np.float32,
                        )
                continue
            query_after_update_count += 1
            actor_positions = actor_positions_tuple
            actor_visibility = (
                bool(
                    visible[
                        actor_positions[0][1],
                        actor_positions[0][0],
                    ]
                ),
                current_unique_visible["a"],
                current_unique_visible["b"],
            )
            distractor_actor = 1 + ((int(seed) + step) % 2)
            query_actor_indices = (
                (distractor_actor, 0)
                if _deterministic_order_bit(int(seed), step, "self")
                else (0, distractor_actor)
            )
            query_validity = tuple(
                bool(
                    actor_visibility[index]
                    and position_counts[actor_positions[index]] == 1
                )
                for index in query_actor_indices
            )
            query_positions = tuple(
                actor_positions[index] if valid else None
                for index, valid in zip(
                    query_actor_indices,
                    query_validity,
                    strict=True,
                )
            )
            permanence_positions: tuple[
                tuple[int, int] | None, tuple[int, int] | None
            ] = (None, None)
            permanence_labels: tuple[float, float] | None = None
            hidden_names = [
                name
                for name in ("a", "b")
                if hidden_steps[name] >= 2
            ]
            if hidden_names:
                selected_name = hidden_names[(int(seed) + step) % len(hidden_names)]
                positive = actor_positions[1 if selected_name == "a" else 2]
                truth = world.truth()
                negative_candidates = [
                    (x, y)
                    for y in range(ARENA_LOW, ARENA_HIGH + 1)
                    for x in range(ARENA_LOW, ARENA_HIGH + 1)
                    if not visible[y, x]
                    and not bool(truth[y, x])
                    and (x, y) not in world.static
                ]
                if negative_candidates:
                    permanence_pair_count += 1
                    permanence_positive_visible_count += int(
                        bool(visible[positive[1], positive[0]])
                    )
                    permanence_min_hidden_steps = (
                        hidden_steps[selected_name]
                        if permanence_min_hidden_steps is None
                        else min(
                            permanence_min_hidden_steps,
                            hidden_steps[selected_name],
                        )
                    )
                    negative = min(
                        negative_candidates,
                        key=lambda cell: (
                            abs(cell[0] - positive[0])
                            + abs(cell[1] - positive[1]),
                            (cell[0] * 17 + cell[1] * 31 + int(seed) + step)
                            % 97,
                        ),
                    )
                    if _deterministic_order_bit(
                        int(seed), step, "permanence"
                    ):
                        permanence_positions = (negative, positive)
                        permanence_labels = (0.0, 1.0)
                    else:
                        permanence_positions = (positive, negative)
                        permanence_labels = (1.0, 0.0)

            identity_positions: tuple[
                tuple[int, int] | None, tuple[int, int] | None
            ] = (None, None)
            identity_labels: tuple[float, float] | None = None
            identity_refs_graph: tuple[np.ndarray | None, np.ndarray | None] = (
                None,
                None,
            )
            identity_refs_raw: tuple[np.ndarray | None, np.ndarray | None] = (
                None,
                None,
            )
            active_identity_names = [
                name
                for name in ("a", "b")
                if identity_window[name] > 0
                and current_unique_visible[name]
                and current_unique_visible["b" if name == "a" else "a"]
                and identity_reference_graph[name] is not None
                and identity_reference_raw[name] is not None
            ]
            selected_identity_role = -1
            opposite_control_reference = np.zeros(
                len(GRAPH_MAP_NAMES), dtype=np.float32
            )
            if active_identity_names:
                selected_name = active_identity_names[
                    identity_event_counter % len(active_identity_names)
                ]
                other_name = "b" if selected_name == "a" else "a"
                same = actor_positions[1 if selected_name == "a" else 2]
                other = actor_positions[1 if other_name == "a" else 2]
                graph_reference = identity_reference_graph[selected_name]
                raw_reference = identity_reference_raw[selected_name]
                identity_pair_count += 1
                identity_min_hidden_steps = (
                    identity_occlusion_steps[selected_name]
                    if identity_min_hidden_steps is None
                    else min(
                        identity_min_hidden_steps,
                        identity_occlusion_steps[selected_name],
                    )
                )
                identity_invalid_boundary_count += int(
                    identity_occlusion_steps[selected_name] < 2
                    or not current_unique_visible[selected_name]
                    or not current_unique_visible[other_name]
                    or graph_reference is None
                    or raw_reference is None
                )
                if (identity_event_counter // 2) % 2:
                    identity_positions = (other, same)
                    identity_labels = (0.0, 1.0)
                else:
                    identity_positions = (same, other)
                    identity_labels = (1.0, 0.0)
                identity_refs_graph = (graph_reference, graph_reference)
                identity_refs_raw = (raw_reference, raw_reference)
                selected_identity_role = 0 if selected_name == "a" else 1
                opposite_control_reference = _sample_maps(
                    _graph_maps(agent, steps=steps),
                    other,
                )
                identity_event_counter += 1

            all_query_positions = (
                *query_positions,
                *permanence_positions,
                *identity_positions,
            )
            graph_references = (
                None,
                None,
                None,
                None,
                *identity_refs_graph,
            )
            raw_references = (
                None,
                None,
                None,
                None,
                *identity_refs_raw,
            )
            item_labels, training_mask, item_groups = _labels_and_masks(
                world,
                visible,
                query_actor_indices=query_actor_indices,
                query_validity=query_validity,
                permanence_labels=permanence_labels,
                identity_labels=identity_labels,
            )
            graph_features.append(
                _graph_features(
                    agent,
                    action,
                    steps=steps,
                    query_positions=all_query_positions,
                    reference_descriptors=graph_references,
                )
            )
            raw_features.append(
                _raw_features(
                    sensed,
                    action,
                    query_positions=all_query_positions,
                    reference_descriptors=raw_references,
                )
            )
            labels.append(item_labels)
            training_masks.append(training_mask)
            for name in GROUP_NAMES:
                group_masks[name].append(item_groups[name])
            episode_ids.append(episode_id)
            identity_reference_roles.append(selected_identity_role)
            identity_opposite_control_references.append(
                opposite_control_reference
            )
            current_maps = _graph_maps(agent, steps=steps)
            for name, position in zip(
                ("a", "b"), actor_positions[1:], strict=True
            ):
                if (
                    current_unique_visible[name]
                    and identity_window[name] == 0
                ):
                    identity_reference_graph[name] = _sample_maps(
                        current_maps, position
                    )
                    identity_reference_raw[name] = np.asarray(
                        [
                            float(
                                sensed[
                                    position[1] - ARENA_LOW,
                                    position[0] - ARENA_LOW,
                                ]
                            )
                        ],
                        dtype=np.float32,
                    )

    collected = CollectedLanguageData(
        graph_features=torch.from_numpy(np.stack(graph_features)).detach(),
        raw_features=torch.from_numpy(np.stack(raw_features)).detach(),
        labels=torch.from_numpy(np.stack(labels)).detach(),
        training_mask=torch.from_numpy(np.stack(training_masks)).detach(),
        group_masks={
            name: torch.from_numpy(np.stack(values)).detach()
            for name, values in group_masks.items()
        },
        episode_ids=torch.tensor(episode_ids, dtype=torch.int64),
        graph_base_feature_count=(
            len(GRAPH_MAP_NAMES)
            * (ARENA_HIGH - ARENA_LOW + 1) ** 2
            + len(ACTION_DELTAS)
        ),
        raw_base_feature_count=(
            (ARENA_HIGH - ARENA_LOW + 1) ** 2
            + len(ACTION_DELTAS)
        ),
        query_count=6,
        graph_query_block_size=(
            (ARENA_HIGH - ARENA_LOW + 1) ** 2
            + 4 * len(GRAPH_MAP_NAMES)
        ),
        raw_query_block_size=(
            (ARENA_HIGH - ARENA_LOW + 1) ** 2 + 4
        ),
        identity_reference_roles=torch.tensor(
            identity_reference_roles, dtype=torch.int64
        ),
        identity_opposite_control_references=torch.from_numpy(
            np.stack(identity_opposite_control_references)
        ).detach(),
    )
    if audit_log is not None:
        audit_log.update(
            {
                "update_count": update_count,
                "evaluated_sample_count": len(labels),
                "queries_constructed_after_update_count": (
                    query_after_update_count
                ),
                "queries_constructed_after_update": (
                    query_after_update_count == len(labels)
                ),
                "world_output_stream_sha256": (
                    expected_input_hasher.hexdigest()
                ),
                "learner_received_stream_sha256": (
                    received_input_hasher.hexdigest()
                ),
                "learner_input_stream_sha256": (
                    received_input_hasher.hexdigest()
                ),
                "world_output_equals_learner_input": (
                    expected_input_hasher.digest()
                    == received_input_hasher.digest()
                ),
                "update_signature": list(
                    inspect.signature(
                        IntegratedBeliefAgentV2.update
                    ).parameters
                ),
                "labels_or_truth_in_update_signature": any(
                    name in {
                        "label",
                        "labels",
                        "truth",
                        "query",
                        "language",
                    }
                    for name in inspect.signature(
                        IntegratedBeliefAgentV2.update
                    ).parameters
                ),
                "paired_query_semantics": {
                    "permanence_pair_count": permanence_pair_count,
                    "permanence_positive_sensor_visible_count": (
                        permanence_positive_visible_count
                    ),
                    "permanence_min_continuous_hidden_steps": (
                        permanence_min_hidden_steps
                    ),
                    "identity_pair_count": identity_pair_count,
                    "identity_invalid_boundary_count": (
                        identity_invalid_boundary_count
                    ),
                    "identity_min_continuous_hidden_steps": (
                        identity_min_hidden_steps
                    ),
                },
            }
        )
    return collected


def time_shuffle_data(
    data: CollectedLanguageData,
    *,
    lag: int,
) -> CollectedLanguageData:
    if lag < 1:
        raise ValueError("time shuffle lag must be positive")
    shuffled = data.graph_features.clone()
    base_count = data.graph_base_feature_count
    for episode_id in torch.unique(data.episode_ids):
        indices = torch.nonzero(
            data.episode_ids == episode_id,
            as_tuple=False,
        ).flatten()
        source = torch.clamp(
            torch.arange(len(indices)) - lag,
            min=0,
        )
        shuffled[indices, :base_count] = data.graph_features[
            indices[source],
            :base_count,
        ]
    arena_cells = (ARENA_HIGH - ARENA_LOW + 1) ** 2
    descriptor_count = len(GRAPH_MAP_NAMES)
    for query_index in range(data.query_count):
        query_start = base_count + query_index * data.graph_query_block_size
        query_mask = shuffled[
            :,
            query_start : query_start + arena_cells,
        ]
        current_start = query_start + arena_cells
        reference_start = current_start + descriptor_count
        product_start = reference_start + descriptor_count
        difference_start = product_start + descriptor_count
        for row in range(shuffled.shape[0]):
            active = torch.nonzero(
                query_mask[row] > 0.5,
                as_tuple=False,
            ).flatten()
            if len(active) != 1:
                shuffled[
                    row,
                    current_start : current_start + descriptor_count,
                ] = 0.0
            else:
                cell_index = int(active[0])
                shuffled[
                    row,
                    current_start : current_start + descriptor_count,
                ] = torch.stack(
                    [
                        shuffled[
                            row,
                            map_index * arena_cells + cell_index,
                        ]
                        for map_index in range(descriptor_count)
                    ]
                )
            current = shuffled[
                row, current_start : current_start + descriptor_count
            ]
            reference = shuffled[
                row, reference_start : reference_start + descriptor_count
            ]
            shuffled[
                row, product_start : product_start + descriptor_count
            ] = current * reference
            shuffled[
                row, difference_start : difference_start + descriptor_count
            ] = torch.abs(current - reference)
    return data.with_graph_features(shuffled)


def referent_swap_data(data: CollectedLanguageData) -> CollectedLanguageData:
    order = torch.tensor((0, 1, 4, 5, 2, 3, 7, 6, 9, 8))
    return data.with_labels(
        data.labels[:, order],
        data.training_mask[:, order],
    )


def identity_scramble_data(
    data: CollectedLanguageData,
    *,
    lag: int = 1,
) -> CollectedLanguageData:
    """Replace saved identity history with the row-local other actor."""

    if lag < 1:
        raise ValueError("identity scramble lag must be positive")
    features = data.graph_features.clone()
    scrambled_roles = data.identity_reference_roles.clone()
    descriptor_count = len(GRAPH_MAP_NAMES)
    arena_cells = (ARENA_HIGH - ARENA_LOW + 1) ** 2
    active = torch.nonzero(
        data.group_masks["identity"][:, 8:10].any(dim=1),
        as_tuple=False,
    ).flatten()
    roles = data.identity_reference_roles[active]
    if bool(((roles != 0) & (roles != 1)).any()):
        raise RuntimeError("active identity row lacks a reference role")
    scrambled_roles[active] = 1 - roles
    references = data.identity_opposite_control_references[active].clone()
    for query_index in (4, 5):
        start = (
            data.graph_base_feature_count
            + query_index * data.graph_query_block_size
            + arena_cells
        )
        current_start = start
        reference_start = current_start + descriptor_count
        product_start = reference_start + descriptor_count
        difference_start = product_start + descriptor_count
        features[
            active,
            reference_start : reference_start + descriptor_count,
        ] = references
        current = features[
            active, current_start : current_start + descriptor_count
        ]
        features[
            active, product_start : product_start + descriptor_count
        ] = current * references
        features[
            active,
            difference_start : difference_start + descriptor_count,
        ] = torch.abs(current - references)
    return replace(
        data,
        graph_features=features,
        identity_reference_roles=scrambled_roles,
    )


def identity_scramble_audit(
    original: CollectedLanguageData,
    scrambled: CollectedLanguageData,
) -> dict[str, float | int | bool]:
    descriptor_count = len(GRAPH_MAP_NAMES)
    arena_cells = (ARENA_HIGH - ARENA_LOW + 1) ** 2
    active = original.group_masks["identity"][:, 8:10].any(dim=1)
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    active_labels = original.labels[active, 8:10]
    valid_label_pairs = bool(
        len(active_indices)
        and ((active_labels == 0.0).sum(dim=1) == 1).all()
        and ((active_labels == 1.0).sum(dim=1) == 1).all()
    )
    negative_queries = torch.argmin(active_labels, dim=1)
    expected_references = torch.empty(
        (len(active_indices), descriptor_count),
        dtype=original.graph_features.dtype,
    )
    for candidate_index, query_index in enumerate((4, 5)):
        selected = negative_queries == candidate_index
        current_start = (
            original.graph_base_feature_count
            + query_index * original.graph_query_block_size
            + arena_cells
        )
        expected_references[selected] = original.graph_features[
            active_indices[selected],
            current_start : current_start + descriptor_count,
        ]
    metadata_matches = int(
        torch.eq(
            original.identity_opposite_control_references[active],
            expected_references,
        )
        .all(dim=1)
        .sum()
    )
    changed = 0
    opposite_motion = 0
    row_local_matches = 0
    recomputed_products = 0
    recomputed_differences = 0
    count = 0
    allowed_changes = torch.zeros(
        original.graph_features.shape[1], dtype=torch.bool
    )
    inactive = ~active
    preserved = (
        torch.equal(original.labels, scrambled.labels)
        and torch.equal(original.training_mask, scrambled.training_mask)
        and torch.equal(original.episode_ids, scrambled.episode_ids)
        and torch.equal(original.raw_features, scrambled.raw_features)
        and all(
            torch.equal(original.group_masks[name], scrambled.group_masks[name])
            for name in GROUP_NAMES
        )
        and torch.equal(
            original.graph_features[inactive],
            scrambled.graph_features[inactive],
        )
    )
    for query_index in (4, 5):
        block_start = (
            original.graph_base_feature_count
            + query_index * original.graph_query_block_size
        )
        current_start = block_start + arena_cells
        reference_start = current_start + descriptor_count
        product_start = reference_start + descriptor_count
        difference_start = product_start + descriptor_count
        allowed_changes[
            reference_start : difference_start + descriptor_count
        ] = True
        before = original.graph_features[
            active,
            reference_start : reference_start + descriptor_count,
        ]
        after = scrambled.graph_features[
            active,
            reference_start : reference_start + descriptor_count,
        ]
        changed += int((before != after).any(dim=1).sum())
        before_motion = torch.argmax(before[:, 8:10], dim=1)
        after_motion = torch.argmax(after[:, 8:10], dim=1)
        opposite_motion += int((before_motion != after_motion).sum())
        row_local_matches += int(
            torch.eq(after, expected_references).all(dim=1).sum()
        )
        current = scrambled.graph_features[
            active, current_start : current_start + descriptor_count
        ]
        product = scrambled.graph_features[
            active, product_start : product_start + descriptor_count
        ]
        difference = scrambled.graph_features[
            active, difference_start : difference_start + descriptor_count
        ]
        recomputed_products += int(
            torch.eq(product, current * after).all(dim=1).sum()
        )
        recomputed_differences += int(
            torch.eq(difference, torch.abs(current - after)).all(dim=1).sum()
        )
        count += int(before.shape[0])
    preserved = (
        preserved
        and torch.equal(
            original.graph_features[:, ~allowed_changes],
            scrambled.graph_features[:, ~allowed_changes],
        )
    )
    active_row_count = len(active_indices)
    return {
        "active_identity_row_count": active_row_count,
        "reference_block_count": count,
        "changed_reference_block_count": changed,
        "opposite_motion_reference_block_count": opposite_motion,
        "row_local_counterfactual_match_count": row_local_matches,
        "metadata_matches_negative_candidate_current_count": metadata_matches,
        "recomputed_product_block_count": recomputed_products,
        "recomputed_difference_block_count": recomputed_differences,
        "all_references_changed": bool(count and changed == count),
        "counterfactual_defined_for_every_active_identity_row": bool(
            valid_label_pairs
            and active_row_count
            and metadata_matches == active_row_count
        ),
        "all_interactions_recomputed": bool(
            count
            and recomputed_products == count
            and recomputed_differences == count
        ),
        "preserved_labels_masks_candidates_and_nonidentity_features": bool(
            preserved
        ),
        "opposite_motion_fraction": (
            float(opposite_motion / count) if count else 0.0
        ),
        "row_local_counterfactual_fraction": (
            float(row_local_matches / count) if count else 0.0
        ),
        "metadata_matches_negative_candidate_current_fraction": (
            float(metadata_matches / active_row_count)
            if active_row_count
            else 0.0
        ),
        "opposite_reference_role_fraction": (
            float(
                (
                    original.identity_reference_roles[active]
                    != scrambled.identity_reference_roles[active]
                )
                .to(torch.float32)
                .mean()
            )
            if bool(active.any())
            else 0.0
        ),
    }


def random_label_data(
    data: CollectedLanguageData,
    *,
    seed: int,
) -> CollectedLanguageData:
    generator = torch.Generator().manual_seed(seed)
    labels = data.labels.clone()
    for column in range(labels.shape[1]):
        permutation = torch.randperm(labels.shape[0], generator=generator)
        labels[:, column] = labels[permutation, column]
    return data.with_labels(labels)


def _standardization(features: Tensor) -> tuple[Tensor, Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return mean, scale


def _masked_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    positive_weight: Tensor,
) -> Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=positive_weight,
    )
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def train_readout(
    data: CollectedLanguageData,
    *,
    representation: str,
    config: ReadoutConfig,
) -> ReadoutModel:
    features = data.features(representation).to(torch.float32)
    labels = data.labels.to(torch.float32)
    mask = data.training_mask.to(torch.bool)
    mean, scale = _standardization(features)
    normalized = (features - mean) / scale

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    probe = LinearLanguageReadout(
        features.shape[1],
        data.base_feature_count(representation),
        query_count=data.query_count,
    ).to(device)
    optimizer = Adam(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positives = (labels * mask).sum(dim=0)
    negatives = ((1.0 - labels) * mask).sum(dim=0)
    positive_weight = (
        negatives / torch.clamp(positives, min=1.0)
    ).to(device)
    loader = DataLoader(
        TensorDataset(normalized, labels, mask),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    final_loss = float("nan")
    for _ in range(config.epochs):
        probe.train()
        loss_sum = 0.0
        weight_sum = 0
        for batch_features, batch_labels, batch_mask in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(
                probe(batch_features),
                batch_labels,
                batch_mask,
                positive_weight=positive_weight,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_features.shape[0]
            weight_sum += batch_features.shape[0]
        final_loss = loss_sum / max(weight_sum, 1)
    return ReadoutModel(
        probe=copy.deepcopy(probe).cpu(),
        mean=mean.detach(),
        scale=scale.detach(),
        final_loss=final_loss,
    )


@torch.no_grad()
def readout_logits(
    model: ReadoutModel,
    data: CollectedLanguageData,
    *,
    representation: str,
) -> Tensor:
    features = data.features(representation).to(torch.float32)
    normalized = (features - model.mean) / model.scale
    model.probe.eval()
    return model.probe(normalized)


def _proposition_metrics(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> list[dict[str, float | int | None]]:
    predictions = logits >= 0.0
    target = labels >= 0.5
    result: list[dict[str, float | int | None]] = []
    for column in range(labels.shape[1]):
        active = mask[:, column].to(torch.bool)
        column_target = target[active, column]
        column_prediction = predictions[active, column]
        positive = column_target
        negative = ~column_target
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        score: float | None = None
        if positive_count and negative_count:
            true_positive_rate = float(
                column_prediction[positive].to(torch.float32).mean()
            )
            true_negative_rate = float(
                (~column_prediction[negative]).to(torch.float32).mean()
            )
            score = (true_positive_rate + true_negative_rate) / 2.0
        result.append(
            {
                "active_count": int(active.sum()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "balanced_accuracy": score,
            }
        )
    return result


def _balanced_accuracy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> tuple[float, int]:
    metrics = _proposition_metrics(logits, labels, mask)
    active = [item for item in metrics if int(item["active_count"]) > 0]
    if not active or any(item["balanced_accuracy"] is None for item in active):
        raise RuntimeError(
            "balanced accuracy requires both classes for every registered column"
        )
    return (
        float(
            sum(float(item["balanced_accuracy"]) for item in active)
            / len(active)
        ),
        sum(int(item["active_count"]) for item in active),
    )


def evaluate_readout(
    model: ReadoutModel,
    data: CollectedLanguageData,
    *,
    representation: str,
) -> dict[str, Any]:
    logits = readout_logits(model, data, representation=representation)
    result: dict[str, Any] = {}
    concept_values: list[float] = []
    proposition_evidence: dict[str, dict[str, float | int | None]] = {}
    for name in GROUP_NAMES:
        metrics = _proposition_metrics(
            logits,
            data.labels,
            data.group_masks[name],
        )
        relevant = [
            (index, item)
            for index, item in enumerate(metrics)
            if int(item["active_count"]) > 0
        ]
        complete = bool(relevant) and all(
            item["balanced_accuracy"] is not None for _, item in relevant
        )
        score = (
            float(
                sum(float(item["balanced_accuracy"]) for _, item in relevant)
                / len(relevant)
            )
            if complete
            else 0.0
        )
        count = sum(int(item["active_count"]) for _, item in relevant)
        result[f"{name}_balanced_accuracy"] = score
        result[f"{name}_sample_count"] = count
        result[f"{name}_all_columns_covered"] = complete
        concept_values.append(score)
        for index, item in relevant:
            proposition_evidence[PROPOSITION_NAMES[index]] = item
    for index, proposition in enumerate(PROPOSITION_NAMES):
        proposition_evidence.setdefault(
            proposition,
            {
                "active_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "balanced_accuracy": None,
            },
        )
    result["per_proposition"] = proposition_evidence
    result["all_registered_columns_covered"] = all(
        item["balanced_accuracy"] is not None
        for item in proposition_evidence.values()
    )
    result["macro_balanced_accuracy"] = float(
        sum(concept_values) / len(concept_values)
    )
    return result


def render_language(
    probabilities: Sequence[float],
    *,
    protocol: Mapping[str, Any],
    template_split: str,
) -> list[dict[str, Any]]:
    templates = protocol["semantic_schema"]["templates"][template_split]
    if len(probabilities) != len(PROPOSITION_NAMES):
        raise ValueError("one probability is required per proposition")
    if len(templates) != len(PROPOSITION_NAMES):
        raise RuntimeError("language template schema mismatch")
    return [
        {
            "proposition": proposition,
            "sentence": sentence,
            "probability_true": float(probability),
            "predicted_true": bool(probability >= 0.5),
        }
        for proposition, sentence, probability in zip(
            PROPOSITION_NAMES,
            templates,
            probabilities,
            strict=True,
        )
    ]


def _readout_config(protocol: Mapping[str, Any]) -> ReadoutConfig:
    fixed = protocol["fixed_execution"]
    return ReadoutConfig(
        epochs=int(fixed["training_epochs"]),
        batch_size=int(fixed["batch_size"]),
        learning_rate=float(fixed["learning_rate"]),
        weight_decay=float(fixed["weight_decay"]),
        seed=int(fixed["probe_seed"]),
        device=str(fixed["device"]),
    )


def _fit_and_evaluate(
    train: CollectedLanguageData,
    evaluation: CollectedLanguageData,
    *,
    representation: str,
    config: ReadoutConfig,
) -> tuple[ReadoutModel, dict[str, float | int]]:
    model = train_readout(
        train,
        representation=representation,
        config=config,
    )
    return model, evaluate_readout(
        model,
        evaluation,
        representation=representation,
    )


def subset_data(
    data: CollectedLanguageData,
    indices: Tensor,
) -> CollectedLanguageData:
    indices = indices.to(torch.int64)
    return CollectedLanguageData(
        graph_features=data.graph_features[indices],
        raw_features=data.raw_features[indices],
        labels=data.labels[indices],
        training_mask=data.training_mask[indices],
        group_masks={
            name: mask[indices] for name, mask in data.group_masks.items()
        },
        episode_ids=data.episode_ids[indices],
        graph_base_feature_count=data.graph_base_feature_count,
        raw_base_feature_count=data.raw_base_feature_count,
        query_count=data.query_count,
        graph_query_block_size=data.graph_query_block_size,
        raw_query_block_size=data.raw_query_block_size,
        identity_reference_roles=data.identity_reference_roles[indices],
        identity_opposite_control_references=(
            data.identity_opposite_control_references[indices]
        ),
    )


def _numeric_id_permutation_audit_one(
    model: ReadoutModel,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    world = _IntegratedWorld(seed)
    agent = IntegratedBeliefAgentV2(seed=seed + 40_000)
    action_rng = np.random.default_rng(seed + 50_000)
    sensed, _ = world.observe()
    agent.update(sensed, 0)
    action = 0
    for _ in range(min(steps, 24)):
        action = int(action_rng.integers(0, len(ACTION_DELTAS)))
        sensed, _ = world.step(action)
        agent.update(sensed, action)
    positions = (
        tuple(map(int, world.self_position)),
        tuple(map(int, world.distractor_a)),
        tuple(map(int, world.distractor_b)),
        None,
        tuple(map(int, world.distractor_a)),
        tuple(map(int, world.distractor_b)),
    )
    maps = _graph_maps(agent, steps=steps)
    reference = _sample_maps(maps, positions[4])
    references = (None, None, None, None, reference, reference)
    original = _graph_features(
        agent,
        action,
        steps=steps,
        query_positions=positions,
        reference_descriptors=references,
    )
    renamed = copy.deepcopy(agent)
    original_ids = sorted(
        {
            int(entity.index)
            for hypothesis in renamed.graph._hypotheses
            for entity in hypothesis.entities
        }
    )
    permutation = {
        identity: 10_003 + rank * 37
        for rank, identity in enumerate(reversed(original_ids))
    }
    renamed_objects: set[int] = set()
    for hypothesis in renamed.graph._hypotheses:
        for entity in hypothesis.entities:
            if id(entity) in renamed_objects:
                continue
            renamed_objects.add(id(entity))
            entity.index = permutation[int(entity.index)]
    shifted = _graph_features(
        renamed,
        action,
        steps=steps,
        query_positions=positions,
        reference_descriptors=references,
    )
    original_tensor = torch.from_numpy(original).unsqueeze(0)
    shifted_tensor = torch.from_numpy(shifted).unsqueeze(0)
    model.probe.eval()
    with torch.no_grad():
        original_logits = model.probe(
            (original_tensor - model.mean) / model.scale
        )
        shifted_logits = model.probe(
            (shifted_tensor - model.mean) / model.scale
        )
    return {
        "seed": seed,
        "entity_count": len(original_ids),
        "permutation_non_identity": bool(
            original_ids
            and all(permutation[item] != item for item in original_ids)
        ),
        "permutation": {str(key): value for key, value in permutation.items()},
        "feature_bytes_equal": bool(np.array_equal(original, shifted)),
        "logits_equal": bool(torch.equal(original_logits, shifted_logits)),
        "original_feature_sha256": hashlib.sha256(
            original.tobytes()
        ).hexdigest(),
        "permuted_feature_sha256": hashlib.sha256(
            shifted.tobytes()
        ).hexdigest(),
    }


def _numeric_id_permutation_audit(
    model: ReadoutModel,
    *,
    seeds: Sequence[int],
    steps: int,
) -> dict[str, Any]:
    cases = [
        _numeric_id_permutation_audit_one(
            model, seed=int(seed), steps=steps
        )
        for seed in seeds
    ]
    return {
        "cases": cases,
        "case_count": len(cases),
        "minimum_entity_count": min(
            int(case["entity_count"]) for case in cases
        ),
        "all_permutations_non_identity": all(
            bool(case["permutation_non_identity"]) for case in cases
        ),
        "feature_bytes_equal": all(
            bool(case["feature_bytes_equal"]) for case in cases
        ),
        "logits_equal": all(bool(case["logits_equal"]) for case in cases),
    }


def _gates(
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    parameter_count: int,
    protocol: Mapping[str, Any],
    formal_per_seed: Mapping[str, Mapping[str, Any]],
    runtime_audits: Mapping[str, Any],
    split: str,
) -> dict[str, bool]:
    fixed = protocol["fixed_gates"]
    formal = conditions["formal_entity_graph"]
    raw = conditions["raw_sensor"]
    shuffled = conditions["time_shuffled"]
    swapped = conditions["referent_swapped"]
    random_labels = conditions["random_labels"]
    no_action = conditions["no_action_entity_graph"]
    all_visible = conditions["assume_all_visible_entity_graph"]
    identity_scrambled = conditions["identity_scrambled_at_occlusion"]
    formal_spatial_identity = (
        float(formal["spatial_balanced_accuracy"])
        + float(formal["identity_balanced_accuracy"])
    ) / 2.0
    swapped_spatial_identity = (
        float(swapped["spatial_balanced_accuracy"])
        + float(swapped["identity_balanced_accuracy"])
    ) / 2.0
    coverage = formal["per_proposition"]
    coverage_spec = protocol["coverage_and_evidence"]
    minimum_per_class = int(
        coverage_spec[
            "development_validation_minimum_positive_per_proposition"
            if split == "development"
            else "holdout_minimum_positive_per_proposition"
        ]
    )
    input_audit = runtime_audits["learner_input_boundary"]
    id_audit = runtime_audits["numeric_id_permutation"]
    query_audit = input_audit["paired_query_semantics"]
    scramble_audit = runtime_audits["identity_scramble_integrity"]
    return {
        "formal_macro_pass": (
            float(formal["macro_balanced_accuracy"])
            >= fixed["formal_macro_balanced_accuracy_minimum"]
        ),
        "formal_self_pass": (
            float(formal["self_balanced_accuracy"])
            >= fixed["formal_self_balanced_accuracy_minimum"]
        ),
        "formal_spatial_pass": (
            float(formal["spatial_balanced_accuracy"])
            >= fixed["formal_spatial_balanced_accuracy_minimum"]
        ),
        "formal_permanence_pass": (
            float(formal["permanence_balanced_accuracy"])
            >= fixed["formal_permanence_balanced_accuracy_minimum"]
        ),
        "formal_identity_pass": (
            float(formal["identity_balanced_accuracy"])
            >= fixed["formal_identity_balanced_accuracy_minimum"]
        ),
        "formal_beats_raw": (
            float(formal["macro_balanced_accuracy"])
            - float(raw["macro_balanced_accuracy"])
            >= fixed["formal_minus_raw_macro_minimum"]
        ),
        "time_shuffle_fails": (
            float(formal["macro_balanced_accuracy"])
            - float(shuffled["macro_balanced_accuracy"])
            >= fixed["time_shuffle_macro_drop_minimum"]
        ),
        "referent_swap_fails": (
            formal_spatial_identity - swapped_spatial_identity
            >= fixed["referent_swap_spatial_identity_drop_minimum"]
        ),
        "random_labels_fail": (
            float(random_labels["macro_balanced_accuracy"])
            <= fixed["random_label_macro_maximum"]
        ),
        "no_action_self_fails": (
            float(formal["self_balanced_accuracy"])
            - float(no_action["self_balanced_accuracy"])
            >= fixed["no_action_self_drop_minimum"]
        ),
        "all_i1_parameters_frozen": (
            fixed["all_i1_parameters_frozen"] is True
            and bool(runtime_audits["frozen_i1_source_hashes_match"])
        ),
        "labels_absent_from_i1": (
            fixed["labels_absent_from_i1"] is True
            and set(input_audit["update_signature"])
            == {"self", "sensed_occupancy", "action"}
            and not input_audit["labels_or_truth_in_update_signature"]
            and input_audit["queries_constructed_after_update"]
            and input_audit["world_output_equals_learner_input"]
        ),
        "numeric_entity_ids_absent_from_probe": (
            fixed["numeric_entity_ids_absent_from_probe"] is True
            and id_audit["feature_bytes_equal"]
            and id_audit["logits_equal"]
            and id_audit["all_permutations_non_identity"]
            and int(id_audit["minimum_entity_count"]) >= 2
        ),
        "probe_parameter_count_pass": (
            parameter_count
            <= fixed["probe_parameter_count_maximum"]
        ),
        "formal_beats_raw_permanence": (
            float(formal["permanence_balanced_accuracy"])
            - float(raw["permanence_balanced_accuracy"])
            >= fixed["formal_minus_raw_permanence_minimum"]
        ),
        "formal_beats_raw_identity": (
            float(formal["identity_balanced_accuracy"])
            - float(raw["identity_balanced_accuracy"])
            >= fixed["formal_minus_raw_identity_minimum"]
        ),
        "all_visible_permanence_fails": (
            float(formal["permanence_balanced_accuracy"])
            - float(all_visible["permanence_balanced_accuracy"])
            >= fixed[
                "formal_minus_assume_all_visible_permanence_minimum"
            ]
        ),
        "identity_scramble_fails": (
            float(formal["identity_balanced_accuracy"])
            - float(identity_scrambled["identity_balanced_accuracy"])
            >= fixed[
                "formal_minus_identity_scrambled_identity_minimum"
            ]
        ),
        "identity_scramble_integrity_pass": (
            bool(scramble_audit["all_references_changed"])
            and float(scramble_audit["opposite_motion_fraction"]) >= 0.9
            and float(
                scramble_audit["opposite_reference_role_fraction"]
            )
            == 1.0
            and (
                int(protocol["protocol_version"]) < 6
                or (
                    int(scramble_audit["active_identity_row_count"])
                    >= int(
                        protocol["control_integrity_gates"][
                            "active_identity_reference_count_minimum"
                        ]
                    )
                    and float(
                        scramble_audit[
                            "row_local_counterfactual_fraction"
                        ]
                    )
                    >= float(
                        protocol["control_integrity_gates"][
                            "row_local_counterfactual_fraction_minimum"
                        ]
                    )
                    and float(
                        scramble_audit[
                            "metadata_matches_negative_candidate_current_fraction"
                        ]
                    )
                    == 1.0
                    and bool(
                        scramble_audit[
                            "counterfactual_defined_for_every_active_identity_row"
                        ]
                    )
                    and bool(
                        scramble_audit["all_interactions_recomputed"]
                    )
                    and bool(
                        scramble_audit[
                            "preserved_labels_masks_candidates_and_nonidentity_features"
                        ]
                    )
                )
            )
        ),
        "all_registered_columns_covered": all(
            int(item["positive_count"]) >= minimum_per_class
            and int(item["negative_count"]) >= minimum_per_class
            and item["balanced_accuracy"] is not None
            for item in coverage.values()
        ),
        "all_formal_per_seed_macro_pass": all(
            float(metrics["macro_balanced_accuracy"])
            >= coverage_spec[
                "formal_per_seed_macro_balanced_accuracy_minimum"
            ]
            and bool(metrics["all_registered_columns_covered"])
            for metrics in formal_per_seed.values()
        ),
        "runtime_input_audit_pass": (
            input_audit["queries_constructed_after_update"]
            and not input_audit["labels_or_truth_in_update_signature"]
            and input_audit["world_output_equals_learner_input"]
        ),
        "paired_query_semantics_pass": (
            int(query_audit["permanence_pair_count"]) > 0
            and int(
                query_audit["permanence_positive_sensor_visible_count"]
            )
            == 0
            and int(
                query_audit["permanence_min_continuous_hidden_steps"]
            )
            >= 2
            and int(query_audit["identity_pair_count"]) > 0
            and int(query_audit["identity_invalid_boundary_count"]) == 0
            and int(query_audit["identity_min_continuous_hidden_steps"])
            >= 2
        ),
        "runtime_numeric_id_permutation_audit_pass": (
            id_audit["feature_bytes_equal"] and id_audit["logits_equal"]
            and id_audit["all_permutations_non_identity"]
            and int(id_audit["minimum_entity_count"]) >= 2
        ),
    }


def _publish_failure_evidence(
    path: Path,
    *,
    remote: str,
    tag: str,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    raw = path.read_bytes()
    failure_sha256 = hashlib.sha256(raw).hexdigest()
    blob = _git_command(
        "hash-object",
        "-w",
        "--stdin",
        cwd=PROJECT_ROOT,
        text=False,
        input_data=raw,
    ).stdout.decode().strip()
    payload = {
        **certificate,
        "certificate_schema_version": 1,
        "certificate_type": "immutable_failure_evidence",
        "failure_sha256": failure_sha256,
        "git_blob": blob,
    }
    _publish_annotated_tag(
        remote=remote,
        tag=tag,
        target=blob,
        certificate=payload,
        cwd=PROJECT_ROOT,
    )
    verified, tag_object_sha = _load_failure_evidence(
        remote=remote,
        tag=tag,
        expected_failure_sha256=failure_sha256,
    )
    return {
        "certificate": verified,
        "tag_object_sha": tag_object_sha,
    }


def _load_failure_evidence(
    *,
    remote: str,
    tag: str,
    expected_failure_sha256: str,
) -> tuple[dict[str, Any], str]:
    certificate, blob, tag_object_sha = _load_registry_certificate(
        remote=remote, tag=tag
    )
    raw = _git_command(
        "cat-file",
        "-p",
        blob,
        cwd=PROJECT_ROOT,
        text=False,
    ).stdout
    if (
        certificate.get("certificate_schema_version") != 1
        or certificate.get("certificate_type")
        != "immutable_failure_evidence"
        or certificate.get("failure_sha256") != expected_failure_sha256
        or certificate.get("failure_sha256")
        != hashlib.sha256(raw).hexdigest()
        or certificate.get("git_blob") != blob
    ):
        raise RuntimeError("invalid V2-L0 immutable failure evidence")
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid V2-L0 failure evidence JSON") from exc
    return certificate, tag_object_sha


def publish_v2_l0_v5_failure_evidence() -> dict[str, Any]:
    protocol, _ = _load_protocol(PROTOCOL_V6)
    provenance = capture_provenance(PROJECT_ROOT)
    if provenance["git_dirty"] or not provenance["git_commit"]:
        raise RuntimeError("failure evidence publication requires a clean commit")
    amendment = protocol["amendment_record"]
    failure_path = PROJECT_ROOT / amendment["consumed_v5_failure_path"]
    reservation_path = PROJECT_ROOT / amendment[
        "consumed_v5_reservation_path"
    ]
    failure_raw = failure_path.read_bytes()
    reservation_raw = reservation_path.read_bytes()
    if (
        hashlib.sha256(failure_raw).hexdigest()
        != amendment["consumed_v5_failure_sha256"]
        or hashlib.sha256(reservation_raw).hexdigest()
        != amendment["consumed_v5_reservation_sha256"]
    ):
        raise RuntimeError("V5 consumed failure evidence changed")
    failure = json.loads(failure_raw)
    reservation = json.loads(reservation_raw)
    if (
        failure.get("outcome") != "consumed_failed_before_result"
        or failure.get("retry_allowed") is not False
        or reservation.get("status") != "consumed_before_first_episode"
        or failure.get("protocol_sha256")
        != KNOWN_PROTOCOL_DIGESTS[5]
        or reservation.get("protocol_sha256")
        != KNOWN_PROTOCOL_DIGESTS[5]
    ):
        raise RuntimeError("V5 consumed failure evidence is invalid")
    return _publish_failure_evidence(
        failure_path,
        remote="origin",
        tag="calmodel-l0-v5-holdout-failure-evidence",
        certificate={
            "split": "holdout",
            "protocol_sha256": KNOWN_PROTOCOL_DIGESTS[5],
            "git_commit": failure["git_commit"],
            "source_sha256": failure["source_sha256"],
            "reservation_sha256": amendment[
                "consumed_v5_reservation_sha256"
            ],
            "archive_commit": "7b5042f9af1dd60e52017fb2eca213c1f0d604d0",
            "status": "consumed_failure_archived_no_retry",
            "publication_commit": provenance["git_commit"],
        },
    )


def _load_registry_certificate(
    *,
    remote: str,
    tag: str,
) -> tuple[dict[str, Any], str, str]:
    if not _remote_tag_exists(remote, tag, cwd=PROJECT_ROOT):
        raise RuntimeError(f"required V2-L0 registry tag is absent: {tag}")
    _git_command(
        "fetch",
        "--force",
        remote,
        f"refs/tags/{tag}:refs/tags/{tag}",
        cwd=PROJECT_ROOT,
    )
    contents = _git_command(
        "for-each-ref",
        "--format=%(contents)",
        f"refs/tags/{tag}",
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    try:
        certificate = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid V2-L0 registry tag: {tag}") from exc
    target = _git_command(
        "rev-parse",
        f"refs/tags/{tag}^{{}}",
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    tag_object_sha = _git_command(
        "rev-parse",
        f"refs/tags/{tag}",
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    return certificate, target, tag_object_sha


def _require_historical_v5_failure_evidence(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = protocol["historical_v5_failure_evidence"]
    certificate, tag_object_sha = _load_failure_evidence(
        remote=expected["remote"],
        tag=expected["tag"],
        expected_failure_sha256=expected["failure_sha256"],
    )
    if (
        tag_object_sha != expected["tag_object_sha"]
        or certificate != expected["certificate"]
    ):
        raise RuntimeError("V2-L0 historical V5 failure evidence mismatch")
    return {
        "certificate": certificate,
        "tag_object_sha": tag_object_sha,
    }


def _certificate_matches_exact(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    nonce = actual.get("publication_nonce")
    return (
        isinstance(nonce, str)
        and len(nonce) == 32
        and {key: value for key, value in actual.items()
             if key != "publication_nonce"}
        == dict(expected)
    )


def _expected_source_lock_certificate(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    version = int(protocol["protocol_version"])
    amendment = protocol["amendment_record"]
    prior = (
        protocol["amendment_record_v7"]
        if version == 8
        else amendment
    )
    certificate = {
        "certificate_schema_version": 1,
        "certificate_type": "l0_exact_source_lock",
        "protocol_sha256": protocol_digest,
        "source_sha256": protocol["exact_source_sha256"],
        "git_commit": provenance["git_commit"],
        "implementation_commit": amendment["implementation_commit"],
        "development_result_sha256": prior[
            "prior_development_result_sha256"
        ],
        "status": protocol["status"],
    }
    if version >= 7:
        certificate.update(
            {
                "review_record_sha256": (
                    amendment["post_fix_review_record_sha256"]
                    if version == 8
                    else amendment["prior_review_record_sha256"]
                ),
                "historical_v5_failure_evidence_tag_object_sha": protocol[
                    "historical_v5_failure_evidence"
                ]["tag_object_sha"],
            }
        )
    if version == 8:
        certificate["superseded_v7_source_lock_tag_object_sha"] = (
            protocol["superseded_v7_source_lock"]["tag_object_sha"]
        )
    return certificate


def _require_source_lock_registry(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    require_authorization: bool,
) -> dict[str, Any]:
    provenance = capture_provenance(PROJECT_ROOT)
    if provenance["git_dirty"]:
        raise RuntimeError("V2-L0 source lock requires a clean worktree")
    registry = protocol["shared_git_registry"]
    source_certificate, source_target, source_tag_object_sha = (
        _load_registry_certificate(
            remote=registry["remote"],
            tag=registry["source_lock_tag"],
        )
    )
    expected_source = _expected_source_lock_certificate(
        protocol, protocol_digest, provenance
    )
    if (
        not _certificate_matches_exact(source_certificate, expected_source)
        or source_target != provenance["git_commit"]
    ):
        raise RuntimeError("V2-L0 source-lock certificate mismatch")
    result = {
        "source_lock": source_certificate,
        "source_lock_tag_object_sha": source_tag_object_sha,
    }
    if int(protocol["protocol_version"]) in {7, 8}:
        result["historical_v5_failure_evidence"] = (
            _require_historical_v5_failure_evidence(protocol)
        )
    if require_authorization:
        (
            authorization_certificate,
            authorization_target,
            authorization_tag_object_sha,
        ) = (
            _load_registry_certificate(
                remote=registry["remote"],
                tag=registry["holdout_authorization_tag"],
            )
        )
        expected_authorization = {
            "certificate_schema_version": 1,
            "certificate_type": "explicit_l0_holdout_authorization",
            "protocol_sha256": protocol_digest,
            "source_sha256": protocol["exact_source_sha256"],
            "git_commit": provenance["git_commit"],
            "source_lock_certificate": source_certificate,
            "status": "authorized_for_exactly_one_holdout_attempt",
        }
        if int(protocol["protocol_version"]) == 8:
            expected_authorization["source_lock_tag_object_sha"] = (
                source_tag_object_sha
            )
        if (
            not _certificate_matches_exact(
                authorization_certificate, expected_authorization
            )
            or authorization_target != provenance["git_commit"]
        ):
            raise RuntimeError("V2-L0 holdout authorization mismatch")
        result["holdout_authorization"] = authorization_certificate
        result["holdout_authorization_tag_object_sha"] = (
            authorization_tag_object_sha
        )
    return result


def publish_v2_l0_source_lock(
    *,
    protocol_path: Path = PROTOCOL_V8,
) -> dict[str, Any]:
    protocol, protocol_digest = _load_protocol(protocol_path)
    if protocol["protocol_version"] not in {5, 7, 8}:
        raise RuntimeError(
            "source lock publication requires V2-L0 V5, V7, or V8"
        )
    provenance = capture_provenance(PROJECT_ROOT)
    if provenance["git_dirty"] or not provenance["git_commit"]:
        raise RuntimeError("source lock publication requires a clean commit")
    registry = protocol["shared_git_registry"]
    certificate = _expected_source_lock_certificate(
        protocol, protocol_digest, provenance
    )
    if int(protocol["protocol_version"]) in {7, 8}:
        _require_historical_v5_failure_evidence(protocol)
    _publish_annotated_tag(
        remote=registry["remote"],
        tag=registry["source_lock_tag"],
        target=provenance["git_commit"],
        certificate=certificate,
        cwd=PROJECT_ROOT,
    )
    _require_source_lock_registry(
        protocol, protocol_digest, require_authorization=False
    )
    return certificate


def publish_v2_l0_holdout_authorization(
    *,
    protocol_path: Path = PROTOCOL_V8,
) -> dict[str, Any]:
    """Publish only after the user explicitly authorizes one-shot holdout."""

    protocol, protocol_digest = _load_protocol(protocol_path)
    registry_evidence = _require_source_lock_registry(
        protocol, protocol_digest, require_authorization=False
    )
    provenance = capture_provenance(PROJECT_ROOT)
    registry = protocol["shared_git_registry"]
    certificate = {
        "certificate_schema_version": 1,
        "certificate_type": "explicit_l0_holdout_authorization",
        "protocol_sha256": protocol_digest,
        "source_sha256": protocol["exact_source_sha256"],
        "git_commit": provenance["git_commit"],
        "source_lock_certificate": registry_evidence["source_lock"],
        "status": "authorized_for_exactly_one_holdout_attempt",
    }
    if int(protocol["protocol_version"]) == 8:
        certificate["source_lock_tag_object_sha"] = registry_evidence[
            "source_lock_tag_object_sha"
        ]
    _publish_annotated_tag(
        remote=registry["remote"],
        tag=registry["holdout_authorization_tag"],
        target=provenance["git_commit"],
        certificate=certificate,
        cwd=PROJECT_ROOT,
    )
    _require_source_lock_registry(
        protocol, protocol_digest, require_authorization=True
    )
    return certificate


def _run_v2_l0_language_readout(
    *,
    split: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_path: Path | None = None,
    attempt_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol, protocol_digest = _load_protocol(protocol_path)
    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    if (
        split == "holdout"
        and (
            protocol["protocol_version"] not in {5, 7, 8}
            or protocol["status"]
            not in {
                "source_locked_awaiting_explicit_holdout_authorization",
                "source_locked_awaiting_explicit_holdout_authorization_v7",
                "source_locked_awaiting_explicit_holdout_authorization_v8",
            }
        )
    ):
        raise RuntimeError(
            "V2-L0 holdout requires a reviewed source-locked protocol amendment"
        )
    expected_destination = (
        PROJECT_ROOT / protocol["result_paths"][split]
    ).resolve()
    destination = (
        output_path.resolve()
        if output_path is not None
        else expected_destination
    )
    if destination != expected_destination:
        raise RuntimeError("output path does not match the frozen protocol")
    if split == "holdout" and destination.exists():
        raise RuntimeError("one-shot V2-L0 holdout result already exists")
    run_start: dict[str, Any] | None = None
    registry_evidence: dict[str, Any] | None = None
    if split == "holdout":
        reservation = (
            PROJECT_ROOT / protocol["result_paths"]["holdout_reservation"]
        )
        if reservation.exists():
            raise RuntimeError("one-shot V2-L0 holdout is already reserved")
        registry = protocol["shared_git_registry"]
        if _remote_tag_exists(
            registry["remote"],
            registry["holdout_evidence_tag"],
            cwd=PROJECT_ROOT,
        ):
            raise RuntimeError("V2-L0 holdout evidence already exists")
        if _remote_tag_exists(
            registry["remote"],
            registry["holdout_failure_evidence_tag"],
            cwd=PROJECT_ROOT,
        ):
            raise RuntimeError(
                "V2-L0 holdout failure evidence already exists"
            )
        if _remote_tag_exists(
            registry["remote"],
            registry["holdout_consumption_tag"],
            cwd=PROJECT_ROOT,
        ):
            raise RuntimeError("V2-L0 holdout is already consumed")
        registry_evidence = _require_source_lock_registry(
            protocol, protocol_digest, require_authorization=True
        )
        run_start = capture_provenance(PROJECT_ROOT)
        if (
            run_start["git_dirty"]
            or run_start["git_commit"]
            != registry_evidence["source_lock"]["git_commit"]
            or run_start["source_sha256"]
            != protocol["exact_source_sha256"]
        ):
            raise RuntimeError(
                "V2-L0 source changed between lock verification and consumption"
            )
        if int(protocol["protocol_version"]) == 8 and attempt_state is None:
            raise RuntimeError("V2-L0 V8 holdout attempt state is absent")
        consumption = _reserve_shared_one_shot(
            remote=registry["remote"],
            tag=registry["holdout_consumption_tag"],
            split="holdout",
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
            source_sha256=run_start["source_sha256"],
            attempt_id=(
                attempt_state["attempt_id"]
                if int(protocol["protocol_version"]) == 8
                else None
            ),
            cwd=PROJECT_ROOT,
        )
        if attempt_state is not None:
            attempt_state.update(
                {
                    "consumption_acquired": True,
                    "consumption_tag_object_sha": consumption[
                        "tag_object_sha"
                    ],
                    "phase": "consumed_before_local_reservation",
                }
            )
        _reserve_one_shot(
            reservation,
            split="holdout",
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
            attempt_id=(
                attempt_state["attempt_id"]
                if int(protocol["protocol_version"]) == 8
                else None
            ),
            consumption_tag_object_sha=(
                consumption["tag_object_sha"]
                if int(protocol["protocol_version"]) == 8
                else None
            ),
        )
        if attempt_state is not None:
            attempt_state["phase"] = "collecting"

    fixed = protocol["fixed_execution"]
    train_seeds = tuple(
        int(seed)
        for seed in protocol["splits"]["development_train"]["seeds"]
    )
    evaluation_key = (
        "development_validation"
        if split == "development"
        else "review_holdout"
    )
    evaluation_seeds = tuple(
        int(seed)
        for seed in protocol["splits"][evaluation_key]["seeds"]
    )
    collection_options = {
        "steps": int(fixed["steps_per_seed"]),
        "warmup": int(fixed["warmup_steps"]),
        "reappearance_window": int(fixed["reappearance_window_steps"]),
    }
    train_audit: dict[str, Any] = {}
    evaluation_audit: dict[str, Any] = {}
    train = collect_language_data(
        train_seeds, audit_log=train_audit, **collection_options
    )
    evaluation = collect_language_data(
        evaluation_seeds,
        audit_log=evaluation_audit,
        **collection_options,
    )
    no_action_train = collect_language_data(
        train_seeds,
        use_action=False,
        **collection_options,
    )
    no_action_evaluation = collect_language_data(
        evaluation_seeds,
        use_action=False,
        **collection_options,
    )
    all_visible_train = collect_language_data(
        train_seeds,
        infer_occlusion=False,
        **collection_options,
    )
    all_visible_evaluation = collect_language_data(
        evaluation_seeds,
        infer_occlusion=False,
        **collection_options,
    )
    config = _readout_config(protocol)

    models: dict[str, ReadoutModel] = {}
    conditions: dict[str, dict[str, Any]] = {}
    models["formal_entity_graph"], conditions["formal_entity_graph"] = (
        _fit_and_evaluate(
            train,
            evaluation,
            representation="formal_entity_graph",
            config=config,
        )
    )
    models["raw_sensor"], conditions["raw_sensor"] = _fit_and_evaluate(
        train,
        evaluation,
        representation="raw_sensor",
        config=config,
    )
    shuffled_train = time_shuffle_data(
        train,
        lag=int(fixed["time_shuffle_lag"]),
    )
    shuffled_evaluation = time_shuffle_data(
        evaluation,
        lag=int(fixed["time_shuffle_lag"]),
    )
    models["time_shuffled"], conditions["time_shuffled"] = (
        _fit_and_evaluate(
            shuffled_train,
            shuffled_evaluation,
            representation="formal_entity_graph",
            config=config,
        )
    )
    models["referent_swapped"], conditions["referent_swapped"] = (
        _fit_and_evaluate(
            referent_swap_data(train),
            evaluation,
            representation="formal_entity_graph",
            config=config,
        )
    )
    models["random_labels"], conditions["random_labels"] = (
        _fit_and_evaluate(
            random_label_data(
                train,
                seed=int(fixed["random_label_seed"]),
            ),
            evaluation,
            representation="formal_entity_graph",
            config=config,
        )
    )
    (
        models["no_action_entity_graph"],
        conditions["no_action_entity_graph"],
    ) = _fit_and_evaluate(
        no_action_train,
        no_action_evaluation,
        representation="formal_entity_graph",
        config=config,
    )
    (
        models["assume_all_visible_entity_graph"],
        conditions["assume_all_visible_entity_graph"],
    ) = _fit_and_evaluate(
        all_visible_train,
        all_visible_evaluation,
        representation="formal_entity_graph",
        config=config,
    )
    scrambled_evaluation = identity_scramble_data(evaluation)
    models["identity_scrambled_at_occlusion"] = formal_model = models[
        "formal_entity_graph"
    ]
    conditions["identity_scrambled_at_occlusion"] = evaluate_readout(
        formal_model,
        scrambled_evaluation,
        representation="formal_entity_graph",
    )

    formal_model = models["formal_entity_graph"]
    formal_per_seed = {
        str(seed): evaluate_readout(
            formal_model,
            subset_data(
                evaluation,
                torch.nonzero(
                    evaluation.episode_ids == episode_id,
                    as_tuple=False,
                ).flatten(),
            ),
            representation="formal_entity_graph",
        )
        for episode_id, seed in enumerate(evaluation_seeds)
    }
    numeric_id_audit = _numeric_id_permutation_audit(
        formal_model,
        seeds=train_seeds[:3],
        steps=int(fixed["steps_per_seed"]),
    )
    frozen_source_matches = all(
        _sha256(PROJECT_ROOT / relative) == expected
        for relative, expected in protocol["frozen_i1_sources"].items()
    )
    runtime_audits = {
        "learner_input_boundary": {
            "train": train_audit,
            "evaluation": evaluation_audit,
            **evaluation_audit,
        },
        "numeric_id_permutation": numeric_id_audit,
        "identity_scramble_integrity": identity_scramble_audit(
            evaluation, scrambled_evaluation
        ),
        "frozen_i1_source_hashes_match": frozen_source_matches,
    }
    parameter_count = sum(
        parameter.numel() for parameter in formal_model.probe.parameters()
    )
    gates = _gates(
        conditions,
        parameter_count=parameter_count,
        protocol=protocol,
        formal_per_seed=formal_per_seed,
        runtime_audits=runtime_audits,
        split=split,
    )
    example_logits = readout_logits(
        formal_model,
        evaluation,
        representation="formal_entity_graph",
    )[0]
    example_language = render_language(
        torch.sigmoid(example_logits).tolist(),
        protocol=protocol,
        template_split=(
            "validation" if split == "development" else "holdout"
        ),
    )
    result = {
        "result_schema_version": 2,
        "experiment": "V2-L0-frozen-entity-language-readout",
        "split": split,
        "protocol_path": str(protocol_path.resolve().relative_to(PROJECT_ROOT)),
        "protocol_sha256": protocol_digest,
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "proposition_names": list(PROPOSITION_NAMES),
        "graph_map_names": list(GRAPH_MAP_NAMES),
        "feature_dimensions": {
            "formal_entity_graph": int(train.graph_features.shape[1]),
            "raw_sensor": int(train.raw_features.shape[1]),
        },
        "sample_counts": {
            "train": int(train.labels.shape[0]),
            "evaluation": int(evaluation.labels.shape[0]),
        },
        "conditions": conditions,
        "formal_per_seed": formal_per_seed,
        "runtime_audits": runtime_audits,
        "source_lock_registry_evidence": registry_evidence,
        "run_start": (
            None
            if run_start is None
            else {
                "git_commit": run_start["git_commit"],
                "source_sha256": run_start["source_sha256"],
                "git_dirty": run_start["git_dirty"],
            }
        ),
        "probe": {
            "type": "single_affine_multilabel_readout",
            "parameter_count": parameter_count,
            "training_final_loss": formal_model.final_loss,
            "i1_receives_gradients": False,
        },
        "example_language": example_language,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "authorize_review_and_source_lock"
            if split == "development" and all(gates.values())
            else (
                "v2_l0_language_readability_verified"
                if split == "holdout" and all(gates.values())
                else "stop_and_report"
            )
        ),
        "provenance": capture_provenance(PROJECT_ROOT),
    }
    if split == "holdout":
        run_end = capture_provenance(PROJECT_ROOT)
        if (
            run_start is None
            or run_end["git_commit"] != run_start["git_commit"]
            or run_end["source_sha256"] != run_start["source_sha256"]
        ):
            raise RuntimeError("V2-L0 source changed during holdout")
        result["run_end"] = {
            "git_commit": run_end["git_commit"],
            "source_sha256": run_end["source_sha256"],
            "git_dirty": run_end["git_dirty"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_raw = (
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    destination.write_bytes(result_raw)
    if attempt_state is not None:
        attempt_state.update(
            {
                "phase": "result_written_pending_evidence",
                "result_created": True,
                "result_sha256": hashlib.sha256(result_raw).hexdigest(),
            }
        )
    if split == "holdout":
        registry = protocol["shared_git_registry"]
        if _remote_tag_exists(
            registry["remote"],
            registry["holdout_failure_evidence_tag"],
            cwd=PROJECT_ROOT,
        ):
            raise RuntimeError(
                "V2-L0 failure evidence exists before result publication"
            )
        publication = _publish_result_evidence(
            destination,
            remote=registry["remote"],
            tag=registry["holdout_evidence_tag"],
            split="holdout",
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
            source_sha256=run_start["source_sha256"],
            extra_certificate=(
                {
                    "attempt_id": attempt_state["attempt_id"],
                    "consumption_tag_object_sha": attempt_state[
                        "consumption_tag_object_sha"
                    ],
                    "source_lock_tag_object_sha": registry_evidence[
                        "source_lock_tag_object_sha"
                    ],
                    "holdout_authorization_tag_object_sha": (
                        registry_evidence[
                            "holdout_authorization_tag_object_sha"
                        ]
                    ),
                }
                if int(protocol["protocol_version"]) == 8
                else None
            ),
            cwd=PROJECT_ROOT,
        )
        if attempt_state is not None:
            attempt_state.update(
                {
                    "phase": "result_evidence_published",
                    "result_evidence_tag_object_sha": publication[
                        "tag_object_sha"
                    ],
                }
            )
    return result


def _record_v8_holdout_failure_if_owned(
    *,
    protocol_path: Path,
    error: Exception,
    attempt_state: Mapping[str, Any],
) -> None:
    document, protocol_digest = _read_protocol_document(protocol_path)
    if int(document["protocol_version"]) != 8:
        return
    if attempt_state.get("consumption_acquired") is not True:
        return
    registry = document["shared_git_registry"]
    consumption_tag = registry["holdout_consumption_tag"]
    reservation_path = (
        PROJECT_ROOT / document["result_paths"]["holdout_reservation"]
    )
    if not _remote_tag_exists(
        registry["remote"], consumption_tag, cwd=PROJECT_ROOT
    ):
        raise RuntimeError("owned V8 consumption tag disappeared from origin")
    consumption, consumption_target, consumption_tag_object_sha = (
        _load_registry_certificate(
            remote=registry["remote"],
            tag=consumption_tag,
        )
    )
    if (
        consumption.get("certificate_schema_version") != 1
        or consumption.get("certificate_type") != "one_shot_consumption"
        or consumption.get("split") != "holdout"
        or consumption.get("protocol_sha256") != protocol_digest
        or consumption.get("status") != "consumed_before_first_episode"
        or consumption.get("git_commit") != consumption_target
        or consumption.get("attempt_id")
        != attempt_state.get("attempt_id")
        or consumption_tag_object_sha
        != attempt_state.get("consumption_tag_object_sha")
    ):
        raise RuntimeError(
            "V8 failure recorder does not own origin consumption"
        )
    reservation_present = reservation_path.exists()
    if reservation_present:
        reservation = json.loads(reservation_path.read_bytes())
        if (
            reservation.get("split") != "holdout"
            or reservation.get("protocol_sha256") != protocol_digest
            or reservation.get("git_commit") != consumption_target
            or reservation.get("status") != "consumed_before_first_episode"
            or reservation.get("attempt_id")
            != attempt_state.get("attempt_id")
            or reservation.get("consumption_tag_object_sha")
            != consumption_tag_object_sha
        ):
            raise RuntimeError("invalid V8 local reservation during failure")
    result_path = PROJECT_ROOT / document["result_paths"]["holdout"]
    result_evidence_tag = registry["holdout_evidence_tag"]
    if _remote_tag_exists(
        registry["remote"], result_evidence_tag, cwd=PROJECT_ROOT
    ):
        result_payload, result_certificate = _load_result_evidence(
            remote=registry["remote"],
            tag=result_evidence_tag,
            split="holdout",
            protocol_digest=protocol_digest,
            cwd=PROJECT_ROOT,
        )
        if (
            result_certificate.get("attempt_id")
            != attempt_state.get("attempt_id")
            or result_certificate.get("consumption_tag_object_sha")
            != consumption_tag_object_sha
            or result_payload.get("protocol_sha256") != protocol_digest
        ):
            raise RuntimeError("invalid V8 result evidence during recovery")
        return
    failure_path = PROJECT_ROOT / document["result_paths"]["holdout_failure"]
    result_created = result_path.exists()
    result_sha256 = (
        hashlib.sha256(result_path.read_bytes()).hexdigest()
        if result_created
        else None
    )
    expected_state_sha = attempt_state.get("result_sha256")
    if (
        expected_state_sha is not None
        and result_sha256 != expected_state_sha
    ):
        raise RuntimeError("V8 result changed before failure archival")
    outcome = (
        "consumed_failed_after_result_write_before_evidence"
        if result_created
        else "consumed_failed_before_result"
    )
    if failure_path.exists():
        failure = json.loads(failure_path.read_bytes())
        if (
            failure.get("protocol_sha256") != protocol_digest
            or failure.get("attempt_id")
            != attempt_state.get("attempt_id")
            or failure.get("outcome") != outcome
            or failure.get("result_created") is not result_created
            or failure.get("result_sha256") != result_sha256
        ):
            raise RuntimeError("invalid existing V8 holdout failure record")
    else:
        run_end = capture_provenance(PROJECT_ROOT)
        failure = {
            "attempt": 1,
            "attempt_id": attempt_state["attempt_id"],
            "consumption_tag": consumption_tag,
            "consumption_tag_object_sha": consumption_tag_object_sha,
            "error": {
                "message": str(error),
                "type": type(error).__name__,
            },
            "failure_evidence_tag": registry[
                "holdout_failure_evidence_tag"
            ],
            "git_commit": consumption_target,
            "local_reservation_present": reservation_present,
            "outcome": outcome,
            "phase": attempt_state.get("phase"),
            "protocol_sha256": protocol_digest,
            "result_created": result_created,
            "result_sha256": result_sha256,
            "retry_allowed": False,
            "run_end": {
                "git_commit": run_end["git_commit"],
                "git_dirty": run_end["git_dirty"],
                "source_sha256": run_end["source_sha256"],
            },
            "source_sha256": consumption["source_sha256"],
            "split": "holdout",
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    evidence_tag = registry["holdout_failure_evidence_tag"]
    if _remote_tag_exists(
        registry["remote"], result_evidence_tag, cwd=PROJECT_ROOT
    ):
        raise RuntimeError(
            "V8 result evidence appeared before failure publication"
        )
    if _remote_tag_exists(
        registry["remote"], evidence_tag, cwd=PROJECT_ROOT
    ):
        certificate, _ = _load_failure_evidence(
            remote=registry["remote"],
            tag=evidence_tag,
            expected_failure_sha256=hashlib.sha256(
                failure_path.read_bytes()
            ).hexdigest(),
        )
        expected_fields = {
            "split": "holdout",
            "protocol_sha256": protocol_digest,
            "git_commit": consumption_target,
            "source_sha256": consumption["source_sha256"],
            "attempt_id": attempt_state["attempt_id"],
            "consumption_tag_object_sha": consumption_tag_object_sha,
            "outcome": outcome,
            "result_sha256": result_sha256,
            "status": "consumed_failure_archived_no_retry",
        }
        if any(
            certificate.get(key) != value
            for key, value in expected_fields.items()
        ):
            raise RuntimeError("invalid existing V8 failure certificate")
        return
    _publish_failure_evidence(
        failure_path,
        remote=registry["remote"],
        tag=evidence_tag,
        certificate={
            "split": "holdout",
            "protocol_sha256": protocol_digest,
            "git_commit": consumption_target,
            "source_sha256": consumption["source_sha256"],
            "attempt_id": attempt_state["attempt_id"],
            "consumption_tag_object_sha": consumption_tag_object_sha,
            "outcome": outcome,
            "result_sha256": result_sha256,
            "status": "consumed_failure_archived_no_retry",
        },
    )


def run_v2_l0_language_readout(
    *,
    split: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_path: Path | None = None,
) -> dict[str, Any]:
    attempt_state = {
        "attempt_id": uuid4().hex,
        "consumption_acquired": False,
        "phase": "preflight",
        "result_created": False,
    }
    try:
        return _run_v2_l0_language_readout(
            split=split,
            protocol_path=protocol_path,
            output_path=output_path,
            attempt_state=attempt_state if split == "holdout" else None,
        )
    except Exception as error:
        if split == "holdout":
            try:
                _record_v8_holdout_failure_if_owned(
                    protocol_path=protocol_path,
                    error=error,
                    attempt_state=attempt_state,
                )
            except Exception as evidence_error:
                raise RuntimeError(
                    f"{error}; failure evidence publication also failed: "
                    f"{evidence_error}"
                ) from error
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--publish-source-lock", action="store_true")
    actions.add_argument(
        "--publish-holdout-authorization", action="store_true"
    )
    actions.add_argument(
        "--publish-v5-failure-evidence", action="store_true"
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.publish_source_lock:
            certificate = publish_v2_l0_source_lock(
                protocol_path=arguments.protocol
            )
            print(
                "source_lock_published="
                f"{certificate['git_commit']}"
            )
            return 0
        if arguments.publish_holdout_authorization:
            certificate = publish_v2_l0_holdout_authorization(
                protocol_path=arguments.protocol
            )
            print(
                "holdout_authorized="
                f"{certificate['git_commit']}"
            )
            return 0
        if arguments.publish_v5_failure_evidence:
            evidence = publish_v2_l0_v5_failure_evidence()
            print(
                "v5_failure_evidence_published="
                f"{evidence['tag_object_sha']}"
            )
            return 0
        if arguments.split is None:
            parser.error(
                "--split is required unless a publication action is used"
            )
        result = run_v2_l0_language_readout(
            split=arguments.split,
            protocol_path=arguments.protocol,
            output_path=arguments.output,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

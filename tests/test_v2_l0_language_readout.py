from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cal.evaluation import v2_l0_language_readout as language
from cal.evaluation.v2_i1_integration import _IntegratedWorld
from cal.model.entity_belief_graph import IntegratedBeliefAgentV2


def _small_data() -> language.CollectedLanguageData:
    return language.collect_language_data(
        (33_000,),
        steps=48,
        warmup=4,
        reappearance_window=5,
    )


def test_protocol_freezes_i1_sources_and_language_boundary() -> None:
    protocol, digest = language._load_protocol()

    assert len(digest) == 64
    assert protocol["learner_boundary"]["language_gradients_reach_i1"] is False
    assert protocol["learner_boundary"]["evaluation_truth_reaches_i1"] is False
    assert protocol["semantic_schema"]["propositions"] == list(
        language.PROPOSITION_NAMES
    )


def test_collection_is_detached_id_invariant_and_has_expected_shapes() -> None:
    data = _small_data()

    assert data.graph_base_feature_count == 15 * 11 * 11 + 5
    assert data.raw_base_feature_count == 11 * 11 + 5
    assert data.graph_features.shape == (
        45,
        data.graph_base_feature_count + 6 * (11 * 11 + 4 * 15),
    )
    assert data.raw_features.shape == (
        45,
        data.raw_base_feature_count + 6 * (11 * 11 + 4),
    )
    assert data.labels.shape == (45, 10)
    assert data.training_mask.shape == data.labels.shape
    assert set(data.group_masks) == set(language.GROUP_NAMES)
    assert not data.graph_features.requires_grad
    assert not data.labels.requires_grad
    assert torch.isfinite(data.graph_features).all()
    self_labels = data.labels[:, :2][data.group_masks["self"][:, :2]]
    assert bool((self_labels == 0).any())
    assert bool((self_labels == 1).any())


def test_graph_features_do_not_encode_numeric_entity_ids() -> None:
    world = _IntegratedWorld(33_000)
    agent = IntegratedBeliefAgentV2(seed=73_000)
    action_rng = np.random.default_rng(83_000)
    sensed, _ = world.observe()
    agent.update(sensed, 0)
    action = 0
    for _ in range(20):
        action = int(action_rng.integers(0, 5))
        sensed, _ = world.step(action)
        agent.update(sensed, action)

    original = language._graph_features(agent, action, steps=200)
    renamed = copy.deepcopy(agent)
    renamed_objects: set[int] = set()
    for hypothesis in renamed.graph._hypotheses:
        for entity in hypothesis.entities:
            if id(entity) in renamed_objects:
                continue
            renamed_objects.add(id(entity))
            entity.index += 10_000
    shifted = language._graph_features(renamed, action, steps=200)

    assert np.array_equal(original, shifted)


def test_i1_update_interface_receives_no_language_or_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    class SpyAgent(IntegratedBeliefAgentV2):
        def update(self, sensed_occupancy: np.ndarray, action: int) -> None:
            calls.append((sensed_occupancy.shape, action))
            super().update(sensed_occupancy, action)

    monkeypatch.setattr(language, "IntegratedBeliefAgentV2", SpyAgent)
    language.collect_language_data(
        (33_000,),
        steps=6,
        warmup=2,
        reappearance_window=2,
    )

    assert set(inspect.signature(SpyAgent.update).parameters) == {
        "self",
        "sensed_occupancy",
        "action",
    }
    assert len(calls) == 7
    assert all(shape == (11, 11) for shape, _ in calls)
    assert all(action in range(5) for _, action in calls)


def test_time_shuffle_stays_inside_each_episode() -> None:
    data = language.collect_language_data(
        (33_000, 33_001),
        steps=8,
        warmup=2,
        reappearance_window=2,
    )
    shuffled = language.time_shuffle_data(data, lag=3)

    for episode_id in (0, 1):
        indices = torch.nonzero(
            data.episode_ids == episode_id,
            as_tuple=False,
        ).flatten()
        assert torch.equal(
            shuffled.graph_features[indices[0]],
            data.graph_features[indices[0]],
        )
        assert torch.equal(
            shuffled.graph_features[
                indices[3],
                : data.graph_base_feature_count,
            ],
            data.graph_features[
                indices[0],
                : data.graph_base_feature_count,
            ],
        )
        arena_cells = 11 * 11
        query_start = data.graph_base_feature_count
        original_query_mask = data.graph_features[
            indices[3],
            query_start : query_start + arena_cells,
        ]
        shuffled_query_mask = shuffled.graph_features[
            indices[3],
            query_start : query_start + arena_cells,
        ]
        assert torch.equal(shuffled_query_mask, original_query_mask)
        active = torch.nonzero(shuffled_query_mask > 0.5).flatten()
        if len(active) == 1:
            cell_index = int(active[0])
            expected_samples = torch.stack(
                [
                    shuffled.graph_features[
                        indices[3],
                        map_index * arena_cells + cell_index,
                    ]
                    for map_index in range(len(language.GRAPH_MAP_NAMES))
                ]
            )
            sampled_start = query_start + arena_cells
            assert torch.equal(
                shuffled.graph_features[
                    indices[3],
                    sampled_start : sampled_start
                    + len(language.GRAPH_MAP_NAMES),
                ],
                expected_samples,
            )


def test_referent_swap_and_random_labels_are_prespecified() -> None:
    data = _small_data()
    swapped = language.referent_swap_data(data)
    random_first = language.random_label_data(data, seed=34_001)
    random_second = language.random_label_data(data, seed=34_001)

    assert torch.equal(swapped.labels[:, 2], data.labels[:, 4])
    assert torch.equal(swapped.labels[:, 3], data.labels[:, 5])
    assert torch.equal(swapped.labels[:, 4], data.labels[:, 2])
    assert torch.equal(swapped.labels[:, 5], data.labels[:, 3])
    assert torch.equal(random_first.labels, random_second.labels)
    assert not torch.equal(random_first.labels, data.labels)


def test_paired_queries_require_real_sensor_occlusion_and_unique_boundaries() -> None:
    audit: dict[str, object] = {}
    data = language.collect_language_data(
        (33_100, 33_101, 33_102, 33_103),
        steps=200,
        warmup=12,
        reappearance_window=5,
        audit_log=audit,
    )
    semantics = audit["paired_query_semantics"]

    assert semantics["permanence_pair_count"] > 0
    assert semantics["permanence_positive_sensor_visible_count"] == 0
    assert semantics["permanence_min_continuous_hidden_steps"] >= 2
    assert semantics["identity_pair_count"] > 0
    assert semantics["identity_invalid_boundary_count"] == 0
    assert semantics["identity_min_continuous_hidden_steps"] >= 2
    assert bool(data.group_masks["identity"][:, 8:10].any())

    scrambled = language.identity_scramble_data(data)
    scramble_audit = language.identity_scramble_audit(data, scrambled)
    assert scramble_audit["all_references_changed"] is True
    assert scramble_audit["opposite_reference_role_fraction"] == 1.0
    assert scramble_audit["row_local_counterfactual_fraction"] == 1.0
    assert (
        scramble_audit[
            "preserved_labels_masks_candidates_and_nonidentity_features"
        ]
        is True
    )
    assert torch.equal(
        data.graph_features[:, : data.graph_base_feature_count],
        scrambled.graph_features[:, : data.graph_base_feature_count],
    )
    assert torch.equal(data.labels, scrambled.labels)
    assert torch.equal(data.training_mask, scrambled.training_mask)

    active = torch.nonzero(
        data.group_masks["identity"][:, 8:10].any(dim=1),
        as_tuple=False,
    ).flatten()
    first = int(active[0])
    same_episode_and_role = active[
        (data.episode_ids[active] == data.episode_ids[first])
        & (
            data.identity_reference_roles[active]
            == data.identity_reference_roles[first]
        )
    ]
    focused = language.subset_data(data, same_episode_and_role)
    focused_scrambled = language.identity_scramble_data(focused)
    focused_audit = language.identity_scramble_audit(
        focused, focused_scrambled
    )
    assert focused_audit["all_references_changed"] is True
    assert focused_audit["row_local_counterfactual_fraction"] == 1.0

    fake_metadata = data.identity_opposite_control_references.clone()
    fake_metadata[active] = 123.0
    fake_data = replace(
        data,
        identity_opposite_control_references=fake_metadata,
    )
    fake_scrambled = language.identity_scramble_data(fake_data)
    fake_audit = language.identity_scramble_audit(
        fake_data, fake_scrambled
    )
    assert fake_audit["row_local_counterfactual_fraction"] == 0.0
    assert (
        fake_audit[
            "counterfactual_defined_for_every_active_identity_row"
        ]
        is False
    )

    broken_interactions = scrambled.graph_features.clone()
    descriptor_count = len(language.GRAPH_MAP_NAMES)
    arena_cells = (language.ARENA_HIGH - language.ARENA_LOW + 1) ** 2
    for query_index in (4, 5):
        reference_start = (
            data.graph_base_feature_count
            + query_index * data.graph_query_block_size
            + arena_cells
            + descriptor_count
        )
        product_start = reference_start + descriptor_count
        difference_start = product_start + descriptor_count
        broken_interactions[
            active,
            product_start : difference_start + descriptor_count,
        ] = 123.0
    broken_audit = language.identity_scramble_audit(
        data,
        scrambled.with_graph_features(broken_interactions),
    )
    assert broken_audit["all_interactions_recomputed"] is False

    inactive = torch.nonzero(
        ~data.group_masks["identity"][:, 8:10].any(dim=1),
        as_tuple=False,
    ).flatten()
    broken_inactive = scrambled.graph_features.clone()
    broken_inactive[int(inactive[0]), -1] += 1.0
    inactive_audit = language.identity_scramble_audit(
        data,
        scrambled.with_graph_features(broken_inactive),
    )
    assert (
        inactive_audit[
            "preserved_labels_masks_candidates_and_nonidentity_features"
        ]
        is False
    )


def test_query_heads_are_isolated_and_identity_hides_absolute_position() -> None:
    torch.manual_seed(4)
    base_count = 5
    block_size = 11 * 11 + 4 * len(language.GRAPH_MAP_NAMES)
    probe = language.LinearLanguageReadout(
        base_count + 6 * block_size,
        base_count,
    )
    baseline = torch.zeros((1, base_count + 6 * block_size))
    original = probe(baseline)

    for query_index, expected_slice in (
        (0, slice(0, 2)),
        (2, slice(6, 8)),
        (4, slice(8, 10)),
    ):
        changed = baseline.clone()
        descriptor_offset = (
            base_count + query_index * block_size + 11 * 11
            + (12 if query_index == 0 else 0)
        )
        changed[0, descriptor_offset] = 1.0
        delta = probe(changed) - original
        outside = torch.ones(10, dtype=torch.bool)
        outside[expected_slice] = False
        assert bool((delta[0, expected_slice] != 0).any())
        assert torch.equal(delta[0, outside], torch.zeros_like(delta[0, outside]))

    position_only = baseline.clone()
    position_only[0, base_count + 4 * block_size] = 1.0
    assert torch.equal(probe(position_only)[:, 8:10], original[:, 8:10])


def test_balanced_accuracy_refuses_single_class_registered_column() -> None:
    logits = torch.zeros((8, 1))
    labels = torch.ones((8, 1))
    mask = torch.ones((8, 1), dtype=torch.bool)

    with pytest.raises(RuntimeError, match="both classes"):
        language._balanced_accuracy(logits, labels, mask)


def test_protocol_and_output_paths_cannot_be_replaced(tmp_path: Path) -> None:
    fake_protocol = tmp_path / "fake-v3.json"
    fake_protocol.write_bytes(language.PROTOCOL_V3.read_bytes())

    with pytest.raises(RuntimeError, match="not canonical"):
        language._load_protocol(fake_protocol)
    with pytest.raises(RuntimeError, match="output path"):
        language.run_v2_l0_language_readout(
            split="development",
            output_path=tmp_path / "forged-result.json",
        )
    assert not (tmp_path / "forged-result.json").exists()


def test_linear_readout_learns_detached_balanced_propositions() -> None:
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(512, 12, generator=generator)
    labels = torch.stack(
        [(features[:, index] > 0).to(torch.float32) for index in range(10)],
        dim=1,
    )
    mask = torch.ones_like(labels, dtype=torch.bool)
    episode_ids = torch.zeros(512, dtype=torch.int64)
    data = language.CollectedLanguageData(
        graph_features=features,
        raw_features=features[:, :4],
        labels=labels,
        training_mask=mask,
        group_masks={
            "self": mask.clone(),
            "spatial": mask.clone(),
            "permanence": mask.clone(),
            "identity": mask.clone(),
        },
        episode_ids=episode_ids,
        graph_base_feature_count=12,
        raw_base_feature_count=4,
        query_count=0,
    )
    model = language.train_readout(
        data,
        representation="formal_entity_graph",
        config=language.ReadoutConfig(
            epochs=30,
            batch_size=64,
            learning_rate=0.05,
            weight_decay=0.0,
            seed=1,
        ),
    )
    metrics = language.evaluate_readout(
        model,
        data,
        representation="formal_entity_graph",
    )

    assert metrics["macro_balanced_accuracy"] > 0.98
    assert sum(parameter.numel() for parameter in model.probe.parameters()) == 130


def test_controlled_language_templates_are_split_and_complete() -> None:
    protocol, _ = language._load_protocol()
    probabilities = [0.1, 0.9] * 5

    for split in ("train", "validation", "holdout"):
        rendered = language.render_language(
            probabilities,
            protocol=protocol,
            template_split=split,
        )
        assert len(rendered) == len(language.PROPOSITION_NAMES)
        assert {item["proposition"] for item in rendered} == set(
            language.PROPOSITION_NAMES
        )
        assert rendered[0]["predicted_true"] is False
        assert rendered[1]["predicted_true"] is True


def test_v1_protocol_refuses_holdout_before_review(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="reviewed source-locked"):
        language.run_v2_l0_language_readout(
            split="holdout",
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_v5_protocol_is_historical_after_locked_source_changes() -> None:
    protocol, digest = language._read_protocol_document(language.PROTOCOL_V5)

    assert len(digest) == 64
    assert protocol["protocol_version"] == 5
    assert (
        protocol["status"]
        == "source_locked_awaiting_explicit_holdout_authorization"
    )
    assert protocol["authorization"]["holdout_authorized"] is False
    assert (
        language._sha256(
            language.PROJECT_ROOT
            / "cal/evaluation/v2_l0_language_readout.py"
        )
        != protocol["exact_source_locks"][
            "cal/evaluation/v2_l0_language_readout.py"
        ]
    )
    with pytest.raises(RuntimeError, match="exact source changed"):
        language._load_protocol(language.PROTOCOL_V5)


def test_consumed_v5_holdout_cannot_collect_after_v6_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_collection(*args: object, **kwargs: object) -> object:
        calls.append("collection")
        raise AssertionError("consumed V5 holdout collection restarted")

    monkeypatch.setattr(
        language, "collect_language_data", forbidden_collection
    )

    with pytest.raises(RuntimeError, match="exact source changed"):
        language.run_v2_l0_language_readout(
            split="holdout",
            protocol_path=language.PROTOCOL_V5,
        )
    assert calls == []


def test_v6_protocol_locks_v5_failure_and_new_unopened_holdout() -> None:
    protocol, digest = language._load_protocol(language.PROTOCOL_V6)

    assert len(digest) == 64
    assert protocol["protocol_version"] == 6
    assert (
        protocol["status"]
        == "frozen_after_v5_consumed_failure_before_row_local_control_implementation"
    )
    assert protocol["authorization"]["holdout_authorized"] is False
    assert protocol["authorization"]["v5_retry_forbidden"] is True
    assert protocol["splits"]["review_holdout"]["seeds"] == [
        33600,
        33601,
        33602,
        33603,
    ]
    assert (
        protocol["control_integrity_gates"][
            "counterfactual_defined_for_every_active_identity_row"
        ]
        is True
    )
    assert language.KNOWN_PROTOCOL_DIGESTS[5] == (
        "51a4f561bceb23de2c9c483895b82e2f5b1cd4168736b22b166e236be6ce1aae"
    )
    assert language.KNOWN_PROTOCOL_DIGESTS[6] == digest


def test_v6_hashes_and_parses_each_historical_artifact_from_one_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watched = {
        language.PROTOCOL_V5.resolve(): 0,
        (
            language.PROJECT_ROOT
            / "results/V2-L0-language-readout-holdout-v5-failure.json"
        ).resolve(): 0,
        (
            language.PROJECT_ROOT
            / "results/V2-L0-language-readout-holdout-v5-reservation.json"
        ).resolve(): 0,
    }
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in watched:
            watched[resolved] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    language._load_protocol(language.PROTOCOL_V6)

    assert set(watched.values()) == {1}


def test_source_lock_publication_certificate_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, digest = language._read_protocol_document(language.PROTOCOL_V5)
    captured: dict[str, object] = {}
    provenance = {
        "git_dirty": False,
        "git_commit": "a" * 40,
        "source_sha256": protocol["exact_source_sha256"],
    }
    monkeypatch.setattr(
        language,
        "_load_protocol",
        lambda path: (protocol, digest),
    )
    monkeypatch.setattr(
        language,
        "capture_provenance",
        lambda root: provenance,
    )

    def capture_publish(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(language, "_publish_annotated_tag", capture_publish)
    monkeypatch.setattr(
        language,
        "_require_source_lock_registry",
        lambda *args, **kwargs: {"source_lock": {}},
    )
    certificate = language.publish_v2_l0_source_lock(
        protocol_path=language.PROTOCOL_V5
    )

    assert certificate["protocol_sha256"] == digest
    assert certificate["source_sha256"] == protocol["exact_source_sha256"]
    assert certificate["git_commit"] == "a" * 40
    assert captured["tag"] == protocol["shared_git_registry"][
        "source_lock_tag"
    ]


def test_v5_failure_evidence_publication_is_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        language,
        "capture_provenance",
        lambda root: {
            "git_dirty": False,
            "git_commit": "a" * 40,
        },
    )

    def capture_publish(
        path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {
            "certificate": {},
            "tag_object_sha": "b" * 40,
        }

    monkeypatch.setattr(
        language,
        "_publish_failure_evidence",
        capture_publish,
    )
    evidence = language.publish_v2_l0_v5_failure_evidence()

    assert evidence["tag_object_sha"] == "b" * 40
    assert captured["tag"] == "calmodel-l0-v5-holdout-failure-evidence"
    certificate = captured["certificate"]
    assert certificate["protocol_sha256"] == (
        language.KNOWN_PROTOCOL_DIGESTS[5]
    )
    assert certificate["status"] == "consumed_failure_archived_no_retry"


def test_v7_consumed_failure_is_recorded_and_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol_digest = "c" * 64
    document = {
        "protocol_version": 7,
        "result_paths": {
            "holdout_reservation": "results/reservation.json",
            "holdout_failure": "results/failure.json",
        },
        "shared_git_registry": {
            "remote": "origin",
            "holdout_consumption_tag": "v7-consumed",
            "holdout_failure_evidence_tag": "v7-failure",
        },
    }
    monkeypatch.setattr(language, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        language,
        "_read_protocol_document",
        lambda path: (document, protocol_digest),
    )
    monkeypatch.setattr(
        language,
        "_remote_tag_exists",
        lambda remote, tag, cwd: tag == "v7-consumed",
    )
    monkeypatch.setattr(
        language,
        "_load_registry_certificate",
        lambda **kwargs: (
            {
                "certificate_type": "one_shot_consumption",
                "protocol_sha256": protocol_digest,
                "status": "consumed_before_first_episode",
                "git_commit": "d" * 40,
                "source_sha256": "e" * 64,
            },
            "d" * 40,
        ),
    )
    monkeypatch.setattr(
        language,
        "capture_provenance",
        lambda root: {
            "git_commit": "d" * 40,
            "git_dirty": True,
            "source_sha256": "e" * 64,
        },
    )
    published: dict[str, object] = {}

    def capture_failure(
        path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        published["path"] = path
        published.update(kwargs)
        return {"certificate": {}, "tag_object_sha": "f" * 40}

    monkeypatch.setattr(
        language,
        "_publish_failure_evidence",
        capture_failure,
    )

    language._record_v7_holdout_failure_if_consumed(
        protocol_path=tmp_path / "protocol.json",
        error=RuntimeError("synthetic failure"),
    )

    failure_path = tmp_path / "results/failure.json"
    failure = json.loads(failure_path.read_text())
    assert failure["outcome"] == "consumed_failed_before_result"
    assert failure["retry_allowed"] is False
    assert failure["local_reservation_present"] is False
    assert failure["error"] == {
        "message": "synthetic failure",
        "type": "RuntimeError",
    }
    assert published["tag"] == "v7-failure"

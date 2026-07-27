"""Deterministic, presentation-only replay generator for the I1 V4 experiment.

The replay recorder observes evaluator-side truth only after the agent update.
Truth is never passed into ``IntegratedBeliefAgentV2.update``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cal.evaluation.v2_i1_integration import (
    ACTION_DELTAS,
    ARENA_HIGH,
    ARENA_LOW,
    _IntegratedWorld,
    _global_visibility,
)
from cal.evaluation.v2_i1_integration_v2 import (
    PERMANENCE_WARMUP,
    WARMUP,
    _identity_metrics,
    _load_frozen_protocol,
)
from cal.model.entity_belief_graph import (
    STATIC_THRESHOLD,
    IntegratedBeliefAgentV2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiments" / "V2_I1_INTEGRATION_PROTOCOL_V4.json"
DEFAULT_SEED = 30_000
DEFAULT_OUTPUT_PATTERN = "docs/experiments/assets/v2_i1_v4_replay_seed{seed}.html"
ACTION_NAMES = ("停留", "向左", "向右", "向上", "向下")
CONDITION_CONFIGS: dict[str, dict[str, Any]] = {
    "formal": {
        "label": "Formal · 正式系统",
        "description": "使用真实动作输入，并推断遮挡区域。",
        "infer_occlusion": True,
        "use_action": True,
        "shuffle_lag": 0,
    },
    "no_action": {
        "label": "No action · 不提供动作",
        "description": (
            "不让系统使用真实动作，检验动作证据是否必要。"
        ),
        "infer_occlusion": True,
        "use_action": False,
        "shuffle_lag": 0,
    },
    "time_shuffled": {
        "label": "Shuffled action · 错时动作",
        "description": "将动作错开 5 步，检验时序对齐是否必要。",
        "infer_occlusion": True,
        "use_action": True,
        "shuffle_lag": 5,
    },
    "assume_all_visible": {
        "label": "Assume all visible · 假设全可见",
        "description": "不推断遮挡，把未看到的区域当作已观察。",
        "infer_occlusion": False,
        "use_action": True,
        "shuffle_lag": 0,
    },
}


@dataclass(frozen=True)
class ReplayCondition:
    name: str
    label: str
    description: str
    infer_occlusion: bool
    use_action: bool
    shuffle_lag: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _condition(name: str) -> ReplayCondition:
    try:
        config = CONDITION_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(CONDITION_CONFIGS)
        raise ValueError(
            f"unknown replay condition {name!r}; choose one of: {choices}"
        ) from exc
    return ReplayCondition(name=name, **config)


def _load_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    protocol, digest = _load_frozen_protocol(path.resolve())
    return protocol, digest


def allowed_calibration_seeds(protocol: dict[str, Any]) -> tuple[int, ...]:
    """Return the only seeds that the presentation replay may consume."""

    return tuple(int(seed) for seed in protocol["calibration"]["seeds"])


def validate_replay_seed(seed: int, protocol: dict[str, Any]) -> None:
    """Reject validation, holdout, and unregistered seeds before simulation."""

    calibration = set(allowed_calibration_seeds(protocol))
    if seed in calibration:
        return

    validation = {int(value) for value in protocol["validation"]["seeds"]}
    holdout = {int(value) for value in protocol["holdout"]["seeds"]}
    if seed in validation:
        reason = "validation seed（一次性证据，禁止演示消费）"
    elif seed in holdout:
        reason = "holdout seed（一次性证据，禁止演示消费）"
    else:
        reason = "未在冻结协议中登记的 calibration seed"
    allowed = ", ".join(str(value) for value in sorted(calibration))
    raise ValueError(f"seed {seed} 是{reason}；可用 seed：{allowed}")


def _to_local(position: tuple[int, int]) -> list[int]:
    return [int(position[0] - ARENA_LOW), int(position[1] - ARENA_LOW)]


def _local_indices(mask: np.ndarray) -> list[int]:
    arena_size = ARENA_HIGH - ARENA_LOW + 1
    local = (
        mask
        if mask.shape == (arena_size, arena_size)
        else mask[ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1]
    )
    rows, columns = np.nonzero(local)
    width = ARENA_HIGH - ARENA_LOW + 1
    return [
        int(row * width + column)
        for row, column in zip(rows, columns, strict=True)
    ]


def _quantized_belief(agent: IntegratedBeliefAgentV2) -> list[int]:
    probability = agent.probability()[
        ARENA_LOW : ARENA_HIGH + 1,
        ARENA_LOW : ARENA_HIGH + 1,
    ]
    return (
        np.rint(np.clip(probability, 0.0, 1.0) * 100.0)
        .astype(np.uint8)
        .ravel()
        .tolist()
    )


def _capture_frame(
    *,
    step: int,
    action: int,
    supplied_action: int,
    world: _IntegratedWorld,
    agent: IntegratedBeliefAgentV2,
    sensed: np.ndarray,
    truth_visibility: np.ndarray,
) -> dict[str, Any]:
    positions = agent.track_positions()
    self_identity = agent.self_track_identity()
    self_posterior = agent.graph.self_posterior()
    tracks = [
        {
            "id": int(identity),
            "x": int(position[0] - ARENA_LOW),
            "y": int(position[1] - ARENA_LOW),
            "selfProbability": round(float(self_posterior.get(identity, 0.0)), 4),
            "isSelf": identity == self_identity,
        }
        for identity, position in positions.items()
    ]
    hypothesis_weights = [
        round(float(hypothesis.weight), 4)
        for hypothesis in sorted(
            agent.graph._hypotheses,
            key=lambda item: -item.weight,
        )
    ]
    return {
        "step": step,
        "action": action,
        "actionName": ACTION_NAMES[action],
        "suppliedAction": supplied_action,
        "suppliedActionName": ACTION_NAMES[supplied_action],
        "truth": {
            "self": _to_local(world.self_position),
            "a": _to_local(world.distractor_a),
            "b": _to_local(world.distractor_b),
        },
        "truthVisible": {
            "self": bool(
                truth_visibility[
                    int(world.self_position[1]),
                    int(world.self_position[0]),
                ]
            ),
            "a": bool(
                truth_visibility[
                    int(world.distractor_a[1]),
                    int(world.distractor_a[0]),
                ]
            ),
            "b": bool(
                truth_visibility[
                    int(world.distractor_b[1]),
                    int(world.distractor_b[0]),
                ]
            ),
        },
        "agentVisible": _local_indices(agent.front_end.last_visibility),
        "sensed": _local_indices(sensed),
        "truthStatic": [
            (y - ARENA_LOW) * (ARENA_HIGH - ARENA_LOW + 1) + (x - ARENA_LOW)
            for x, y in sorted(world.static)
        ],
        "learnedStatic": _local_indices(
            agent.front_end.static_score >= STATIC_THRESHOLD
        ),
        "belief": _quantized_belief(agent),
        "tracks": tracks,
        "selfId": self_identity,
        "selfPosterior": {
            str(track_id): round(float(probability), 4)
            for track_id, probability in sorted(self_posterior.items())
        },
        "hypothesisWeights": hypothesis_weights,
    }


def _unique_truth_track(frame: dict[str, Any], actor: str) -> int | None:
    position = frame["truth"][actor]
    matches = [
        int(track["id"])
        for track in frame["tracks"]
        if [track["x"], track["y"]] == position
    ]
    return matches[0] if len(matches) == 1 else None


def _detect_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{"step": 0, "type": "start", "label": "开始"}]
    previous = frames[0]
    previous_mapping = {
        actor: _unique_truth_track(previous, actor)
        for actor in ("a", "b")
    }
    previous_merged = len({tuple(value) for value in previous["truth"].values()}) < 3

    for frame in frames[1:]:
        step = int(frame["step"])
        if previous["selfId"] is None and frame["selfId"] is not None:
            events.append(
                {
                    "step": step,
                    "type": "self_acquired",
                    "label": f"第 {step} 步：识别出自己",
                }
            )
        elif previous["selfId"] is not None and frame["selfId"] is None:
            events.append(
                {
                    "step": step,
                    "type": "self_lost",
                    "label": f"第 {step} 步：自我身份暂时丢失",
                }
            )

        for actor, actor_label in (("a", "A"), ("b", "B")):
            was_visible = bool(previous["truthVisible"][actor])
            is_visible = bool(frame["truthVisible"][actor])
            if was_visible and not is_visible:
                events.append(
                    {
                        "step": step,
                        "type": f"{actor}_hidden",
                        "label": f"第 {step} 步：目标 {actor_label} 进入遮挡",
                    }
                )
            elif not was_visible and is_visible:
                events.append(
                    {
                        "step": step,
                        "type": f"{actor}_reappeared",
                        "label": f"第 {step} 步：目标 {actor_label} 重新出现",
                    }
                )

        merged = len({tuple(value) for value in frame["truth"].values()}) < 3
        if merged and not previous_merged:
            events.append(
                {
                    "step": step,
                    "type": "merge",
                    "label": f"第 {step} 步：多个目标位置重合",
                }
            )

        if not merged:
            for actor, actor_label in (("a", "A"), ("b", "B")):
                mapping = _unique_truth_track(frame, actor)
                old_mapping = previous_mapping[actor]
                if (
                    mapping is not None
                    and old_mapping is not None
                    and mapping != old_mapping
                ):
                    events.append(
                        {
                            "step": step,
                            "type": "identity_change",
                            "label": (
                                f"第 {step} 步：目标 {actor_label} "
                                "的轨迹身份改变"
                            ),
                        }
                    )
                if mapping is not None:
                    previous_mapping[actor] = mapping

        previous = frame
        previous_merged = merged

    events.append(
        {
            "step": int(frames[-1]["step"]),
            "type": "end",
            "label": f"第 {frames[-1]['step']} 步：结束",
        }
    )
    return events


def record_condition(
    *,
    seed: int,
    steps: int,
    condition_name: str,
) -> dict[str, Any]:
    """Record one condition and reproduce the formal runner's metric arithmetic."""

    condition = _condition(condition_name)
    world = _IntegratedWorld(seed)
    agent = IntegratedBeliefAgentV2(
        grid_size=world.grid_size,
        infer_occlusion=condition.infer_occlusion,
        use_action=condition.use_action,
        seed=seed + 40_000,
    )
    action_rng = np.random.default_rng(seed + 50_000)
    executed_actions = [0]
    sensed, local_visibility = world.observe()
    truth_visibility = _global_visibility(local_visibility, world.grid_size)
    agent.update(sensed, 0)
    frames = [
        _capture_frame(
            step=0,
            action=0,
            supplied_action=0,
            world=world,
            agent=agent,
            sensed=sensed,
            truth_visibility=truth_visibility,
        )
    ]

    true_positive = false_positive = false_negative = 0
    hidden_probabilities: list[float] = []
    identity_map: dict[str, dict[int, int]] = {"a": {}, "b": {}}
    identity_opportunities: dict[str, int] = {"a": 0, "b": 0}
    identity_detections: dict[str, int] = {"a": 0, "b": 0}

    for step in range(1, steps + 1):
        action = int(action_rng.integers(0, len(ACTION_DELTAS)))
        world.step(action)
        executed_actions.append(action)
        supplied_action = executed_actions[
            max(0, len(executed_actions) - 1 - condition.shuffle_lag)
        ]
        sensed, local_visibility = world.observe()
        agent.update(sensed, supplied_action)

        truth_visibility = _global_visibility(local_visibility, world.grid_size)
        positions = agent.track_positions()
        if step >= WARMUP:
            self_identity = agent.self_track_identity()
            true_self = (
                int(world.self_position[0]),
                int(world.self_position[1]),
            )
            if truth_visibility[true_self[1], true_self[0]]:
                predicted = (
                    positions.get(self_identity)
                    if self_identity is not None
                    else None
                )
                if predicted == true_self:
                    true_positive += 1
                else:
                    false_negative += 1
                    if predicted is not None:
                        false_positive += 1

        truth_cells = {
            "self": (int(world.self_position[0]), int(world.self_position[1])),
            "a": (int(world.distractor_a[0]), int(world.distractor_a[1])),
            "b": (int(world.distractor_b[0]), int(world.distractor_b[1])),
        }
        for name, point in (("a", world.distractor_a), ("b", world.distractor_b)):
            cell = (int(point[0]), int(point[1]))
            if (
                truth_visibility[cell[1], cell[0]]
                and sum(other_cell == cell for other_cell in truth_cells.values()) == 1
            ):
                identity_opportunities[name] += 1
                matched = [
                    identity
                    for identity, position in positions.items()
                    if position == cell
                ]
                if len(matched) == 1:
                    identity = matched[0]
                    identity_detections[name] += 1
                    identity_map[name][identity] = (
                        identity_map[name].get(identity, 0) + 1
                    )

        if step >= PERMANENCE_WARMUP:
            probability = agent.probability()
            for point in (world.distractor_a, world.distractor_b):
                cell = (int(point[0]), int(point[1]))
                if not truth_visibility[cell[1], cell[0]]:
                    hidden_probabilities.append(float(probability[cell[1], cell[0]]))

        frames.append(
            _capture_frame(
                step=step,
                action=action,
                supplied_action=supplied_action,
                world=world,
                agent=agent,
                sensed=sensed,
                truth_visibility=truth_visibility,
            )
        )

    denominator = 2 * true_positive + false_positive + false_negative
    identity_consistency, visible_identity_coverage = _identity_metrics(
        identity_map,
        identity_opportunities,
        identity_detections,
    )
    metrics = {
        "seed": seed,
        "self_f1": 2 * true_positive / denominator if denominator else 0.0,
        "identity_consistency": identity_consistency,
        "visible_identity_coverage": visible_identity_coverage,
        "visible_identity_opportunities": sum(identity_opportunities.values()),
        "distractor_hidden_probability": (
            statistics.mean(hidden_probabilities) if hidden_probabilities else 0.0
        ),
        "hidden_sample_count": len(hidden_probabilities),
    }
    return {
        "name": condition.name,
        "label": condition.label,
        "description": condition.description,
        "config": {
            "inferOcclusion": condition.infer_occlusion,
            "useAction": condition.use_action,
            "shuffleLag": condition.shuffle_lag,
        },
        "metrics": metrics,
        "events": _detect_events(frames),
        "frames": frames,
        "actionScheduleSha256": _sha256_bytes(_canonical_json(executed_actions)),
    }


def _source_manifest() -> dict[str, str]:
    paths = {
        "model": PROJECT_ROOT / "cal/model/entity_belief_graph.py",
        "world": PROJECT_ROOT / "cal/evaluation/v2_i1_integration.py",
        "formalRunner": PROJECT_ROOT / "cal/evaluation/v2_i1_integration_v2.py",
        "replayRecorder": Path(__file__).resolve(),
        "replayTemplate": Path(__file__).with_name("v2_i1_replay_template.py"),
    }
    return {
        name: _sha256_file(path)
        for name, path in paths.items()
    }


def build_replay_payload(
    *,
    seed: int = DEFAULT_SEED,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol, protocol_digest = _load_protocol(protocol_path)
    validate_replay_seed(seed, protocol)
    steps = int(protocol["fixed_runtime"]["steps_per_seed"])

    conditions = {
        name: record_condition(seed=seed, steps=steps, condition_name=name)
        for name in CONDITION_CONFIGS
    }
    action_hashes = {item["actionScheduleSha256"] for item in conditions.values()}
    if len(action_hashes) != 1:
        raise RuntimeError("condition action schedules diverged")

    calibration_result_path = PROJECT_ROOT / protocol["calibration"]["result_path"]
    calibration_result = json.loads(calibration_result_path.read_text(encoding="utf-8"))
    condition_data_hash = _sha256_bytes(_canonical_json(conditions))
    return {
        "schemaVersion": 1,
        "experiment": "V2-I1-unified-entity-belief-graph",
        "title": "I1 V4 Calibration Replay",
        "seed": seed,
        "steps": steps,
        "arena": {
            "low": ARENA_LOW,
            "high": ARENA_HIGH,
            "size": ARENA_HIGH - ARENA_LOW + 1,
        },
        "conditionOrder": list(CONDITION_CONFIGS),
        "conditions": conditions,
        "presentationOnly": True,
        "learnerInput": ["局部二值占据栅格", "动作编号"],
        "evaluatorTruthUsedForLearning": False,
        "protocol": {
            "path": str(protocol_path.relative_to(PROJECT_ROOT)),
            "sha256": protocol_digest,
        },
        "formalEvidence": {
            "resultPath": str(calibration_result_path.relative_to(PROJECT_ROOT)),
            "resultSha256": _sha256_file(calibration_result_path),
            "implementationCommit": calibration_result["run_start"]["git_commit"],
            "formalSourceSha256": calibration_result["run_start"]["source_sha256"],
        },
        "sourceFiles": _source_manifest(),
        "actionScheduleSha256": next(iter(action_hashes)),
        "conditionDataSha256": condition_data_hash,
    }


def render_replay(
    *,
    seed: int = DEFAULT_SEED,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> str:
    from cal.evaluation.v2_i1_replay_template import render_replay_html

    payload = build_replay_payload(
        seed=seed,
        protocol_path=protocol_path,
    )
    return render_replay_html(payload)


def default_output(seed: int) -> Path:
    return PROJECT_ROOT / DEFAULT_OUTPUT_PATTERN.format(seed=seed)


def write_replay(
    *,
    seed: int = DEFAULT_SEED,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_path: Path | None = None,
) -> tuple[Path, str]:
    destination = (output_path or default_output(seed)).resolve()
    html = render_replay(seed=seed, protocol_path=protocol_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination, _sha256_bytes(html.encode("utf-8"))


def check_replay(
    *,
    seed: int,
    replay_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[bool, str, str]:
    expected = render_replay(seed=seed, protocol_path=protocol_path).encode("utf-8")
    actual = replay_path.resolve().read_bytes()
    expected_sha = _sha256_bytes(expected)
    actual_sha = _sha256_bytes(actual)
    return expected == actual, expected_sha, actual_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cal-v2-i1-replay",
        description="生成或校验 I1 V4 calibration 交互式回放。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        type=Path,
        metavar="HTML",
        help="重新生成并逐字节校验现有 HTML，不写文件。",
    )
    parser.add_argument(
        "--list-seeds",
        action="store_true",
        help="列出冻结协议允许用于演示的 calibration seeds。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol, _ = _load_protocol(args.protocol)
        if args.list_seeds:
            print(" ".join(str(seed) for seed in allowed_calibration_seeds(protocol)))
            return 0
        validate_replay_seed(args.seed, protocol)
        if args.check:
            matches, expected_sha, actual_sha = check_replay(
                seed=args.seed,
                replay_path=args.check,
                protocol_path=args.protocol,
            )
            if matches:
                print(f"PASS {args.check.resolve()} sha256={actual_sha}")
                return 0
            print(
                f"FAIL {args.check.resolve()} "
                f"expected={expected_sha} actual={actual_sha}",
                file=sys.stderr,
            )
            return 1

        destination, digest = write_replay(
            seed=args.seed,
            protocol_path=args.protocol,
            output_path=args.output,
        )
        print(f"WROTE {destination} sha256={digest}")
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

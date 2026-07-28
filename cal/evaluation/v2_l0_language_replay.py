"""Deterministic, development-only interactive replay for the L0 readout.

The replay trains the frozen linear probes on the preregistered development
training seeds and renders one preregistered development-validation episode.
It never loads or consumes the one-shot V8 holdout split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cal.evaluation import v2_i1_replay
from cal.evaluation import v2_l0_language_readout as language


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = language.PROTOCOL_V6
DEFAULT_SEED = 33_100
DEFAULT_OUTPUT_PATTERN = (
    "docs/experiments/assets/v2_l0_language_replay_seed{seed}.html"
)
FROZEN_DEVELOPMENT_RESULT = (
    PROJECT_ROOT / "results/V2-L0-language-readout-development-v6.json"
)
REPRESENTATIONS: dict[str, dict[str, str]] = {
    "formal_entity_graph": {
        "label": "I1 实体信念图",
        "description": (
            "从冻结的 I1 实体状态读取：包含自我、运动、遮挡与身份历史。"
        ),
    },
    "raw_sensor": {
        "label": "原始局部传感器（对照）",
        "description": (
            "只从当前 11×11 二值占据和动作读取，用来判断 I1 表征是否真的有增益。"
        ),
    },
}
GROUP_LABELS = {
    "self": "自我",
    "spatial": "空间关系",
    "permanence": "遮挡后仍存在",
    "identity": "重现后身份",
}
PROPOSITION_GROUPS = (
    "self",
    "self",
    "spatial",
    "spatial",
    "spatial",
    "spatial",
    "permanence",
    "permanence",
    "identity",
    "identity",
)
PROPOSITION_QUERY_BLOCKS = (0, 1, None, None, None, None, 2, 3, 4, 5)


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


def allowed_development_seeds(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    """Return repeatable development-validation seeds only."""

    return tuple(
        int(seed)
        for seed in protocol["splits"]["development_validation"]["seeds"]
    )


def validate_replay_seed(seed: int, protocol: Mapping[str, Any]) -> None:
    """Reject holdout, training, and unknown seeds before any simulation."""

    allowed = set(allowed_development_seeds(protocol))
    if seed in allowed:
        return

    training = {
        int(value)
        for value in protocol["splits"]["development_train"]["seeds"]
    }
    holdout = {
        int(value)
        for value in protocol["splits"]["review_holdout"]["seeds"]
    }
    if seed in holdout:
        reason = "V8 一次性 holdout seed，演示程序永久拒绝访问"
    elif seed in training:
        reason = "训练 seed，不用于展示泛化结果"
    else:
        reason = "未在冻结协议中登记的 development-validation seed"
    choices = ", ".join(
        str(value) for value in sorted(allowed)
    )
    raise ValueError(f"seed {seed} 是{reason}；可用 seed：{choices}")


def _source_manifest() -> dict[str, str]:
    paths = {
        "model": PROJECT_ROOT / "cal/model/entity_belief_graph.py",
        "languageEvaluator": Path(language.__file__).resolve(),
        "i1ReplayRecorder": Path(v2_i1_replay.__file__).resolve(),
        "languageReplayRecorder": Path(__file__).resolve(),
        "languageReplayTemplate": Path(__file__).with_name(
            "v2_l0_language_replay_template.py"
        ),
    }
    return {
        name: _sha256_file(path)
        for name, path in paths.items()
    }


def _language_frames(
    *,
    data: language.CollectedLanguageData,
    logits: torch.Tensor,
    protocol: Mapping[str, Any],
    steps: int,
    warmup: int,
) -> list[dict[str, Any]]:
    expected_samples = steps - warmup + 1
    if data.labels.shape[0] != expected_samples:
        raise RuntimeError(
            "L0 sample/frame alignment changed: "
            f"expected {expected_samples}, got {data.labels.shape[0]}"
        )
    probabilities = torch.sigmoid(logits).tolist()
    templates = protocol["semantic_schema"]["templates"]["validation"]
    arena_size = language.ARENA_HIGH - language.ARENA_LOW + 1
    query_cell_count = arena_size**2
    frames: list[dict[str, Any]] = []
    for step in range(steps + 1):
        if step < warmup:
            frames.append(
                {
                    "step": step,
                    "ready": False,
                    "activeCount": 0,
                    "correctCount": 0,
                    "items": [],
                }
            )
            continue

        row = step - warmup
        labels = data.labels[row].to(torch.bool).tolist()
        active = data.training_mask[row].to(torch.bool).tolist()
        items: list[dict[str, Any]] = []
        correct_count = 0
        active_count = 0
        for index, (
            proposition,
            group,
            sentence,
            probability,
            truth,
            is_active,
            query_block,
        ) in enumerate(
            zip(
                language.PROPOSITION_NAMES,
                PROPOSITION_GROUPS,
                templates,
                probabilities[row],
                labels,
                active,
                PROPOSITION_QUERY_BLOCKS,
                strict=True,
            )
        ):
            predicted = bool(probability >= 0.5)
            correct = predicted == truth if is_active else None
            query_position: list[int] | None = None
            if query_block is not None:
                start = (
                    data.graph_base_feature_count
                    + query_block * data.graph_query_block_size
                )
                query_mask = data.graph_features[
                    row,
                    start : start + query_cell_count,
                ]
                marked = torch.nonzero(
                    query_mask > 0.5,
                    as_tuple=False,
                ).flatten()
                if marked.numel() == 1:
                    index_value = int(marked.item())
                    query_position = [
                        index_value % arena_size,
                        index_value // arena_size,
                    ]
            active_count += int(is_active)
            correct_count += int(correct is True)
            items.append(
                {
                    "index": index,
                    "proposition": proposition,
                    "group": group,
                    "groupLabel": GROUP_LABELS[group],
                    "sentence": sentence,
                    "probabilityTrue": round(float(probability), 6),
                    "predictedTrue": predicted,
                    "truthTrue": bool(truth),
                    "active": bool(is_active),
                    "correct": correct,
                    "queryPosition": query_position,
                }
            )
        frames.append(
            {
                "step": step,
                "ready": True,
                "activeCount": active_count,
                "correctCount": correct_count,
                "items": items,
            }
        )
    return frames


def _language_events(
    frames: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for group, label in (
        ("permanence", "首次出现遮挡后仍存在命题"),
        ("identity", "首次出现重现后身份命题"),
    ):
        match = next(
            (
                frame
                for frame in frames
                if any(
                    item["group"] == group and item["active"]
                    for item in frame["items"]
                )
            ),
            None,
        )
        if match is not None:
            events.append(
                {
                    "step": int(match["step"]),
                    "type": f"language_{group}",
                    "label": f"第 {match['step']} 步：{label}",
                }
            )
    return events


def build_replay_payload(
    *,
    seed: int = DEFAULT_SEED,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Build a standalone replay payload without touching holdout state."""

    protocol_path = protocol_path.resolve()
    protocol, protocol_digest = language._load_protocol(protocol_path)
    if int(protocol["protocol_version"]) != 6:
        raise ValueError(
            "L0 replay only accepts the repeatable V6 development protocol"
        )
    validate_replay_seed(seed, protocol)

    fixed = protocol["fixed_execution"]
    steps = int(fixed["steps_per_seed"])
    warmup = int(fixed["warmup_steps"])
    collection_options = {
        "steps": steps,
        "warmup": warmup,
        "reappearance_window": int(
            fixed["reappearance_window_steps"]
        ),
    }
    train_seeds = tuple(
        int(value)
        for value in protocol["splits"]["development_train"]["seeds"]
    )
    train = language.collect_language_data(
        train_seeds,
        **collection_options,
    )
    evaluation = language.collect_language_data(
        (seed,),
        **collection_options,
    )
    config = language._readout_config(protocol)

    language_conditions: dict[str, dict[str, Any]] = {}
    for representation, presentation in REPRESENTATIONS.items():
        model = language.train_readout(
            train,
            representation=representation,
            config=config,
        )
        logits = language.readout_logits(
            model,
            evaluation,
            representation=representation,
        )
        language_conditions[representation] = {
            **presentation,
            "metrics": language.evaluate_readout(
                model,
                evaluation,
                representation=representation,
            ),
            "frames": _language_frames(
                data=evaluation,
                logits=logits,
                protocol=protocol,
                steps=steps,
                warmup=warmup,
            ),
        }

    visual = v2_i1_replay.record_condition(
        seed=seed,
        steps=steps,
        condition_name="formal",
    )
    formal_events = language_conditions["formal_entity_graph"]["frames"]
    events = sorted(
        [*visual["events"], *_language_events(formal_events)],
        key=lambda item: (int(item["step"]), str(item["type"])),
    )

    result_raw = FROZEN_DEVELOPMENT_RESULT.read_bytes()
    frozen_result = json.loads(result_raw)
    if (
        frozen_result.get("protocol_sha256") != protocol_digest
        or frozen_result.get("passed") is not True
        or frozen_result.get("decision")
        != "authorize_review_and_source_lock"
    ):
        raise RuntimeError("frozen V6 development evidence is invalid")
    aggregate_metrics = {
        name: frozen_result["conditions"][name]
        for name in REPRESENTATIONS
    }
    data_digest = _sha256_bytes(
        _canonical_json(
            {
                "visualFrames": visual["frames"],
                "languageConditions": language_conditions,
            }
        )
    )
    return {
        "schemaVersion": 1,
        "experiment": "V2-L0-frozen-entity-language-readout",
        "title": "L0 Development Language Replay",
        "seed": seed,
        "steps": steps,
        "warmup": warmup,
        "arena": {
            "low": language.ARENA_LOW,
            "high": language.ARENA_HIGH,
            "size": language.ARENA_HIGH - language.ARENA_LOW + 1,
        },
        "presentationOnly": True,
        "evidenceLevel": "repeatable development-validation",
        "holdoutSeedsAccessed": False,
        "learnerInput": ["局部二值占据栅格", "动作编号"],
        "languageReadoutInput": [
            "冻结且 detached 的 I1 实体表征",
            "受控命题查询",
        ],
        "evaluatorTruthUsedForI1": False,
        "evaluatorTruthUsedForReadoutTraining": True,
        "languageGradientsReachI1": False,
        "representationOrder": list(REPRESENTATIONS),
        "languageConditions": language_conditions,
        "aggregateMetrics": aggregate_metrics,
        "visualFrames": visual["frames"],
        "events": events,
        "protocol": {
            "path": str(protocol_path.relative_to(PROJECT_ROOT)),
            "sha256": protocol_digest,
        },
        "formalEvidence": {
            "resultPath": str(
                FROZEN_DEVELOPMENT_RESULT.relative_to(PROJECT_ROOT)
            ),
            "resultSha256": _sha256_bytes(result_raw),
            "passed": frozen_result["passed"],
            "decision": frozen_result["decision"],
            "gateCount": len(frozen_result["gates"]),
            "allGatesPassed": all(frozen_result["gates"].values()),
        },
        "sourceLock": {
            "tag": "calmodel-l0-v8-source-locked",
            "targetCommit": "e26c613e4648528f38f7125b662c6daf89448983",
            "note": (
                "本演示分支不改变该锁定 tag；正式 holdout 只能从 tag "
                "锁定源码另行授权运行。"
            ),
        },
        "sourceFiles": _source_manifest(),
        "actionScheduleSha256": visual["actionScheduleSha256"],
        "replayDataSha256": data_digest,
    }


def render_replay(
    *,
    seed: int = DEFAULT_SEED,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> str:
    from cal.evaluation.v2_l0_language_replay_template import (
        render_replay_html,
    )

    return render_replay_html(
        build_replay_payload(seed=seed, protocol_path=protocol_path)
    )


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
    expected = render_replay(
        seed=seed,
        protocol_path=protocol_path,
    ).encode("utf-8")
    actual = replay_path.resolve().read_bytes()
    expected_sha = _sha256_bytes(expected)
    actual_sha = _sha256_bytes(actual)
    return expected == actual, expected_sha, actual_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cal-v2-l0-language-replay",
        description="生成或校验 L0 development-validation 交互式回放。",
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
        help="列出冻结协议允许重复演示的 development-validation seeds。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol, _ = language._load_protocol(args.protocol.resolve())
        if args.list_seeds:
            print(
                " ".join(
                    str(seed)
                    for seed in allowed_development_seeds(protocol)
                )
            )
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

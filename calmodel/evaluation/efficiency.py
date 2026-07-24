"""Architecture cost proxies and measured CPU latency for M1 models."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

import torch
from torch.optim import Adam

from calmodel.learning.dataset import ACTION_VOCABULARY
from calmodel.infra.provenance import capture_provenance
from calmodel.learning.trainer import (
    count_trainable_parameters,
    load_checkpoint,
    prediction_losses,
)
from calmodel.model.predictors import SensorimotorPredictor


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    repeats: int
    mean_milliseconds: float
    median_milliseconds: float
    p95_milliseconds: float


def estimate_macs_per_step(model: SensorimotorPredictor) -> int:
    """Count dense multiply-accumulates; elementwise operations are excluded."""

    config = model.config
    width, height = config.image_size
    macs = 0
    if config.use_vision:
        macs += width * height * 8 * (1 * 3 * 3)
        second_width = (width + 2 - 3) // 2 + 1
        second_height = (height + 2 - 3) // 2 + 1
        macs += second_width * second_height * 16 * (8 * 3 * 3)
        macs += 16 * 4 * 4 * config.vision_latent_size
    if config.use_proprioception:
        size = config.proprioception_latent_size
        macs += 4 * size + size * size
    if config.use_touch:
        size = config.touch_latent_size
        macs += 2 * size + size * size
    if config.use_motion:
        size = config.motion_size
        macs += 10 * size + size * size

    encoder_size = sum(
        size
        for enabled, size in (
            (config.use_vision, config.vision_latent_size),
            (config.use_proprioception, config.proprioception_latent_size),
            (config.use_touch, config.touch_latent_size),
            (config.use_action, config.action_latent_size),
        )
        if enabled
    )
    hidden = config.hidden_size
    if config.core_type == "gru":
        # Three gates, each with input and recurrent affine transforms.
        macs += 3 * (encoder_size * hidden + hidden * hidden)
    else:
        macs += encoder_size * hidden + hidden * hidden

    representation_size = hidden + (
        10 if config.use_motion and config.motion_passthrough else 0
    )
    macs += representation_size * (width * height + 4 + 2)
    if config.use_inverse_dynamics:
        macs += representation_size * 2 * len(ACTION_VOCABULARY)
    return macs


def active_parameter_proxy(model: SensorimotorPredictor) -> int:
    """Dense active parameters, counting only one accessed embedding row."""

    total = count_trainable_parameters(model)
    action = model.encoder.action
    if action is None:
        return total
    embedding_parameters = action.embedding.weight.numel()
    active_embedding_parameters = action.embedding.embedding_dim
    return total - embedding_parameters + active_embedding_parameters


def persistent_state_io_bytes_per_step(
    model: SensorimotorPredictor,
) -> tuple[int, int]:
    """Bytes read and written by state carried across environment steps."""

    if model.config.core_type != "gru":
        return 0, 0
    state_bytes = model.config.hidden_size * 4
    return state_bytes, state_bytes


def profile_checkpoint(
    checkpoint_path: str | Path,
    *,
    inference_repeats: int = 500,
    training_repeats: int = 100,
    warmup: int = 20,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure one-step batch-one latency under a controlled local setup."""

    if min(inference_repeats, training_repeats, warmup) <= 0:
        raise ValueError("benchmark repeat counts must be positive")
    torch.manual_seed(0)
    if device == "cpu":
        torch.set_num_threads(1)
    resolved_device = torch.device(device)
    model = load_checkpoint(checkpoint_path, device=device)
    batch = _benchmark_batch(model, resolved_device)

    model.eval()
    for _ in range(warmup):
        with torch.inference_mode():
            _forward(model, batch)
    inference_times = []
    with torch.inference_mode():
        for _ in range(inference_repeats):
            started = time.perf_counter_ns()
            _forward(model, batch)
            inference_times.append(time.perf_counter_ns() - started)

    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        losses = prediction_losses(_forward(model, batch), batch)
        losses["total"].backward()
        optimizer.step()
    training_times = []
    for _ in range(training_repeats):
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        losses = prediction_losses(_forward(model, batch), batch)
        losses["total"].backward()
        optimizer.step()
        training_times.append(time.perf_counter_ns() - started)

    state_read, state_write = persistent_state_io_bytes_per_step(model)
    checkpoint = Path(checkpoint_path)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return {
        "result_schema_version": 1,
        "checkpoint": str(checkpoint),
        "model_config": model.config.to_dict(),
        "parameter_count": count_trainable_parameters(model),
        "active_parameter_proxy": active_parameter_proxy(model),
        "parameter_bytes": parameter_bytes,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "macs_per_step": estimate_macs_per_step(model),
        "persistent_state_read_bytes_per_step": state_read,
        "persistent_state_write_bytes_per_step": state_write,
        "representation_write_bytes_per_step": model.representation_size * 4,
        "inference_latency": asdict(_latency(inference_times)),
        "training_latency": asdict(_latency(training_times)),
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "benchmark": {
            "batch_size": 1,
            "sequence_length": 1,
            "warmup": warmup,
            "device": device,
            "dtype": "float32",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "mac_definition": (
                "dense Conv/Linear/GRU multiply-accumulates; bias, "
                "normalization, activation, pooling and elementwise "
                "operations excluded"
            ),
            "active_parameter_definition": (
                "all dense trainable parameters plus one action embedding row"
            ),
            "memory_definition": (
                "maximum resident set size of the isolated benchmark process; "
                "includes Python and PyTorch runtime"
            ),
        },
        "provenance": capture_provenance(),
    }


def _benchmark_batch(
    model: SensorimotorPredictor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    width, height = model.config.image_size
    vision = torch.rand((1, 1, 1, height, width), device=device)
    proprioception = torch.rand((1, 1, 4), device=device) * 2.0 - 1.0
    touch = torch.zeros((1, 1, 2), device=device)
    actions = torch.randint(
        0,
        len(ACTION_VOCABULARY),
        (1, 1),
        device=device,
    )
    return {
        "vision": vision,
        "proprioception": proprioception,
        "touch": touch,
        "actions": actions,
        "next_vision": vision.clone(),
        "next_proprioception": proprioception.clone(),
        "next_touch": touch.clone(),
    }


def _forward(
    model: SensorimotorPredictor,
    batch: dict[str, torch.Tensor],
) -> Any:
    return model(
        vision=batch["vision"],
        proprioception=batch["proprioception"],
        touch=batch["touch"],
        actions=batch["actions"],
    )


def _latency(nanoseconds: Sequence[int]) -> LatencyMetrics:
    ordered = sorted(value / 1_000_000.0 for value in nanoseconds)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return LatencyMetrics(
        repeats=len(ordered),
        mean_milliseconds=fmean(ordered),
        median_milliseconds=median(ordered),
        p95_milliseconds=ordered[p95_index],
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Cal parameter, MAC, state and latency costs."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-repeats", type=int, default=500)
    parser.add_argument("--training-repeats", type=int, default=100)
    arguments = parser.parse_args(argv)
    summary = profile_checkpoint(
        arguments.checkpoint,
        inference_repeats=arguments.inference_repeats,
        training_repeats=arguments.training_repeats,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"parameters={summary['parameter_count']}, "
        f"MACs/step={summary['macs_per_step']}, "
        f"inference median="
        f"{summary['inference_latency']['median_milliseconds']:.4f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

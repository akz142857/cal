"""Small online control-identification agent used by V2-M1."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from calmodel.env.point_world import PointAction


@dataclass(slots=True)
class _Track:
    index: int
    position: np.ndarray
    last_seen: int
    theta: np.ndarray
    covariance: np.ndarray
    error_ema: float = 1.0
    evidence: float = 0.0
    probability: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlAgentConfig:
    maximum_tracks: int = 5
    forgetting: float = 0.98
    observation_variance: float = 0.16
    active: bool = True
    use_action: bool = True
    use_failure_update: bool = True
    use_uncertainty: bool = True
    failure_memory_size: int = 32


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Compact label-free record used by the bounded failure-data loop."""

    step: int
    action: int
    predicted_positions: tuple[tuple[int, int], ...]
    observed_positions: tuple[tuple[int, int], ...]
    residual_magnitudes: tuple[float, ...]
    weakened_tracks: tuple[int, ...]
    enhanced_tracks: tuple[int, ...]
    next_action_information: tuple[float, float, float, float]
    priority: float


class OnlineControlAgent:
    """Track anonymous points and infer which trajectory is action-controlled.

    Each track owns a 2x4 linear control matrix.  It is updated once per
    transition by recursive least squares.  Membership evidence is the
    improvement of its learned action-conditioned prediction over a
    no-control prediction.
    """

    def __init__(
        self,
        config: ControlAgentConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config or ControlAgentConfig()
        self._rng = np.random.default_rng(seed)
        self._tracks: list[_Track] = []
        self._step = 0
        self._action_counts = np.zeros(5, dtype=np.int64)
        self._last_action = PointAction.NOOP
        self._next_index = 0
        self._failure_memory: list[FailureRecord] = []

    @property
    def learnable_parameter_count(self) -> int:
        return self.config.maximum_tracks * (2 * 4 + 4 * 4 + 3)

    @property
    def active_state_bytes(self) -> int:
        track_bytes = self.config.maximum_tracks * (2 + 8 + 16 + 3) * 8
        failure_bytes = self.config.failure_memory_size * 1536
        return track_bytes + failure_bytes + self._action_counts.nbytes

    @property
    def estimated_mac_per_step(self) -> int:
        return self.config.maximum_tracks * 160

    def reset(self, frame: np.ndarray) -> None:
        self._tracks = []
        self._step = 0
        self._action_counts.fill(0)
        self._last_action = PointAction.NOOP
        self._next_index = 0
        self._failure_memory = []
        for detection in self._detections(frame):
            self._tracks.append(self._new_track(detection))
        self._normalize_probabilities()

    def choose_action(self) -> PointAction:
        if not self.config.active:
            return PointAction(int(self._rng.integers(0, 5)))
        if self.identification_ready():
            return PointAction.NOOP
        # RLS predictive variance u^T P u is the local information gain.
        utilities = []
        for action in tuple(PointAction)[1:]:
            feature = self._feature(action)
            gain = sum(
                float(feature @ track.covariance @ feature)
                for track in self._tracks
            )
            novelty = 1.0 / (1.0 + self._action_counts[int(action)])
            utilities.append((gain + 4.0 * novelty, -int(action), action))
        return max(utilities)[2]

    def observe_transition(
        self,
        frame: np.ndarray,
        action: PointAction,
    ) -> None:
        self._step += 1
        self._action_counts[int(action)] += 1
        detections = self._detections(frame)
        assignments = self._assign(detections)
        feature = self._feature(action)
        predictions: list[tuple[int, int]] = []
        residual_magnitudes: list[float] = []
        old_evidence = {track.index: track.evidence for track in self._tracks}
        for track, detection in assignments:
            previous = track.position.copy()
            displacement = detection - previous
            predicted_position = previous + track.theta @ feature
            predictions.append(
                (int(round(predicted_position[0])), int(round(predicted_position[1])))
            )
            residual_magnitudes.append(
                float(np.linalg.norm(detection - predicted_position))
            )
            track.position = detection
            track.last_seen = self._step
            if not self.config.use_failure_update:
                continue
            prediction_before = track.theta @ feature
            baseline_error = float(displacement @ displacement)
            residual = displacement - prediction_before
            conditioned_error = float(residual @ residual)
            denominator = (
                self.config.forgetting
                + float(feature @ track.covariance @ feature)
            )
            if np.any(feature):
                gain = (track.covariance @ feature) / denominator
                track.theta += np.outer(residual, gain)
                track.covariance = (
                    track.covariance - np.outer(gain, feature) @ track.covariance
                ) / self.config.forgetting
            prediction_after = track.theta @ feature
            learned_residual = displacement - prediction_after
            learned_error = float(learned_residual @ learned_residual)
            track.error_ema = 0.9 * track.error_ema + 0.1 * learned_error
            improvement = baseline_error - min(conditioned_error, learned_error)
            # Prediction failures weaken a hypothesis; successful controlled
            # motion strengthens it. Both directions are online updates.
            track.evidence = 0.88 * track.evidence + 1.8 * improvement
            if np.any(feature) and baseline_error < 0.25:
                track.evidence -= 0.35
        for detection in detections:
            if not any(np.array_equal(detection, item[1]) for item in assignments):
                if len(self._tracks) < self.config.maximum_tracks:
                    self._tracks.append(self._new_track(detection))
        self._normalize_probabilities()
        self._last_action = action
        changed = [
            (track.index, track.evidence - old_evidence.get(track.index, 0.0))
            for track in self._tracks
        ]
        information = tuple(
            self._information_gain(candidate)
            for candidate in tuple(PointAction)[1:]
        )
        record = FailureRecord(
            step=self._step,
            action=int(action),
            predicted_positions=tuple(predictions),
            observed_positions=tuple(
                (int(item[0]), int(item[1])) for item in detections
            ),
            residual_magnitudes=tuple(residual_magnitudes),
            weakened_tracks=tuple(index for index, delta in changed if delta < 0.0),
            enhanced_tracks=tuple(index for index, delta in changed if delta > 0.0),
            next_action_information=information,
            priority=max(residual_magnitudes, default=0.0) + self.entropy(),
        )
        self._failure_memory.append(record)
        self._failure_memory.sort(key=lambda item: item.priority, reverse=True)
        del self._failure_memory[self.config.failure_memory_size :]

    def probabilities(self) -> dict[int, float]:
        return {track.index: track.probability for track in self._tracks}

    def track_positions(self) -> dict[int, tuple[int, int]]:
        return {
            track.index: (int(track.position[0]), int(track.position[1]))
            for track in self._tracks
        }

    def confidence(self) -> float:
        return max(self.probabilities().values(), default=0.0)

    def identification_ready(self) -> bool:
        if self.confidence() < 0.95 or not self._tracks:
            return False
        return max(
            float(np.diag(track.covariance).max()) for track in self._tracks
        ) <= 1.25

    def entropy(self) -> float:
        return -sum(
            probability * log(max(probability, 1e-12))
            for probability in self.probabilities().values()
        )

    def failure_memory(self) -> tuple[FailureRecord, ...]:
        return tuple(self._failure_memory)

    def _information_gain(self, action: PointAction) -> float:
        feature = self._feature(action)
        return sum(
            float(feature @ track.covariance @ feature)
            for track in self._tracks
        )

    def _feature(self, action: PointAction) -> np.ndarray:
        feature = np.zeros(4, dtype=np.float64)
        if self.config.use_action and action is not PointAction.NOOP:
            feature[int(action) - 1] = 1.0
        return feature

    def _assign(
        self,
        detections: list[np.ndarray],
    ) -> list[tuple[_Track, np.ndarray]]:
        candidates = []
        for track in self._tracks:
            age = self._step - track.last_seen
            limit = 1.5 + min(age, 3)
            for detection in detections:
                distance = float(np.linalg.norm(detection - track.position))
                if distance <= limit:
                    candidates.append((distance, track.index, track, detection))
        assigned_tracks: set[int] = set()
        assigned_detections: set[tuple[int, int]] = set()
        result = []
        for _, _, track, detection in sorted(candidates, key=lambda item: item[:2]):
            key = (int(detection[0]), int(detection[1]))
            if track.index in assigned_tracks or key in assigned_detections:
                continue
            assigned_tracks.add(track.index)
            assigned_detections.add(key)
            result.append((track, detection))
        return result

    def _normalize_probabilities(self) -> None:
        if not self._tracks:
            return
        if not self.config.use_failure_update:
            values = np.ones(len(self._tracks), dtype=np.float64)
        else:
            scores = np.array([track.evidence for track in self._tracks])
            scores -= float(scores.max())
            temperature = 0.45 if self.config.use_uncertainty else 0.15
            values = np.exp(scores / temperature)
        values /= float(values.sum())
        for track, value in zip(self._tracks, values, strict=True):
            track.probability = float(value)

    def _new_track(self, position: np.ndarray) -> _Track:
        track = _Track(
            index=self._next_index,
            position=position.copy(),
            last_seen=self._step,
            theta=np.zeros((2, 4), dtype=np.float64),
            covariance=np.eye(4, dtype=np.float64) * 8.0,
        )
        self._next_index += 1
        return track

    @staticmethod
    def _detections(frame: np.ndarray) -> list[np.ndarray]:
        ys, xs = np.nonzero(frame)
        return [
            np.array((x, y), dtype=np.float64)
            for x, y in sorted(zip(xs.tolist(), ys.tolist(), strict=True))
        ]

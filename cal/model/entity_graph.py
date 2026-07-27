"""Online sparse visual entity graph for the V2-M2/M3 stages."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np


@dataclass(slots=True)
class GraphTrack:
    index: int
    position: np.ndarray
    previous: np.ndarray
    last_seen: int
    theta: np.ndarray
    covariance: np.ndarray
    autonomous_velocity: np.ndarray
    control_evidence: float = 0.0
    probability: float = 0.5


@dataclass(slots=True)
class _Edge:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, distance: float) -> None:
        self.count += 1
        delta = distance - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (distance - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / max(1, self.count - 1)


class OnlineEntityGraph:
    """RLS control estimates plus a streaming pairwise geometry graph."""

    def __init__(
        self,
        action_dimensions: int,
        *,
        maximum_tracks: int = 16,
        match_radius: float = 3.0,
        forgetting: float = 0.96,
        association_beam_width: int = 64,
        association_commitment: float = 0.58,
        association_miss_cost: float = 7.0,
        association_mode: str = "probabilistic",
        reacquisition_window: int = 6,
        identity_switch_penalty_weight: float = 0.0,
        drift_reset_after: int | None = None,
        confidence_adaptive_gating_weight: float = 0.0,
    ) -> None:
        self.action_dimensions = action_dimensions
        self.maximum_tracks = maximum_tracks
        self.match_radius = match_radius
        self.forgetting = forgetting
        self.association_beam_width = association_beam_width
        self.association_commitment = association_commitment
        self.association_miss_cost = association_miss_cost
        if reacquisition_window < 1:
            raise ValueError("reacquisition_window must be positive")
        self.reacquisition_window = reacquisition_window
        if identity_switch_penalty_weight < 0.0:
            raise ValueError("identity_switch_penalty_weight must be non-negative")
        self.identity_switch_penalty_weight = identity_switch_penalty_weight
        if drift_reset_after is not None and drift_reset_after < 1:
            raise ValueError("drift_reset_after must be positive")
        self.drift_reset_after = drift_reset_after
        if confidence_adaptive_gating_weight < 0.0:
            raise ValueError(
                "confidence_adaptive_gating_weight must be non-negative"
            )
        self.confidence_adaptive_gating_weight = confidence_adaptive_gating_weight
        if association_mode not in {"probabilistic", "hard_map", "nearest"}:
            raise ValueError("invalid association mode")
        self.association_mode = association_mode
        self._tracks: list[GraphTrack] = []
        self._edges: dict[tuple[int, int], _Edge] = {}
        self._step = 0
        self._next_index = 0
        self._association_hypotheses: list[
            tuple[float, tuple[tuple[int, int], ...]]
        ] = []
        self._association_marginals: dict[tuple[int, int], float] = {}
        self._association_entropy = 0.0
        self._last_detections: list[np.ndarray] = []
        self._prequential_errors: dict[int, float] = {}
        self._causal_prediction_errors: dict[int, float] = {}
        self._maximum_observed_detections = 0

    @property
    def learnable_parameter_count(self) -> int:
        per_track = (
            2 * self.action_dimensions
            + self.action_dimensions * self.action_dimensions
            + 6
        )
        return self.maximum_tracks * per_track

    @property
    def active_state_bytes(self) -> int:
        tracks = self.learnable_parameter_count * 8
        edges = self.maximum_tracks * (self.maximum_tracks - 1) // 2 * 32
        association = self.association_beam_width * self.maximum_tracks * 16
        return tracks + edges + association

    @property
    def estimated_mac_per_step(self) -> int:
        total = self.maximum_tracks * (80 + 12 * self.action_dimensions)
        if self.identity_switch_penalty_weight > 0.0:
            # For each active track's candidate detection, worst case
            # compares against every other active track's predicted
            # position. Disabled callers (every existing one, at the
            # default weight 0.0) do zero extra work here:
            # _identity_switch_penalty short-circuits before touching
            # predicted_positions, so omitting this term at the default
            # matches the exact computation performed, not just a rough
            # estimate of it.
            total += self.maximum_tracks * self.maximum_tracks * 4
        if self.drift_reset_after is not None:
            # Worst case: every unmatched track qualifies for a reset in
            # the same step, each rebuilding a theta/covariance of size
            # action_dimensions and action_dimensions^2 respectively.
            # Disabled callers (drift_reset_after is None, the default)
            # never reach _reset_track_state, so this term is correctly
            # omitted for them too.
            total += self.maximum_tracks * (
                2 * self.action_dimensions + self.action_dimensions**2
            )
        return total

    def reset(self, detections: np.ndarray) -> None:
        self._tracks = []
        self._edges = {}
        self._step = 0
        self._next_index = 0
        self._association_hypotheses = []
        self._association_marginals = {}
        self._association_entropy = 0.0
        self._last_detections = []
        self._prequential_errors = {}
        self._causal_prediction_errors = {}
        self._maximum_observed_detections = len(detections)
        for detection in self._sorted(detections):
            self._tracks.append(self._new_track(detection))

    def update(self, detections: np.ndarray, action: np.ndarray) -> None:
        if action.shape != (self.action_dimensions,):
            raise ValueError("unexpected action shape")
        self._step += 1
        ordered = self._sorted(detections)
        previous_detection_maximum = self._maximum_observed_detections
        self._maximum_observed_detections = max(
            self._maximum_observed_detections, len(ordered)
        )
        self._prequential_errors = {}
        self._causal_prediction_errors = {}
        self._last_detections = [item.copy() for item in ordered]
        if self.association_mode == "nearest":
            assignments = self._nearest_assign(ordered)
            claimed_detections = {
                detection_index for _, detection_index in assignments
            }
            self._association_hypotheses = []
            self._association_marginals = {
                (track.index, detection_index): 1.0
                for track, detection_index in assignments
            }
            self._association_entropy = 0.0
        else:
            assignments, claimed_detections = self._probabilistic_assign(
                ordered, action
            )
        assigned_detections: set[int] = set()
        visible: list[GraphTrack] = []
        for track, detection_index in assignments:
            detection = ordered[detection_index]
            assigned_detections.add(detection_index)
            track.previous = track.position.copy()
            displacement = detection - track.position
            action_prediction = track.theta @ action
            autonomous_residual = displacement - track.autonomous_velocity
            residual = autonomous_residual - action_prediction
            self._causal_prediction_errors[track.index] = float(
                np.linalg.norm(displacement - action_prediction)
            )
            baseline = float(autonomous_residual @ autonomous_residual)
            denominator = self.forgetting + float(
                action @ track.covariance @ action
            )
            if np.any(action):
                gain = track.covariance @ action / denominator
                track.theta += np.outer(residual, gain)
                track.covariance = (
                    track.covariance
                    - np.outer(gain, action) @ track.covariance
                ) / self.forgetting
            # Membership evidence is prequential: score the prediction made
            # before seeing this transition.  Using the post-update residual
            # here would reward fitting autonomous noise on the same sample.
            improvement = baseline - float(residual @ residual)
            self._prequential_errors[track.index] = float(
                np.linalg.norm(residual)
            )
            track.control_evidence = (
                0.93 * track.control_evidence + 1.5 * improvement
            )
            if np.any(action) and baseline < 1e-3:
                track.control_evidence -= 0.08
            evidence_probability = 1.0 / (
                1.0 + exp(-3.0 * (track.control_evidence - 2.00))
            )
            control_strength = max(
                np.linalg.norm(track.theta[:, column])
                for column in range(self.action_dimensions)
            )
            resolved_effect = 1.0 / (
                1.0 + exp(-20.0 * (float(control_strength) - 0.20))
            )
            track.probability = evidence_probability * resolved_effect
            autonomous_sample = displacement - track.theta @ action
            track.autonomous_velocity = (
                0.80 * track.autonomous_velocity + 0.20 * autonomous_sample
            )
            track.position = detection.copy()
            track.last_seen = self._step
            visible.append(track)
        for index, detection in enumerate(ordered):
            if (
                index not in assigned_detections
                and index not in claimed_detections
                and len(self._tracks) < self.maximum_tracks
            ):
                track = self._new_track(detection)
                self._tracks.append(track)
                visible.append(track)
        visible_indexes = {track.index for track in visible}
        for track in self._tracks:
            if track.index not in visible_indexes:
                if (
                    ordered
                    and len(ordered) >= previous_detection_maximum
                    and track.index not in self._causal_prediction_errors
                ):
                    predicted = self._predicted_position(track, action)
                    self._causal_prediction_errors[track.index] = min(
                        float(np.linalg.norm(detection - predicted))
                        for detection in ordered
                    )
                track.previous = track.position.copy()
                track.position = self._predicted_position(track, action)
                age = self._step - track.last_seen
                if age > 3:
                    track.control_evidence *= 0.80
                    track.probability *= 0.80
                if (
                    self.drift_reset_after is not None
                    and age > self.drift_reset_after
                    and track.control_evidence < self._DRIFT_RESET_CONTROL_EVIDENCE_CEILING
                ):
                    self._reset_track_state(track)
        self._update_edges(visible)
        self._propagate_membership()

    def probabilities(self) -> dict[int, float]:
        return {track.index: track.probability for track in self._tracks}

    def positions(self) -> dict[int, tuple[float, float]]:
        return {
            track.index: (float(track.position[0]), float(track.position[1]))
            for track in self._tracks
        }

    def predict_displacements(
        self,
        action: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        if action.shape != (self.action_dimensions,):
            raise ValueError("unexpected action shape")
        return {
            track.index: tuple(float(value) for value in track.theta @ action)
            for track in self._tracks
        }

    def control_strengths(self) -> dict[int, float]:
        return {
            track.index: max(
                float(np.linalg.norm(track.theta[:, column]))
                for column in range(self.action_dimensions)
            )
            for track in self._tracks
        }

    def control_matrices(self) -> dict[int, np.ndarray]:
        return {
            track.index: track.theta.copy()
            for track in self._tracks
        }

    def prequential_errors(self) -> dict[int, float]:
        return dict(self._prequential_errors)

    def causal_prediction_errors(self) -> dict[int, float]:
        """Errors of the pre-update action-effect model, excluding autonomy."""
        return dict(self._causal_prediction_errors)

    def association_entropy(self) -> float:
        return self._association_entropy

    def association_hypothesis_count(self) -> int:
        return len(self._association_hypotheses)

    def association_distribution(
        self,
        track_index: int,
    ) -> dict[tuple[float, float] | None, float]:
        result: dict[tuple[float, float] | None, float] = {}
        assigned_mass = 0.0
        for detection_index, detection in enumerate(self._last_detections):
            probability = self._association_marginals.get(
                (track_index, detection_index), 0.0
            )
            if probability <= 0.0:
                continue
            key = (float(detection[0]), float(detection[1]))
            result[key] = result.get(key, 0.0) + probability
            assigned_mass += probability
        result[None] = max(0.0, 1.0 - assigned_mass)
        return result

    def detection_positions(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(detection[0]), float(detection[1]))
            for detection in self._last_detections
        )

    def rigid_edges(
        self,
        *,
        minimum_samples: int = 12,
        maximum_variance: float = 0.02,
        maximum_length: float = 4.2,
    ) -> set[tuple[int, int]]:
        return {
            key
            for key, edge in self._edges.items()
            if (
                edge.count >= minimum_samples
                and edge.variance <= maximum_variance
                and edge.mean <= maximum_length
            )
        }

    def self_tracks(self, threshold: float = 0.62) -> set[int]:
        return {
            track.index
            for track in self._tracks
            if track.probability >= threshold
        }

    def _propagate_membership(self) -> None:
        edges = self.rigid_edges()
        probabilities = self.probabilities()
        seeds = {index for index, value in probabilities.items() if value >= 0.95}
        changed = True
        while changed:
            changed = False
            for left, right in edges:
                if left in seeds and right not in seeds:
                    seeds.add(right)
                    changed = True
                if right in seeds and left not in seeds:
                    seeds.add(left)
                    changed = True
        for track in self._tracks:
            if track.index in seeds:
                track.probability = max(track.probability, 0.82)

    def _update_edges(self, visible: list[GraphTrack]) -> None:
        for offset, left in enumerate(visible):
            for right in visible[offset + 1 :]:
                key = tuple(sorted((left.index, right.index)))
                distance = float(np.linalg.norm(left.position - right.position))
                self._edges.setdefault(key, _Edge()).update(distance)

    def _probabilistic_assign(
        self,
        detections: list[np.ndarray],
        action: np.ndarray,
    ) -> tuple[list[tuple[GraphTrack, int]], set[int]]:
        # How long an unseen track may still compete for a new detection.
        # The default of 6 matches this class's native V2-M2/M3 worlds,
        # whose occlusions are brief (about 2 of every 23 steps); callers
        # composing this mechanism with longer-occlusion environments (see
        # V2-I1) may pass a wider window at construction time.
        active_tracks = [
            track
            for track in self._tracks
            if self._step - track.last_seen <= self.reacquisition_window
        ]
        # Predicted positions for every active track, computed once up
        # front (not order-dependent) so the identity-switch penalty below
        # can compare a candidate detection's fit against every OTHER
        # track's own prediction, independent of which track this beam
        # happens to process first.
        predicted_positions = {
            other.index: self._predicted_position(other, action)
            for other in active_tracks
        }
        beams: list[
            tuple[float, tuple[tuple[int, int], ...], frozenset[int]]
        ] = [(0.0, (), frozenset())]
        for track in active_tracks:
            predicted = predicted_positions[track.index]
            age = self._step - track.last_seen
            radius = self.match_radius + min(age, 3)
            candidates = [
                (
                    index,
                    float(np.linalg.norm(detection - predicted)),
                )
                for index, detection in enumerate(detections)
                if float(np.linalg.norm(detection - predicted)) <= radius
            ]
            expanded = []
            for cost, assignments, used in beams:
                miss_cost = self.association_miss_cost + 0.5 * min(age, 4)
                expanded.append(
                    (cost + miss_cost, assignments + ((track.index, -1),), used)
                )
                for detection_index, distance in candidates:
                    if detection_index in used:
                        continue
                    geometry_cost = self._geometry_assignment_cost(
                        track,
                        detections[detection_index],
                        assignments,
                        detections,
                    )
                    motion_cost = (
                        distance / self._motion_cost_scale(track, action)
                    ) ** 2
                    switch_penalty = self._identity_switch_penalty(
                        track.index,
                        detections[detection_index],
                        distance,
                        predicted_positions,
                    )
                    expanded.append(
                        (
                            cost + motion_cost + geometry_cost + switch_penalty,
                            assignments + ((track.index, detection_index),),
                            used | {detection_index},
                        )
                    )
            beams = sorted(expanded, key=lambda item: item[0])[
                : self.association_beam_width
            ]
        if not beams:
            self._association_hypotheses = []
            self._association_marginals = {}
            self._association_entropy = 0.0
            return [], set()
        minimum = beams[0][0]
        weights = np.asarray(
            [exp(-0.5 * min(80.0, cost - minimum)) for cost, _, _ in beams],
            dtype=np.float64,
        )
        weights /= float(weights.sum())
        self._association_hypotheses = [
            (float(weight), assignments)
            for weight, (_, assignments, _) in zip(weights, beams, strict=True)
        ]
        self._association_marginals = {}
        claimed_detections: set[int] = set()
        for weight, assignments in self._association_hypotheses:
            for track_index, detection_index in assignments:
                if detection_index < 0:
                    continue
                key = (track_index, detection_index)
                self._association_marginals[key] = (
                    self._association_marginals.get(key, 0.0) + weight
                )
                if weight >= 0.01:
                    claimed_detections.add(detection_index)
        self._association_entropy = -sum(
            weight * log(max(weight, 1e-12))
            for weight, _ in self._association_hypotheses
        )
        best_assignments = dict(self._association_hypotheses[0][1])
        tracks_by_index = {track.index: track for track in active_tracks}
        committed = []
        for track_index, detection_index in best_assignments.items():
            if detection_index < 0:
                continue
            marginal = self._association_marginals.get(
                (track_index, detection_index), 0.0
            )
            if (
                self.association_mode == "hard_map"
                or marginal >= self.association_commitment
            ):
                committed.append((tracks_by_index[track_index], detection_index))
        return committed, claimed_detections

    def _nearest_assign(
        self,
        detections: list[np.ndarray],
    ) -> list[tuple[GraphTrack, int]]:
        candidates = []
        for track in self._tracks:
            age = self._step - track.last_seen
            radius = self.match_radius + min(age, 3)
            for index, detection in enumerate(detections):
                distance = float(np.linalg.norm(detection - track.position))
                if distance <= radius:
                    candidates.append((distance, track.index, index, track))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        result = []
        for _, track_index, detection_index, track in sorted(
            candidates, key=lambda item: item[:3]
        ):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            result.append((track, detection_index))
        return result

    def _predicted_position(
        self,
        track: GraphTrack,
        action: np.ndarray,
    ) -> np.ndarray:
        return track.position + track.theta @ action + track.autonomous_velocity

    # Reference prediction-variance level: action @ covariance @ action for
    # a unit-norm (one-hot) action against _new_track's own initial
    # covariance (eye(action_dimensions) * 6.0). A track at exactly this
    # variance - i.e. exactly as confident as a brand-new track - gets
    # exactly the disabled-path scale (0.32); less confident tracks get a
    # tighter scale, more confident ones a wider one (see the direction
    # note on _motion_cost_scale below for why). Chosen this way so
    # "just created" behaves identically whether or not the mechanism is
    # enabled, rather than picking an arbitrary unrelated reference point.
    _MOTION_COST_SCALE_REFERENCE_VARIANCE = 6.0

    def _motion_cost_scale(self, track: GraphTrack, action: np.ndarray) -> float:
        """The distance scale motion_cost normalizes against for this track.

        Disabled by default (confidence_adaptive_gating_weight == 0.0):
        returns exactly 0.32 (the fixed scale every existing caller has
        always used), a hard no-op rather than a numerically-close one.

        When enabled, widens or tightens that scale per track using
        `action @ track.covariance @ action` - the same RLS-derived
        prediction-variance term already computed inline during the
        theta update (as `denominator - forgetting`), just not
        previously used for anything at candidate-scoring time.

        Direction (empirically corrected, not the first guess): a
        well-established, confident track (lower variance than the
        reference) gets a WIDER tolerance, and a still-uncertain one
        (freshly created, long unmatched, or otherwise less confident)
        gets a TIGHTER one. The original hypothesis was the opposite -
        tighten confident tracks so a rival can't steal a close
        encounter - reasoning by analogy from Kalman-filter gating,
        where the state covariance IS the position uncertainty. Here
        `track.covariance` is uncertainty in theta (the control-effect
        estimate), not in position, so that analogy doesn't transfer
        directly; a pre-registered calibration-set sweep (see
        docs/experiments/V2_I1_INTEGRATION_REPORT.md) measured the
        originally-hypothesized direction making identity_consistency
        WORSE at every tested weight, and the opposite direction
        measurably better. The likely mechanism: a still-uncertain
        track is the one most at risk of being prematurely fragmented
        (a single ordinary detection jitter read as a miss, killing a
        track that was still learning) - loosening its tolerance lets
        it survive to become well-established, rather than protecting
        already-stable tracks that were not the source of the churn.
        Exponential in the variance gap (rather than linear) keeps the
        scale positive and bounded for any covariance value, including
        the degenerate prediction_variance == 0 case a "stay" action
        produces for every track uniformly (no discrimination that
        step, but no blow-up either).
        """

        if self.confidence_adaptive_gating_weight <= 0.0:
            return 0.32
        prediction_variance = float(action @ track.covariance @ action)
        gap = self._MOTION_COST_SCALE_REFERENCE_VARIANCE - prediction_variance
        return 0.32 * exp(self.confidence_adaptive_gating_weight * gap)

    def _geometry_assignment_cost(
        self,
        track: GraphTrack,
        detection: np.ndarray,
        assignments: tuple[tuple[int, int], ...],
        detections: list[np.ndarray],
    ) -> float:
        cost = 0.0
        for other_index, detection_index in assignments:
            if detection_index < 0:
                continue
            edge = self._edges.get(tuple(sorted((track.index, other_index))))
            if edge is None or edge.count < 12 or edge.variance > 0.03:
                continue
            observed = float(
                np.linalg.norm(detection - detections[detection_index])
            )
            cost += ((observed - edge.mean) / 0.20) ** 2
        return cost

    def _identity_switch_penalty(
        self,
        track_index: int,
        detection: np.ndarray,
        distance: float,
        predicted_positions: dict[int, np.ndarray],
    ) -> float:
        """Extra cost for claiming a detection that fits a rival track better.

        Disabled by default (identity_switch_penalty_weight == 0.0): this is
        a no-op for every existing caller and adds exactly 0.0 to the cost,
        so default behavior is bit-for-bit unchanged from before this
        parameter existed. When enabled, it discourages a track from being
        assigned a detection that some OTHER active track's own prediction
        is closer to - the concrete precondition for an identity switch
        (two tracks close enough that a detection near their crossing point
        gets attributed to the wrong one, silently swapping which physical
        entity a track index follows from then on, since the corrupted
        association also poisons that track's theta/autonomous_velocity on
        the very same update).
        """

        if self.identity_switch_penalty_weight <= 0.0:
            return 0.0
        best_rival_gap = 0.0
        for other_index, other_predicted in predicted_positions.items():
            if other_index == track_index:
                continue
            other_distance = float(np.linalg.norm(detection - other_predicted))
            if other_distance < distance:
                best_rival_gap = max(best_rival_gap, distance - other_distance)
        if best_rival_gap <= 0.0:
            return 0.0
        return self.identity_switch_penalty_weight * (best_rival_gap / 0.32) ** 2

    def _new_track(self, detection: np.ndarray) -> GraphTrack:
        track = GraphTrack(
            index=self._next_index,
            position=detection.copy(),
            previous=detection.copy(),
            last_seen=self._step,
            theta=np.zeros((2, self.action_dimensions), dtype=np.float64),
            covariance=np.eye(self.action_dimensions, dtype=np.float64) * 6.0,
            autonomous_velocity=np.zeros(2, dtype=np.float64),
        )
        self._next_index += 1
        return track

    # Deliberately NOT `probability`: in a world where control_evidence
    # rarely climbs high even for a genuinely correct track (e.g. one
    # driven by an intermittent/weak action-correlation signal),
    # probability can sit low for track after track regardless of whether
    # any of them are actually corrupted - it can't discriminate a real
    # mis-association from ordinary weak-but-real evidence. Raw
    # control_evidence going NEGATIVE is a stronger, rarer signal: it
    # means recent residuals were worse than the autonomous-only baseline,
    # which a track merely riding out low-but-real signal does not do.
    _DRIFT_RESET_CONTROL_EVIDENCE_CEILING = 0.0

    def _reset_track_state(self, track: GraphTrack) -> None:
        """Wipe a track's learned control/motion state, keeping its index.

        Disabled by default (drift_reset_after is None): a mis-associated
        track's theta/autonomous_velocity get updated from a wrong
        displacement on the very step the bad association happens, and
        every prediction after that is generated from this now-corrupted
        state - it cannot "remember" the right answer no matter how long
        it coasts unmatched, and no amount of retuning
        identity_switch_penalty_weight (which only prevents a bad
        association, never undoes one) changes that. Once a track has
        gone unmatched long enough that reacquisition_window itself would
        soon give up on it AND its control_evidence has already gone
        negative (i.e. it is not a confident track merely riding out a
        normal occlusion), there is no more evidence left to protect -
        resetting it to the same blank state _new_track uses lets it
        start learning cleanly the next time it is matched, instead of
        dragging a poisoned theta into a match it might otherwise have
        made well. Position, last_seen and index are untouched: this
        does not create a new identity, it un-learns a corrupted one
        under the same index. Also drops this track's pairwise geometry
        statistics from self._edges: they were built from a position
        history that may itself be corrupted, and _propagate_membership
        would otherwise be free to read a stale rigid edge to a
        confident neighbor and immediately raise probability right back
        up in this same update() call, undoing the reset before any
        caller ever observes the neutral prior.
        """

        track.theta = np.zeros((2, self.action_dimensions), dtype=np.float64)
        track.covariance = np.eye(self.action_dimensions, dtype=np.float64) * 6.0
        track.autonomous_velocity = np.zeros(2, dtype=np.float64)
        track.control_evidence = 0.0
        track.probability = 0.5
        self._edges = {
            key: edge
            for key, edge in self._edges.items()
            if track.index not in key
        }

    @staticmethod
    def _sorted(detections: np.ndarray) -> list[np.ndarray]:
        if detections.ndim != 2 or detections.shape[1] != 2:
            raise ValueError("detections must be [points, 2]")
        return [
            np.asarray(item, dtype=np.float64)
            for item in sorted(detections.tolist(), key=lambda item: (item[1], item[0]))
        ]

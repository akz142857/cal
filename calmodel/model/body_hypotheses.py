"""Mutually exclusive posterior over complete articulated body graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable

import numpy as np

from calmodel.model.entity_graph import OnlineEntityGraph


@dataclass(frozen=True, slots=True, order=True)
class BodyGraphCandidate:
    """One complete base--joint--endpoint world hypothesis."""

    base: int
    joint: int
    endpoint: int

    @property
    def nodes(self) -> tuple[int, int, int]:
        return (self.base, self.joint, self.endpoint)

    @property
    def edges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            tuple(sorted((self.base, self.joint))),
            tuple(sorted((self.joint, self.endpoint))),
        )


@dataclass(frozen=True, slots=True)
class WeightedBodyGraph:
    candidate: BodyGraphCandidate
    probability: float


class BodyGraphHypothesisFilter:
    """Categorical Bayesian filter over full, mutually exclusive body graphs.

    The filter never receives a simulator identity. Candidate graphs are
    generated from learned link stability and action-Jacobian roles. Their
    posterior is updated only by prequential visual prediction residuals.
    """

    def __init__(
        self,
        *,
        maximum_hypotheses: int = 8,
        base_control_maximum: float = 0.20,
        joint_shoulder_minimum: float = 0.20,
        joint_elbow_maximum: float = 0.20,
        endpoint_elbow_minimum: float = 0.20,
        residual_sigma: float = 0.16,
        equivalent_log_likelihood_tolerance: float = 0.50,
        log_weight_forgetting: float = 0.985,
        lock_at_hypothesis_count: int | None = 2,
    ) -> None:
        self.maximum_hypotheses = maximum_hypotheses
        self.base_control_maximum = base_control_maximum
        self.joint_shoulder_minimum = joint_shoulder_minimum
        self.joint_elbow_maximum = joint_elbow_maximum
        self.endpoint_elbow_minimum = endpoint_elbow_minimum
        self.residual_sigma = residual_sigma
        self.equivalent_log_likelihood_tolerance = (
            equivalent_log_likelihood_tolerance
        )
        self.log_weight_forgetting = log_weight_forgetting
        self.lock_at_hypothesis_count = lock_at_hypothesis_count
        self._log_weights: dict[BodyGraphCandidate, float] = {}
        self._last_log_likelihoods: dict[BodyGraphCandidate, float] = {}
        self._candidate_space_locked = False

    @property
    def learnable_parameter_count(self) -> int:
        return self.maximum_hypotheses

    @property
    def active_state_bytes(self) -> int:
        # Three node ids, one log weight, one likelihood per hypothesis.
        return self.maximum_hypotheses * 5 * 8

    @property
    def estimated_mac_per_step(self) -> int:
        return self.maximum_hypotheses * 32

    def update_from_entity_graph(
        self,
        graph: OnlineEntityGraph,
    ) -> None:
        candidates = self.discover_candidates(graph)
        self.update(candidates, graph.causal_prediction_errors())

    def update(
        self,
        candidates: Iterable[BodyGraphCandidate],
        prediction_errors: dict[int, float],
    ) -> None:
        discovered = set(candidates)
        if self._candidate_space_locked:
            resolved = tuple(sorted(self._log_weights))
        else:
            resolved = tuple(
                sorted(discovered | set(self._log_weights))
            )[: self.maximum_hypotheses]
        if not resolved:
            self._log_weights = {}
            self._last_log_likelihoods = {}
            return
        if set(resolved) != set(self._log_weights):
            # Candidate discovery is asynchronous.  When a newly observable
            # complete graph enters the categorical space, evidence collected
            # before it existed is not comparable, so restart from an
            # exchangeable prior.
            probability = 1.0 / len(resolved)
            self._log_weights = {
                candidate: log(probability) for candidate in resolved
            }
        if (
            self.lock_at_hypothesis_count is not None
            and len(resolved) == self.lock_at_hypothesis_count
        ):
            self._candidate_space_locked = True
        complete_evidence = all(
            all(
                index in prediction_errors
                for index in (candidate.joint, candidate.endpoint)
            )
            for candidate in resolved
        )
        likelihoods = {
            candidate: self._candidate_log_likelihood(
                candidate, prediction_errors
            )
            for candidate in resolved
        }
        if not complete_evidence:
            likelihoods = {candidate: 0.0 for candidate in resolved}
        finite = [value for value in likelihoods.values() if np.isfinite(value)]
        if finite and max(finite) - min(finite) <= (
            self.equivalent_log_likelihood_tolerance
        ):
            # Equal observable evidence must not arbitrarily break symmetry.
            likelihoods = {candidate: 0.0 for candidate in resolved}
        for candidate in resolved:
            self._log_weights[candidate] = (
                self.log_weight_forgetting * self._log_weights[candidate]
                + likelihoods[candidate]
            )
        self._last_log_likelihoods = likelihoods
        self._normalize()

    def discover_candidates(
        self,
        graph: OnlineEntityGraph,
    ) -> tuple[BodyGraphCandidate, ...]:
        matrices = graph.control_matrices()
        edges = graph.rigid_edges(
            minimum_samples=12,
            maximum_variance=0.03,
            maximum_length=3.35,
        )
        neighbors: dict[int, set[int]] = {}
        for left, right in edges:
            neighbors.setdefault(left, set()).add(right)
            neighbors.setdefault(right, set()).add(left)
        roles = {
            index: self._role_strengths(matrix)
            for index, matrix in matrices.items()
        }
        candidates = []
        for base, (base_shoulder, base_elbow) in roles.items():
            if max(base_shoulder, base_elbow) > self.base_control_maximum:
                continue
            for joint in neighbors.get(base, set()):
                joint_shoulder, joint_elbow = roles.get(joint, (0.0, 0.0))
                if (
                    joint_shoulder < self.joint_shoulder_minimum
                    or joint_elbow > self.joint_elbow_maximum
                ):
                    continue
                for endpoint in neighbors.get(joint, set()) - {base}:
                    _, endpoint_elbow = roles.get(endpoint, (0.0, 0.0))
                    if endpoint_elbow < self.endpoint_elbow_minimum:
                        continue
                    candidates.append(
                        BodyGraphCandidate(base, joint, endpoint)
                    )
        return tuple(sorted(set(candidates)))[: self.maximum_hypotheses]

    def hypotheses(self) -> tuple[WeightedBodyGraph, ...]:
        probabilities = self.probability_map()
        return tuple(
            WeightedBodyGraph(candidate, probabilities[candidate])
            for candidate in sorted(probabilities)
        )

    def probability_map(self) -> dict[BodyGraphCandidate, float]:
        if not self._log_weights:
            return {}
        values = np.asarray(list(self._log_weights.values()), dtype=np.float64)
        values -= float(values.max())
        weights = np.exp(values)
        weights /= float(weights.sum())
        return {
            candidate: float(weight)
            for candidate, weight in zip(
                self._log_weights, weights, strict=True
            )
        }

    def entropy(self) -> float:
        return -sum(
            probability * log(max(probability, 1e-12))
            for probability in self.probability_map().values()
        )

    def last_log_likelihoods(self) -> dict[BodyGraphCandidate, float]:
        return dict(self._last_log_likelihoods)

    def _candidate_log_likelihood(
        self,
        candidate: BodyGraphCandidate,
        errors: dict[int, float],
    ) -> float:
        relevant = [
            errors[index]
            for index in (candidate.joint, candidate.endpoint)
            if index in errors
        ]
        if len(relevant) < 2:
            return 0.0
        squared = sum(error * error for error in relevant)
        return -0.5 * squared / (self.residual_sigma**2)

    @staticmethod
    def _role_strengths(matrix: np.ndarray) -> tuple[float, float]:
        shoulder = max(
            float(np.linalg.norm(matrix[:, column])) for column in (0, 1)
        )
        elbow = max(
            float(np.linalg.norm(matrix[:, column])) for column in (2, 3)
        )
        return shoulder, elbow

    def _normalize(self) -> None:
        maximum = max(self._log_weights.values())
        normalizer = maximum + log(
            sum(exp(value - maximum) for value in self._log_weights.values())
        )
        for candidate in self._log_weights:
            self._log_weights[candidate] -= normalizer

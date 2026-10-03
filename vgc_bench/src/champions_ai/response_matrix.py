"""Turn-response matrix for Champions AI.

This module combines:
- our legal joint actions,
- estimated opponent action probabilities,
- a pluggable outcome evaluator.

It does not yet simulate Pokemon mechanics by itself. Instead, it provides the
mathematical structure that later damage, speed, KO, and state-transition
logic will plug into.
"""

from collections.abc import Callable
from dataclasses import dataclass

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.opponent_model import ActionProbability


OutcomeEvaluator = Callable[[JointAction, JointAction], float]


@dataclass(frozen=True)
class ResponseCell:
    """One pairing of our action and one opponent response."""

    our_action: JointAction
    opponent_action: JointAction
    opponent_probability: float
    outcome_value: float
    weighted_value: float


@dataclass(frozen=True)
class ActionResponseSummary:
    """Expected value of one of our actions across opponent responses."""

    our_action: JointAction
    expected_value: float
    worst_case_value: float
    best_case_value: float
    cells: tuple[ResponseCell, ...]

    @property
    def most_likely_response(self) -> ResponseCell:
        if not self.cells:
            raise ValueError("summary has no response cells")
        return max(
            self.cells,
            key=lambda cell: (
                cell.opponent_probability,
                cell.opponent_action.label,
            ),
        )


def _validate_probability_distribution(
    opponent_estimates: tuple[ActionProbability, ...],
) -> None:
    if not opponent_estimates:
        raise ValueError("at least one opponent action estimate is required")

    if any(
        estimate.probability < 0 or estimate.probability > 1
        for estimate in opponent_estimates
    ):
        raise ValueError("opponent action probabilities must be between 0 and 1")

    total = sum(estimate.probability for estimate in opponent_estimates)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "opponent action probabilities must sum to 1.0 "
            f"(received {total})"
        )


def build_response_matrix(
    our_actions: tuple[JointAction, ...],
    opponent_estimates: tuple[ActionProbability, ...],
    evaluator: OutcomeEvaluator,
) -> tuple[ActionResponseSummary, ...]:
    """Evaluate each of our actions against every opponent response.

    The evaluator returns a numeric value for one action pairing. At this stage
    that value is generic. Later it can be replaced with a calibrated position
    value or true conditional win probability.
    """

    if not our_actions:
        return ()

    _validate_probability_distribution(opponent_estimates)

    summaries: list[ActionResponseSummary] = []

    for our_action in our_actions:
        cells: list[ResponseCell] = []

        for opponent_estimate in opponent_estimates:
            opponent_action = opponent_estimate.candidate.action
            probability = opponent_estimate.probability
            outcome_value = float(evaluator(our_action, opponent_action))

            cells.append(
                ResponseCell(
                    our_action=our_action,
                    opponent_action=opponent_action,
                    opponent_probability=probability,
                    outcome_value=outcome_value,
                    weighted_value=probability * outcome_value,
                )
            )

        expected_value = sum(cell.weighted_value for cell in cells)

        summaries.append(
            ActionResponseSummary(
                our_action=our_action,
                expected_value=expected_value,
                worst_case_value=min(cell.outcome_value for cell in cells),
                best_case_value=max(cell.outcome_value for cell in cells),
                cells=tuple(cells),
            )
        )

    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                -summary.expected_value,
                summary.our_action.label,
            ),
        )
    )

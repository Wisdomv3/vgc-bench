"""Mechanics-backed action-pair and response-matrix evaluation.

This module is the bridge between exact/probabilistic turn simulation and the
existing response matrix. It still returns a transparent position heuristic,
not a calibrated match win probability.
"""

from dataclasses import dataclass
from typing import Callable

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.hidden_scenarios import HiddenStateScenario
from vgc_bench.src.champions_ai.matchup import MoveProfile
from vgc_bench.src.champions_ai.opponent_model import ActionProbability
from vgc_bench.src.champions_ai.position_value import (
    PositionWeights,
    evaluate_position,
)
from vgc_bench.src.champions_ai.response_matrix import (
    ActionResponseSummary,
    build_response_matrix,
)
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_branching import (
    BranchingPolicy,
    TurnOutcomeDistribution,
    WeightedTurnOutcome,
    simulate_turn_distribution,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


@dataclass(frozen=True)
class MechanicsResponseAnalysis:
    """Response matrix plus the exact action-pair simulations behind it."""

    summaries: tuple[ActionResponseSummary, ...]
    pair_evaluations: tuple["ActionPairEvaluation", ...]


@dataclass(frozen=True)
class ActionPairEvaluation:
    """Expected simulated value for one our-action/opponent-action pairing."""

    our_action: JointAction
    opponent_action: JointAction
    initial_score: float
    expected_final_score: float
    expected_score_delta: float
    worst_final_score: float
    best_final_score: float
    outcome_count: int
    distribution: TurnOutcomeDistribution
    model: str = "mechanics_position_v0"


def evaluate_action_pair(
    initial_state: ExactTurnState,
    our_action: JointAction,
    opponent_action: JointAction,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    gen: int = 9,
) -> ActionPairEvaluation:
    """Simulate one action pairing and evaluate its weighted resulting states."""

    distribution = simulate_turn_distribution(
        initial_state,
        our_action,
        opponent_action,
        speed_states,
        move_profiles,
        policy=branching_policy,
        gen=gen,
    )
    if not distribution.outcomes:
        raise ValueError("turn simulation produced no outcomes")

    initial_score = evaluate_position(
        initial_state,
        weights=position_weights,
    ).score

    scored_outcomes = tuple(
        (
            outcome.probability,
            evaluate_position(
                outcome.result.state,
                weights=position_weights,
            ).score,
        )
        for outcome in distribution.outcomes
    )

    expected_final_score = sum(
        probability * score
        for probability, score in scored_outcomes
    )

    return ActionPairEvaluation(
        our_action=our_action,
        opponent_action=opponent_action,
        initial_score=initial_score,
        expected_final_score=expected_final_score,
        expected_score_delta=expected_final_score - initial_score,
        worst_final_score=min(score for _probability, score in scored_outcomes),
        best_final_score=max(score for _probability, score in scored_outcomes),
        outcome_count=len(distribution.outcomes),
        distribution=distribution,
    )


def evaluate_action_pair_scenarios(
    scenarios: tuple[HiddenStateScenario, ...],
    our_action: JointAction,
    opponent_action: JointAction,
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    gen: int = 9,
) -> ActionPairEvaluation:
    """Average one action pairing across conditioned hidden-set worlds."""

    if not scenarios:
        raise ValueError("hidden-set scenario provider produced no scenarios")

    total_probability = sum(
        scenario.probability
        for scenario in scenarios
    )
    if total_probability <= 0:
        raise ValueError("hidden-set scenario probability mass is zero")

    evaluations: list[tuple[float, ActionPairEvaluation]] = []
    combined_outcomes: list[WeightedTurnOutcome] = []

    for scenario in scenarios:
        scenario_probability = (
            scenario.probability / total_probability
        )
        evaluation = evaluate_action_pair(
            scenario.state,
            our_action,
            opponent_action,
            scenario.speed_states,
            move_profiles,
            branching_policy=branching_policy,
            position_weights=position_weights,
            gen=gen,
        )
        evaluations.append(
            (scenario_probability, evaluation)
        )

        for outcome in evaluation.distribution.outcomes:
            combined_outcomes.append(
                WeightedTurnOutcome(
                    probability=(
                        scenario_probability
                        * outcome.probability
                    ),
                    result=outcome.result,
                    decisions=outcome.decisions,
                )
            )

    distribution = TurnOutcomeDistribution(
        tuple(
            sorted(
                combined_outcomes,
                key=lambda outcome: -outcome.probability,
            )
        )
    )
    if abs(distribution.total_probability - 1.0) > 1e-9:
        raise RuntimeError(
            "hidden-set mechanics probability mass does not sum to 1.0 "
            f"(received {distribution.total_probability})"
        )

    initial_score = sum(
        probability * evaluation.initial_score
        for probability, evaluation in evaluations
    )
    expected_final_score = sum(
        probability * evaluation.expected_final_score
        for probability, evaluation in evaluations
    )

    return ActionPairEvaluation(
        our_action=our_action,
        opponent_action=opponent_action,
        initial_score=initial_score,
        expected_final_score=expected_final_score,
        expected_score_delta=expected_final_score - initial_score,
        worst_final_score=min(
            evaluation.worst_final_score
            for _probability, evaluation in evaluations
        ),
        best_final_score=max(
            evaluation.best_final_score
            for _probability, evaluation in evaluations
        ),
        outcome_count=len(distribution.outcomes),
        distribution=distribution,
        model="mechanics_position_hidden_v0",
    )


def build_mechanics_response_analysis(
    initial_state: ExactTurnState,
    our_actions: tuple[JointAction, ...],
    opponent_estimates: tuple[ActionProbability, ...],
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    scenario_provider: Callable[
        [JointAction],
        tuple[HiddenStateScenario, ...],
    ] | None = None,
    gen: int = 9,
) -> MechanicsResponseAnalysis:
    """Evaluate the full action matrix once and retain every pair simulation."""

    pair_evaluations: dict[
        tuple[tuple, tuple],
        ActionPairEvaluation,
    ] = {}

    def evaluator(
        our_action: JointAction,
        opponent_action: JointAction,
    ) -> float:
        key = (our_action.key, opponent_action.key)
        if key not in pair_evaluations:
            if scenario_provider is None:
                pair_evaluations[key] = evaluate_action_pair(
                    initial_state,
                    our_action,
                    opponent_action,
                    speed_states,
                    move_profiles,
                    branching_policy=branching_policy,
                    position_weights=position_weights,
                    gen=gen,
                )
            else:
                pair_evaluations[key] = evaluate_action_pair_scenarios(
                    scenario_provider(opponent_action),
                    our_action,
                    opponent_action,
                    move_profiles,
                    branching_policy=branching_policy,
                    position_weights=position_weights,
                    gen=gen,
                )
        return pair_evaluations[key].expected_score_delta

    summaries = build_response_matrix(
        our_actions,
        opponent_estimates,
        evaluator,
    )

    ordered_pairs = tuple(
        pair_evaluations[
            (summary.our_action.key, cell.opponent_action.key)
        ]
        for summary in summaries
        for cell in summary.cells
    )

    return MechanicsResponseAnalysis(
        summaries=summaries,
        pair_evaluations=ordered_pairs,
    )


def build_mechanics_response_matrix(
    initial_state: ExactTurnState,
    our_actions: tuple[JointAction, ...],
    opponent_estimates: tuple[ActionProbability, ...],
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    scenario_provider: Callable[
        [JointAction],
        tuple[HiddenStateScenario, ...],
    ] | None = None,
    gen: int = 9,
) -> tuple[ActionResponseSummary, ...]:
    """Rank our actions using simulated expected position-score change."""

    return build_mechanics_response_analysis(
        initial_state,
        our_actions,
        opponent_estimates,
        speed_states,
        move_profiles,
        branching_policy=branching_policy,
        position_weights=position_weights,
        scenario_provider=scenario_provider,
        gen=gen,
    ).summaries

"""Mechanics-backed action-pair and response-matrix evaluation.

This module is the bridge between exact/probabilistic turn simulation and the
existing response matrix. It still returns a transparent position heuristic,
not a calibrated match win probability.
"""

from dataclasses import dataclass

from vgc_bench.src.champions_ai.actions import JointAction
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
    simulate_turn_distribution,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


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


def build_mechanics_response_matrix(
    initial_state: ExactTurnState,
    our_actions: tuple[JointAction, ...],
    opponent_estimates: tuple[ActionProbability, ...],
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    gen: int = 9,
) -> tuple[ActionResponseSummary, ...]:
    """Rank our actions using simulated expected position-score change."""

    cache: dict[tuple[tuple, tuple], float] = {}

    def evaluator(
        our_action: JointAction,
        opponent_action: JointAction,
    ) -> float:
        key = (our_action.key, opponent_action.key)
        if key not in cache:
            cache[key] = evaluate_action_pair(
                initial_state,
                our_action,
                opponent_action,
                speed_states,
                move_profiles,
                branching_policy=branching_policy,
                position_weights=position_weights,
                gen=gen,
            ).expected_score_delta
        return cache[key]

    return build_response_matrix(
        our_actions,
        opponent_estimates,
        evaluator,
    )

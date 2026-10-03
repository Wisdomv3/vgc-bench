"""Exact probability branching for one Champions AI turn.

The deterministic simulator raises RandomDecisionRequired whenever it reaches a
random event whose result is not fixed. This module recursively answers those
requests, replays the turn, and returns weighted terminal outcomes.

Currently supported probability branches:
- independent damage rolls, grouped by identical damage values;
- move accuracy for each target;
- modern critical-hit odds;
- damaging-move secondary effects, including flinches and stat/status effects;
- repeated Protect/Detect-style success odds;
- exact Speed ties.
"""

from dataclasses import dataclass

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.matchup import MoveProfile
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    RandomChoiceKey,
    RandomChoiceValue,
    RandomDecisionRequired,
    TurnSimulationConfig,
    TurnSimulationResult,
    simulate_turn,
)


class BranchLimitExceeded(RuntimeError):
    """Raised when exact branching grows beyond the configured safety limit."""


@dataclass(frozen=True)
class BranchingPolicy:
    branch_damage_rolls: bool = True
    branch_accuracy: bool = True
    branch_critical_hits: bool = True
    branch_secondary_effects: bool = True
    branch_protect: bool = True
    branch_speed_ties: bool = True
    fixed_damage_roll_index: int = 7
    merge_equivalent_states: bool = True
    max_nodes: int = 200_000

    def __post_init__(self) -> None:
        if not 0 <= self.fixed_damage_roll_index <= 15:
            raise ValueError("fixed_damage_roll_index must be between 0 and 15")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")


@dataclass(frozen=True)
class WeightedTurnOutcome:
    probability: float
    result: TurnSimulationResult
    decisions: tuple[tuple[RandomChoiceKey, RandomChoiceValue], ...]


@dataclass(frozen=True)
class TurnOutcomeDistribution:
    outcomes: tuple[WeightedTurnOutcome, ...]

    @property
    def total_probability(self) -> float:
        return sum(outcome.probability for outcome in self.outcomes)

    @property
    def most_likely(self) -> WeightedTurnOutcome:
        if not self.outcomes:
            raise ValueError("distribution has no outcomes")
        return max(
            self.outcomes,
            key=lambda outcome: outcome.probability,
        )


def _profile_signature(profile) -> tuple:
    return (
        profile.current_hp,
        profile.max_hp,
        profile.status,
        tuple(sorted(profile.boosts.items())),
        profile.item,
        profile.ability,
        tuple(profile.types),
    )


def _state_signature(state: ExactTurnState) -> tuple:
    profiles = tuple(
        sorted(
            (
                side.value,
                name,
                _profile_signature(profile),
            )
            for (side, name), profile in state.profiles.items()
        )
    )
    active_slots = tuple(
        sorted(
            (
                side.value,
                slot,
                name,
            )
            for (side, slot), name in state.active_slots.items()
        )
    )
    tailwind = tuple(sorted(side.value for side in state.tailwind_sides))
    protect_streaks = tuple(
        sorted(
            (
                side.value,
                name,
                streak,
            )
            for (side, name), streak in state.protect_streaks.items()
            if streak
        )
    )
    toxic_stages = tuple(
        sorted(
            (
                side.value,
                name,
                stage,
            )
            for (side, name), stage in state.toxic_stages.items()
            if stage
        )
    )
    return (
        profiles,
        active_slots,
        state.weather,
        state.terrain,
        state.trick_room,
        tailwind,
        protect_streaks,
        toxic_stages,
        tuple(sorted(state.field_conditions)),
    )


def _merge_equivalent(
    outcomes: list[WeightedTurnOutcome],
) -> tuple[WeightedTurnOutcome, ...]:
    merged: dict[tuple, WeightedTurnOutcome] = {}

    for outcome in outcomes:
        signature = _state_signature(outcome.result.state)
        if signature not in merged:
            merged[signature] = outcome
            continue

        previous = merged[signature]
        merged[signature] = WeightedTurnOutcome(
            probability=previous.probability + outcome.probability,
            result=previous.result,
            decisions=previous.decisions,
        )

    return tuple(
        sorted(
            merged.values(),
            key=lambda outcome: -outcome.probability,
        )
    )


def simulate_turn_distribution(
    initial_state: ExactTurnState,
    our_action: JointAction,
    opponent_action: JointAction,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    policy: BranchingPolicy | None = None,
    protect_success_overrides: dict[tuple[TurnSide, str], bool] | None = None,
    gen: int = 9,
) -> TurnOutcomeDistribution:
    """Return the exact weighted outcomes reachable under the enabled branches."""

    policy = policy or BranchingPolicy()
    protect_success_overrides = protect_success_overrides or {}

    pending: list[
        tuple[
            dict[RandomChoiceKey, RandomChoiceValue],
            float,
        ]
    ] = [({}, 1.0)]
    leaves: list[WeightedTurnOutcome] = []
    visited_nodes = 0

    while pending:
        decisions, probability = pending.pop()
        if probability <= 0:
            continue

        visited_nodes += 1
        if visited_nodes > policy.max_nodes:
            raise BranchLimitExceeded(
                "exact turn branching exceeded "
                f"{policy.max_nodes} nodes; use a narrower policy or sampling"
            )

        config = TurnSimulationConfig(
            damage_roll_index=(
                None
                if policy.branch_damage_rolls
                else policy.fixed_damage_roll_index
            ),
            protect_success=protect_success_overrides,
            random_choices=decisions,
            branch_damage_rolls=policy.branch_damage_rolls,
            branch_accuracy=policy.branch_accuracy,
            branch_critical_hits=policy.branch_critical_hits,
            branch_secondary_effects=policy.branch_secondary_effects,
            branch_protect=policy.branch_protect,
            branch_speed_ties=policy.branch_speed_ties,
        )

        try:
            result = simulate_turn(
                initial_state,
                our_action,
                opponent_action,
                speed_states,
                move_profiles,
                config,
                gen=gen,
            )
        except RandomDecisionRequired as request:
            for value, option_probability in request.options:
                if option_probability <= 0:
                    continue

                child_decisions = dict(decisions)
                child_decisions[request.key] = value
                pending.append(
                    (
                        child_decisions,
                        probability * option_probability,
                    )
                )
            continue

        leaves.append(
            WeightedTurnOutcome(
                probability=probability,
                result=result,
                decisions=tuple(sorted(decisions.items())),
            )
        )

    if policy.merge_equivalent_states:
        final_outcomes = _merge_equivalent(leaves)
    else:
        final_outcomes = tuple(
            sorted(
                leaves,
                key=lambda outcome: -outcome.probability,
            )
        )

    distribution = TurnOutcomeDistribution(final_outcomes)
    if final_outcomes and abs(distribution.total_probability - 1.0) > 1e-9:
        raise RuntimeError(
            "turn outcome probability mass does not sum to 1.0 "
            f"(received {distribution.total_probability})"
        )

    return distribution

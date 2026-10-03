"""Transparent one-turn position evaluator for Champions AI.

This is not a win-probability model. It converts concrete board consequences
from the turn simulator into a stable numeric score so legal actions can be
compared before the later calibrated match-value model exists.

Positive scores favor the player; negative scores favor the opponent.
"""

from dataclasses import dataclass

from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


@dataclass(frozen=True)
class PositionWeights:
    """Explicit weights for the temporary position heuristic."""

    pokemon_alive: float = 100.0
    total_hp_fraction: float = 25.0
    status: float = 6.0
    active_stat_stage: float = 2.0
    tailwind: float = 5.0


@dataclass(frozen=True)
class PositionFeatures:
    player_alive: int
    opponent_alive: int
    player_hp_fraction: float
    opponent_hp_fraction: float
    player_status_count: int
    opponent_status_count: int
    player_active_stage_sum: int
    opponent_active_stage_sum: int
    player_tailwind: bool
    opponent_tailwind: bool


@dataclass(frozen=True)
class PositionEvaluation:
    score: float
    features: PositionFeatures
    model: str = "position_heuristic_v0"


def _side_profiles(state: ExactTurnState, side: TurnSide):
    return tuple(
        profile
        for (profile_side, _name), profile in state.profiles.items()
        if profile_side is side
    )


def _alive_count(state: ExactTurnState, side: TurnSide) -> int:
    return sum(
        profile.current_hp > 0
        for profile in _side_profiles(state, side)
    )


def _hp_fraction(state: ExactTurnState, side: TurnSide) -> float:
    profiles = _side_profiles(state, side)
    if not profiles:
        return 0.0

    return sum(
        profile.current_hp / profile.max_hp
        for profile in profiles
    )


def _status_count(state: ExactTurnState, side: TurnSide) -> int:
    return sum(
        profile.current_hp > 0 and profile.status is not None
        for profile in _side_profiles(state, side)
    )


def _active_stage_sum(state: ExactTurnState, side: TurnSide) -> int:
    total = 0
    for (active_side, _slot), name in state.active_slots.items():
        if active_side is not side:
            continue

        profile = state.profile(side, name)
        if profile.current_hp <= 0:
            continue

        total += sum(
            int(profile.boosts.get(stat, 0))
            for stat in ("atk", "def", "spa", "spd", "spe")
        )

    return total


def position_features(state: ExactTurnState) -> PositionFeatures:
    """Extract concrete board features from an exact simulated state."""

    return PositionFeatures(
        player_alive=_alive_count(state, TurnSide.PLAYER),
        opponent_alive=_alive_count(state, TurnSide.OPPONENT),
        player_hp_fraction=_hp_fraction(state, TurnSide.PLAYER),
        opponent_hp_fraction=_hp_fraction(state, TurnSide.OPPONENT),
        player_status_count=_status_count(state, TurnSide.PLAYER),
        opponent_status_count=_status_count(state, TurnSide.OPPONENT),
        player_active_stage_sum=_active_stage_sum(state, TurnSide.PLAYER),
        opponent_active_stage_sum=_active_stage_sum(state, TurnSide.OPPONENT),
        player_tailwind=TurnSide.PLAYER in state.tailwind_sides,
        opponent_tailwind=TurnSide.OPPONENT in state.tailwind_sides,
    )


def evaluate_position(
    state: ExactTurnState,
    *,
    weights: PositionWeights | None = None,
) -> PositionEvaluation:
    """Score one exact board state from the player's perspective."""

    weights = weights or PositionWeights()
    features = position_features(state)

    alive_advantage = features.player_alive - features.opponent_alive
    hp_advantage = (
        features.player_hp_fraction
        - features.opponent_hp_fraction
    )
    status_advantage = (
        features.opponent_status_count
        - features.player_status_count
    )
    stage_advantage = (
        features.player_active_stage_sum
        - features.opponent_active_stage_sum
    )
    tailwind_advantage = (
        int(features.player_tailwind)
        - int(features.opponent_tailwind)
    )

    score = (
        weights.pokemon_alive * alive_advantage
        + weights.total_hp_fraction * hp_advantage
        + weights.status * status_advantage
        + weights.active_stat_stage * stage_advantage
        + weights.tailwind * tailwind_advantage
    )

    return PositionEvaluation(
        score=score,
        features=features,
    )

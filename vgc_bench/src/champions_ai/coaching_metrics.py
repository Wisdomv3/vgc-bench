"""Immutable, next-turn event probabilities from the existing simulations.

These are weighted model probabilities, conditional on supplied opponent
responses, hidden worlds and enabled RNG branches. They are not match-win
probabilities. State-based events remain valid when equivalent outcome states
are merged; representative event logs cannot supply reliable hit/miss odds.
"""

from dataclasses import dataclass
from math import isfinite

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.mechanics_evaluator import MechanicsResponseAnalysis
from vgc_bench.src.champions_ai.opponent_model import ActionProbability
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.turn_order import TurnSide


@dataclass(frozen=True)
class TurnOutcomeMetrics:
    """End-of-this-turn odds; survival includes a successful switch to the bench."""

    action: JointAction
    opponent_ko_probability: float
    player_ko_probability: float
    starting_actives_survive_probability: float | None
    both_starting_actives_faint_probability: float | None
    player_tailwind_probability: float
    opponent_faint_probabilities: tuple[tuple[str, float], ...]
    player_survival_probabilities: tuple[tuple[str, float], ...]
    horizon: int = 1


def build_turn_outcome_metrics(
    snapshot: DecisionSnapshot,
    analysis: MechanicsResponseAnalysis,
    opponent_estimates: tuple[ActionProbability, ...],
) -> tuple[TurnOutcomeMetrics, ...]:
    """Aggregate root distributions before multi-turn summaries replace scores.

    Previously fainted Pokemon do not count as new KOs. An absent profile or
    incomplete probability mass is an error, rather than fabricated certainty.
    """

    response_mass: dict[tuple, float] = {}
    for estimate in opponent_estimates:
        probability = estimate.probability
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                "opponent probabilities must be finite and between 0 and 1"
            )
        key = estimate.candidate.action.key
        response_mass[key] = response_mass.get(key, 0.0) + probability
    if abs(sum(response_mass.values()) - 1.0) > 1e-9:
        raise ValueError("opponent probabilities must sum to 1")

    alive = {
        side: tuple(
            pokemon.name
            for pokemon in side_snapshot.pokemon
            if not pokemon.fainted and pokemon.hp_percent > 0
        )
        for side, side_snapshot in (
            (TurnSide.PLAYER, snapshot.player),
            (TurnSide.OPPONENT, snapshot.opponent),
        )
    }
    starters = tuple(
        name
        for _slot, name in snapshot.player.active_slots
        if name in alive[TurnSide.PLAYER]
    )
    pairs = {
        (pair.our_action.key, pair.opponent_action.key): pair
        for pair in analysis.pair_evaluations
    }
    output = []
    for summary in analysis.summaries:
        opponent_faints = dict.fromkeys(alive[TurnSide.OPPONENT], 0.0)
        player_survivals = dict.fromkeys(starters, 0.0)
        any_foe_ko = any_player_ko = all_survive = both_faint = tailwind = mass = 0.0
        for response_key, probability in response_mass.items():
            if probability == 0:
                continue
            pair = pairs.get((summary.our_action.key, response_key))
            if pair is None:
                raise ValueError("missing positive-probability response simulation")
            if not pair.distribution.outcomes:
                raise ValueError("turn distribution has no outcomes")
            if abs(pair.distribution.total_probability - 1.0) > 1e-9:
                raise ValueError("turn outcome probabilities must sum to 1")
            for outcome in pair.distribution.outcomes:
                if (
                    not isfinite(outcome.probability)
                    or not 0 <= outcome.probability <= 1
                ):
                    raise ValueError(
                        "turn probabilities must be finite and between 0 and 1"
                    )
                weight = probability * outcome.probability
                if weight == 0:
                    continue
                state = outcome.result.state
                survives = {
                    side: {
                        name: state.profile(side, name).current_hp > 0 for name in names
                    }
                    for side, names in alive.items()
                }
                mass += weight
                any_foe_ko += weight * any(
                    not ok for ok in survives[TurnSide.OPPONENT].values()
                )
                any_player_ko += weight * any(
                    not ok for ok in survives[TurnSide.PLAYER].values()
                )
                all_survive += weight * all(
                    survives[TurnSide.PLAYER][name] for name in starters
                )
                both_faint += weight * all(
                    not survives[TurnSide.PLAYER][name] for name in starters
                )
                tailwind += weight * (TurnSide.PLAYER in state.tailwind_sides)
                for name in opponent_faints:
                    opponent_faints[name] += weight * (
                        not survives[TurnSide.OPPONENT][name]
                    )
                for name in player_survivals:
                    player_survivals[name] += weight * survives[TurnSide.PLAYER][name]
        if abs(mass - 1.0) > 1e-9:
            raise ValueError("weighted turn probability mass must sum to 1")

        def odds(value):
            return min(1.0, max(0.0, value))

        output.append(
            TurnOutcomeMetrics(
                action=summary.our_action,
                opponent_ko_probability=odds(any_foe_ko),
                player_ko_probability=odds(any_player_ko),
                starting_actives_survive_probability=odds(all_survive)
                if starters
                else None,
                both_starting_actives_faint_probability=odds(both_faint)
                if len(starters) == 2
                else None,
                player_tailwind_probability=odds(tailwind),
                opponent_faint_probabilities=tuple(
                    (name, odds(p)) for name, p in opponent_faints.items()
                ),
                player_survival_probabilities=tuple(
                    (name, odds(p)) for name, p in player_survivals.items()
                ),
            )
        )
    return tuple(output)


def probability_assumptions(policy) -> tuple[str, ...]:
    """Expose disabled randomness instead of presenting fixed scenarios as exact."""

    from vgc_bench.src.champions_ai.turn_branching import BranchingPolicy

    policy = policy or BranchingPolicy()
    features = (
        ("branch_damage_rolls", "damage rolls"),
        ("branch_accuracy", "accuracy"),
        ("branch_critical_hits", "critical hits"),
        ("branch_secondary_effects", "secondary effects"),
        ("branch_before_move_status", "before-move status"),
        ("branch_protect", "repeated Protect"),
        ("branch_speed_ties", "Speed ties"),
    )
    fixed = tuple(
        label for attribute, label in features if not getattr(policy, attribute)
    )
    return () if not fixed else ("Fixed RNG assumptions: " + ", ".join(fixed) + ".",)

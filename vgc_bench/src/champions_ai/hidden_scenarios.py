"""Hidden-set state scenarios for mechanics evaluation.

Opponent actions can imply information about the hidden set. For example, an
unrevealed Heat Wave action is only compatible with hypotheses that contain
Heat Wave. This module conditions the hidden-set posterior on the modeled
opponent action, replaces the opponent's exact stats/item/ability with each
compatible set, preserves all public battle information, and refreshes Speed.
"""

from dataclasses import dataclass, field, replace
from itertools import product

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction
from vgc_bench.src.champions_ai.hidden_sets import (
    SetHypothesis,
    WeightedHypothesis,
    posterior_hypotheses,
)
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
)
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.speed_context import (
    refresh_speed_states_for_exact_state,
)
from vgc_bench.src.champions_ai.spread import normalize_move_id
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


CHOICE_ITEMS = {
    "choiceband",
    "choicescarf",
    "choicespecs",
}


@dataclass(frozen=True)
class HiddenStateScenario:
    """One conditioned hidden-set world used by the mechanics simulator."""

    probability: float
    state: ExactTurnState
    speed_states: dict[tuple[TurnSide, str], SpeedState]
    labels: tuple[str, ...]
    move_sets: dict[tuple[TurnSide, str], frozenset[str]] = field(
        default_factory=dict
    )


def _snapshot_by_name(
    snapshot: DecisionSnapshot,
    name: str,
) -> PokemonSnapshot | None:
    normalized = normalize_move_id(name)
    return next(
        (
            pokemon
            for pokemon in snapshot.opponent.pokemon
            if normalize_move_id(pokemon.name) == normalized
        ),
        None,
    )


def _hypotheses_for_name(
    hidden_hypotheses: dict[str, tuple[SetHypothesis, ...]],
    name: str,
) -> tuple[SetHypothesis, ...]:
    if name in hidden_hypotheses:
        return hidden_hypotheses[name]

    normalized = normalize_move_id(name)
    for candidate_name, hypotheses in hidden_hypotheses.items():
        if normalize_move_id(candidate_name) == normalized:
            return hypotheses
    return ()


def _action_requirements(
    opponent_action: JointAction,
) -> tuple[dict[str, set[str]], set[str]]:
    required_moves: dict[str, set[str]] = {}
    relevant_names: set[str] = set()

    for slot_action in (
        opponent_action.first,
        opponent_action.second,
    ):
        if slot_action.actor is not None:
            relevant_names.add(slot_action.actor)

        if (
            slot_action.kind is ActionKind.MOVE
            and slot_action.actor is not None
            and slot_action.move is not None
        ):
            required_moves.setdefault(
                slot_action.actor,
                set(),
            ).add(normalize_move_id(slot_action.move))

        if (
            slot_action.kind is ActionKind.SWITCH
            and slot_action.switch_to is not None
        ):
            relevant_names.add(slot_action.switch_to)

    return required_moves, relevant_names


def _choice_compatible(
    state: ExactTurnState,
    *,
    name: str,
    hypothesis: SetHypothesis,
    required_moves: set[str],
) -> bool:
    if not required_moves:
        return True

    item = normalize_move_id(hypothesis.profile.item or "")
    ability = normalize_move_id(hypothesis.profile.ability or "")
    if item not in CHOICE_ITEMS and ability != "gorillatactics":
        return True

    last_move = state.last_moves.get((TurnSide.OPPONENT, name))
    if last_move in {None, "struggle"}:
        return True

    return required_moves == {normalize_move_id(last_move)}


def _conditioned_hypotheses(
    pokemon_snapshot: PokemonSnapshot,
    hypotheses: tuple[SetHypothesis, ...],
    *,
    required_moves: set[str],
    state: ExactTurnState,
    name: str,
) -> tuple[WeightedHypothesis, ...]:
    posterior = posterior_hypotheses(
        pokemon_snapshot,
        hypotheses,
    )
    if not posterior:
        return ()

    compatible = tuple(
        entry
        for entry in posterior
        if required_moves.issubset(
            entry.hypothesis.normalized_moves
        )
        and _choice_compatible(
            state,
            name=name,
            hypothesis=entry.hypothesis,
            required_moves=required_moves,
        )
    )
    if not compatible:
        return ()

    total = sum(entry.probability for entry in compatible)
    if total <= 0:
        equal = 1.0 / len(compatible)
        return tuple(
            WeightedHypothesis(
                hypothesis=entry.hypothesis,
                probability=equal,
            )
            for entry in compatible
        )

    return tuple(
        WeightedHypothesis(
            hypothesis=entry.hypothesis,
            probability=entry.probability / total,
        )
        for entry in compatible
    )


def _scenario_profile(
    snapshot: PokemonSnapshot,
    public_profile,
    hypothesis: SetHypothesis,
):
    hidden = hypothesis.profile

    if snapshot.fainted or snapshot.hp_percent <= 0:
        current_hp = 0
    else:
        current_hp = int(
            round(hidden.max_hp * snapshot.hp_percent / 100.0)
        )
        current_hp = max(1, min(hidden.max_hp, current_hp))

    return replace(
        hidden,
        name=public_profile.name,
        current_hp=current_hp,
        types=public_profile.types,
        boosts=dict(public_profile.boosts),
        item=(
            snapshot.item
            if snapshot.item is not None
            else hidden.item
        ),
        ability=(
            snapshot.ability
            if snapshot.ability is not None
            else hidden.ability
        ),
        status=public_profile.status,
        gender=(
            public_profile.gender
            if public_profile.gender is not None
            else hidden.gender
        ),
    )


def hidden_state_scenarios(
    snapshot: DecisionSnapshot,
    base_state: ExactTurnState,
    opponent_action: JointAction,
    hidden_hypotheses: dict[str, tuple[SetHypothesis, ...]],
    base_speed_states: dict[tuple[TurnSide, str], SpeedState],
    *,
    max_scenarios: int | None = 64,
) -> tuple[HiddenStateScenario, ...]:
    """Return hidden worlds conditioned on one modeled opponent action."""

    if max_scenarios is not None and max_scenarios <= 0:
        raise ValueError("max_scenarios must be positive")

    required_moves, relevant_names = _action_requirements(
        opponent_action
    )
    # Keep idle and bench Pokemon's hidden sets fixed for future search turns.
    relevant_names.update(
        name for (side, name) in base_state.profiles
        if side is TurnSide.OPPONENT
        and _hypotheses_for_name(hidden_hypotheses, name)
    )

    option_groups: list[
        tuple[
            str,
            tuple[WeightedHypothesis, ...],
            PokemonSnapshot,
        ]
    ] = []

    for name in sorted(
        relevant_names,
        key=normalize_move_id,
    ):
        hypotheses = _hypotheses_for_name(
            hidden_hypotheses,
            name,
        )
        if not hypotheses:
            continue

        pokemon_snapshot = _snapshot_by_name(
            snapshot,
            name,
        )
        if pokemon_snapshot is None:
            raise ValueError(
                f"opponent snapshot is missing hidden-set Pokemon {name}"
            )

        conditioned = _conditioned_hypotheses(
            pokemon_snapshot,
            hypotheses,
            required_moves=required_moves.get(name, set()),
            state=base_state,
            name=name,
        )
        if not conditioned:
            return ()

        option_groups.append(
            (name, conditioned, pokemon_snapshot)
        )

    if not option_groups:
        return (
            HiddenStateScenario(
                probability=1.0,
                state=base_state.copy(),
                speed_states=dict(base_speed_states),
                labels=(),
            ),
        )

    raw: list[
        tuple[
            float,
            tuple[
                tuple[
                    str,
                    WeightedHypothesis,
                    PokemonSnapshot,
                ],
                ...,
            ],
        ]
    ] = []

    groups = [
        tuple(
            (name, entry, pokemon_snapshot)
            for entry in entries
        )
        for name, entries, pokemon_snapshot in option_groups
    ]

    for combination in product(*groups):
        probability = 1.0
        for _name, entry, _snapshot in combination:
            probability *= entry.probability
        if probability > 0:
            raw.append((probability, combination))

    raw.sort(
        key=lambda row: (
            -row[0],
            tuple(
                (
                    normalize_move_id(name),
                    entry.hypothesis.label,
                )
                for name, entry, _snapshot in row[1]
            ),
        )
    )
    if max_scenarios is not None:
        raw = raw[:max_scenarios]

    total = sum(probability for probability, _combo in raw)
    if total <= 0:
        return ()

    scenarios: list[HiddenStateScenario] = []
    for probability, combination in raw:
        state = base_state.copy()
        labels: list[str] = []
        move_sets: dict[tuple[TurnSide, str], frozenset[str]] = {}

        for name, entry, pokemon_snapshot in combination:
            key = (TurnSide.OPPONENT, name)
            if key not in state.profiles:
                raise ValueError(
                    f"exact state is missing hidden-set Pokemon {name}"
                )

            state.profiles[key] = _scenario_profile(
                pokemon_snapshot,
                state.profiles[key],
                entry.hypothesis,
            )
            move_sets[key] = frozenset(entry.hypothesis.normalized_moves)
            labels.append(
                f"{name}:{entry.hypothesis.label}"
            )

        scenarios.append(
            HiddenStateScenario(
                probability=probability / total,
                state=state,
                speed_states=refresh_speed_states_for_exact_state(
                    state,
                    base_speed_states,
                ),
                labels=tuple(labels),
                move_sets=move_sets,
            )
        )

    return tuple(scenarios)

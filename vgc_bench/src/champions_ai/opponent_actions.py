"""Automatic plausible opponent joint-action generation.

The opponent's exact request is private, so this module generates plausible
responses from information we are allowed to know:

- currently active opponent Pokemon;
- revealed moves and PP;
- compatible hidden-set hypotheses;
- revealed, non-fainted bench Pokemon;
- public restriction state already synchronized into ExactTurnState.

The generator does not invent usage statistics. Hidden-set posterior weights and
an explicit switch prior become transparent candidate prior weights.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from poke_env.battle import Move
from poke_env.battle.move_category import MoveCategory

from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.hidden_sets import (
    SetHypothesis,
    posterior_hypotheses,
)
from vgc_bench.src.champions_ai.opponent_model import OpponentActionCandidate
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
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
class _WeightedSlotAction:
    action: SlotAction
    weight: float
    source: str


def _pokemon_label(pokemon) -> str | None:
    if pokemon is None:
        return None
    return pokemon.species or pokemon.name


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


def _move_prior_weights(
    pokemon_snapshot: PokemonSnapshot,
    hypotheses: tuple[SetHypothesis, ...],
) -> dict[str, tuple[float, str]]:
    """Return move inclusion weights from reveals plus compatible hidden sets."""

    revealed = {
        normalize_move_id(move)
        for move in pokemon_snapshot.revealed_moves
    }

    posterior = posterior_hypotheses(
        pokemon_snapshot,
        hypotheses,
    )
    hidden_weights: dict[str, float] = defaultdict(float)
    for entry in posterior:
        for move_id in entry.hypothesis.normalized_moves:
            hidden_weights[move_id] += entry.probability

    move_ids = revealed | set(hidden_weights)
    output: dict[str, tuple[float, str]] = {}
    for move_id in move_ids:
        hidden_weight = hidden_weights.get(move_id, 0.0)
        if move_id in revealed:
            weight = max(1.0, hidden_weight)
            source = "revealed"
            if hidden_weight > 0:
                source = "revealed+hidden-set"
        else:
            weight = hidden_weight
            source = "hidden-set"

        if weight > 0:
            output[move_id] = (weight, source)

    return output


def _active_names(battle, side: TurnSide) -> tuple[str | None, str | None]:
    active = (
        battle.opponent_active_pokemon
        if side is TurnSide.OPPONENT
        else battle.active_pokemon
    )
    names = tuple(_pokemon_label(pokemon) for pokemon in active)
    if len(names) != 2:
        raise ValueError("doubles battle must expose exactly two active slots")
    return names


def _target_variants(
    battle,
    *,
    actor_slot: int,
    move: Move,
) -> tuple[tuple[str | None, int | None, float], ...]:
    """Return target name, simulator-relative position, and neutral target weight."""

    target_name = move.target.name if move.target is not None else "SCRIPTED"
    our_active = _active_names(battle, TurnSide.PLAYER)
    opponent_active = _active_names(battle, TurnSide.OPPONENT)

    self_position = -(actor_slot + 1)
    ally_slot = 1 - actor_slot
    ally_position = -(ally_slot + 1)

    def existing(
        positions: tuple[tuple[str | None, int], ...],
    ) -> list[tuple[str, int]]:
        return [
            (name, position)
            for name, position in positions
            if name is not None
        ]

    foe_positions = existing(
        tuple(
            (name, slot + 1)
            for slot, name in enumerate(our_active)
        )
    )
    ally = (
        []
        if opponent_active[ally_slot] is None
        else [(opponent_active[ally_slot], ally_position)]
    )
    self_target = (
        []
        if opponent_active[actor_slot] is None
        else [(opponent_active[actor_slot], self_position)]
    )

    selectable: list[tuple[str, int]]
    if target_name == "ADJACENT_ALLY":
        selectable = ally
    elif target_name == "ADJACENT_ALLY_OR_SELF":
        selectable = ally + self_target
    elif target_name == "ADJACENT_FOE":
        selectable = foe_positions
    elif target_name in {"ANY", "NORMAL"}:
        selectable = ally + foe_positions
    elif target_name == "RANDOM_NORMAL":
        selectable = foe_positions
    else:
        return ((None, None, 1.0),)

    if not selectable:
        return ()

    weight = 1.0 / len(selectable)
    return tuple(
        (name, position, weight)
        for name, position in selectable
    )


def _move_allowed_by_public_state(
    state: ExactTurnState,
    *,
    actor_name: str,
    move: Move,
) -> bool:
    key = (TurnSide.OPPONENT, actor_name)
    move_id = move.id

    if key in state.must_recharge:
        return False

    encore = state.encore_locks.get(key)
    if encore is not None and move_id != encore[0]:
        return False

    disabled = state.disabled_moves.get(key)
    if disabled is not None and move_id == disabled[0]:
        return False

    if (
        key in state.taunt_turns
        and move.category is MoveCategory.STATUS
        and move_id != "mefirst"
    ):
        return False

    if (
        key in state.tormented
        and state.last_moves.get(key) == move_id
        and move_id != "struggle"
    ):
        return False

    if (
        bool(move.entry.get("flags", {}).get("cantusetwice", False))
        and state.last_moves.get(key) == move_id
    ):
        return False

    pp = state.move_pp.get((TurnSide.OPPONENT, actor_name, move_id))
    if pp is not None and pp <= 0:
        return False

    profile = state.profiles.get(key)
    if profile is not None:
        item = normalize_move_id(profile.item or "")
        ability = normalize_move_id(profile.ability or "")
        locked = item in CHOICE_ITEMS or ability == "gorillatactics"
        last_move = state.last_moves.get(key)
        if (
            locked
            and last_move not in {None, "struggle"}
            and move_id != last_move
        ):
            return False

    return True


def _move_slot_actions(
    battle,
    snapshot: DecisionSnapshot,
    state: ExactTurnState,
    *,
    slot: int,
    actor_name: str,
    hidden_hypotheses: dict[str, tuple[SetHypothesis, ...]],
    gen: int,
) -> tuple[_WeightedSlotAction, ...]:
    pokemon_snapshot = _snapshot_by_name(snapshot, actor_name)
    if pokemon_snapshot is None:
        raise ValueError(
            f"opponent snapshot is missing active Pokemon {actor_name}"
        )

    hypotheses = _hypotheses_for_name(
        hidden_hypotheses,
        actor_name,
    )
    move_weights = _move_prior_weights(
        pokemon_snapshot,
        hypotheses,
    )

    output: list[_WeightedSlotAction] = []
    for move_id, (move_weight, source) in sorted(move_weights.items()):
        try:
            move = Move(move_id, gen)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"cannot resolve opponent move {move_id!r} in generation {gen}"
            ) from exc

        if not _move_allowed_by_public_state(
            state,
            actor_name=actor_name,
            move=move,
        ):
            continue

        targets = _target_variants(
            battle,
            actor_slot=slot,
            move=move,
        )
        for target, target_position, target_weight in targets:
            output.append(
                _WeightedSlotAction(
                    action=SlotAction(
                        slot=slot,
                        kind=ActionKind.MOVE,
                        actor=actor_name,
                        move=move_id,
                        target=target,
                        target_position=target_position,
                    ),
                    weight=move_weight * target_weight,
                    source=source,
                )
            )

    return tuple(output)


def _bench_switches(battle) -> tuple[str, ...]:
    active = {
        normalize_move_id(name)
        for name in _active_names(battle, TurnSide.OPPONENT)
        if name is not None
    }

    seen: set[str] = set()
    output: list[str] = []
    for pokemon in battle.opponent_team.values():
        if getattr(pokemon, "fainted", False):
            continue

        name = _pokemon_label(pokemon)
        if name is None:
            continue

        normalized = normalize_move_id(name)
        if normalized in active or normalized in seen:
            continue

        seen.add(normalized)
        output.append(name)

    return tuple(output)


def _switch_slot_actions(
    *,
    slot: int,
    actor_name: str | None,
    bench: tuple[str, ...],
    switch_prior_weight: float,
) -> tuple[_WeightedSlotAction, ...]:
    return tuple(
        _WeightedSlotAction(
            action=SlotAction(
                slot=slot,
                kind=ActionKind.SWITCH,
                actor=actor_name,
                switch_to=target,
            ),
            weight=switch_prior_weight,
            source="revealed-switch",
        )
        for target in bench
    )


def _pass_slot_action(
    *,
    slot: int,
    actor_name: str | None,
    source: str,
) -> _WeightedSlotAction:
    return _WeightedSlotAction(
        action=SlotAction(
            slot=slot,
            kind=ActionKind.PASS,
            actor=actor_name,
        ),
        weight=1.0,
        source=source,
    )


def _same_switch_target(
    first: SlotAction,
    second: SlotAction,
) -> bool:
    return (
        first.kind is ActionKind.SWITCH
        and second.kind is ActionKind.SWITCH
        and first.switch_to is not None
        and first.switch_to == second.switch_to
    )


def generate_opponent_action_candidates(
    battle,
    snapshot: DecisionSnapshot,
    state: ExactTurnState,
    *,
    hidden_hypotheses: dict[str, tuple[SetHypothesis, ...]] | None = None,
    include_switches: bool = True,
    switch_prior_weight: float = 1.0,
    max_candidates: int | None = None,
    require_move_coverage: bool = True,
    gen: int = 9,
) -> tuple[OpponentActionCandidate, ...]:
    """Generate plausible opponent joint actions from public battle information.

    switch_prior_weight is deliberately explicit; it is not a learned switch
    frequency. Hidden-set weights are inherited from posterior_hypotheses.

    If an active opponent has no revealed or hidden-set move coverage, the
    default is to fail rather than silently pretend the opponent can only switch.
    """

    if switch_prior_weight < 0:
        raise ValueError("switch_prior_weight cannot be negative")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    hidden_hypotheses = hidden_hypotheses or {}
    active = _active_names(battle, TurnSide.OPPONENT)
    bench = _bench_switches(battle) if include_switches else ()

    per_slot: list[tuple[_WeightedSlotAction, ...]] = []
    for slot, actor_name in enumerate(active):
        if actor_name is None:
            switches = _switch_slot_actions(
                slot=slot,
                actor_name=None,
                bench=bench,
                switch_prior_weight=switch_prior_weight,
            )
            per_slot.append(
                switches
                if switches
                else (
                    _pass_slot_action(
                        slot=slot,
                        actor_name=None,
                        source="empty-slot",
                    ),
                )
            )
            continue

        key = (TurnSide.OPPONENT, actor_name)
        if key in state.must_recharge:
            per_slot.append(
                (
                    _pass_slot_action(
                        slot=slot,
                        actor_name=actor_name,
                        source="forced-recharge",
                    ),
                )
            )
            continue

        moves = _move_slot_actions(
            battle,
            snapshot,
            state,
            slot=slot,
            actor_name=actor_name,
            hidden_hypotheses=hidden_hypotheses,
            gen=gen,
        )

        if require_move_coverage and not moves:
            raise ValueError(
                "no plausible moves for active opponent "
                f"{actor_name}; provide compatible hidden-set hypotheses "
                "or revealed move information"
            )

        switches = (
            _switch_slot_actions(
                slot=slot,
                actor_name=actor_name,
                bench=bench,
                switch_prior_weight=switch_prior_weight,
            )
            if include_switches
            else ()
        )
        options = moves + switches
        if not options:
            options = (
                _pass_slot_action(
                    slot=slot,
                    actor_name=actor_name,
                    source="no-action",
                ),
            )
        per_slot.append(options)

    combined_weights: dict[JointAction, float] = defaultdict(float)
    combined_sources: dict[JointAction, set[str]] = defaultdict(set)

    for first, second in product(per_slot[0], per_slot[1]):
        if _same_switch_target(first.action, second.action):
            continue

        action = JointAction(
            first=first.action,
            second=second.action,
        )
        combined_weights[action] += first.weight * second.weight
        combined_sources[action].update((first.source, second.source))

    candidates = [
        OpponentActionCandidate(
            action=action,
            prior_weight=weight,
            source="auto:" + "+".join(sorted(combined_sources[action])),
        )
        for action, weight in combined_weights.items()
        if weight > 0
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.prior_weight,
            candidate.action.label,
        )
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    if not candidates:
        raise ValueError("no plausible opponent joint actions were generated")

    return tuple(candidates)

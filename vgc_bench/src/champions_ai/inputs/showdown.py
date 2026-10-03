"""Pokemon Showdown input adapter for Champions AI.

This module converts poke-env's structured DoubleBattle object into the same
BattleEvent/BattleState representation used by manual and future OBS inputs.
It also synchronizes live poke-env observations into ExactTurnState templates
used by the mechanics simulator.
"""

from dataclasses import replace

from poke_env.battle import DoubleBattle, Effect, Field, SideCondition
from poke_env.data import GenData

from vgc_bench.src.champions_ai.events import BattleEvent
from vgc_bench.src.champions_ai.matchup import CombatantProfile
from vgc_bench.src.champions_ai.inputs.showdown_actions import (
    legal_joint_actions_from_showdown,
)
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.state import BattleState
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


TAILWIND_DURATION = 4
TRICK_ROOM_DURATION = 5


def _remaining_turns(current_turn: int, started_turn: int, duration: int) -> int:
    return max(0, duration - (current_turn - started_turn))


def _pokemon_label(pokemon) -> str:
    species = getattr(pokemon, "species", None)
    if species:
        return str(species)
    return str(pokemon.name)


def _status_name(pokemon) -> str | None:
    status = getattr(pokemon, "status", None)
    if status is None:
        return None
    return status.name.lower()


def showdown_events(battle: DoubleBattle) -> list[BattleEvent]:
    """Return current structured Showdown observations as universal events.

    These events describe the currently known board state. They are not intended
    to reconstruct the exact chronological protocol history; that will be handled
    by the later replay/history adapter.
    """

    events: list[BattleEvent] = [
        BattleEvent(
            type="turn_started",
            turn=battle.turn,
            source="showdown",
        )
    ]

    side_specs = [
        ("player", battle.team, battle.active_pokemon),
        ("opponent", battle.opponent_team, battle.opponent_active_pokemon),
    ]

    for side, team, active in side_specs:
        for slot, pokemon in enumerate(active):
            if pokemon is None:
                continue
            events.append(
                BattleEvent(
                    type="switch_in",
                    side=side,
                    slot=slot,
                    pokemon=_pokemon_label(pokemon),
                    source="showdown",
                )
            )

        seen: set[int] = set()
        for pokemon in team.values():
            object_id = id(pokemon)
            if object_id in seen:
                continue
            seen.add(object_id)

            name = _pokemon_label(pokemon)

            if pokemon.fainted:
                events.append(
                    BattleEvent(
                        type="fainted",
                        side=side,
                        pokemon=name,
                        source="showdown",
                    )
                )
            elif getattr(pokemon, "max_hp", 0) > 0:
                events.append(
                    BattleEvent(
                        type="hp_updated",
                        side=side,
                        pokemon=name,
                        hp_percent=max(
                            0.0,
                            min(100.0, pokemon.current_hp_fraction * 100),
                        ),
                        source="showdown",
                    )
                )

            status = _status_name(pokemon)
            if status is not None and status != "fnt":
                events.append(
                    BattleEvent(
                        type="status_changed",
                        side=side,
                        pokemon=name,
                        status=status,
                        source="showdown",
                    )
                )

            item = getattr(pokemon, "item", None)
            if item not in {None, "", GenData.UNKNOWN_ITEM}:
                events.append(
                    BattleEvent(
                        type="item_revealed",
                        side=side,
                        pokemon=name,
                        item=item,
                        source="showdown",
                    )
                )

            ability = getattr(pokemon, "ability", None)
            if ability:
                events.append(
                    BattleEvent(
                        type="ability_revealed",
                        side=side,
                        pokemon=name,
                        ability=ability,
                        source="showdown",
                    )
                )

            for move_id in getattr(pokemon, "moves", {}):
                events.append(
                    BattleEvent(
                        type="move_revealed",
                        side=side,
                        pokemon=name,
                        move=move_id,
                        source="showdown",
                    )
                )

            for stat, stage in getattr(pokemon, "boosts", {}).items():
                events.append(
                    BattleEvent(
                        type="stat_stage_changed",
                        side=side,
                        pokemon=name,
                        field=stat,
                        value=int(stage),
                        source="showdown",
                    )
                )

            protect_counter = int(getattr(pokemon, "protect_counter", 0))
            events.append(
                BattleEvent(
                    type="protect_streak_changed",
                    side=side,
                    pokemon=name,
                    value=protect_counter,
                    source="showdown",
                )
            )

            first_turn = getattr(pokemon, "first_turn", None)
            if isinstance(first_turn, bool):
                events.append(
                    BattleEvent(
                        type="first_turn_changed",
                        side=side,
                        pokemon=name,
                        value=first_turn,
                        source="showdown",
                    )
                )

    for side, conditions in [
        ("player", battle.side_conditions),
        ("opponent", battle.opponent_side_conditions),
    ]:
        for condition, started_or_layers in conditions.items():
            condition_name = condition.name.lower()
            events.append(
                BattleEvent(
                    type="side_condition_changed",
                    side=side,
                    field=condition_name,
                    value=True,
                    source="showdown",
                )
            )

            if condition is SideCondition.TAILWIND:
                events.append(
                    BattleEvent(
                        type="tailwind_changed",
                        side=side,
                        value=_remaining_turns(
                            battle.turn,
                            started_or_layers,
                            TAILWIND_DURATION,
                        ),
                        source="showdown",
                    )
                )

    for field, started_turn in battle.fields.items():
        if field is Field.TRICK_ROOM:
            events.append(
                BattleEvent(
                    type="trick_room_changed",
                    value=_remaining_turns(
                        battle.turn,
                        started_turn,
                        TRICK_ROOM_DURATION,
                    ),
                    source="showdown",
                )
            )
        elif field.is_terrain:
            events.append(
                BattleEvent(
                    type="terrain_changed",
                    value=field.name.lower(),
                    source="showdown",
                )
            )
        else:
            events.append(
                BattleEvent(
                    type="field_condition_changed",
                    field=field.name.lower(),
                    value=True,
                    source="showdown",
                )
            )

    for weather in battle.weather:
        events.append(
            BattleEvent(
                type="weather_changed",
                value=weather.name.lower(),
                source="showdown",
            )
        )
        break

    return events


def battle_state_from_showdown(battle: DoubleBattle) -> BattleState:
    """Build the universal current BattleState from a live Showdown battle."""

    state = BattleState()
    for event in showdown_events(battle):
        state.apply(event)
    return state


def decision_snapshot_from_showdown(
    battle: DoubleBattle,
    legal_actions: tuple[str, ...] | None = None,
) -> DecisionSnapshot:
    """Freeze the current Showdown decision state for coaching/evaluation."""

    if legal_actions is None:
        legal_actions = tuple(
            action.label for action in legal_joint_actions_from_showdown(battle)
        )

    return DecisionSnapshot.from_state(
        battle_state_from_showdown(battle),
        legal_actions=legal_actions,
    )


def _known_item(pokemon) -> str | None:
    item = getattr(pokemon, "item", None)
    if item in {None, "", GenData.UNKNOWN_ITEM}:
        return None
    return str(item)


def _known_gender(pokemon) -> str | None:
    gender = getattr(pokemon, "gender", None)
    if gender is None:
        return None
    name = getattr(gender, "name", None)
    if name is None:
        return str(gender).lower()
    return str(name).lower()


def _known_types(pokemon) -> tuple[str, ...]:
    types = getattr(pokemon, "types", ()) or ()
    resolved: list[str] = []
    for pokemon_type in types:
        name = getattr(pokemon_type, "name", None)
        if name is None:
            resolved.append(str(pokemon_type).lower())
        else:
            resolved.append(str(name).lower())
    return tuple(resolved)


def _scaled_current_hp(pokemon, profile: CombatantProfile) -> int:
    if getattr(pokemon, "fainted", False):
        return 0

    observed_max = int(getattr(pokemon, "max_hp", 0) or 0)
    observed_current = int(getattr(pokemon, "current_hp", 0) or 0)
    if observed_max > 0 and observed_max == profile.max_hp:
        return max(0, min(profile.max_hp, observed_current))

    fraction = float(getattr(pokemon, "current_hp_fraction", 1.0))
    return max(
        0,
        min(
            profile.max_hp,
            int(round(profile.max_hp * fraction)),
        ),
    )


def _remaining_effect_turns(
    effects: dict,
    effect: Effect,
    base_duration: int,
) -> int | None:
    if effect not in effects:
        return None
    elapsed = int(effects.get(effect, 0) or 0)
    return max(1, base_duration - elapsed)


def _disabled_move_from_request(pokemon) -> str | None:
    request = getattr(pokemon, "_last_request", None)
    if not isinstance(request, dict):
        return None

    disabled = [
        move.get("id")
        for move in request.get("moves", ())
        if move.get("disabled", False) and move.get("id")
    ]
    if len(disabled) == 1:
        return str(disabled[0])
    return None


def exact_turn_state_from_showdown(
    battle: DoubleBattle,
    profiles: dict[tuple[TurnSide, str], CombatantProfile],
) -> ExactTurnState:
    """Synchronize a live Showdown battle into an ExactTurnState template.

    Exact stats still come from the supplied profiles so opponent hidden-set
    hypotheses remain explicit. Observable battle fields are refreshed from
    poke-env before simulation.
    """

    synced_profiles = dict(profiles)
    active_slots: dict[tuple[TurnSide, int], str] = {}
    protect_streaks: dict[tuple[TurnSide, str], int] = {}
    toxic_stages: dict[tuple[TurnSide, str], int] = {}
    must_recharge: set[tuple[TurnSide, str]] = set()
    last_moves: dict[tuple[TurnSide, str], str] = {}
    disabled_moves: dict[tuple[TurnSide, str], tuple[str, int]] = {}
    taunt_turns: dict[tuple[TurnSide, str], int] = {}
    encore_locks: dict[tuple[TurnSide, str], tuple[str, int]] = {}
    imprison_users: set[tuple[TurnSide, str]] = set()
    tormented: set[tuple[TurnSide, str]] = set()
    known_moves: dict[tuple[TurnSide, str], set[str]] = {}
    move_pp: dict[tuple[TurnSide, str, str], int] = {}

    side_specs = (
        (
            TurnSide.PLAYER,
            battle.team,
            battle.active_pokemon,
        ),
        (
            TurnSide.OPPONENT,
            battle.opponent_team,
            battle.opponent_active_pokemon,
        ),
    )

    for side, team, active in side_specs:
        for slot, pokemon in enumerate(active):
            if pokemon is None:
                continue
            active_slots[(side, slot)] = _pokemon_label(pokemon)

        seen: set[int] = set()
        for pokemon in team.values():
            object_id = id(pokemon)
            if object_id in seen:
                continue
            seen.add(object_id)

            name = _pokemon_label(pokemon)
            key = (side, name)
            if key not in profiles:
                raise ValueError(
                    f"missing CombatantProfile template for {side.value} {name}"
                )

            template = profiles[key]
            observed_item = _known_item(pokemon)
            observed_ability = getattr(pokemon, "ability", None)
            observed_types = _known_types(pokemon)
            observed_gender = _known_gender(pokemon)
            status = _status_name(pokemon)

            synced_profiles[key] = replace(
                template,
                current_hp=_scaled_current_hp(pokemon, template),
                types=observed_types or template.types,
                boosts={
                    stat: int(stage)
                    for stat, stage in getattr(pokemon, "boosts", {}).items()
                }
                or template.boosts,
                item=(
                    observed_item
                    if observed_item is not None
                    else template.item
                ),
                ability=(
                    str(observed_ability)
                    if observed_ability
                    else template.ability
                ),
                status=None if status == "fnt" else status,
                gender=(
                    observed_gender
                    if observed_gender is not None
                    else template.gender
                ),
            )

            protect_streak = int(getattr(pokemon, "protect_counter", 0) or 0)
            if protect_streak:
                protect_streaks[key] = protect_streak

            if status == "tox":
                toxic_stages[key] = int(
                    getattr(pokemon, "status_counter", 0) or 0
                )

            if bool(getattr(pokemon, "must_recharge", False)):
                must_recharge.add(key)

            last_move = getattr(pokemon, "last_move", None)
            last_move_id = getattr(last_move, "id", None)
            if last_move_id:
                last_moves[key] = str(last_move_id)

            moves = getattr(pokemon, "moves", {}) or {}
            known_moves[key] = set()
            for move_id, move in moves.items():
                normalized_id = str(getattr(move, "id", move_id))
                known_moves[key].add(normalized_id)
                current_pp = getattr(move, "current_pp", None)
                if current_pp is not None:
                    move_pp[(side, name, normalized_id)] = int(current_pp)

            effects = getattr(pokemon, "effects", {}) or {}

            disable_remaining = _remaining_effect_turns(
                effects,
                Effect.DISABLE,
                4,
            )
            if disable_remaining is not None:
                disabled_move = _disabled_move_from_request(pokemon)
                if disabled_move is None and last_move_id:
                    disabled_move = str(last_move_id)
                if disabled_move is not None:
                    disabled_moves[key] = (
                        disabled_move,
                        disable_remaining,
                    )

            taunt_remaining = _remaining_effect_turns(
                effects,
                Effect.TAUNT,
                3,
            )
            if taunt_remaining is not None:
                taunt_turns[key] = taunt_remaining

            encore_remaining = _remaining_effect_turns(
                effects,
                Effect.ENCORE,
                3,
            )
            if encore_remaining is not None and last_move_id:
                encore_locks[key] = (
                    str(last_move_id),
                    encore_remaining,
                )

            if Effect.IMPRISON in effects:
                imprison_users.add(key)

            if Effect.TORMENT in effects:
                tormented.add(key)

    tailwind_sides: set[TurnSide] = set()
    if SideCondition.TAILWIND in battle.side_conditions:
        tailwind_sides.add(TurnSide.PLAYER)
    if SideCondition.TAILWIND in battle.opponent_side_conditions:
        tailwind_sides.add(TurnSide.OPPONENT)

    terrain: str | None = None
    trick_room = False
    field_conditions: set[str] = set()
    for field in battle.fields:
        if field is Field.TRICK_ROOM:
            trick_room = True
        elif field.is_terrain:
            terrain = field.name.lower()
        else:
            field_conditions.add(field.name.lower())

    weather: str | None = None
    for current_weather in battle.weather:
        weather = current_weather.name.lower()
        break

    return ExactTurnState(
        profiles=synced_profiles,
        active_slots=active_slots,
        weather=weather,
        terrain=terrain,
        trick_room=trick_room,
        tailwind_sides=tailwind_sides,
        protect_streaks=protect_streaks,
        toxic_stages=toxic_stages,
        must_recharge=must_recharge,
        last_moves=last_moves,
        disabled_moves=disabled_moves,
        taunt_turns=taunt_turns,
        encore_locks=encore_locks,
        imprison_users=imprison_users,
        tormented=tormented,
        known_moves=known_moves,
        move_pp=move_pp,
        field_conditions=field_conditions,
    )

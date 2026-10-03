"""Pokemon Showdown input adapter for Champions AI.

This module converts poke-env's structured DoubleBattle object into the same
BattleEvent/BattleState representation used by manual and future OBS inputs.
"""

from poke_env.battle import DoubleBattle, Field, SideCondition
from poke_env.data import GenData

from vgc_bench.src.champions_ai.events import BattleEvent
from vgc_bench.src.champions_ai.inputs.showdown_actions import (
    legal_joint_actions_from_showdown,
)
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.state import BattleState


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

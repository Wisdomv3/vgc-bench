from types import SimpleNamespace

from poke_env.battle import Field, SideCondition, Weather

from vgc_bench.src.champions_ai.inputs.showdown import (
    battle_state_from_showdown,
    showdown_events,
)


def _pokemon(
    species: str,
    *,
    hp_fraction: float = 1.0,
    fainted: bool = False,
    status=None,
    item: str | None = None,
    ability: str | None = None,
    moves: tuple[str, ...] = (),
    boosts: dict[str, int] | None = None,
    protect_counter: int = 0,
    first_turn: bool = False,
):
    return SimpleNamespace(
        species=species,
        name=species,
        max_hp=100,
        current_hp_fraction=hp_fraction,
        fainted=fainted,
        status=status,
        item=item,
        ability=ability,
        moves={move: SimpleNamespace(id=move) for move in moves},
        boosts=boosts
        or {
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "accuracy": 0,
            "evasion": 0,
        },
        protect_counter=protect_counter,
        first_turn=first_turn,
    )


def test_showdown_adapter_builds_current_board_state() -> None:
    garchomp = _pokemon(
        "garchomp",
        hp_fraction=0.75,
        item="lifeorb",
        ability="roughskin",
        moves=("protect", "dragonclaw"),
        boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 1, "accuracy": 0, "evasion": 0},
        first_turn=True,
    )
    whimsicott = _pokemon("whimsicott", hp_fraction=0.90)
    salamence = _pokemon(
        "salamence",
        hp_fraction=0.60,
        status=SimpleNamespace(name="PAR"),
        item="choicescarf",
        ability="intimidate",
        moves=("hypervoice", "protect"),
        protect_counter=2,
    )

    battle = SimpleNamespace(
        turn=5,
        team={"p1: Garchomp": garchomp, "p1: Whimsicott": whimsicott},
        opponent_team={"p2: Salamence": salamence},
        active_pokemon=[garchomp, whimsicott],
        opponent_active_pokemon=[salamence, None],
        side_conditions={
            SideCondition.TAILWIND: 4,
            SideCondition.REFLECT: 2,
        },
        opponent_side_conditions={},
        fields={
            Field.TRICK_ROOM: 3,
            Field.PSYCHIC_TERRAIN: 4,
            Field.GRAVITY: 5,
        },
        weather={Weather.RAINDANCE: 5},
    )

    state = battle_state_from_showdown(battle)

    assert state.turn == 5
    assert state.player.active_slots == {0: "garchomp", 1: "whimsicott"}
    assert state.opponent.active_slots == {0: "salamence"}
    assert state.player.pokemon["garchomp"].hp_percent == 75
    assert state.opponent.pokemon["salamence"].hp_percent == 60
    assert state.opponent.pokemon["salamence"].status == "par"
    assert state.opponent.pokemon["salamence"].item == "choicescarf"
    assert state.opponent.pokemon["salamence"].ability == "intimidate"
    assert "hypervoice" in state.opponent.pokemon["salamence"].revealed_moves
    assert state.player.pokemon["garchomp"].stat_stages["spe"] == 1
    assert state.player.pokemon["garchomp"].first_turn is True
    assert state.opponent.pokemon["salamence"].protect_streak == 2
    assert state.opponent.pokemon["salamence"].first_turn is False

    assert state.player.tailwind_turns == 3
    assert "reflect" in state.player.side_conditions
    assert state.trick_room_turns == 3
    assert state.terrain == "psychic_terrain"
    assert state.weather == "raindance"
    assert "gravity" in state.field_conditions


def test_showdown_adapter_marks_source_as_showdown() -> None:
    garchomp = _pokemon("garchomp")
    battle = SimpleNamespace(
        turn=1,
        team={"p1: Garchomp": garchomp},
        opponent_team={},
        active_pokemon=[garchomp, None],
        opponent_active_pokemon=[None, None],
        side_conditions={},
        opponent_side_conditions={},
        fields={},
        weather={},
    )

    events = showdown_events(battle)

    assert events
    assert all(event.source == "showdown" for event in events)


def test_unknown_opponent_item_is_not_treated_as_revealed() -> None:
    salamence = _pokemon("salamence", item="unknown_item")
    battle = SimpleNamespace(
        turn=1,
        team={},
        opponent_team={"p2: Salamence": salamence},
        active_pokemon=[None, None],
        opponent_active_pokemon=[salamence, None],
        side_conditions={},
        opponent_side_conditions={},
        fields={},
        weather={},
    )

    state = battle_state_from_showdown(battle)

    assert state.opponent.pokemon["salamence"].item is None

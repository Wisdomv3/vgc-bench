from types import SimpleNamespace

import pytest
from poke_env.battle import Effect, Field, SideCondition, Weather

from vgc_bench.src.champions_ai.inputs.showdown import (
    exact_turn_state_from_showdown,
)
from vgc_bench.src.champions_ai.matchup import CombatantProfile
from vgc_bench.src.champions_ai.turn_order import TurnSide


def _move(move_id: str, pp: int):
    return SimpleNamespace(id=move_id, current_pp=pp)


def _pokemon(
    species: str,
    *,
    hp_fraction: float = 1.0,
    max_hp: int = 100,
    current_hp: int | None = None,
    fainted: bool = False,
    status=None,
    status_counter: int = 0,
    item: str | None = None,
    ability: str | None = None,
    gender: str = "NEUTRAL",
    types: tuple[str, ...] = ("NORMAL",),
    moves: dict[str, object] | None = None,
    boosts: dict[str, int] | None = None,
    protect_counter: int = 0,
    must_recharge: bool = False,
    last_move: str | None = None,
    effects: dict | None = None,
    request_moves: list[dict] | None = None,
):
    if current_hp is None:
        current_hp = int(round(max_hp * hp_fraction))

    return SimpleNamespace(
        species=species,
        name=species,
        max_hp=max_hp,
        current_hp=current_hp,
        current_hp_fraction=hp_fraction,
        fainted=fainted,
        status=status,
        status_counter=status_counter,
        item=item,
        ability=ability,
        gender=SimpleNamespace(name=gender),
        types=tuple(SimpleNamespace(name=name) for name in types),
        moves=moves or {},
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
        must_recharge=must_recharge,
        last_move=(
            SimpleNamespace(id=last_move)
            if last_move is not None
            else None
        ),
        effects=effects or {},
        _last_request=(
            {"moves": request_moves}
            if request_moves is not None
            else None
        ),
    )


def _profile(
    name: str,
    *,
    max_hp: int = 200,
    item: str | None = None,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=max_hp,
        max_hp=max_hp,
        types=("normal",),
        stats={
            "atk": 130,
            "def": 110,
            "spa": 120,
            "spd": 110,
            "spe": 100,
        },
        item=item,
        ability=ability,
    )


def _battle(player, opponent):
    return SimpleNamespace(
        turn=6,
        team={"p1: Player": player},
        opponent_team={"p2: Opponent": opponent},
        active_pokemon=[player, None],
        opponent_active_pokemon=[opponent, None],
        side_conditions={SideCondition.TAILWIND: 5},
        opponent_side_conditions={},
        fields={
            Field.TRICK_ROOM: 5,
            Field.PSYCHIC_TERRAIN: 5,
            Field.GRAVITY: 5,
        },
        weather={Weather.RAINDANCE: 5},
    )


def test_showdown_sync_refreshes_observable_profile_and_pp_state() -> None:
    player = _pokemon(
        "player",
        hp_fraction=0.60,
        max_hp=100,
        current_hp=60,
        status=SimpleNamespace(name="TOX"),
        status_counter=2,
        item="lifeorb",
        ability="roughskin",
        gender="MALE",
        types=("DRAGON", "GROUND"),
        moves={
            "protect": _move("protect", 7),
            "dragonclaw": _move("dragonclaw", 12),
        },
        boosts={
            "atk": 1,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 2,
            "accuracy": 0,
            "evasion": 0,
        },
        protect_counter=2,
        must_recharge=True,
        last_move="dragonclaw",
    )
    opponent = _pokemon(
        "opponent",
        moves={"tackle": _move("tackle", 30)},
        gender="FEMALE",
    )
    profiles = {
        (TurnSide.PLAYER, "player"): _profile("player"),
        (TurnSide.OPPONENT, "opponent"): _profile("opponent"),
    }

    state = exact_turn_state_from_showdown(
        _battle(player, opponent),
        profiles,
    )

    synced = state.profile(TurnSide.PLAYER, "player")
    assert synced.current_hp == 120
    assert synced.status == "tox"
    assert synced.item == "lifeorb"
    assert synced.ability == "roughskin"
    assert synced.gender == "male"
    assert synced.types == ("dragon", "ground")
    assert synced.boosts["atk"] == 1
    assert synced.boosts["spe"] == 2

    key = (TurnSide.PLAYER, "player")
    assert state.protect_streaks[key] == 2
    assert state.toxic_stages[key] == 2
    assert key in state.must_recharge
    assert state.last_moves[key] == "dragonclaw"
    assert state.move_pp[(TurnSide.PLAYER, "player", "protect")] == 7
    assert state.move_pp[
        (TurnSide.PLAYER, "player", "dragonclaw")
    ] == 12
    assert state.known_moves[key] == {"protect", "dragonclaw"}

    assert state.active_slots[
        (TurnSide.PLAYER, 0)
    ] == "player"
    assert state.tailwind_sides == {TurnSide.PLAYER}
    assert state.trick_room is True
    assert state.terrain == "psychic_terrain"
    assert state.weather == "raindance"
    assert "gravity" in state.field_conditions


def test_showdown_sync_imports_supported_volatile_restrictions() -> None:
    player = _pokemon("player")
    opponent = _pokemon(
        "opponent",
        moves={
            "tackle": _move("tackle", 20),
            "protect": _move("protect", 9),
        },
        last_move="protect",
        effects={
            Effect.DISABLE: 1,
            Effect.TAUNT: 1,
            Effect.ENCORE: 1,
            Effect.IMPRISON: 0,
            Effect.TORMENT: 0,
        },
        request_moves=[
            {"id": "tackle", "disabled": True},
            {"id": "protect", "disabled": False},
        ],
    )
    profiles = {
        (TurnSide.PLAYER, "player"): _profile("player"),
        (TurnSide.OPPONENT, "opponent"): _profile("opponent"),
    }

    state = exact_turn_state_from_showdown(
        _battle(player, opponent),
        profiles,
    )

    key = (TurnSide.OPPONENT, "opponent")
    assert state.disabled_moves[key] == ("tackle", 3)
    assert state.taunt_turns[key] == 2
    assert state.encore_locks[key] == ("protect", 2)
    assert key in state.imprison_users
    assert key in state.tormented


def test_showdown_sync_preserves_hidden_item_and_ability_when_unknown() -> None:
    player = _pokemon("player")
    opponent = _pokemon(
        "opponent",
        item="unknown_item",
        ability=None,
        moves={"tackle": _move("tackle", 31)},
    )
    profiles = {
        (TurnSide.PLAYER, "player"): _profile("player"),
        (
            TurnSide.OPPONENT,
            "opponent",
        ): _profile(
            "opponent",
            item="choicescarf",
            ability="intimidate",
        ),
    }

    state = exact_turn_state_from_showdown(
        _battle(player, opponent),
        profiles,
    )

    synced = state.profile(TurnSide.OPPONENT, "opponent")
    assert synced.item == "choicescarf"
    assert synced.ability == "intimidate"


def test_showdown_sync_requires_profile_template_for_each_known_pokemon() -> None:
    player = _pokemon("player")
    opponent = _pokemon("opponent")

    with pytest.raises(
        ValueError,
        match="missing CombatantProfile template",
    ):
        exact_turn_state_from_showdown(
            _battle(player, opponent),
            {
                (TurnSide.PLAYER, "player"): _profile("player"),
            },
        )

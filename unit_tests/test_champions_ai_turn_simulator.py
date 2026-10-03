import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    SimulationEventType,
    TurnSimulationConfig,
    UnresolvedSpeedTie,
    simulate_turn,
)


def _profile(
    name: str,
    *,
    hp: int = 150,
    max_hp: int = 150,
    atk: int = 150,
    defense: int = 100,
    spa: int = 150,
    spd: int = 100,
    spe: int = 100,
    types: tuple[str, ...] = ("normal",),
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=max_hp,
        types=types,
        stats={
            "atk": atk,
            "def": defense,
            "spa": spa,
            "spd": spd,
            "spe": spe,
        },
        ability=ability,
    )


def _move(
    slot: int,
    actor: str,
    move: str,
    target: str | None = None,
    target_position: int | None = None,
):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target=target,
        target_position=target_position,
    )


def _switch(slot: int, actor: str, switch_to: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.SWITCH,
        actor=actor,
        switch_to=switch_to,
    )


def _pass(slot: int, actor: str):
    return SlotAction(slot=slot, kind=ActionKind.PASS, actor=actor)


def _joint(first, second):
    return JointAction(first=first, second=second)


def _base_state():
    profiles = {
        (TurnSide.PLAYER, "garchomp"): _profile("garchomp"),
        (TurnSide.PLAYER, "whimsicott"): _profile(
            "whimsicott",
            ability="prankster",
        ),
        (TurnSide.PLAYER, "kingambit"): _profile(
            "kingambit",
            hp=180,
            max_hp=180,
            defense=140,
        ),
        (TurnSide.OPPONENT, "salamence"): _profile("salamence"),
        (TurnSide.OPPONENT, "sneasler"): _profile("sneasler"),
    }
    return ExactTurnState(
        profiles=profiles,
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.PLAYER, 1): "whimsicott",
            (TurnSide.OPPONENT, 0): "salamence",
            (TurnSide.OPPONENT, 1): "sneasler",
        },
    )


def _speeds():
    return {
        (TurnSide.PLAYER, "garchomp"): SpeedState(100),
        (TurnSide.PLAYER, "whimsicott"): SpeedState(100),
        (TurnSide.PLAYER, "kingambit"): SpeedState(70),
        (TurnSide.OPPONENT, "salamence"): SpeedState(150),
        (TurnSide.OPPONENT, "sneasler"): SpeedState(140),
    }


def _profile_map(*entries):
    return {
        (side, actor, move.move_id): move
        for side, actor, move in entries
    }


def test_switch_happens_first_and_attack_hits_incoming_slot() -> None:
    state = _base_state()
    ours = _joint(
        _switch(0, "garchomp", "kingambit"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    body_slam = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.OPPONENT, "salamence", body_slam)),
        TurnSimulationConfig(damage_roll_index=15),
    )

    assert result.state.active_name(TurnSide.PLAYER, 0) == "kingambit"
    assert result.state.profile(TurnSide.PLAYER, "garchomp").current_hp == 150
    assert result.state.profile(TurnSide.PLAYER, "kingambit").current_hp < 180


def test_protect_blocks_targeted_damage() -> None:
    state = _base_state()
    ours = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )
    body_slam = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.OPPONENT, "salamence", body_slam)),
        TurnSimulationConfig(damage_roll_index=15),
    )

    assert result.state.profile(TurnSide.PLAYER, "garchomp").current_hp == 150
    assert any(event.type is SimulationEventType.BLOCKED for event in result.events)


def test_fainted_pokemon_does_not_execute_queued_move() -> None:
    state = _base_state()
    state.profiles[(TurnSide.OPPONENT, "salamence")] = _profile(
        "salamence",
        hp=40,
        max_hp=150,
        defense=50,
    )
    speeds = _speeds()
    speeds[(TurnSide.PLAYER, "garchomp")] = SpeedState(200)

    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    body_slam = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        speeds,
        _profile_map(
            (TurnSide.PLAYER, "garchomp", body_slam),
            (TurnSide.OPPONENT, "salamence", body_slam),
        ),
        TurnSimulationConfig(damage_roll_index=15),
    )

    assert result.state.profile(TurnSide.OPPONENT, "salamence").current_hp == 0
    assert result.state.profile(TurnSide.PLAYER, "garchomp").current_hp == 150
    assert any(
        event.type is SimulationEventType.SKIPPED
        and event.actor == "salamence"
        for event in result.events
    )


def test_spread_move_hits_both_foes() -> None:
    state = _base_state()
    ours = _joint(
        _move(0, "garchomp", "Hyper Voice"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    hyper_voice = MoveProfile(
        move_id="hypervoice",
        base_power=90,
        category="special",
        move_type="normal",
        spread=True,
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "garchomp", hyper_voice)),
        TurnSimulationConfig(damage_roll_index=15),
    )

    assert result.state.profile(TurnSide.OPPONENT, "salamence").current_hp < 150
    assert result.state.profile(TurnSide.OPPONENT, "sneasler").current_hp < 150


def test_all_adjacent_spread_move_can_damage_partner() -> None:
    state = _base_state()
    ours = _joint(
        _move(0, "garchomp", "Earthquake"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    earthquake = MoveProfile(
        move_id="earthquake",
        base_power=100,
        category="physical",
        move_type="ground",
        spread=True,
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "garchomp", earthquake)),
        TurnSimulationConfig(damage_roll_index=15),
    )

    assert result.state.profile(TurnSide.PLAYER, "whimsicott").current_hp < 150
    assert result.state.profile(TurnSide.OPPONENT, "salamence").current_hp < 150
    assert result.state.profile(TurnSide.OPPONENT, "sneasler").current_hp < 150


def test_prankster_tailwind_dynamically_reorders_remaining_gen9_moves() -> None:
    state = _base_state()
    speeds = _speeds()
    speeds[(TurnSide.PLAYER, "whimsicott")] = SpeedState(50)
    speeds[(TurnSide.PLAYER, "garchomp")] = SpeedState(100)
    speeds[(TurnSide.OPPONENT, "salamence")] = SpeedState(150)

    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _move(1, "whimsicott", "Tailwind"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    body_slam = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        speeds,
        _profile_map(
            (TurnSide.PLAYER, "garchomp", body_slam),
            (TurnSide.OPPONENT, "salamence", body_slam),
        ),
        TurnSimulationConfig(damage_roll_index=0),
    )

    executed = [
        event.actor
        for event in result.events
        if event.type in {SimulationEventType.FIELD, SimulationEventType.DAMAGE}
    ]

    assert executed[:3] == ["whimsicott", "garchomp", "salamence"]
    assert TurnSide.PLAYER in result.state.tailwind_sides


def test_deterministic_simulator_refuses_to_guess_speed_tie() -> None:
    state = _base_state()
    speeds = _speeds()
    speeds[(TurnSide.PLAYER, "garchomp")] = SpeedState(150)
    speeds[(TurnSide.OPPONENT, "salamence")] = SpeedState(150)

    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )
    body_slam = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    with pytest.raises(UnresolvedSpeedTie):
        simulate_turn(
            state,
            ours,
            theirs,
            speeds,
            _profile_map(
                (TurnSide.PLAYER, "garchomp", body_slam),
                (TurnSide.OPPONENT, "salamence", body_slam),
            ),
            TurnSimulationConfig(damage_roll_index=15),
        )

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    SimulationEventType,
    TurnSimulationConfig,
    simulate_turn,
)


TACKLE = MoveProfile(
    move_id="tackle",
    base_power=40,
    category="physical",
    move_type="normal",
)

BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=85,
    category="physical",
    move_type="normal",
)

GIGATON_HAMMER = MoveProfile(
    move_id="gigatonhammer",
    base_power=160,
    category="physical",
    move_type="steel",
)


def _profile(
    name: str,
    *,
    hp: int = 500,
    item: str | None = None,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=500,
        types=("normal",),
        stats={
            "atk": 120,
            "def": 120,
            "spa": 120,
            "spd": 120,
            "spe": 100,
        },
        item=item,
        ability=ability,
    )


def _state(
    *,
    attacker_item: str | None = None,
    attacker_ability: str | None = None,
):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "attacker"): _profile(
                "attacker",
                item=attacker_item,
                ability=attacker_ability,
            ),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile("target"),
            (TurnSide.OPPONENT, "partner2"): _profile("partner2"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "attacker",
            (TurnSide.PLAYER, 1): "partner",
            (TurnSide.OPPONENT, 0): "target",
            (TurnSide.OPPONENT, 1): "partner2",
        },
    )


def _move(slot: int, actor: str, move: str, target_position: int | None = None):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target_position=target_position,
    )


def _pass(slot: int, actor: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.PASS,
        actor=actor,
    )


def _joint(first, second):
    return JointAction(first=first, second=second)


def _ours(move: str):
    return _joint(
        _move(0, "attacker", move, 1),
        _pass(1, "partner"),
    )


def _theirs():
    return _joint(
        _pass(0, "target"),
        _pass(1, "partner2"),
    )


def _speeds(extra: dict | None = None):
    speeds = {
        (TurnSide.PLAYER, "attacker"): SpeedState(200),
        (TurnSide.PLAYER, "partner"): SpeedState(90),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "partner2"): SpeedState(80),
    }
    if extra:
        speeds.update(extra)
    return speeds


def _moves(*profiles: MoveProfile):
    return {
        (TurnSide.PLAYER, "attacker", profile.move_id): profile
        for profile in profiles
    }


def _config():
    return TurnSimulationConfig(
        damage_roll_index=7,
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=False,
        branch_protect=False,
        branch_speed_ties=False,
    )


def test_truant_alternates_act_and_loaf_turns() -> None:
    first = simulate_turn(
        _state(attacker_ability="truant"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )
    hp_after_first = first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp
    assert hp_after_first < 500
    assert (TurnSide.PLAYER, "attacker") in first.state.truant_loaf

    second = simulate_turn(
        first.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )
    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == hp_after_first
    assert (TurnSide.PLAYER, "attacker") not in second.state.truant_loaf
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "Truant" in event.detail
        for event in second.events
    )

    third = simulate_turn(
        second.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )
    assert third.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < hp_after_first


def test_switch_clears_truant_cycle() -> None:
    state = _state(attacker_ability="truant")
    state.profiles[(TurnSide.PLAYER, "bench")] = _profile("bench")
    state.truant_loaf.add((TurnSide.PLAYER, "attacker"))

    ours = _joint(
        SlotAction(
            slot=0,
            kind=ActionKind.SWITCH,
            actor="attacker",
            switch_to="bench",
        ),
        _pass(1, "partner"),
    )

    result = simulate_turn(
        state,
        ours,
        _theirs(),
        _speeds({
            (TurnSide.PLAYER, "bench"): SpeedState(100),
        }),
        {},
        _config(),
    )

    assert (TurnSide.PLAYER, "attacker") not in result.state.truant_loaf


def test_choice_item_locks_into_first_committed_move() -> None:
    first = simulate_turn(
        _state(attacker_item="choiceband"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )

    assert first.state.choice_locks[
        (TurnSide.PLAYER, "attacker")
    ] == "tackle"

    hp_after_first = first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp

    second = simulate_turn(
        first.state,
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == hp_after_first
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "locked into tackle" in event.detail
        for event in second.events
    )


def test_choice_item_allows_locked_move_again() -> None:
    first = simulate_turn(
        _state(attacker_item="choicescarf"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )
    hp_after_first = first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp

    second = simulate_turn(
        first.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )

    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < hp_after_first


def test_gorilla_tactics_uses_same_move_lock_rule() -> None:
    first = simulate_turn(
        _state(attacker_ability="gorillatactics"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )

    second = simulate_turn(
        first.state,
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "locked into tackle" in event.detail
        for event in second.events
    )


def test_switch_clears_choice_lock_and_last_move() -> None:
    state = _state(attacker_item="choiceband")
    state.profiles[(TurnSide.PLAYER, "bench")] = _profile("bench")
    state.choice_locks[(TurnSide.PLAYER, "attacker")] = "tackle"
    state.last_moves[(TurnSide.PLAYER, "attacker")] = "tackle"

    ours = _joint(
        SlotAction(
            slot=0,
            kind=ActionKind.SWITCH,
            actor="attacker",
            switch_to="bench",
        ),
        _pass(1, "partner"),
    )

    result = simulate_turn(
        state,
        ours,
        _theirs(),
        _speeds({
            (TurnSide.PLAYER, "bench"): SpeedState(100),
        }),
        {},
        _config(),
    )

    key = (TurnSide.PLAYER, "attacker")
    assert key not in result.state.choice_locks
    assert key not in result.state.last_moves


def test_gigaton_hammer_cannot_be_used_twice_in_a_row() -> None:
    first = simulate_turn(
        _state(),
        _ours("Gigaton Hammer"),
        _theirs(),
        _speeds(),
        _moves(GIGATON_HAMMER),
        _config(),
    )
    hp_after_first = first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp

    second = simulate_turn(
        first.state,
        _ours("Gigaton Hammer"),
        _theirs(),
        _speeds(),
        _moves(GIGATON_HAMMER),
        _config(),
    )

    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == hp_after_first
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "twice in a row" in event.detail
        for event in second.events
    )


def test_different_move_resets_cant_use_twice_restriction() -> None:
    first = simulate_turn(
        _state(),
        _ours("Gigaton Hammer"),
        _theirs(),
        _speeds(),
        _moves(GIGATON_HAMMER),
        _config(),
    )

    second = simulate_turn(
        first.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _moves(TACKLE),
        _config(),
    )
    hp_after_second = second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp

    third = simulate_turn(
        second.state,
        _ours("Gigaton Hammer"),
        _theirs(),
        _speeds(),
        _moves(GIGATON_HAMMER),
        _config(),
    )

    assert third.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < hp_after_second

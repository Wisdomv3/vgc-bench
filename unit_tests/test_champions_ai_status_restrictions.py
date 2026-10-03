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


def _profile(name: str, hp: int = 300):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=300,
        types=("normal",),
        stats={
            "atk": 130,
            "def": 110,
            "spa": 120,
            "spd": 110,
            "spe": 100,
        },
    )


def _state() -> ExactTurnState:
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "restrictor"): _profile("restrictor"),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile("target"),
            (TurnSide.OPPONENT, "partner2"): _profile("partner2"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "restrictor",
            (TurnSide.PLAYER, 1): "partner",
            (TurnSide.OPPONENT, 0): "target",
            (TurnSide.OPPONENT, 1): "partner2",
        },
    )


def _move(
    slot: int,
    actor: str,
    move: str,
    target_position: int | None = None,
):
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


def _our_status(move: str):
    return _joint(
        _move(0, "restrictor", move, 1),
        _pass(1, "partner"),
    )


def _our_pass():
    return _joint(
        _pass(0, "restrictor"),
        _pass(1, "partner"),
    )


def _their_move(move: str):
    return _joint(
        _move(0, "target", move, 1),
        _pass(1, "partner2"),
    )


def _their_tailwind():
    return _joint(
        _move(0, "target", "Tailwind"),
        _pass(1, "partner2"),
    )


def _speeds(extra: dict | None = None):
    speeds = {
        (TurnSide.PLAYER, "restrictor"): SpeedState(200),
        (TurnSide.PLAYER, "partner"): SpeedState(90),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "partner2"): SpeedState(80),
    }
    if extra:
        speeds.update(extra)
    return speeds


def _profiles(*entries):
    return {
        (side, actor, profile.move_id): profile
        for side, actor, profile in entries
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


def test_disable_blocks_targets_last_move() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"

    result = simulate_turn(
        state,
        _our_status("Disable"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp == 300
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "disabled" in event.detail
        for event in result.events
    )


def test_disable_allows_a_different_move() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"

    result = simulate_turn(
        state,
        _our_status("Disable"),
        _their_move("Body Slam"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", BODY_SLAM)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300


def test_taunt_blocks_status_move() -> None:
    result = simulate_turn(
        _state(),
        _our_status("Taunt"),
        _their_tailwind(),
        _speeds(),
        {},
        _config(),
    )

    assert TurnSide.OPPONENT not in result.state.tailwind_sides
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "Taunt" in event.detail
        for event in result.events
    )


def test_taunt_allows_damaging_move() -> None:
    result = simulate_turn(
        _state(),
        _our_status("Taunt"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300


def test_encore_overrides_pending_move_with_last_move() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"

    result = simulate_turn(
        state,
        _our_status("Encore"),
        _their_move("Body Slam"),
        _speeds(),
        _profiles(
            (TurnSide.OPPONENT, "target", TACKLE),
            (TurnSide.OPPONENT, "target", BODY_SLAM),
        ),
        _config(),
    )

    damage_events = [
        event
        for event in result.events
        if event.type is SimulationEventType.DAMAGE
        and event.actor == "target"
    ]
    assert damage_events
    assert damage_events[0].move == "tackle"


def test_encore_expires_after_three_target_turns() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"
    move_profiles = _profiles(
        (TurnSide.OPPONENT, "target", TACKLE),
        (TurnSide.OPPONENT, "target", BODY_SLAM),
    )

    first = simulate_turn(
        state,
        _our_status("Encore"),
        _their_move("Body Slam"),
        _speeds(),
        move_profiles,
        _config(),
    )
    second = simulate_turn(
        first.state,
        _our_pass(),
        _their_move("Body Slam"),
        _speeds(),
        move_profiles,
        _config(),
    )
    third = simulate_turn(
        second.state,
        _our_pass(),
        _their_move("Body Slam"),
        _speeds(),
        move_profiles,
        _config(),
    )

    assert (
        TurnSide.OPPONENT,
        "target",
    ) not in third.state.encore_locks

    fourth = simulate_turn(
        third.state,
        _our_pass(),
        _their_move("Body Slam"),
        _speeds(),
        move_profiles,
        _config(),
    )

    assert any(
        event.type is SimulationEventType.DAMAGE
        and event.actor == "target"
        and event.move == "bodyslam"
        for event in fourth.events
    )


def test_imprison_blocks_shared_move() -> None:
    state = _state()
    state.known_moves[(TurnSide.PLAYER, "restrictor")] = {
        "imprison",
        "tackle",
    }

    result = simulate_turn(
        state,
        _our_status("Imprison"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp == 300
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "Imprison" in event.detail
        for event in result.events
    )


def test_imprison_allows_unshared_move() -> None:
    state = _state()
    state.known_moves[(TurnSide.PLAYER, "restrictor")] = {
        "imprison",
        "bodyslam",
    }

    result = simulate_turn(
        state,
        _our_status("Imprison"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300


def test_switch_clears_volatile_move_restrictions() -> None:
    state = _state()
    state.profiles[(TurnSide.PLAYER, "bench")] = _profile("bench")
    key = (TurnSide.PLAYER, "restrictor")
    state.disabled_moves[key] = ("tackle", 4)
    state.taunt_turns[key] = 3
    state.encore_locks[key] = ("tackle", 3)
    state.imprison_users.add(key)

    ours = _joint(
        SlotAction(
            slot=0,
            kind=ActionKind.SWITCH,
            actor="restrictor",
            switch_to="bench",
        ),
        _pass(1, "partner"),
    )

    result = simulate_turn(
        state,
        ours,
        _joint(
            _pass(0, "target"),
            _pass(1, "partner2"),
        ),
        _speeds({
            (TurnSide.PLAYER, "bench"): SpeedState(100),
        }),
        {},
        _config(),
    )

    assert key not in result.state.disabled_moves
    assert key not in result.state.taunt_turns
    assert key not in result.state.encore_locks
    assert key not in result.state.imprison_users

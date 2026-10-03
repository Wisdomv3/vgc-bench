import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_branching import (
    BranchingPolicy,
    simulate_turn_distribution,
)
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


def _profile(
    name: str,
    *,
    hp: int = 300,
    gender: str | None = None,
    ability: str | None = None,
):
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
        gender=gender,
        ability=ability,
    )


def _state(
    *,
    restrictor_gender: str | None = "male",
    target_gender: str | None = "female",
    target_ability: str | None = None,
) -> ExactTurnState:
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "restrictor"): _profile(
                "restrictor",
                gender=restrictor_gender,
            ),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile(
                "target",
                gender=target_gender,
                ability=target_ability,
            ),
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


def _our_move(move: str):
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


def _their_pass():
    return _joint(
        _pass(0, "target"),
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


def _branch_policy():
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=True,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=7,
        merge_equivalent_states=False,
    )


def test_torment_does_not_cancel_already_selected_repeated_move() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"

    result = simulate_turn(
        state,
        _our_status("Torment"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300
    assert (TurnSide.OPPONENT, "target") in result.state.tormented


def test_torment_blocks_repeating_last_move_next_turn() -> None:
    state = _state()
    state.last_moves[(TurnSide.OPPONENT, "target")] = "tackle"

    first = simulate_turn(
        state,
        _our_status("Torment"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )
    hp_after_first = first.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp

    second = simulate_turn(
        first.state,
        _our_pass(),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert second.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp == hp_after_first
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "Torment" in event.detail
        for event in second.events
    )


def test_torment_allows_a_different_move() -> None:
    state = _state()
    key = (TurnSide.OPPONENT, "target")
    state.last_moves[key] = "tackle"
    state.tormented.add(key)

    result = simulate_turn(
        state,
        _our_pass(),
        _their_move("Body Slam"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", BODY_SLAM)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300


def test_attract_branches_fifty_fifty_for_compatible_genders() -> None:
    distribution = simulate_turn_distribution(
        _state(),
        _our_status("Attract"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        policy=_branch_policy(),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([0.5, 0.5])
    assert any(
        any(
            event.type is SimulationEventType.CANNOT_MOVE
            and "love" in event.detail
            for event in outcome.result.events
        )
        for outcome in distribution.outcomes
    )


def test_attract_fails_for_same_gender() -> None:
    result = simulate_turn(
        _state(target_gender="male"),
        _our_status("Attract"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert (TurnSide.OPPONENT, "target") not in result.state.attractions
    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp < 300


def test_oblivious_blocks_attract() -> None:
    result = simulate_turn(
        _state(target_ability="oblivious"),
        _our_status("Attract"),
        _their_move("Tackle"),
        _speeds(),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert (TurnSide.OPPONENT, "target") not in result.state.attractions
    assert any(
        event.type is SimulationEventType.BLOCKED
        and "Oblivious" in event.detail
        for event in result.events
    )


def test_switching_attraction_source_clears_attraction() -> None:
    state = _state()
    state.profiles[(TurnSide.PLAYER, "bench")] = _profile("bench")
    state.attractions[
        (TurnSide.OPPONENT, "target")
    ] = (TurnSide.PLAYER, "restrictor")

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
        _their_move("Tackle"),
        _speeds({
            (TurnSide.PLAYER, "bench"): SpeedState(100),
        }),
        _profiles((TurnSide.OPPONENT, "target", TACKLE)),
        _config(),
    )

    assert (TurnSide.OPPONENT, "target") not in result.state.attractions


def test_move_pp_is_decremented_and_pressure_costs_extra_pp() -> None:
    state = _state()
    state.profiles[(TurnSide.OPPONENT, "target")] = _profile(
        "target",
        gender="female",
        ability="pressure",
    )
    state.move_pp[(TurnSide.PLAYER, "restrictor", "tackle")] = 2

    result = simulate_turn(
        state,
        _our_move("Tackle"),
        _their_pass(),
        _speeds(),
        _profiles((TurnSide.PLAYER, "restrictor", TACKLE)),
        _config(),
    )

    assert result.state.move_pp[
        (TurnSide.PLAYER, "restrictor", "tackle")
    ] == 0


def test_zero_pp_move_is_blocked_when_another_move_has_pp() -> None:
    state = _state()
    state.move_pp[(TurnSide.PLAYER, "restrictor", "tackle")] = 0
    state.move_pp[(TurnSide.PLAYER, "restrictor", "bodyslam")] = 1

    result = simulate_turn(
        state,
        _our_move("Tackle"),
        _their_pass(),
        _speeds(),
        _profiles((TurnSide.PLAYER, "restrictor", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == 300
    assert any(
        event.type is SimulationEventType.CANNOT_MOVE
        and "no PP" in event.detail
        for event in result.events
    )


def test_all_pp_exhausted_falls_back_to_struggle_with_direct_recoil() -> None:
    state = _state()
    state.profiles[(TurnSide.PLAYER, "restrictor")] = _profile(
        "restrictor",
        gender="male",
        ability="magicguard",
    )
    state.move_pp[(TurnSide.PLAYER, "restrictor", "tackle")] = 0

    result = simulate_turn(
        state,
        _our_move("Tackle"),
        _their_pass(),
        _speeds(),
        _profiles((TurnSide.PLAYER, "restrictor", TACKLE)),
        _config(),
    )

    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 300
    assert result.state.profile(
        TurnSide.PLAYER,
        "restrictor",
    ).current_hp == 225
    assert any(
        event.type is SimulationEventType.RECOIL
        and "Struggle" in event.detail
        for event in result.events
    )

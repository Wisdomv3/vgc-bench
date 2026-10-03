import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.field_mechanics import is_grounded
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


BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=85,
    category="physical",
    move_type="normal",
)

FAKE_OUT = MoveProfile(
    move_id="fakeout",
    base_power=40,
    category="physical",
    move_type="normal",
)

HYPER_VOICE = MoveProfile(
    move_id="hypervoice",
    base_power=90,
    category="special",
    move_type="normal",
    spread=True,
)

ROCK_SLIDE = MoveProfile(
    move_id="rockslide",
    base_power=75,
    category="physical",
    move_type="rock",
    spread=True,
)


def _profile(
    name: str,
    *,
    hp: int = 120,
    types: tuple[str, ...] = ("normal",),
    ability: str | None = None,
    item: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=120,
        types=types,
        stats={
            "atk": 140,
            "def": 100,
            "spa": 140,
            "spd": 100,
            "spe": 100,
        },
        ability=ability,
        item=item,
    )


def _state(
    *,
    terrain: str | None = None,
    attacker_types: tuple[str, ...] = ("normal",),
    attacker_ability: str | None = None,
    attacker_item: str | None = None,
    target_types: tuple[str, ...] = ("normal",),
    target_ability: str | None = None,
    target_item: str | None = None,
):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "attacker"): _profile(
                "attacker",
                types=attacker_types,
                ability=attacker_ability,
                item=attacker_item,
            ),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile(
                "target",
                types=target_types,
                ability=target_ability,
                item=target_item,
            ),
            (TurnSide.OPPONENT, "redirector"): _profile("redirector"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "attacker",
            (TurnSide.PLAYER, 1): "partner",
            (TurnSide.OPPONENT, 0): "target",
            (TurnSide.OPPONENT, 1): "redirector",
        },
        terrain=terrain,
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


def _speeds():
    return {
        (TurnSide.PLAYER, "attacker"): SpeedState(150),
        (TurnSide.PLAYER, "partner"): SpeedState(80),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "redirector"): SpeedState(90),
    }


def _fixed_config():
    return TurnSimulationConfig(
        damage_roll_index=7,
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_protect=False,
        branch_speed_ties=False,
    )


def _profile_map(*entries):
    return {
        (side, actor, profile.move_id): profile
        for side, actor, profile in entries
    }


def test_psychic_terrain_blocks_priority_against_grounded_foe() -> None:
    state = _state(terrain="psychic_terrain")
    ours = _joint(
        _move(0, "attacker", "Fake Out", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _pass(1, "redirector"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", FAKE_OUT)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp == 120
    assert any(
        event.type is SimulationEventType.PRIORITY_BLOCKED
        for event in result.events
    )


def test_psychic_terrain_does_not_block_priority_against_airborne_foe() -> None:
    state = _state(
        terrain="psychic_terrain",
        target_types=("flying",),
    )
    ours = _joint(
        _move(0, "attacker", "Fake Out", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _pass(1, "redirector"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", FAKE_OUT)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp < 120


def test_gravity_makes_flying_target_grounded_for_psychic_terrain() -> None:
    profile = _profile("target", types=("flying",))
    assert is_grounded(profile) is False
    assert is_grounded(profile, field_conditions={"gravity"}) is True


def test_wide_guard_blocks_spread_damage_to_both_allies() -> None:
    state = _state()
    ours = _joint(
        _move(0, "attacker", "Hyper Voice"),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Wide Guard"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", HYPER_VOICE)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp == 120
    assert result.state.profile(TurnSide.OPPONENT, "redirector").current_hp == 120
    assert any(
        event.type is SimulationEventType.WIDE_GUARD
        for event in result.events
    )


def test_wide_guard_does_not_block_single_target_move() -> None:
    state = _state()
    ours = _joint(
        _move(0, "attacker", "Body Slam", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Wide Guard"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", BODY_SLAM)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp < 120


def test_follow_me_redirects_single_target_attack() -> None:
    state = _state()
    ours = _joint(
        _move(0, "attacker", "Body Slam", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Follow Me"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", BODY_SLAM)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp == 120
    assert result.state.profile(TurnSide.OPPONENT, "redirector").current_hp < 120


def test_spread_move_bypasses_follow_me() -> None:
    state = _state()
    ours = _joint(
        _move(0, "attacker", "Rock Slide"),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Follow Me"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", ROCK_SLIDE)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp < 120
    assert result.state.profile(TurnSide.OPPONENT, "redirector").current_hp < 120


def test_grass_attacker_ignores_rage_powder() -> None:
    state = _state(attacker_types=("grass",))
    ours = _joint(
        _move(0, "attacker", "Body Slam", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Rage Powder"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", BODY_SLAM)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp < 120
    assert result.state.profile(TurnSide.OPPONENT, "redirector").current_hp == 120


def test_stalwart_bypasses_follow_me() -> None:
    state = _state(attacker_ability="stalwart")
    ours = _joint(
        _move(0, "attacker", "Body Slam", 1),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Follow Me"),
    )

    result = simulate_turn(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", BODY_SLAM)),
        _fixed_config(),
    )

    assert result.state.profile(TurnSide.OPPONENT, "target").current_hp < 120
    assert result.state.profile(TurnSide.OPPONENT, "redirector").current_hp == 120


def test_repeated_wide_guard_uses_shared_one_in_three_stall_odds() -> None:
    state = _state()
    state.protect_streaks[(TurnSide.OPPONENT, "redirector")] = 1

    ours = _joint(
        _move(0, "attacker", "Hyper Voice"),
        _pass(1, "partner"),
    )
    theirs = _joint(
        _pass(0, "target"),
        _move(1, "redirector", "Wide Guard"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(),
        _profile_map((TurnSide.PLAYER, "attacker", HYPER_VOICE)),
        policy=BranchingPolicy(
            branch_damage_rolls=False,
            branch_accuracy=False,
            branch_critical_hits=False,
            branch_secondary_effects=False,
            branch_protect=True,
            branch_speed_ties=False,
            fixed_damage_roll_index=7,
            merge_equivalent_states=False,
        ),
    )

    probabilities = sorted(
        outcome.probability
        for outcome in distribution.outcomes
    )
    assert probabilities == pytest.approx([1 / 3, 2 / 3])

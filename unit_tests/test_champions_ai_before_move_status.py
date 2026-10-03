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

SNORE = MoveProfile(
    move_id="snore",
    base_power=50,
    category="special",
    move_type="normal",
)

FLAME_WHEEL = MoveProfile(
    move_id="flamewheel",
    base_power=60,
    category="physical",
    move_type="fire",
)

FLAMETHROWER = MoveProfile(
    move_id="flamethrower",
    base_power=90,
    category="special",
    move_type="fire",
)


def _profile(
    name: str,
    *,
    hp: int = 160,
    status: str | None = None,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=160,
        types=("normal",),
        stats={
            "atk": 140,
            "def": 100,
            "spa": 140,
            "spd": 100,
            "spe": 100,
        },
        status=status,
        ability=ability,
    )


def _state(
    *,
    attacker_status: str | None = None,
    attacker_ability: str | None = None,
    target_status: str | None = None,
    sleep_turns: int | None = None,
):
    state = ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "attacker"): _profile(
                "attacker",
                status=attacker_status,
                ability=attacker_ability,
            ),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile(
                "target",
                status=target_status,
            ),
            (TurnSide.OPPONENT, "partner2"): _profile("partner2"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "attacker",
            (TurnSide.PLAYER, 1): "partner",
            (TurnSide.OPPONENT, 0): "target",
            (TurnSide.OPPONENT, 1): "partner2",
        },
    )

    if sleep_turns is not None:
        state.sleep_turns[(TurnSide.PLAYER, "attacker")] = sleep_turns

    return state


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


def _speeds():
    return {
        (TurnSide.PLAYER, "attacker"): SpeedState(200),
        (TurnSide.PLAYER, "partner"): SpeedState(90),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "partner2"): SpeedState(80),
    }


def _move_map(profile: MoveProfile):
    return {
        (TurnSide.PLAYER, "attacker", profile.move_id): profile,
    }


def _status_policy(*, merge: bool = False):
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=True,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=7,
        merge_equivalent_states=merge,
    )


def _fixed_config():
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


def test_paralysis_branches_one_quarter_full_paralysis() -> None:
    distribution = simulate_turn_distribution(
        _state(attacker_status="par"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_status_policy(),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([0.25, 0.75])

    assert any(
        any(
            event.type is SimulationEventType.CANNOT_MOVE
            for event in outcome.result.events
        )
        for outcome in distribution.outcomes
    )


def test_freeze_branches_twenty_percent_thaw() -> None:
    distribution = simulate_turn_distribution(
        _state(attacker_status="frz"),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_status_policy(),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([0.2, 0.8])

    statuses = {
        outcome.result.state.profile(
            TurnSide.PLAYER,
            "attacker",
        ).status
        for outcome in distribution.outcomes
    }
    assert statuses == {None, "frz"}


def test_known_sleep_counter_blocks_then_wakes_next_turn() -> None:
    state = _state(
        attacker_status="slp",
        sleep_turns=2,
    )

    first = simulate_turn(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert first.state.sleep_turns[
        (TurnSide.PLAYER, "attacker")
    ] == 1
    assert first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == 160

    second = simulate_turn(
        first.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert second.state.profile(
        TurnSide.PLAYER,
        "attacker",
    ).status is None
    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 160


def test_new_sleep_duration_branches_equally_over_three_lengths() -> None:
    distribution = simulate_turn_distribution(
        _state(
            attacker_status="slp",
            sleep_turns=0,
        ),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_status_policy(),
    )

    assert len(distribution.outcomes) == 3
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    remaining = {
        outcome.result.state.sleep_turns[
            (TurnSide.PLAYER, "attacker")
        ]
        for outcome in distribution.outcomes
    }
    assert remaining == {1, 2, 3}


def test_snore_can_be_used_while_still_asleep() -> None:
    result = simulate_turn(
        _state(
            attacker_status="slp",
            sleep_turns=3,
        ),
        _ours("Snore"),
        _theirs(),
        _speeds(),
        _move_map(SNORE),
        _fixed_config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "attacker",
    ).status == "slp"
    assert result.state.sleep_turns[
        (TurnSide.PLAYER, "attacker")
    ] == 2
    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 160


def test_early_bird_wakes_twice_as_fast() -> None:
    result = simulate_turn(
        _state(
            attacker_status="slp",
            attacker_ability="earlybird",
            sleep_turns=2,
        ),
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "attacker",
    ).status is None
    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 160


def test_defrost_move_cures_frozen_user_and_executes() -> None:
    result = simulate_turn(
        _state(attacker_status="frz"),
        _ours("Flame Wheel"),
        _theirs(),
        _speeds(),
        _move_map(FLAME_WHEEL),
        _fixed_config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER,
        "attacker",
    ).status is None
    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 160


def test_damaging_fire_move_thaws_frozen_target() -> None:
    result = simulate_turn(
        _state(target_status="frz"),
        _ours("Flamethrower"),
        _theirs(),
        _speeds(),
        _move_map(FLAMETHROWER),
        _fixed_config(),
    )

    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).status is None
    assert any(
        event.type is SimulationEventType.STATUS_CURED
        and event.actor == "target"
        for event in result.events
    )

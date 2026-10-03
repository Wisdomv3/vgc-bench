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

HURRICANE = MoveProfile(
    move_id="hurricane",
    base_power=110,
    category="special",
    move_type="flying",
)

HYPER_BEAM = MoveProfile(
    move_id="hyperbeam",
    base_power=150,
    category="special",
    move_type="normal",
)


def _profile(
    name: str,
    *,
    hp: int = 200,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=200,
        types=("normal",),
        stats={
            "atk": 140,
            "def": 100,
            "spa": 150,
            "spd": 100,
            "spe": 100,
        },
        ability=ability,
    )


def _state(
    *,
    attacker_ability: str | None = None,
    attacker_hp: int = 200,
):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "attacker"): _profile(
                "attacker",
                hp=attacker_hp,
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


def _speeds():
    return {
        (TurnSide.PLAYER, "attacker"): SpeedState(200),
        (TurnSide.PLAYER, "partner"): SpeedState(90),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "partner2"): SpeedState(80),
    }


def _move_map(*profiles: MoveProfile):
    return {
        (TurnSide.PLAYER, "attacker", profile.move_id): profile
        for profile in profiles
    }


def _branch_policy(
    *,
    secondary: bool = False,
    damage_rolls: bool = False,
    merge: bool = False,
):
    return BranchingPolicy(
        branch_damage_rolls=damage_rolls,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=secondary,
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


def test_confusion_self_hit_is_thirty_three_percent() -> None:
    state = _state()
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 2

    distribution = simulate_turn_distribution(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_branch_policy(),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([0.33, 0.67])

    assert any(
        any(
            event.type is SimulationEventType.CANNOT_MOVE
            and "confusion" in event.detail
            for event in outcome.result.events
        )
        for outcome in distribution.outcomes
    )


def test_confusion_ends_before_self_hit_check_when_counter_reaches_zero() -> None:
    state = _state()
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 1

    result = simulate_turn(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert (TurnSide.PLAYER, "attacker") not in result.state.confusion_turns
    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 200


def test_new_confusion_duration_branches_over_two_to_five_turns() -> None:
    state = _state()
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 0

    distribution = simulate_turn_distribution(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_branch_policy(),
    )

    assert len(distribution.outcomes) == 8
    assert distribution.total_probability == pytest.approx(1.0)

    remaining = {
        outcome.result.state.confusion_turns[
            (TurnSide.PLAYER, "attacker")
        ]
        for outcome in distribution.outcomes
    }
    assert remaining == {1, 2, 3, 4}


def test_own_tempo_clears_confusion_without_branching() -> None:
    state = _state(attacker_ability="owntempo")
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 3

    result = simulate_turn(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert (TurnSide.PLAYER, "attacker") not in result.state.confusion_turns
    assert result.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp < 200


def test_hurricane_secondary_can_apply_confusion() -> None:
    distribution = simulate_turn_distribution(
        _state(),
        _ours("Hurricane"),
        _theirs(),
        _speeds(),
        _move_map(HURRICANE),
        policy=_branch_policy(
            secondary=True,
            merge=False,
        ),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(
        outcome.probability
        for outcome in distribution.outcomes
    ) == pytest.approx([0.3, 0.7])

    confusion_states = {
        (TurnSide.OPPONENT, "target")
        in outcome.result.state.confusion_turns
        for outcome in distribution.outcomes
    }
    assert confusion_states == {False, True}


def test_confusion_self_hit_uses_damage_roll_branching() -> None:
    state = _state()
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 2

    distribution = simulate_turn_distribution(
        state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        policy=_branch_policy(
            damage_rolls=True,
            merge=False,
        ),
    )

    self_hit = [
        outcome
        for outcome in distribution.outcomes
        if any(
            event.move == "confused"
            and event.type is SimulationEventType.DAMAGE
            for event in outcome.result.events
        )
    ]

    assert self_hit
    assert sum(
        outcome.probability
        for outcome in self_hit
    ) == pytest.approx(0.33)


def test_hyper_beam_creates_recharge_turn() -> None:
    first = simulate_turn(
        _state(),
        _ours("Hyper Beam"),
        _theirs(),
        _speeds(),
        _move_map(HYPER_BEAM),
        _fixed_config(),
    )

    assert (TurnSide.PLAYER, "attacker") in first.state.must_recharge
    hp_after_hyper_beam = first.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp

    second = simulate_turn(
        first.state,
        _ours("Tackle"),
        _theirs(),
        _speeds(),
        _move_map(TACKLE),
        _fixed_config(),
    )

    assert (TurnSide.PLAYER, "attacker") not in second.state.must_recharge
    assert second.state.profile(
        TurnSide.OPPONENT,
        "target",
    ).current_hp == hp_after_hyper_beam
    assert any(
        event.type is SimulationEventType.RECHARGE
        for event in second.events
    )


def test_switch_clears_confusion_and_recharge_volatiles() -> None:
    state = _state()
    state.profiles[(TurnSide.PLAYER, "bench")] = _profile("bench")
    state.confusion_turns[(TurnSide.PLAYER, "attacker")] = 3
    state.must_recharge.add((TurnSide.PLAYER, "attacker"))

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
        {
            **_speeds(),
            (TurnSide.PLAYER, "bench"): SpeedState(100),
        },
        {},
        _fixed_config(),
    )

    assert (TurnSide.PLAYER, "attacker") not in result.state.confusion_turns
    assert (TurnSide.PLAYER, "attacker") not in result.state.must_recharge

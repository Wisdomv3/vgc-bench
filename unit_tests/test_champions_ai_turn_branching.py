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
)


def _profile(
    name: str,
    *,
    hp: int = 200,
    max_hp: int = 200,
    atk: int = 150,
    defense: int = 100,
    spa: int = 150,
    spd: int = 100,
    spe: int = 100,
    ability: str | None = None,
    item: str | None = None,
    types: tuple[str, ...] = ("normal",),
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
        item=item,
    )


def _move(
    slot: int,
    actor: str,
    move: str,
    *,
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


def _state(
    *,
    player_hp: int = 200,
    opponent_hp: int = 200,
    protect_streak: int = 0,
    player_ability: str | None = None,
    opponent_ability: str | None = None,
    opponent_item: str | None = None,
):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "garchomp"): _profile(
                "garchomp",
                hp=player_hp,
                max_hp=200,
                ability=player_ability,
            ),
            (TurnSide.PLAYER, "whimsicott"): _profile(
                "whimsicott",
                ability="prankster",
            ),
            (TurnSide.OPPONENT, "salamence"): _profile(
                "salamence",
                hp=opponent_hp,
                max_hp=200,
                ability=opponent_ability,
                item=opponent_item,
            ),
            (TurnSide.OPPONENT, "sneasler"): _profile("sneasler"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.PLAYER, 1): "whimsicott",
            (TurnSide.OPPONENT, 0): "salamence",
            (TurnSide.OPPONENT, 1): "sneasler",
        },
        protect_streaks=(
            {(TurnSide.PLAYER, "garchomp"): protect_streak}
            if protect_streak
            else {}
        ),
    )


def _speeds(
    *,
    garchomp: int = 100,
    whimsicott: int = 80,
    salamence: int = 150,
    sneasler: int = 140,
):
    return {
        (TurnSide.PLAYER, "garchomp"): SpeedState(garchomp),
        (TurnSide.PLAYER, "whimsicott"): SpeedState(whimsicott),
        (TurnSide.OPPONENT, "salamence"): SpeedState(salamence),
        (TurnSide.OPPONENT, "sneasler"): SpeedState(sneasler),
    }


def _profiles(*entries):
    return {
        (side, actor, profile.move_id): profile
        for side, actor, profile in entries
    }


BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=85,
    category="physical",
    move_type="normal",
)

ZAP_CANNON = MoveProfile(
    move_id="zapcannon",
    base_power=120,
    category="special",
    move_type="electric",
)


def test_damage_roll_probability_mass_is_exact() -> None:
    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", BODY_SLAM)),
        policy=BranchingPolicy(
            branch_damage_rolls=True,
            branch_accuracy=False,
            branch_critical_hits=False,
            branch_secondary_effects=False,
            branch_protect=False,
            branch_speed_ties=False,
            merge_equivalent_states=False,
        ),
    )

    assert distribution.total_probability == pytest.approx(1.0)
    assert len(distribution.outcomes) > 1
    assert all(
        outcome.probability > 0
        for outcome in distribution.outcomes
    )


def test_accuracy_branches_hit_and_miss_at_move_accuracy() -> None:
    ours = _joint(
        _move(0, "garchomp", "Zap Cannon", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", ZAP_CANNON)),
        policy=BranchingPolicy(
            branch_damage_rolls=False,
            branch_accuracy=True,
            branch_critical_hits=False,
            branch_secondary_effects=False,
            branch_protect=False,
            branch_speed_ties=False,
            fixed_damage_roll_index=15,
            merge_equivalent_states=False,
        ),
    )

    assert len(distribution.outcomes) == 2
    probabilities = sorted(outcome.probability for outcome in distribution.outcomes)
    assert probabilities == pytest.approx([0.5, 0.5])

    event_types = [
        {event.type for event in outcome.result.events}
        for outcome in distribution.outcomes
    ]
    assert any(SimulationEventType.MISS in types for types in event_types)
    assert any(SimulationEventType.DAMAGE in types for types in event_types)


def test_normal_critical_hit_branches_at_one_in_twenty_four() -> None:
    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", BODY_SLAM)),
        policy=BranchingPolicy(
            branch_damage_rolls=False,
            branch_accuracy=False,
            branch_critical_hits=True,
            branch_secondary_effects=False,
            branch_protect=False,
            branch_speed_ties=False,
            fixed_damage_roll_index=15,
            merge_equivalent_states=False,
        ),
    )

    probabilities = sorted(outcome.probability for outcome in distribution.outcomes)
    assert probabilities == pytest.approx([1 / 24, 23 / 24])
    assert any(
        any(event.type is SimulationEventType.CRITICAL for event in outcome.result.events)
        for outcome in distribution.outcomes
    )


def test_speed_tie_branches_fifty_fifty() -> None:
    state = _state(player_hp=80, opponent_hp=80)
    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(garchomp=150, salamence=150),
        _profiles(
            (TurnSide.PLAYER, "garchomp", BODY_SLAM),
            (TurnSide.OPPONENT, "salamence", BODY_SLAM),
        ),
        policy=BranchingPolicy(
            branch_damage_rolls=False,
            branch_accuracy=False,
            branch_critical_hits=False,
            branch_secondary_effects=False,
            branch_protect=False,
            branch_speed_ties=True,
            fixed_damage_roll_index=15,
            merge_equivalent_states=False,
        ),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(outcome.probability for outcome in distribution.outcomes) == pytest.approx(
        [0.5, 0.5]
    )

    survivors = {
        (
            outcome.result.state.profile(TurnSide.PLAYER, "garchomp").current_hp > 0,
            outcome.result.state.profile(TurnSide.OPPONENT, "salamence").current_hp > 0,
        )
        for outcome in distribution.outcomes
    }
    assert survivors == {(True, False), (False, True)}


def test_repeated_protect_branches_one_in_three() -> None:
    state = _state(protect_streak=1)
    ours = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(),
        _profiles((TurnSide.OPPONENT, "salamence", BODY_SLAM)),
        policy=BranchingPolicy(
            branch_damage_rolls=False,
            branch_accuracy=False,
            branch_critical_hits=False,
            branch_secondary_effects=False,
            branch_protect=True,
            branch_speed_ties=False,
            fixed_damage_roll_index=15,
            merge_equivalent_states=False,
        ),
    )

    assert len(distribution.outcomes) == 2
    assert sorted(outcome.probability for outcome in distribution.outcomes) == pytest.approx(
        [1 / 3, 2 / 3]
    )

    hp_values = sorted(
        outcome.result.state.profile(TurnSide.PLAYER, "garchomp").current_hp
        for outcome in distribution.outcomes
    )
    assert hp_values[0] < hp_values[1]
    assert hp_values[1] == 200


ROCK_SLIDE = MoveProfile(
    move_id="rockslide",
    base_power=75,
    category="physical",
    move_type="rock",
    spread=True,
)

MOONBLAST = MoveProfile(
    move_id="moonblast",
    base_power=95,
    category="special",
    move_type="fairy",
)

ICY_WIND = MoveProfile(
    move_id="icywind",
    base_power=55,
    category="special",
    move_type="ice",
    spread=True,
)

DIRE_CLAW = MoveProfile(
    move_id="direclaw",
    base_power=80,
    category="physical",
    move_type="poison",
)


def _secondary_only_policy(*, merge: bool = True) -> BranchingPolicy:
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=True,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=7,
        merge_equivalent_states=merge,
    )


def test_rock_slide_flinch_branches_and_can_cancel_move() -> None:
    state = _state()
    state.active_slots.pop((TurnSide.OPPONENT, 1))

    ours = _joint(
        _move(0, "garchomp", "Rock Slide"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(garchomp=200, salamence=100),
        _profiles(
            (TurnSide.PLAYER, "garchomp", ROCK_SLIDE),
            (TurnSide.OPPONENT, "salamence", BODY_SLAM),
        ),
        policy=_secondary_only_policy(merge=False),
    )

    assert sorted(outcome.probability for outcome in distribution.outcomes) == pytest.approx(
        [0.3, 0.7]
    )
    assert any(
        any(event.type is SimulationEventType.FLINCHED for event in outcome.result.events)
        for outcome in distribution.outcomes
    )
    assert any(
        outcome.result.state.profile(TurnSide.PLAYER, "garchomp").current_hp == 200
        for outcome in distribution.outcomes
    )


def test_inner_focus_prevents_rock_slide_flinch() -> None:
    state = _state(opponent_ability="innerfocus")
    state.active_slots.pop((TurnSide.OPPONENT, 1))

    ours = _joint(
        _move(0, "garchomp", "Rock Slide"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Body Slam", target_position=1),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(garchomp=200, salamence=100),
        _profiles(
            (TurnSide.PLAYER, "garchomp", ROCK_SLIDE),
            (TurnSide.OPPONENT, "salamence", BODY_SLAM),
        ),
        policy=_secondary_only_policy(),
    )

    assert len(distribution.outcomes) == 1
    assert distribution.outcomes[0].result.state.profile(
        TurnSide.PLAYER, "garchomp"
    ).current_hp < 200


def test_body_slam_paralysis_branches_thirty_seventy() -> None:
    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", BODY_SLAM)),
        policy=_secondary_only_policy(merge=False),
    )

    assert sorted(outcome.probability for outcome in distribution.outcomes) == pytest.approx(
        [0.3, 0.7]
    )
    statuses = {
        outcome.result.state.profile(TurnSide.OPPONENT, "salamence").status
        for outcome in distribution.outcomes
    }
    assert statuses == {None, "par"}


def test_moonblast_special_attack_drop_branches_thirty_seventy() -> None:
    ours = _joint(
        _move(0, "garchomp", "Moonblast", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", MOONBLAST)),
        policy=_secondary_only_policy(merge=False),
    )

    assert sorted(outcome.probability for outcome in distribution.outcomes) == pytest.approx(
        [0.3, 0.7]
    )
    stages = {
        outcome.result.state.profile(TurnSide.OPPONENT, "salamence").boosts.get(
            "spa", 0
        )
        for outcome in distribution.outcomes
    }
    assert stages == {-1, 0}


def test_guaranteed_icy_wind_drop_triggers_defiant() -> None:
    state = _state(player_ability="defiant")

    ours = _joint(
        _pass(0, "garchomp"),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _move(0, "salamence", "Icy Wind"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(),
        _profiles((TurnSide.OPPONENT, "salamence", ICY_WIND)),
        policy=_secondary_only_policy(),
    )

    garchomp = distribution.outcomes[0].result.state.profile(
        TurnSide.PLAYER, "garchomp"
    )
    assert garchomp.boosts["spe"] == -1
    assert garchomp.boosts["atk"] == 2


def test_covert_cloak_blocks_damaging_move_secondary_effects() -> None:
    state = _state(opponent_item="covertcloak")
    ours = _joint(
        _move(0, "garchomp", "Body Slam", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        state,
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", BODY_SLAM)),
        policy=_secondary_only_policy(),
    )

    assert len(distribution.outcomes) == 1
    assert distribution.outcomes[0].result.state.profile(
        TurnSide.OPPONENT, "salamence"
    ).status is None


def test_dire_claw_branches_no_status_and_three_equal_statuses() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dire Claw", target_position=1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    distribution = simulate_turn_distribution(
        _state(),
        ours,
        theirs,
        _speeds(garchomp=200),
        _profiles((TurnSide.PLAYER, "garchomp", DIRE_CLAW)),
        policy=_secondary_only_policy(merge=False),
    )

    by_status = {
        outcome.result.state.profile(
            TurnSide.OPPONENT, "salamence"
        ).status: outcome.probability
        for outcome in distribution.outcomes
    }

    assert by_status[None] == pytest.approx(0.5)
    assert by_status["psn"] == pytest.approx(1 / 6)
    assert by_status["par"] == pytest.approx(1 / 6)
    assert by_status["slp"] == pytest.approx(1 / 6)

import pytest

from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.decision_pipeline import (
    rank_decision,
    speed_states_from_exact_state,
)
from vgc_bench.src.champions_ai.events import Side
from vgc_bench.src.champions_ai.guards import GuardCode
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.opponent_model import (
    OpponentActionCandidate,
    OpponentHabitTracker,
)
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.state import BattleState
from vgc_bench.src.champions_ai.turn_branching import BranchingPolicy
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=85,
    category="physical",
    move_type="normal",
)

TACKLE = MoveProfile(
    move_id="tackle",
    base_power=40,
    category="physical",
    move_type="normal",
)

HYPER_VOICE = MoveProfile(
    move_id="hypervoice",
    base_power=300,
    category="special",
    move_type="normal",
    spread=True,
)


def _profile(
    name: str,
    hp: int = 100,
    *,
    speed: int = 100,
    item: str | None = None,
    ability: str | None = None,
    status: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=100,
        types=("normal",),
        stats={
            "atk": 150,
            "def": 100,
            "spa": 150,
            "spd": 100,
            "spe": speed,
        },
        item=item,
        ability=ability,
        status=status,
    )


def _state(
    *,
    player_hp: int = 100,
    partner_hp: int = 100,
    opponent_hp: int = 35,
) -> ExactTurnState:
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "garchomp"): _profile(
                "garchomp",
                player_hp,
                speed=200,
            ),
            (TurnSide.PLAYER, "whimsicott"): _profile(
                "whimsicott",
                partner_hp,
                speed=150,
            ),
            (TurnSide.OPPONENT, "salamence"): _profile(
                "salamence",
                opponent_hp,
                speed=120,
            ),
            (TurnSide.OPPONENT, "sneasler"): _profile(
                "sneasler",
                100,
                speed=110,
            ),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.PLAYER, 1): "whimsicott",
            (TurnSide.OPPONENT, 0): "salamence",
            (TurnSide.OPPONENT, 1): "sneasler",
        },
    )


def _snapshot(
    *,
    first_turn: bool | None = True,
    protect_streak: int = 0,
) -> DecisionSnapshot:
    state = BattleState()

    garchomp = state.get_or_create_pokemon(
        Side.PLAYER,
        "garchomp",
    )
    garchomp.active_slot = 0
    garchomp.first_turn = first_turn
    garchomp.protect_streak = protect_streak
    state.player.active_slots[0] = "garchomp"

    whimsicott = state.get_or_create_pokemon(
        Side.PLAYER,
        "whimsicott",
    )
    whimsicott.active_slot = 1
    whimsicott.first_turn = True
    state.player.active_slots[1] = "whimsicott"

    salamence = state.get_or_create_pokemon(
        Side.OPPONENT,
        "salamence",
    )
    salamence.active_slot = 0
    state.opponent.active_slots[0] = "salamence"

    sneasler = state.get_or_create_pokemon(
        Side.OPPONENT,
        "sneasler",
    )
    sneasler.active_slot = 1
    state.opponent.active_slots[1] = "sneasler"

    return DecisionSnapshot.from_state(state)


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


def _attack():
    return _joint(
        _move(0, "garchomp", "Body Slam", 1),
        _pass(1, "whimsicott"),
    )


def _protect():
    return _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )


def _opponent_pass():
    return _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )


def _speeds():
    return {
        (TurnSide.PLAYER, "garchomp"): SpeedState(200),
        (TurnSide.PLAYER, "whimsicott"): SpeedState(150),
        (TurnSide.OPPONENT, "salamence"): SpeedState(120),
        (TurnSide.OPPONENT, "sneasler"): SpeedState(110),
    }


def _policy():
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=False,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=15,
    )


def test_pipeline_recommends_best_nonblocked_action() -> None:
    report = rank_decision(
        _snapshot(first_turn=True),
        _state(),
        (_protect(), _attack()),
        (OpponentActionCandidate(_opponent_pass()),),
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
        },
        branching_policy=_policy(),
    )

    assert report.recommendation is not None
    assert report.best_play == _attack()
    assert report.recommendation.rank == 1
    assert report.recommendation.expected_value is not None
    assert report.recommendation.expected_value > 0

    protect_option = next(
        option
        for option in report.blocked_options
        if option.action == _protect()
    )
    assert any(
        finding.code is GuardCode.STRICTLY_DOMINATED
        for finding in protect_option.findings
    )


def test_rule_guard_blocks_impossible_fake_out_before_simulation() -> None:
    fake_out = _joint(
        _move(0, "garchomp", "Fake Out", 1),
        _pass(1, "whimsicott"),
    )

    report = rank_decision(
        _snapshot(first_turn=False),
        _state(),
        (fake_out, _attack()),
        (OpponentActionCandidate(_opponent_pass()),),
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
        },
        branching_policy=_policy(),
    )

    assert report.best_play == _attack()
    blocked = next(
        option
        for option in report.blocked_options
        if option.action == fake_out
    )
    assert blocked.expected_value is None
    assert any(
        finding.code is GuardCode.FAKE_OUT_NOT_FIRST_TURN
        for finding in blocked.findings
    )


def test_pipeline_preserves_habit_weighted_opponent_distribution() -> None:
    opponent_attack = _joint(
        _move(0, "salamence", "Tackle", 1),
        _pass(1, "sneasler"),
    )
    opponent_protect = _joint(
        _move(0, "salamence", "Protect"),
        _pass(1, "sneasler"),
    )

    tracker = OpponentHabitTracker()
    tracker.record(
        "threatened",
        "protect+passive",
    )

    report = rank_decision(
        _snapshot(),
        _state(opponent_hp=100),
        (_attack(),),
        (
            OpponentActionCandidate(
                opponent_attack,
                prior_weight=1,
            ),
            OpponentActionCandidate(
                opponent_protect,
                prior_weight=0,
            ),
        ),
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
            (TurnSide.OPPONENT, "salamence", "tackle"): TACKLE,
        },
        tracker=tracker,
        context_key="threatened",
        habit_weight=0.5,
        branching_policy=_policy(),
    )

    by_behavior = {
        estimate.candidate.behavior: estimate.probability
        for estimate in report.opponent_estimates
    }
    assert sum(by_behavior.values()) == pytest.approx(1.0)
    assert by_behavior["protect+passive"] == pytest.approx(0.5)
    assert by_behavior["passive+targeted_move"] == pytest.approx(0.5)


def test_pipeline_blocks_action_that_guarantees_immediate_match_loss() -> None:
    state = _state(
        player_hp=5,
        partner_hp=5,
        opponent_hp=100,
    )
    opponent_spread = _joint(
        _move(0, "salamence", "Hyper Voice"),
        _pass(1, "sneasler"),
    )

    report = rank_decision(
        _snapshot(),
        state,
        (_opponent_pass(),),
        (OpponentActionCandidate(opponent_spread),),
        _speeds(),
        {
            (TurnSide.OPPONENT, "salamence", "hypervoice"): HYPER_VOICE,
        },
        branching_policy=_policy(),
    )

    assert report.recommendation is None
    assert len(report.blocked_options) == 1
    assert any(
        finding.code is GuardCode.GUARANTEED_MATCH_LOSS
        for finding in report.blocked_options[0].findings
    )


def test_speed_state_builder_uses_supported_live_modifiers() -> None:
    state = ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "swift"): _profile(
                "swift",
                speed=100,
                item="choicescarf",
                ability="swiftswim",
                status="par",
            ),
            (TurnSide.OPPONENT, "sneasler"): _profile(
                "sneasler",
                speed=120,
                ability="unburden",
            ),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "swift",
            (TurnSide.OPPONENT, 0): "sneasler",
        },
        weather="raindance",
        tailwind_sides={TurnSide.PLAYER},
    )

    speeds = speed_states_from_exact_state(
        state,
        unburden_active=frozenset({
            (TurnSide.OPPONENT, "sneasler"),
        }),
    )

    player = speeds[(TurnSide.PLAYER, "swift")]
    assert player.speed_stat == 100
    assert player.tailwind is True
    assert player.choice_scarf is True
    assert player.weather_speed_boost is True
    assert player.paralyzed is True

    opponent = speeds[(TurnSide.OPPONENT, "sneasler")]
    assert opponent.unburden is True

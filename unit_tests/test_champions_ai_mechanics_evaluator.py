from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.mechanics_evaluator import (
    build_mechanics_response_matrix,
    evaluate_action_pair,
)
from vgc_bench.src.champions_ai.opponent_model import (
    OpponentActionCandidate,
    estimate_action_probabilities,
)
from vgc_bench.src.champions_ai.speed import SpeedState
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


def _profile(name: str, hp: int, *, defense: int = 100):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=100,
        types=("normal",),
        stats={
            "atk": 150,
            "def": defense,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        },
    )


def _state(opponent_hp: int = 35) -> ExactTurnState:
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "garchomp"): _profile("garchomp", 100),
            (TurnSide.PLAYER, "whimsicott"): _profile("whimsicott", 100),
            (TurnSide.OPPONENT, "salamence"): _profile(
                "salamence",
                opponent_hp,
                defense=70,
            ),
            (TurnSide.OPPONENT, "sneasler"): _profile("sneasler", 100),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.PLAYER, 1): "whimsicott",
            (TurnSide.OPPONENT, 0): "salamence",
            (TurnSide.OPPONENT, 1): "sneasler",
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


def _speeds():
    return {
        (TurnSide.PLAYER, "garchomp"): SpeedState(200),
        (TurnSide.PLAYER, "whimsicott"): SpeedState(100),
        (TurnSide.OPPONENT, "salamence"): SpeedState(120),
        (TurnSide.OPPONENT, "sneasler"): SpeedState(110),
    }


def _policy():
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=15,
    )


def test_action_pair_uses_real_simulated_position_delta() -> None:
    ours = _joint(
        _move(0, "garchomp", "Body Slam", 1),
        _pass(1, "whimsicott"),
    )
    theirs = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    evaluation = evaluate_action_pair(
        _state(),
        ours,
        theirs,
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
        },
        branching_policy=_policy(),
    )

    assert evaluation.expected_score_delta > 100
    assert evaluation.outcome_count == 1


def test_response_matrix_ranks_ko_line_over_passive_line() -> None:
    attack = _joint(
        _move(0, "garchomp", "Body Slam", 1),
        _pass(1, "whimsicott"),
    )
    protect = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )
    opponent_pass = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    opponent_estimates = estimate_action_probabilities(
        (OpponentActionCandidate(opponent_pass),)
    )

    matrix = build_mechanics_response_matrix(
        _state(),
        (protect, attack),
        opponent_estimates,
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
        },
        branching_policy=_policy(),
    )

    assert matrix[0].our_action == attack
    assert matrix[0].expected_value > matrix[1].expected_value


def test_response_matrix_weights_multiple_opponent_responses() -> None:
    attack = _joint(
        _move(0, "garchomp", "Body Slam", 1),
        _pass(1, "whimsicott"),
    )
    protect = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )
    opponent_attack = _joint(
        _move(0, "salamence", "Tackle", 1),
        _pass(1, "sneasler"),
    )
    opponent_pass = _joint(
        _pass(0, "salamence"),
        _pass(1, "sneasler"),
    )

    opponent_estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(opponent_attack, prior_weight=3),
            OpponentActionCandidate(opponent_pass, prior_weight=1),
        )
    )

    matrix = build_mechanics_response_matrix(
        _state(opponent_hp=100),
        (attack, protect),
        opponent_estimates,
        _speeds(),
        {
            (TurnSide.PLAYER, "garchomp", "bodyslam"): BODY_SLAM,
            (TurnSide.OPPONENT, "salamence", "tackle"): TACKLE,
        },
        branching_policy=_policy(),
    )

    assert len(matrix) == 2
    assert matrix[0].cells[0].opponent_probability in {0.75, 0.25}
    assert abs(sum(cell.opponent_probability for cell in matrix[0].cells) - 1.0) < 1e-9

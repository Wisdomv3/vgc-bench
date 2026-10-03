from dataclasses import replace

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.guards import (
    GuardCode,
    GuardSeverity,
    guaranteed_match_loss_finding,
    inspect_action_rules,
    is_guaranteed_match_loss,
    strictly_dominated_findings,
)
from vgc_bench.src.champions_ai.mechanics_evaluator import ActionPairEvaluation
from vgc_bench.src.champions_ai.opponent_model import OpponentActionCandidate
from vgc_bench.src.champions_ai.response_matrix import build_response_matrix
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.state import BattleState
from vgc_bench.src.champions_ai.turn_branching import (
    TurnOutcomeDistribution,
    WeightedTurnOutcome,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    TurnSimulationResult,
)


def _move(slot: int, actor: str, move: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
    )


def _pass(slot: int, actor: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.PASS,
        actor=actor,
    )


def _joint(first, second):
    return JointAction(first=first, second=second)


def _snapshot(*, first_turn: bool | None, protect_streak: int = 0):
    state = BattleState()
    garchomp = state.get_or_create_pokemon(
        __import__(
            "vgc_bench.src.champions_ai.events",
            fromlist=["Side"],
        ).Side.PLAYER,
        "garchomp",
    )
    garchomp.active_slot = 0
    garchomp.first_turn = first_turn
    garchomp.protect_streak = protect_streak
    state.player.active_slots[0] = "garchomp"

    partner = state.get_or_create_pokemon(
        __import__(
            "vgc_bench.src.champions_ai.events",
            fromlist=["Side"],
        ).Side.PLAYER,
        "whimsicott",
    )
    partner.active_slot = 1
    partner.first_turn = True
    state.player.active_slots[1] = "whimsicott"

    return DecisionSnapshot.from_state(state)


def test_fake_out_after_first_turn_is_blocked() -> None:
    action = _joint(
        _move(0, "garchomp", "Fake Out"),
        _pass(1, "whimsicott"),
    )

    findings = inspect_action_rules(
        _snapshot(first_turn=False),
        action,
    )

    assert len(findings) == 1
    assert findings[0].code is GuardCode.FAKE_OUT_NOT_FIRST_TURN
    assert findings[0].severity is GuardSeverity.BLOCK


def test_fake_out_first_turn_is_not_blocked() -> None:
    action = _joint(
        _move(0, "garchomp", "Fake Out"),
        _pass(1, "whimsicott"),
    )

    assert inspect_action_rules(
        _snapshot(first_turn=True),
        action,
    ) == ()


def test_unknown_first_turn_does_not_create_false_block() -> None:
    action = _joint(
        _move(0, "garchomp", "Fake Out"),
        _pass(1, "whimsicott"),
    )

    assert inspect_action_rules(
        _snapshot(first_turn=None),
        action,
    ) == ()


def test_repeated_protect_creates_warning() -> None:
    action = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )

    findings = inspect_action_rules(
        _snapshot(first_turn=False, protect_streak=1),
        action,
    )

    assert len(findings) == 1
    assert findings[0].code is GuardCode.REPEATED_PROTECT_RISK
    assert findings[0].severity is GuardSeverity.WARNING
    assert "33.333%" in findings[0].message


def test_strict_dominance_blocks_only_dominated_action() -> None:
    attack = _joint(
        _move(0, "garchomp", "Dragon Claw"),
        _pass(1, "whimsicott"),
    )
    passive = _joint(
        _move(0, "garchomp", "Protect"),
        _pass(1, "whimsicott"),
    )
    response_a = _joint(
        _move(0, "salamence", "Draco Meteor"),
        _pass(1, "sneasler"),
    )
    response_b = _joint(
        _move(0, "salamence", "Protect"),
        _pass(1, "sneasler"),
    )

    from vgc_bench.src.champions_ai.opponent_model import (
        estimate_action_probabilities,
    )

    opponent_estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(response_a),
            OpponentActionCandidate(response_b),
        )
    )

    values = {
        (attack.key, response_a.key): 3.0,
        (attack.key, response_b.key): 2.0,
        (passive.key, response_a.key): 1.0,
        (passive.key, response_b.key): 2.0,
    }

    matrix = build_response_matrix(
        (attack, passive),
        opponent_estimates,
        lambda ours, theirs: values[(ours.key, theirs.key)],
    )

    findings = strictly_dominated_findings(matrix)

    assert len(findings) == 1
    assert findings[0].action == passive
    assert findings[0].alternative == attack


def _terminal_state(player_hp: int, opponent_hp: int) -> ExactTurnState:
    from vgc_bench.src.champions_ai.matchup import CombatantProfile

    def profile(name: str, hp: int):
        return CombatantProfile(
            name=name,
            level=50,
            current_hp=hp,
            max_hp=100,
            types=("normal",),
            stats={
                "atk": 100,
                "def": 100,
                "spa": 100,
                "spd": 100,
                "spe": 100,
            },
        )

    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "garchomp"): profile("garchomp", player_hp),
            (TurnSide.OPPONENT, "salamence"): profile("salamence", opponent_hp),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.OPPONENT, 0): "salamence",
        },
    )


def _pair_evaluation(outcomes):
    action = _joint(
        _move(0, "garchomp", "Dragon Claw"),
        _pass(1, "whimsicott"),
    )
    opponent = _joint(
        _move(0, "salamence", "Draco Meteor"),
        _pass(1, "sneasler"),
    )

    distribution = TurnOutcomeDistribution(
        tuple(
            WeightedTurnOutcome(
                probability=probability,
                result=TurnSimulationResult(state=state, events=()),
                decisions=(),
            )
            for probability, state in outcomes
        )
    )

    return ActionPairEvaluation(
        our_action=action,
        opponent_action=opponent,
        initial_score=0,
        expected_final_score=-100,
        expected_score_delta=-100,
        worst_final_score=-100,
        best_final_score=-100,
        outcome_count=len(distribution.outcomes),
        distribution=distribution,
    )


def test_guaranteed_match_loss_requires_every_branch_to_lose() -> None:
    evaluation = _pair_evaluation(
        (
            (0.5, _terminal_state(0, 50)),
            (0.5, _terminal_state(20, 50)),
        )
    )

    assert is_guaranteed_match_loss(evaluation) is False
    assert guaranteed_match_loss_finding(evaluation) is None


def test_guaranteed_match_loss_is_blocked_when_all_branches_lose() -> None:
    evaluation = _pair_evaluation(
        (
            (0.6, _terminal_state(0, 50)),
            (0.4, _terminal_state(0, 10)),
        )
    )

    assert is_guaranteed_match_loss(evaluation) is True
    finding = guaranteed_match_loss_finding(evaluation)
    assert finding is not None
    assert finding.code is GuardCode.GUARANTEED_MATCH_LOSS
    assert finding.severity is GuardSeverity.BLOCK

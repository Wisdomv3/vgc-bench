import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.opponent_model import (
    OpponentActionCandidate,
    OpponentHabitTracker,
    action_features,
    behavior_key_from_action,
    estimate_action_probabilities,
    feature_probability,
)


def _move(
    slot: int,
    actor: str,
    move: str,
    target: str | None = None,
) -> SlotAction:
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target=target,
    )


def _joint(first: SlotAction, second: SlotAction) -> JointAction:
    return JointAction(first=first, second=second)


def test_baseline_prior_weights_are_normalized() -> None:
    protect_attack = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    double_attack = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(protect_attack, prior_weight=3),
            OpponentActionCandidate(double_attack, prior_weight=1),
        )
    )

    assert estimates[0].baseline_probability == 0.75
    assert estimates[0].probability == 0.75
    assert estimates[1].probability == 0.25


def test_behavior_key_is_coarse_and_order_independent() -> None:
    first = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    second = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Protect"),
    )

    assert behavior_key_from_action(first) == "protect+targeted_move"
    assert behavior_key_from_action(second) == "protect+targeted_move"


def test_habit_evidence_shifts_probability_only_in_same_context() -> None:
    protect_attack = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    double_attack = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    candidates = (
        OpponentActionCandidate(protect_attack, prior_weight=1),
        OpponentActionCandidate(double_attack, prior_weight=1),
    )

    tracker = OpponentHabitTracker()
    for _ in range(3):
        tracker.record("low_hp_threatened", "protect+targeted_move")
    tracker.record("low_hp_threatened", "targeted_move+targeted_move")
    tracker.record("different_context", "targeted_move+targeted_move")

    estimates = estimate_action_probabilities(
        candidates,
        tracker=tracker,
        context_key="low_hp_threatened",
        habit_weight=0.5,
    )

    by_behavior = {
        result.candidate.behavior: result
        for result in estimates
    }

    assert by_behavior["protect+targeted_move"].probability == pytest.approx(0.625)
    assert by_behavior["targeted_move+targeted_move"].probability == pytest.approx(
        0.375
    )
    assert all(result.comparable_observations == 4 for result in estimates)


def test_no_comparable_observations_falls_back_to_baseline() -> None:
    protect_attack = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    double_attack = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    tracker = OpponentHabitTracker()
    tracker.record("other_context", "protect+targeted_move")

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(protect_attack, prior_weight=3),
            OpponentActionCandidate(double_attack, prior_weight=1),
        ),
        tracker=tracker,
        context_key="current_context",
        habit_weight=1.0,
    )

    assert estimates[0].probability == 0.75
    assert estimates[1].probability == 0.25
    assert estimates[0].comparable_observations == 0


def test_actions_in_same_behavior_preserve_baseline_share() -> None:
    action_a = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    action_b = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Dire Claw", "whimsicott"),
    )
    action_c = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Dire Claw", "whimsicott"),
    )

    tracker = OpponentHabitTracker()
    tracker.record("threatened", "protect+targeted_move")

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(action_a, prior_weight=3),
            OpponentActionCandidate(action_b, prior_weight=1),
            OpponentActionCandidate(action_c, prior_weight=1),
        ),
        tracker=tracker,
        context_key="threatened",
        habit_weight=1.0,
    )

    protect_actions = [
        result
        for result in estimates
        if result.candidate.behavior == "protect+targeted_move"
    ]

    assert protect_actions[0].probability == pytest.approx(0.75)
    assert protect_actions[1].probability == pytest.approx(0.25)


def test_feature_probability_reports_target_tendency() -> None:
    target_garchomp = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Protect"),
    )
    target_whimsicott = _joint(
        _move(0, "salamence", "Draco Meteor", "whimsicott"),
        _move(1, "sneasler", "Protect"),
    )

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(target_garchomp, prior_weight=3),
            OpponentActionCandidate(target_whimsicott, prior_weight=1),
        )
    )

    assert "target:garchomp" in action_features(target_garchomp)
    assert feature_probability(estimates, "target:garchomp") == pytest.approx(0.75)
    assert feature_probability(estimates, "target:whimsicott") == pytest.approx(
        0.25
    )


def test_invalid_habit_weight_is_rejected() -> None:
    action = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    with pytest.raises(ValueError, match="habit_weight"):
        estimate_action_probabilities(
            (OpponentActionCandidate(action),),
            habit_weight=1.1,
        )


def test_zero_baseline_observed_behavior_keeps_empirical_mass() -> None:
    protect_attack = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    double_attack = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    tracker = OpponentHabitTracker()
    tracker.record("threatened", "protect+targeted_move")

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(protect_attack, prior_weight=0),
            OpponentActionCandidate(double_attack, prior_weight=1),
        ),
        tracker=tracker,
        context_key="threatened",
        habit_weight=1.0,
    )

    by_behavior = {
        estimate.candidate.behavior: estimate
        for estimate in estimates
    }

    assert sum(
        estimate.probability
        for estimate in estimates
    ) == pytest.approx(1.0)
    assert by_behavior[
        "protect+targeted_move"
    ].baseline_probability == 0
    assert by_behavior[
        "protect+targeted_move"
    ].empirical_probability == pytest.approx(1.0)
    assert by_behavior[
        "protect+targeted_move"
    ].probability == pytest.approx(1.0)
    assert by_behavior[
        "targeted_move+targeted_move"
    ].probability == pytest.approx(0.0)


def test_zero_baseline_behavior_splits_empirical_mass_equally() -> None:
    action_a = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    action_b = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Dire Claw", "whimsicott"),
    )
    action_c = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Dire Claw", "whimsicott"),
    )

    tracker = OpponentHabitTracker()
    tracker.record("threatened", "protect+targeted_move")

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(action_a, prior_weight=0),
            OpponentActionCandidate(action_b, prior_weight=0),
            OpponentActionCandidate(action_c, prior_weight=1),
        ),
        tracker=tracker,
        context_key="threatened",
        habit_weight=1.0,
    )

    protect = [
        estimate
        for estimate in estimates
        if estimate.candidate.behavior == "protect+targeted_move"
    ]

    assert len(protect) == 2
    assert all(
        estimate.empirical_probability == pytest.approx(0.5)
        for estimate in protect
    )
    assert all(
        estimate.probability == pytest.approx(0.5)
        for estimate in protect
    )
    assert sum(
        estimate.probability
        for estimate in estimates
    ) == pytest.approx(1.0)


def test_zero_baseline_habit_blend_remains_normalized() -> None:
    protect_attack = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    double_attack = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    tracker = OpponentHabitTracker()
    tracker.record("threatened", "protect+targeted_move")

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(protect_attack, prior_weight=0),
            OpponentActionCandidate(double_attack, prior_weight=1),
        ),
        tracker=tracker,
        context_key="threatened",
        habit_weight=0.25,
    )

    by_behavior = {
        estimate.candidate.behavior: estimate.probability
        for estimate in estimates
    }

    assert by_behavior["protect+targeted_move"] == pytest.approx(0.25)
    assert by_behavior[
        "targeted_move+targeted_move"
    ] == pytest.approx(0.75)
    assert sum(by_behavior.values()) == pytest.approx(1.0)


def test_all_zero_priors_use_neutral_baseline() -> None:
    action_a = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    action_b = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(action_a, prior_weight=0),
            OpponentActionCandidate(action_b, prior_weight=0),
        )
    )

    assert all(
        estimate.baseline_probability == pytest.approx(0.5)
        for estimate in estimates
    )
    assert all(
        estimate.probability == pytest.approx(0.5)
        for estimate in estimates
    )


@pytest.mark.parametrize("prior_weight", [float("inf"), float("nan")])
def test_nonfinite_prior_weight_is_rejected(prior_weight: float) -> None:
    action = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    with pytest.raises(ValueError, match="finite"):
        OpponentActionCandidate(
            action,
            prior_weight=prior_weight,
        )

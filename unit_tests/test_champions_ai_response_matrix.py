import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.opponent_model import (
    OpponentActionCandidate,
    estimate_action_probabilities,
)
from vgc_bench.src.champions_ai.response_matrix import build_response_matrix


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


def _actions():
    protect_pressure = _joint(
        _move(0, "garchomp", "Protect"),
        _move(1, "whimsicott", "Moonblast", "salamence"),
    )
    attack_tailwind = _joint(
        _move(0, "garchomp", "Draco Meteor", "salamence"),
        _move(1, "whimsicott", "Tailwind"),
    )

    opponent_spread = _joint(
        _move(0, "salamence", "Hyper Voice"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )
    opponent_protect = _joint(
        _move(0, "salamence", "Protect"),
        _move(1, "sneasler", "Close Combat", "garchomp"),
    )

    return (
        protect_pressure,
        attack_tailwind,
        opponent_spread,
        opponent_protect,
    )


def test_expected_value_is_probability_weighted() -> None:
    protect_pressure, _, opponent_spread, opponent_protect = _actions()

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(opponent_spread, prior_weight=3),
            OpponentActionCandidate(opponent_protect, prior_weight=1),
        )
    )

    values = {
        (protect_pressure.label, opponent_spread.label): 0.8,
        (protect_pressure.label, opponent_protect.label): 0.4,
    }

    summaries = build_response_matrix(
        (protect_pressure,),
        estimates,
        lambda ours, theirs: values[(ours.label, theirs.label)],
    )

    assert summaries[0].expected_value == pytest.approx(
        0.75 * 0.8 + 0.25 * 0.4
    )


def test_actions_are_ranked_by_expected_value() -> None:
    (
        protect_pressure,
        attack_tailwind,
        opponent_spread,
        opponent_protect,
    ) = _actions()

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(opponent_spread, prior_weight=3),
            OpponentActionCandidate(opponent_protect, prior_weight=1),
        )
    )

    values = {
        (protect_pressure.label, opponent_spread.label): 0.8,
        (protect_pressure.label, opponent_protect.label): 0.4,
        (attack_tailwind.label, opponent_spread.label): 0.1,
        (attack_tailwind.label, opponent_protect.label): 0.9,
    }

    summaries = build_response_matrix(
        (attack_tailwind, protect_pressure),
        estimates,
        lambda ours, theirs: values[(ours.label, theirs.label)],
    )

    assert summaries[0].our_action == protect_pressure
    assert summaries[0].expected_value == pytest.approx(0.7)
    assert summaries[1].expected_value == pytest.approx(0.3)


def test_worst_and_best_case_values_are_preserved() -> None:
    protect_pressure, _, opponent_spread, opponent_protect = _actions()

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(opponent_spread, prior_weight=1),
            OpponentActionCandidate(opponent_protect, prior_weight=1),
        )
    )

    values = {
        (protect_pressure.label, opponent_spread.label): -0.2,
        (protect_pressure.label, opponent_protect.label): 0.9,
    }

    summary = build_response_matrix(
        (protect_pressure,),
        estimates,
        lambda ours, theirs: values[(ours.label, theirs.label)],
    )[0]

    assert summary.worst_case_value == -0.2
    assert summary.best_case_value == 0.9


def test_most_likely_response_is_available_for_explanations() -> None:
    protect_pressure, _, opponent_spread, opponent_protect = _actions()

    estimates = estimate_action_probabilities(
        (
            OpponentActionCandidate(opponent_spread, prior_weight=4),
            OpponentActionCandidate(opponent_protect, prior_weight=1),
        )
    )

    summary = build_response_matrix(
        (protect_pressure,),
        estimates,
        lambda _ours, _theirs: 0.0,
    )[0]

    assert summary.most_likely_response.opponent_action == opponent_spread
    assert summary.most_likely_response.opponent_probability == pytest.approx(0.8)


def test_empty_our_action_list_returns_empty_matrix() -> None:
    _, _, opponent_spread, _ = _actions()

    estimates = estimate_action_probabilities(
        (OpponentActionCandidate(opponent_spread),)
    )

    assert build_response_matrix((), estimates, lambda _ours, _theirs: 0.0) == ()


def test_missing_opponent_distribution_is_rejected() -> None:
    protect_pressure, _, _, _ = _actions()

    with pytest.raises(ValueError, match="opponent action estimate"):
        build_response_matrix(
            (protect_pressure,),
            (),
            lambda _ours, _theirs: 0.0,
        )

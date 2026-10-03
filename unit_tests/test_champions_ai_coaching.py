from dataclasses import FrozenInstanceError, replace

import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.coaching import (
    ProvisionalGradeRubric,
    build_coaching_report,
    coach_decision,
    render_coaching_output,
)
from vgc_bench.src.champions_ai.coaching_metrics import build_turn_outcome_metrics
from vgc_bench.src.champions_ai.decision_pipeline import rank_decision
from vgc_bench.src.champions_ai.grades import GRADE_ORDER, DecisionGrade
from vgc_bench.src.champions_ai.matchup import MoveProfile
from vgc_bench.src.champions_ai.mechanics_evaluator import (
    ActionPairEvaluation,
    MechanicsResponseAnalysis,
)
from vgc_bench.src.champions_ai.opponent_model import (
    OpponentActionCandidate,
    OpponentHabitTracker,
    estimate_action_probabilities,
)
from vgc_bench.src.champions_ai.response_matrix import build_response_matrix
from vgc_bench.src.champions_ai.search import SearchConfig
from vgc_bench.src.champions_ai.search_demo import demo_inputs
from vgc_bench.src.champions_ai.speed_context import speed_states_from_exact_state
from vgc_bench.src.champions_ai.turn_branching import (
    TurnOutcomeDistribution,
    WeightedTurnOutcome,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import TurnSimulationResult


def _raw(depth=2, **kwargs):
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    kwargs.setdefault(
        "search_config", SearchConfig(depth=depth, max_observation_branches=None)
    )
    kwargs.setdefault("include_coaching_metrics", True)
    report = rank_decision(
        snapshot,
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy,
        **kwargs,
    )
    return report, immediate, setup


def _faint(state, side, name):
    state.profiles[(side, name)] = replace(state.profile(side, name), current_hp=0)


def _analysis(distributions, estimates, action):
    pairs = tuple(
        ActionPairEvaluation(
            action,
            estimate.candidate.action,
            0,
            0,
            0,
            0,
            0,
            len(distribution.outcomes),
            distribution,
        )
        for estimate, distribution in zip(estimates, distributions)
    )
    return MechanicsResponseAnalysis(
        build_response_matrix((action,), estimates, lambda ours, theirs: 0), pairs
    )


def _distribution(*weighted_states):
    return TurnOutcomeDistribution(
        tuple(
            WeightedTurnOutcome(probability, TurnSimulationResult(state, ()), ())
            for probability, state in weighted_states
        )
    )


def test_two_turn_coach_uses_search_scores_but_preserves_this_turn_event_odds():
    raw, immediate, setup = _raw()
    coach = build_coaching_report(raw, immediate)
    assert coach.best_action == setup
    assert coach.chosen.score_loss == pytest.approx(129.25)
    assert coach.chosen.grade is DecisionGrade.THROWING
    assert coach.recommendation.grade is DecisionGrade.STUPENDOUS
    assert coach.chosen.metrics.opponent_ko_probability == 1
    assert coach.recommendation.metrics.opponent_ko_probability == 0
    assert coach.recommendation.metrics.player_tailwind_probability == 1
    assert coach.recommendation.metrics.horizon == 1
    assert any("continuation" in reason for reason in coach.reasons)
    one_turn, _, _ = _raw(depth=1)
    assert one_turn.best_play == immediate
    assert coach.chosen.metrics == next(
        m for m in one_turn.turn_metrics if m.action == immediate
    )


def test_secondary_mode_defaults_to_two_turn_search_and_does_not_mutate_board():
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    before = state.copy()
    coach = coach_decision(
        snapshot,
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy,
        chosen_action=immediate,
    )
    assert coach.search_diagnostics.requested_depth == 2
    assert state == before
    assert coach.snapshot is snapshot


def test_saved_report_and_habit_evidence_cannot_read_later_information():
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    protect = JointAction(
        SlotAction(0, ActionKind.MOVE, actor="threat", move="protect"),
        SlotAction(1, ActionKind.MOVE, actor="weak", move="protect"),
    )
    tracker = OpponentHabitTracker()
    tracker.record_action("same-context", theirs)
    for _ in range(3):
        tracker.record_action("same-context", protect)
    raw = rank_decision(
        snapshot,
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs, 3), OpponentActionCandidate(protect, 1)),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy,
        include_coaching_metrics=True,
        tracker=tracker,
        context_key="same-context",
        habit_weight=0.5,
    )
    before = build_coaching_report(raw, immediate)
    text_before = render_coaching_output(before)
    # These represent observations that arrive only after the decision.
    _faint(state, TurnSide.PLAYER, "hero")
    state.tailwind_sides.add(TurnSide.OPPONENT)
    for _ in range(20):
        tracker.record_action("same-context", theirs)
    after = build_coaching_report(raw, immediate)
    assert before == after
    assert render_coaching_output(after) == text_before
    reads = {r.behavior: r for r in after.opponent_reads}
    assert reads["protect+protect"].observed_count == 3
    assert reads["protect+protect"].comparable_observations == 4
    assert reads["protect+protect"].modeled_probability == pytest.approx(0.5)
    assert reads["targeted_move+targeted_move"].baseline_probability == pytest.approx(
        0.75
    )
    assert "observed 3/4" in text_before
    assert "IF OPPONENT (modeled 50.0%)" in text_before
    assert "your plan" in text_before
    with pytest.raises(FrozenInstanceError):
        after.snapshot.turn = 2


def test_no_habit_observations_are_reported_as_baseline_only():
    raw, immediate, _ = _raw(depth=1)
    coach = build_coaching_report(raw, immediate)
    assert all(r.observed_count is None for r in coach.opponent_reads)
    assert (
        "no comparable habit observations; baseline weights only"
        in render_coaching_output(coach)
    )


@pytest.mark.parametrize(
    "loss, expected", zip((0, 2, 5, 10, 20, 35, 60, 90, 120, 130), GRADE_ORDER)
)
def test_provisional_loss_bands_follow_all_ten_requested_labels(loss, expected):
    assert ProvisionalGradeRubric().grade(loss) is expected


@pytest.mark.parametrize("reference", (0, -1, float("nan"), float("inf")))
def test_invalid_grade_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        ProvisionalGradeRubric(reference)


def test_grades_depend_on_loss_not_rank_worst_option_or_absolute_score():
    raw, immediate, _ = _raw(depth=1)
    best = raw.recommendation
    alternative = replace(
        best,
        action=raw.blocked_options[0].action,
        rank=2,
        expected_value=best.expected_value - 20,
        blocked=False,
    )
    controlled = replace(raw, ranked_options=(best, alternative), blocked_options=())
    baseline = build_coaching_report(controlled).option_for(alternative.action)
    # Adding an unrelated terrible candidate must not shrink relative regret.
    terrible_action = JointAction(
        SlotAction(0, ActionKind.PASS), SlotAction(1, ActionKind.PASS)
    )
    terrible = replace(
        alternative, action=terrible_action, rank=3, expected_value=-10000
    )
    extended = replace(controlled, ranked_options=(best, alternative, terrible))
    assert (
        build_coaching_report(extended).option_for(alternative.action).grade
        == baseline.grade
    )
    shifted = replace(
        controlled,
        ranked_options=tuple(
            replace(option, expected_value=option.expected_value - 1000)
            for option in controlled.ranked_options
        ),
    )
    assert (
        build_coaching_report(shifted).option_for(alternative.action).grade
        == baseline.grade
    )
    assert baseline.grade is DecisionGrade.GREAT
    # A tied plan gets the same grade despite its second-place tie-break rank.
    tied = replace(
        controlled,
        ranked_options=(best, replace(alternative, expected_value=best.expected_value)),
    )
    assert (
        build_coaching_report(tied).option_for(alternative.action).grade
        is DecisionGrade.STUPENDOUS
    )
    assert immediate == best.action


def test_custom_position_weights_change_grade_reference_consistently():
    from vgc_bench.src.champions_ai.position_value import PositionWeights

    raw, immediate, _ = _raw(
        position_weights=PositionWeights(pokemon_alive=200, total_hp_fraction=50)
    )
    assert raw.grade_reference_points == 250
    assert build_coaching_report(raw, immediate).reference_points == 250


def test_unknown_plan_and_missing_frozen_snapshot_are_rejected():
    raw, _, _ = _raw(depth=1)
    absent = JointAction(SlotAction(0, ActionKind.PASS), SlotAction(1, ActionKind.PASS))
    with pytest.raises(ValueError, match="not evaluated"):
        build_coaching_report(raw, absent)
    with pytest.raises(ValueError, match="pre-turn snapshot"):
        build_coaching_report(replace(raw, snapshot=None))


def test_missing_event_metrics_require_explicit_recalculation():
    raw, _, _ = _raw(depth=1, include_coaching_metrics=False)
    assert raw.turn_metrics == ()
    with pytest.raises(ValueError, match="include_coaching_metrics=True"):
        build_coaching_report(raw)


def test_rule_blocked_fake_out_is_explained_but_not_given_a_fabricated_grade():
    snapshot, state, immediate, _, theirs, profiles, policy = demo_inputs()
    invalid = JointAction(
        replace(immediate.first, move="fakeout", target_position=1), immediate.second
    )
    raw = rank_decision(
        snapshot,
        state,
        (immediate, invalid),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy,
        include_coaching_metrics=True,
    )
    coach = build_coaching_report(raw, invalid)
    assert coach.chosen.blocked
    assert coach.chosen.grade is None
    assert coach.chosen.metrics is None
    assert any("Fake Out" in warning for warning in coach.chosen.warnings)
    assert coach.recommendation.grade is None  # Only one simulated choice.


def test_all_rule_blocked_plans_produce_explanation_without_a_recommendation():
    snapshot, state, immediate, _, theirs, profiles, policy = demo_inputs()
    invalid = JointAction(
        replace(immediate.first, move="fakeout", target_position=1), immediate.second
    )
    raw = rank_decision(
        snapshot,
        state,
        (invalid,),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy,
        include_coaching_metrics=True,
    )
    coach = build_coaching_report(raw, invalid)
    assert coach.recommendation is None
    assert "no safe recommendation" in render_coaching_output(coach)


def test_search_budget_and_fixed_rng_are_disclosed():
    raw, immediate, _ = _raw(search_config=SearchConfig(depth=3, max_simulations=1))
    coach = build_coaching_report(raw, immediate)
    assert coach.search_diagnostics.completed_depth == 1
    text = render_coaching_output(coach)
    assert "Requested 3 turns; completed 1" in text
    assert "Fixed RNG assumptions" in text
    assert "shorter horizons" in text
    assert "Battle-win %: unavailable" in text


def test_opponent_targets_are_displayed_in_opponents_perspective():
    raw, immediate, _ = _raw()
    text = render_coaching_output(build_coaching_report(raw, immediate))
    assert "hero: Tackle -> weak" in text
    watch = next(line for line in text.splitlines() if line.startswith("WATCH:"))
    assert "threat: Body Slam -> hero" in watch
    assert "weak: Tackle -> ally" in watch
    assert "foe 1" not in watch


def test_real_accuracy_branch_produces_seventy_percent_ko_odds():
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    focus = JointAction(replace(immediate.first, move="focusblast"), immediate.second)
    profiles[(TurnSide.PLAYER, "hero", "focusblast")] = MoveProfile(
        "focusblast", 200, "special", "fighting"
    )
    raw = rank_decision(
        snapshot,
        state,
        (focus, setup),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=replace(policy, branch_accuracy=True),
        include_coaching_metrics=True,
    )
    metrics = next(m for m in raw.turn_metrics if m.action == focus)
    assert metrics.opponent_ko_probability == pytest.approx(0.7)
    assert dict(metrics.opponent_faint_probabilities)["weak"] == pytest.approx(0.7)
    assert metrics.starting_actives_survive_probability == 1


def test_event_odds_mix_opponent_response_and_hidden_world_outcome_mass():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    protect = JointAction(
        replace(theirs.first, move="protect", target_position=None),
        replace(theirs.second, move="protect", target_position=None),
    )
    estimates = estimate_action_probabilities(
        (OpponentActionCandidate(theirs, 3), OpponentActionCandidate(protect, 1))
    )
    success = state.copy()
    _faint(success, TurnSide.OPPONENT, "weak")
    both_dead = state.copy()
    for name in ("hero", "ally"):
        _faint(both_dead, TurnSide.PLAYER, name)
    analysis = _analysis(
        (_distribution((0.2, success), (0.8, state)), _distribution((1, both_dead))),
        estimates,
        action,
    )
    metrics = build_turn_outcome_metrics(snapshot, analysis, estimates)[0]
    assert metrics.opponent_ko_probability == pytest.approx(0.15)
    assert metrics.player_ko_probability == pytest.approx(0.25)
    assert metrics.both_starting_actives_faint_probability == pytest.approx(0.25)
    assert metrics.starting_actives_survive_probability == pytest.approx(0.75)


def test_previously_fainted_opponent_is_not_counted_as_a_new_ko():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    _faint(state, TurnSide.OPPONENT, "weak")
    snapshot = replace(
        snapshot,
        opponent=replace(
            snapshot.opponent,
            pokemon=tuple(
                replace(p, hp_percent=0, fainted=True) if p.name == "weak" else p
                for p in snapshot.opponent.pokemon
            ),
        ),
    )
    estimates = estimate_action_probabilities((OpponentActionCandidate(theirs),))
    metrics = build_turn_outcome_metrics(
        snapshot, _analysis((_distribution((1, state)),), estimates, action), estimates
    )[0]
    assert metrics.opponent_ko_probability == 0
    assert dict(metrics.opponent_faint_probabilities) == {"threat": 0}


@pytest.mark.parametrize("bad_mass", (0.5, float("nan"), -1))
def test_invalid_outcome_mass_is_rejected_instead_of_renormalized(bad_mass):
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    estimates = estimate_action_probabilities((OpponentActionCandidate(theirs),))
    analysis = _analysis((_distribution((bad_mass, state)),), estimates, action)
    with pytest.raises(ValueError, match="probabilit"):
        build_turn_outcome_metrics(snapshot, analysis, estimates)


def test_missing_positive_probability_pair_is_an_error():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    estimates = estimate_action_probabilities((OpponentActionCandidate(theirs),))
    analysis = replace(
        _analysis((_distribution((1, state)),), estimates, action), pair_evaluations=()
    )
    with pytest.raises(ValueError, match="missing positive"):
        build_turn_outcome_metrics(snapshot, analysis, estimates)


def test_saved_coaching_can_grade_another_plan_without_recomputing():
    raw, immediate, setup = _raw()
    first = build_coaching_report(raw, immediate)
    second = build_coaching_report(raw, setup)
    assert first.snapshot == second.snapshot
    assert first.options == second.options
    assert second.chosen.grade is DecisionGrade.STUPENDOUS


def test_bad_rubric_and_display_limit_are_rejected():
    with pytest.raises(ValueError):
        ProvisionalGradeRubric(max_loss_units=(0,) * 10)
    with pytest.raises(ValueError):
        ProvisionalGradeRubric().grade(float("nan"))
    raw, _, _ = _raw(depth=1)
    with pytest.raises(ValueError, match="max_options"):
        render_coaching_output(build_coaching_report(raw), max_options=0)


def test_switched_starting_pokemon_counts_as_surviving_on_the_bench():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    del state.active_slots[(TurnSide.PLAYER, 0)]
    estimates = estimate_action_probabilities((OpponentActionCandidate(theirs),))
    analysis = _analysis((_distribution((1, state)),), estimates, action)
    metrics = build_turn_outcome_metrics(snapshot, analysis, estimates)[0]
    assert metrics.starting_actives_survive_probability == 1
    assert dict(metrics.player_survival_probabilities)["hero"] == 1


def test_zero_probability_response_does_not_need_an_outcome_simulation():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    protect = JointAction(
        replace(theirs.first, move="protect", target_position=None),
        replace(theirs.second, move="protect", target_position=None),
    )
    estimates = estimate_action_probabilities(
        (OpponentActionCandidate(theirs, 1), OpponentActionCandidate(protect, 0))
    )
    analysis = _analysis((_distribution((1, state)),), estimates, action)
    assert (
        build_turn_outcome_metrics(snapshot, analysis, estimates)[
            0
        ].player_ko_probability
        == 0
    )


def test_unknown_profile_is_not_silently_counted_as_surviving():
    snapshot, state, action, _, theirs, _, _ = demo_inputs()
    del state.profiles[(TurnSide.PLAYER, "hero")]
    estimates = estimate_action_probabilities((OpponentActionCandidate(theirs),))
    analysis = _analysis((_distribution((1, state)),), estimates, action)
    with pytest.raises(ValueError, match="missing profile"):
        build_turn_outcome_metrics(snapshot, analysis, estimates)


def test_nonfinite_score_cannot_receive_a_grade():
    raw, _, _ = _raw(depth=1)
    broken = replace(
        raw, ranked_options=(replace(raw.recommendation, expected_value=float("nan")),)
    )
    with pytest.raises(ValueError, match="finite evaluated"):
        build_coaching_report(broken)

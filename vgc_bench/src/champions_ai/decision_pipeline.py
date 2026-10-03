"""End-to-end Champions AI decision orchestration.

This module connects:
- frozen decision snapshots and rule guards;
- legal joint actions;
- opponent action priors / habits;
- probabilistic mechanics simulation;
- response-matrix ranking;
- catastrophic dominance / immediate-loss guards.

The current value model is still the transparent position heuristic used by
mechanics_evaluator, not a calibrated match win probability.
"""

from dataclasses import dataclass, replace
from typing import Callable

from poke_env.battle import DoubleBattle

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.coaching_metrics import (
    TurnOutcomeMetrics,
    build_turn_outcome_metrics,
    probability_assumptions,
)
from vgc_bench.src.champions_ai.guards import (
    GuardCode,
    GuardFinding,
    GuardSeverity,
    blocked_actions,
    inspect_action_rules,
    is_guaranteed_match_loss,
    strictly_dominated_findings,
)
from vgc_bench.src.champions_ai.hidden_scenarios import (
    HiddenStateScenario,
    hidden_state_scenarios,
)
from vgc_bench.src.champions_ai.hidden_sets import SetHypothesis
from vgc_bench.src.champions_ai.inputs.showdown import (
    decision_snapshot_from_showdown,
    exact_turn_state_from_showdown,
)
from vgc_bench.src.champions_ai.inputs.showdown_actions import (
    legal_joint_actions_from_showdown,
)
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.mechanics_evaluator import (
    MechanicsResponseAnalysis,
    build_mechanics_response_analysis,
)
from vgc_bench.src.champions_ai.opponent_actions import (
    generate_opponent_action_candidates,
)
from vgc_bench.src.champions_ai.opponent_model import (
    ActionProbability,
    OpponentActionCandidate,
    OpponentHabitTracker,
    estimate_action_probabilities,
)
from vgc_bench.src.champions_ai.position_value import PositionWeights
from vgc_bench.src.champions_ai.response_matrix import ActionResponseSummary
from vgc_bench.src.champions_ai.search import (
    SearchConfig,
    SearchDiagnostics,
    build_search_analysis,
    prepare_search_state,
)
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.speed_context import speed_states_from_exact_state
from vgc_bench.src.champions_ai.turn_branching import BranchingPolicy
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


@dataclass(frozen=True)
class DecisionOption:
    """One legal action after mechanics evaluation and guard inspection."""

    action: JointAction
    rank: int | None
    expected_value: float | None
    worst_case_value: float | None
    best_case_value: float | None
    blocked: bool
    findings: tuple[GuardFinding, ...]
    most_likely_response: JointAction | None = None
    most_likely_response_probability: float | None = None


@dataclass(frozen=True)
class DecisionReport:
    """Complete output of one Champions AI decision calculation."""

    ranked_options: tuple[DecisionOption, ...]
    blocked_options: tuple[DecisionOption, ...]
    opponent_estimates: tuple[ActionProbability, ...]
    response_summaries: tuple[ActionResponseSummary, ...]
    findings: tuple[GuardFinding, ...]
    model: str = "mechanics_decision_pipeline_v0"
    search_diagnostics: SearchDiagnostics | None = None
    snapshot: DecisionSnapshot | None = None
    turn_metrics: tuple[TurnOutcomeMetrics, ...] = ()
    probability_assumptions: tuple[str, ...] = ()
    grade_reference_points: float = 125.0
    habit_weight: float = 0.0
    habit_context: str | None = None

    @property
    def recommendation(self) -> DecisionOption | None:
        return self.ranked_options[0] if self.ranked_options else None

    @property
    def best_play(self) -> JointAction | None:
        recommendation = self.recommendation
        return None if recommendation is None else recommendation.action


def _dedupe_actions(actions: tuple[JointAction, ...]) -> tuple[JointAction, ...]:
    seen: set[tuple] = set()
    output: list[JointAction] = []
    for action in actions:
        if action.key in seen:
            continue
        seen.add(action.key)
        output.append(action)
    return tuple(output)


def _guaranteed_loss_findings(
    analysis: MechanicsResponseAnalysis,
    opponent_estimates: tuple[ActionProbability, ...],
) -> tuple[GuardFinding, ...]:
    positive_responses = {
        estimate.candidate.action.key
        for estimate in opponent_estimates
        if estimate.probability > 0
    }
    if not positive_responses:
        return ()

    evaluations_by_action: dict[JointAction, dict[tuple, bool]] = {}
    for evaluation in analysis.pair_evaluations:
        evaluations_by_action.setdefault(evaluation.our_action, {})[
            evaluation.opponent_action.key
        ] = is_guaranteed_match_loss(evaluation)

    findings: list[GuardFinding] = []
    for action, response_results in evaluations_by_action.items():
        if not positive_responses.issubset(response_results.keys()):
            continue
        if all(response_results[response_key] for response_key in positive_responses):
            findings.append(
                GuardFinding(
                    code=GuardCode.GUARANTEED_MATCH_LOSS,
                    severity=GuardSeverity.BLOCK,
                    action=action,
                    message=(
                        "Every simulated branch loses immediately against every "
                        "modeled opponent response with nonzero probability."
                    ),
                )
            )

    return tuple(findings)


def _option_from_summary(
    summary: ActionResponseSummary,
    *,
    rank: int | None,
    action_findings: tuple[GuardFinding, ...],
    blocked: bool,
) -> DecisionOption:
    most_likely = summary.most_likely_response
    return DecisionOption(
        action=summary.our_action,
        rank=rank,
        expected_value=summary.expected_value,
        worst_case_value=summary.worst_case_value,
        best_case_value=summary.best_case_value,
        blocked=blocked,
        findings=action_findings,
        most_likely_response=most_likely.opponent_action,
        most_likely_response_probability=most_likely.opponent_probability,
    )


def rank_decision(
    snapshot: DecisionSnapshot,
    exact_state: ExactTurnState,
    legal_actions: tuple[JointAction, ...],
    opponent_candidates: tuple[OpponentActionCandidate, ...],
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    tracker: OpponentHabitTracker | None = None,
    context_key: str | None = None,
    habit_weight: float = 0.0,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    search_config: SearchConfig | None = None,
    include_coaching_metrics: bool = False,
    scenario_provider: Callable[[JointAction], tuple[HiddenStateScenario, ...]]
    | None = None,
    gen: int = 9,
) -> DecisionReport:
    """Run the complete current Champions AI ranking pipeline."""

    actions = _dedupe_actions(legal_actions)
    if not actions:
        raise ValueError("at least one legal joint action is required")
    if not opponent_candidates:
        raise ValueError("at least one opponent action candidate is required")

    rule_findings = tuple(
        finding
        for action in actions
        for finding in inspect_action_rules(snapshot, action)
    )
    rule_blocked = blocked_actions(rule_findings)
    simulation_actions = tuple(
        action for action in actions if action not in rule_blocked
    )

    opponent_estimates = estimate_action_probabilities(
        opponent_candidates,
        tracker=tracker,
        context_key=context_key,
        habit_weight=habit_weight,
    )

    if not simulation_actions:
        blocked_options = tuple(
            DecisionOption(
                action=action,
                rank=None,
                expected_value=None,
                worst_case_value=None,
                best_case_value=None,
                blocked=True,
                findings=tuple(
                    finding for finding in rule_findings if finding.action == action
                ),
            )
            for action in actions
        )
        return DecisionReport(
            ranked_options=(),
            blocked_options=blocked_options,
            opponent_estimates=opponent_estimates,
            response_summaries=(),
            findings=rule_findings,
            snapshot=snapshot,
            probability_assumptions=probability_assumptions(branching_policy),
            grade_reference_points=(position_weights or PositionWeights()).pokemon_alive
            + (position_weights or PositionWeights()).total_hp_fraction,
            habit_weight=habit_weight,
            habit_context=context_key,
        )

    if search_config is not None and search_config.depth > 1:
        exact_state = prepare_search_state(snapshot, exact_state)
        if scenario_provider is not None:
            raw_provider = scenario_provider
            scenario_cache: dict[tuple, tuple[HiddenStateScenario, ...]] = {}

            def prepared_scenarios(action: JointAction):
                if action.key not in scenario_cache:
                    scenario_cache[action.key] = tuple(
                        replace(
                            scenario,
                            state=prepare_search_state(snapshot, scenario.state),
                        )
                        for scenario in raw_provider(action)
                    )
                return scenario_cache[action.key]

            scenario_provider = prepared_scenarios

    analysis = build_mechanics_response_analysis(
        exact_state,
        simulation_actions,
        opponent_estimates,
        speed_states,
        move_profiles,
        branching_policy=branching_policy,
        position_weights=position_weights,
        scenario_provider=scenario_provider,
        gen=gen,
    )

    turn_metrics = (
        build_turn_outcome_metrics(snapshot, analysis, opponent_estimates)
        if include_coaching_metrics
        else ()
    )
    search_diagnostics = None
    if search_config is not None:
        search = build_search_analysis(
            snapshot,
            exact_state,
            analysis,
            simulation_actions,
            opponent_estimates,
            speed_states,
            move_profiles,
            config=search_config,
            branching_policy=branching_policy,
            position_weights=position_weights,
            scenario_provider=scenario_provider,
            gen=gen,
        )
        analysis = search.analysis
        search_diagnostics = search.diagnostics
    dominance_findings = strictly_dominated_findings(analysis.summaries)
    loss_findings = _guaranteed_loss_findings(analysis, opponent_estimates)
    all_findings = rule_findings + dominance_findings + loss_findings
    all_blocked = blocked_actions(all_findings)

    summary_by_action = {summary.our_action: summary for summary in analysis.summaries}
    findings_by_action = {
        action: tuple(finding for finding in all_findings if finding.action == action)
        for action in actions
    }

    ranked_options: list[DecisionOption] = []
    rank = 1
    for summary in analysis.summaries:
        if summary.our_action in all_blocked:
            continue
        ranked_options.append(
            _option_from_summary(
                summary,
                rank=rank,
                action_findings=findings_by_action[summary.our_action],
                blocked=False,
            )
        )
        rank += 1

    blocked_options: list[DecisionOption] = []
    for action in actions:
        if action not in all_blocked:
            continue

        summary = summary_by_action.get(action)
        if summary is None:
            blocked_options.append(
                DecisionOption(
                    action=action,
                    rank=None,
                    expected_value=None,
                    worst_case_value=None,
                    best_case_value=None,
                    blocked=True,
                    findings=findings_by_action[action],
                )
            )
        else:
            blocked_options.append(
                _option_from_summary(
                    summary,
                    rank=None,
                    action_findings=findings_by_action[action],
                    blocked=True,
                )
            )

    return DecisionReport(
        ranked_options=tuple(ranked_options),
        blocked_options=tuple(blocked_options),
        opponent_estimates=opponent_estimates,
        response_summaries=analysis.summaries,
        findings=all_findings,
        model=(
            "mechanics_search_pipeline_v1"
            if search_diagnostics is not None and search_diagnostics.completed_depth > 1
            else "mechanics_decision_pipeline_v0"
        ),
        search_diagnostics=search_diagnostics,
        snapshot=snapshot,
        turn_metrics=turn_metrics,
        probability_assumptions=probability_assumptions(branching_policy),
        grade_reference_points=(position_weights or PositionWeights()).pokemon_alive
        + (position_weights or PositionWeights()).total_hp_fraction,
        habit_weight=habit_weight,
        habit_context=context_key,
    )


def rank_showdown_decision(
    battle: DoubleBattle,
    profiles: dict[tuple[TurnSide, str], CombatantProfile],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    opponent_candidates: tuple[OpponentActionCandidate, ...] | None = None,
    *,
    hidden_hypotheses: dict[str, tuple[SetHypothesis, ...]] | None = None,
    include_opponent_switches: bool = True,
    opponent_switch_prior_weight: float = 1.0,
    max_opponent_candidates: int | None = None,
    max_hidden_scenarios: int | None = 64,
    speed_states: dict[tuple[TurnSide, str], SpeedState] | None = None,
    unburden_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    paradox_speed_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    tracker: OpponentHabitTracker | None = None,
    context_key: str | None = None,
    habit_weight: float = 0.0,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    search_config: SearchConfig | None = None,
    include_coaching_metrics: bool = False,
    gen: int = 9,
) -> DecisionReport:
    """Build live Showdown inputs and run one end-to-end decision calculation.

    When opponent_candidates is omitted, plausible opponent responses are built
    automatically from the public board, revealed moves, revealed switches, and
    caller-supplied hidden-set hypotheses.
    """

    legal_actions = tuple(legal_joint_actions_from_showdown(battle))
    snapshot = decision_snapshot_from_showdown(
        battle, legal_actions=tuple(action.label for action in legal_actions)
    )
    exact_state = exact_turn_state_from_showdown(battle, profiles)

    if opponent_candidates is None:
        opponent_candidates = generate_opponent_action_candidates(
            battle,
            snapshot,
            exact_state,
            hidden_hypotheses=hidden_hypotheses,
            include_switches=include_opponent_switches,
            switch_prior_weight=opponent_switch_prior_weight,
            max_candidates=max_opponent_candidates,
            gen=gen,
        )

    if speed_states is None:
        speed_states = speed_states_from_exact_state(
            exact_state,
            unburden_active=unburden_active,
            paradox_speed_active=paradox_speed_active,
        )

    scenario_provider = None
    if hidden_hypotheses:

        def scenario_provider(opponent_action):
            return hidden_state_scenarios(
                snapshot,
                exact_state,
                opponent_action,
                hidden_hypotheses,
                speed_states,
                max_scenarios=max_hidden_scenarios,
            )

    return rank_decision(
        snapshot,
        exact_state,
        legal_actions,
        opponent_candidates,
        speed_states,
        move_profiles,
        tracker=tracker,
        context_key=context_key,
        habit_weight=habit_weight,
        branching_policy=branching_policy,
        position_weights=position_weights,
        search_config=search_config,
        include_coaching_metrics=include_coaching_metrics,
        scenario_provider=scenario_provider,
        gen=gen,
    )

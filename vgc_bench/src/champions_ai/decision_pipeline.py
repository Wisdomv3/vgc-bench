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

from dataclasses import dataclass

from poke_env.battle import DoubleBattle

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.guards import (
    GuardCode,
    GuardFinding,
    GuardSeverity,
    blocked_actions,
    inspect_action_rules,
    is_guaranteed_match_loss,
    strictly_dominated_findings,
)
from vgc_bench.src.champions_ai.inputs.showdown import (
    decision_snapshot_from_showdown,
    exact_turn_state_from_showdown,
)
from vgc_bench.src.champions_ai.inputs.showdown_actions import (
    legal_joint_actions_from_showdown,
)
from vgc_bench.src.champions_ai.hidden_sets import SetHypothesis
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


def speed_states_from_exact_state(
    state: ExactTurnState,
    *,
    unburden_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    paradox_speed_active: frozenset[tuple[TurnSide, str]] = frozenset(),
) -> dict[tuple[TurnSide, str], SpeedState]:
    """Build supported live SpeedState inputs from an ExactTurnState.

    Unburden and Paradox Speed activation require explicit caller confirmation
    because their activation history is not yet fully represented in
    ExactTurnState.
    """

    weather = normalize_move_id(state.weather or "")
    weather_abilities = WEATHER_SPEED_ABILITIES.get(weather, set())

    output: dict[tuple[TurnSide, str], SpeedState] = {}
    for key, profile in state.profiles.items():
        side, _name = key
        ability = normalize_move_id(profile.ability or "")
        item = normalize_move_id(profile.item or "")

        output[key] = SpeedState(
            speed_stat=int(profile.stats["spe"]),
            stage=int(profile.boosts.get("spe", 0)),
            tailwind=side in state.tailwind_sides,
            unburden=key in unburden_active,
            choice_scarf=item == "choicescarf",
            weather_speed_boost=ability in weather_abilities,
            paradox_speed_boost=key in paradox_speed_active,
            paralyzed=normalize_move_id(profile.status or "") == "par",
        )

    return output


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
        evaluations_by_action.setdefault(
            evaluation.our_action,
            {},
        )[evaluation.opponent_action.key] = is_guaranteed_match_loss(evaluation)

    findings: list[GuardFinding] = []
    for action, response_results in evaluations_by_action.items():
        if not positive_responses.issubset(response_results.keys()):
            continue
        if all(
            response_results[response_key]
            for response_key in positive_responses
        ):
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
        action
        for action in actions
        if action not in rule_blocked
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
                    finding
                    for finding in rule_findings
                    if finding.action == action
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
        )

    analysis = build_mechanics_response_analysis(
        exact_state,
        simulation_actions,
        opponent_estimates,
        speed_states,
        move_profiles,
        branching_policy=branching_policy,
        position_weights=position_weights,
        gen=gen,
    )

    dominance_findings = strictly_dominated_findings(analysis.summaries)
    loss_findings = _guaranteed_loss_findings(
        analysis,
        opponent_estimates,
    )
    all_findings = (
        rule_findings
        + dominance_findings
        + loss_findings
    )
    all_blocked = blocked_actions(all_findings)

    summary_by_action = {
        summary.our_action: summary
        for summary in analysis.summaries
    }
    findings_by_action = {
        action: tuple(
            finding
            for finding in all_findings
            if finding.action == action
        )
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
    speed_states: dict[tuple[TurnSide, str], SpeedState] | None = None,
    unburden_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    paradox_speed_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    tracker: OpponentHabitTracker | None = None,
    context_key: str | None = None,
    habit_weight: float = 0.0,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    gen: int = 9,
) -> DecisionReport:
    """Build live Showdown inputs and run one end-to-end decision calculation.

    When opponent_candidates is omitted, plausible opponent responses are built
    automatically from the public board, revealed moves, revealed switches, and
    caller-supplied hidden-set hypotheses.
    """

    legal_actions = tuple(legal_joint_actions_from_showdown(battle))
    snapshot = decision_snapshot_from_showdown(
        battle,
        legal_actions=tuple(action.label for action in legal_actions),
    )
    exact_state = exact_turn_state_from_showdown(
        battle,
        profiles,
    )

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
        gen=gen,
    )

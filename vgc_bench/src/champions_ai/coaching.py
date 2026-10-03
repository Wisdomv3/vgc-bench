"""Pre-turn coaching over the same decision calculation used by the bot.

No current battle, later result or mutable habit tracker is accepted when a
coaching report is built: all evidence comes from the frozen DecisionReport.
Grades use a documented provisional position-score rubric until match-value
and grade thresholds are calibrated. No heuristic is converted to a win %.
"""

from dataclasses import dataclass, replace
from math import inf, isfinite

from vgc_bench.src.champions_ai.actions import JointAction
from vgc_bench.src.champions_ai.coaching_metrics import TurnOutcomeMetrics
from vgc_bench.src.champions_ai.decision_pipeline import (
    DecisionReport,
    rank_decision,
    rank_showdown_decision,
)
from vgc_bench.src.champions_ai.grades import GRADE_ORDER, DecisionGrade
from vgc_bench.src.champions_ai.live_output import format_joint_action
from vgc_bench.src.champions_ai.search import SearchConfig, SearchDiagnostics
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.spread import is_spread_move


@dataclass(frozen=True)
class ProvisionalGradeRubric:
    """Prototype loss bands in units of one healthy Pokemon's heuristic value.

    The default reference is alive weight + full HP weight (100 + 25 points).
    Thresholds are explicit tuning parameters, not statistically calibrated
    claims. They depend on score loss, never rank or the worst candidate's score.
    Equal-value choices share the same grade. Top grade means zero modeled loss;
    it does not establish tactical brilliance or a guaranteed best play.
    """

    reference_points: float = 125.0
    max_loss_units: tuple[float, ...] = (
        0.0,
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
        inf,
    )

    def __post_init__(self):
        if not isfinite(self.reference_points) or self.reference_points <= 0:
            raise ValueError("grade reference must be finite and positive")
        limits = self.max_loss_units
        if len(limits) != len(GRADE_ORDER) or limits[0] != 0 or limits[-1] != inf:
            raise ValueError("rubric needs ten ordered bands, from zero to infinity")
        if any(not isfinite(x) or x < 0 for x in limits[:-1]):
            raise ValueError("finite grade bands must be nonnegative")
        if any(a >= b for a, b in zip(limits, limits[1:])):
            raise ValueError("grade bands must strictly increase")

    def grade(self, score_loss: float) -> DecisionGrade:
        if not isfinite(score_loss) or score_loss < -1e-9:
            raise ValueError("score loss must be finite and nonnegative")
        loss = max(0.0, score_loss)
        for grade, limit in zip(GRADE_ORDER, self.max_loss_units):
            if loss <= limit * self.reference_points + 1e-9:
                return grade
        raise AssertionError("the final grade band must cover every finite loss")


@dataclass(frozen=True)
class OpponentRead:
    """Joint behavior category, rather than a fabricated move-specific habit."""

    behavior: str
    modeled_probability: float
    baseline_probability: float
    observed_count: int | None
    comparable_observations: int
    habit_weight: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CoachedOption:
    action: JointAction
    rank: int | None
    expected_value: float | None
    score_loss: float | None
    grade: DecisionGrade | None
    blocked: bool
    metrics: TurnOutcomeMetrics | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ResponseComparison:
    response: JointAction
    probability: float
    best_plan_value: float
    proposed_plan_value: float | None


@dataclass(frozen=True)
class CoachingReport:
    snapshot: DecisionSnapshot
    options: tuple[CoachedOption, ...]
    best_action: JointAction | None
    chosen_action: JointAction | None
    opponent_reads: tuple[OpponentRead, ...]
    response_comparisons: tuple[ResponseComparison, ...]
    likely_response: JointAction | None
    likely_response_probability: float | None
    reasons: tuple[str, ...]
    board_notes: tuple[str, ...]
    caveats: tuple[str, ...]
    search_diagnostics: SearchDiagnostics | None
    reference_points: float
    habit_context: str | None
    model: str = "pre_turn_coaching_v1_provisional"

    def option_for(self, action: JointAction) -> CoachedOption:
        for option in self.options:
            if option.action.key == action.key:
                return option
        raise ValueError("the proposed action was not evaluated in this decision")

    @property
    def recommendation(self) -> CoachedOption | None:
        return None if self.best_action is None else self.option_for(self.best_action)

    @property
    def chosen(self) -> CoachedOption | None:
        return (
            None if self.chosen_action is None else self.option_for(self.chosen_action)
        )


def _opponent_reads(report: DecisionReport) -> tuple[OpponentRead, ...]:
    groups = {}
    for estimate in report.opponent_estimates:
        groups.setdefault(estimate.candidate.behavior, []).append(estimate)
    reads = []
    for behavior, estimates in groups.items():
        count = estimates[0].comparable_observations
        empirical = [e.empirical_probability for e in estimates]
        observed = None
        if count > 0 and all(p is not None for p in empirical):
            observed = round(sum(empirical) * count)
        reads.append(
            OpponentRead(
                behavior=behavior,
                modeled_probability=sum(e.probability for e in estimates),
                baseline_probability=sum(e.baseline_probability for e in estimates),
                observed_count=observed,
                comparable_observations=count,
                habit_weight=report.habit_weight,
                sources=tuple(
                    sorted(
                        {
                            e.candidate.source or "caller-supplied weights"
                            for e in estimates
                        }
                    )
                ),
            )
        )
    return tuple(
        sorted(reads, key=lambda read: (-read.modeled_probability, read.behavior))
    )


def _explanations(
    best: CoachedOption | None, chosen: CoachedOption | None, depth: int
) -> tuple[str, ...]:
    if best is None:
        return (
            "The guards excluded all evaluated plans; "
            "no safe recommendation is available.",
        )
    reasons = [
        f"Best found among evaluated actions: {best.expected_value:+.1f} modeled "
        f"position change over up to {depth} turn(s)."
    ]
    if chosen is None:
        return tuple(reasons)
    if chosen.grade is None:
        reasons.append(
            "This plan cannot receive a comparative grade "
            "from the available evaluations."
        )
    elif chosen.score_loss <= 1e-9:
        reasons.append("Your plan matches the best modeled value found for this board.")
    else:
        reasons.append(
            f"Your plan gives up {chosen.score_loss:.1f} position points "
            "versus the best found."
        )
    if best.metrics is None or chosen.metrics is None:
        return tuple(reasons)
    before, after = chosen.metrics, best.metrics
    comparisons = (
        (
            "both starting Pokemon survive",
            before.starting_actives_survive_probability,
            after.starting_actives_survive_probability,
        ),
        (
            "at least one new opposing KO",
            before.opponent_ko_probability,
            after.opponent_ko_probability,
        ),
        (
            "your Tailwind remains active",
            before.player_tailwind_probability,
            after.player_tailwind_probability,
        ),
    )
    for label, old, new in comparisons:
        if old is not None and new is not None and abs(new - old) > 0.005:
            direction = "increases" if new > old else "decreases"
            reasons.append(
                f"Choosing the best plan {direction} this-turn odds that {label}: "
                f"{old:.1%} -> {new:.1%}."
            )
    if (
        before.opponent_ko_probability > after.opponent_ko_probability + 0.005
        and depth > 1
    ):
        reasons.append(
            "The higher search value trades immediate KO odds "
            "for a stronger modeled continuation."
        )
    return tuple(reasons)


def build_coaching_report(
    report: DecisionReport,
    chosen_action: JointAction | None = None,
    *,
    rubric: ProvisionalGradeRubric | None = None,
) -> CoachingReport:
    """Explain only the evidence captured when rank_decision was called.

    Choosing a plan here does not rerun search or add later information. If it
    was outside the original evaluated action set, explicitly recalculate from
    the same pre-turn inputs instead. A rule-blocked unsimulated plan is unrated.
    """

    if report.snapshot is None:
        raise ValueError(
            "coaching requires the pre-turn snapshot captured by rank_decision"
        )
    rubric = rubric or ProvisionalGradeRubric(report.grade_reference_points)
    candidates = report.ranked_options + report.blocked_options
    if chosen_action is not None and not any(
        o.action.key == chosen_action.key for o in candidates
    ):
        raise ValueError("the proposed action was not evaluated in this decision")
    if report.response_summaries and not report.turn_metrics:
        raise ValueError(
            "recalculate with include_coaching_metrics=True to capture outcome odds"
        )
    if any(
        o.expected_value is not None and not isfinite(o.expected_value)
        for o in candidates
    ):
        raise ValueError("coaching requires finite evaluated position scores")
    metrics = {m.action.key: m for m in report.turn_metrics}
    best = report.recommendation
    evaluated_count = sum(o.expected_value is not None for o in candidates)
    options = []
    for candidate in candidates:
        loss = None
        grade = None
        if best is not None and candidate.expected_value is not None:
            difference = best.expected_value - candidate.expected_value
            if difference >= -1e-9:
                loss = max(0.0, difference)
                if evaluated_count > 1:
                    grade = rubric.grade(loss)
        options.append(
            CoachedOption(
                action=candidate.action,
                rank=candidate.rank,
                expected_value=candidate.expected_value,
                score_loss=loss,
                grade=grade,
                blocked=candidate.blocked,
                metrics=metrics.get(candidate.action.key),
                warnings=tuple(dict.fromkeys(f.message for f in candidate.findings)),
            )
        )
    snapshot = report.snapshot
    active_foes = set(dict(snapshot.opponent.active_slots).values())
    board_notes = tuple(
        f"Revealed spread threat: {p.name} has {move}."
        for p in snapshot.opponent.pokemon
        if p.name in active_foes and not p.fainted
        for move in p.revealed_moves
        if is_spread_move(move)
    )
    caveats = [
        "Grades are provisional position-score bands, "
        "not calibrated win-probability loss.",
        "Event odds describe this turn only, conditional on modeled responses, "
        "hidden sets and supported mechanics.",
        "Top grade means zero modeled score loss; "
        "it is not proof of a uniquely brilliant play.",
    ]
    caveats.extend(report.probability_assumptions)
    search = report.search_diagnostics
    if search is not None:
        if search.completed_depth < search.requested_depth:
            caveats.append(
                f"Requested {search.requested_depth} turns; "
                f"completed {search.completed_depth}."
            )
        if search.cutoffs or search.budget_exhausted:
            reasons = ", ".join(
                f"{reason} ({count})" for reason, count in search.cutoffs
            )
            caveats.append(
                "Some search branches use shorter horizons: "
                + (reasons or "budget exhausted")
                + "."
            )
        if search.completed_depth > 1:
            caveats.append(
                "Future opponent responses use neutral continuation priors, "
                "not learned optimal play."
            )
    if evaluated_count < 2:
        caveats.append(
            "Fewer than two plans were simulated; comparative grades are unavailable."
        )
    best_option = next(
        (o for o in options if best and o.action.key == best.action.key), None
    )
    chosen_option = next(
        (o for o in options if chosen_action and o.action.key == chosen_action.key),
        None,
    )
    summaries = {s.our_action.key: s for s in report.response_summaries}
    comparisons = []
    if best is not None and best.action.key in summaries:
        proposed = summaries.get(chosen_action.key) if chosen_action else None
        proposed_values = (
            {cell.opponent_action.key: cell.outcome_value for cell in proposed.cells}
            if proposed is not None
            else {}
        )
        for cell in summaries[best.action.key].cells:
            if cell.opponent_probability > 0:
                comparisons.append(
                    ResponseComparison(
                        cell.opponent_action,
                        cell.opponent_probability,
                        cell.outcome_value,
                        proposed_values.get(cell.opponent_action.key),
                    )
                )
        comparisons.sort(key=lambda c: (-c.probability, c.response.label))
    return CoachingReport(
        snapshot=snapshot,
        options=tuple(options),
        best_action=None if best is None else best.action,
        chosen_action=chosen_action,
        opponent_reads=_opponent_reads(report),
        response_comparisons=tuple(comparisons),
        likely_response=None if best is None else best.most_likely_response,
        likely_response_probability=None
        if best is None
        else best.most_likely_response_probability,
        reasons=_explanations(
            best_option, chosen_option, search.completed_depth if search else 1
        ),
        board_notes=board_notes,
        caveats=tuple(caveats),
        search_diagnostics=search,
        reference_points=rubric.reference_points,
        habit_context=report.habit_context,
    )


def coach_decision(*args, chosen_action=None, rubric=None, **kwargs) -> CoachingReport:
    """Secondary mode: same rank_decision inputs, optional proposed joint action.

    Defaults to bounded two-turn search. Pass SearchConfig(depth=1) explicitly
    for one-turn coaching. Call before recording the current opponent response.
    """

    kwargs["include_coaching_metrics"] = True
    kwargs.setdefault("search_config", SearchConfig(depth=2))
    return build_coaching_report(
        rank_decision(*args, **kwargs), chosen_action, rubric=rubric
    )


def coach_showdown_decision(
    *args, chosen_action=None, rubric=None, **kwargs
) -> CoachingReport:
    """Showdown input adapter for coach_decision's secondary mode."""

    kwargs["include_coaching_metrics"] = True
    kwargs.setdefault("search_config", SearchConfig(depth=2))
    return build_coaching_report(
        rank_showdown_decision(*args, **kwargs), chosen_action, rubric=rubric
    )


def _grade_text(option: CoachedOption) -> str:
    return "Unrated" if option.grade is None else f"{option.grade.value} (provisional)"


def _metric_text(metrics: TurnOutcomeMetrics) -> str:
    parts = [
        f"new opposing KO {metrics.opponent_ko_probability:.1%}",
        f"own new faint {metrics.player_ko_probability:.1%}",
    ]
    if metrics.starting_actives_survive_probability is not None:
        parts.append(
            "starting Pokemon survive together "
            f"{metrics.starting_actives_survive_probability:.1%}"
        )
    parts.append(
        f"Tailwind active at turn end {metrics.player_tailwind_probability:.1%}"
    )
    return "; ".join(parts)


def _format_plan(
    action: JointAction,
    snapshot: DecisionSnapshot,
    *,
    opponent: bool = False,
    gen: int = 9,
) -> str:
    """Resolve target slots in the actor's perspective to the actual board names."""

    own = snapshot.opponent if opponent else snapshot.player
    foe = snapshot.player if opponent else snapshot.opponent

    def slot_action(slot):
        position = slot.target_position
        if slot.target is not None or position not in (-2, -1, 1, 2):
            return slot
        side = foe if position > 0 else own
        name = dict(side.active_slots).get(abs(position) - 1)
        return slot if name is None else replace(slot, target=name)

    display = JointAction(slot_action(action.first), slot_action(action.second))
    return format_joint_action(display, gen=gen)


def render_coaching_output(
    report: CoachingReport, *, max_options: int = 3, gen: int = 9
) -> str:
    """Readable live report; preserve score units, timing and evidence labels."""

    if (
        isinstance(max_options, bool)
        or not isinstance(max_options, int)
        or max_options < 1
    ):
        raise ValueError("max_options must be a positive integer")
    snapshot = report.snapshot
    lines = [f"COACH | Turn {snapshot.turn} | PRE-TURN INFORMATION ONLY"]
    for label, side in (("YOU", snapshot.player), ("FOE", snapshot.opponent)):
        pokemon = {p.name: p for p in side.pokemon}
        names = []
        for slot, name in side.active_slots:
            p = pokemon[name]
            status = "" if p.status is None else f" {p.status}"
            names.append(f"{slot + 1}: {name} {p.hp_percent:.0f}%{status}")
        lines.append(f"{label}: " + " | ".join(names))
    lines.append(
        f"FIELD: Trick Room {snapshot.trick_room_turns}; "
        f"your Tailwind {snapshot.player.tailwind_turns}; "
        f"foe Tailwind {snapshot.opponent.tailwind_turns}; "
        f"weather {snapshot.weather or 'none'}; "
        f"terrain {snapshot.terrain or 'none'}."
    )
    depth = (
        1
        if report.search_diagnostics is None
        else report.search_diagnostics.completed_depth
    )
    lines.append(
        f"VALUE: position points over up to {depth} turn(s); "
        f"grade reference {report.reference_points:g} points. "
        "Battle-win %: unavailable."
    )
    chosen = report.chosen
    if chosen is not None:
        lines.append(
            f"YOUR PLAN: {_grade_text(chosen)} | "
            f"{_format_plan(chosen.action, snapshot, gen=gen)}"
        )
        if chosen.metrics is not None:
            lines.append("YOUR THIS-TURN ODDS: " + _metric_text(chosen.metrics))
        lines.extend("CAUTION: " + warning for warning in chosen.warnings)
    best = report.recommendation
    if best is None:
        lines.append("BEST FOUND: no safe recommendation.")
    else:
        lines.append(
            f"BEST FOUND: {_grade_text(best)} | "
            f"{_format_plan(best.action, snapshot, gen=gen)}"
        )
        if best.metrics is not None:
            lines.append("BEST THIS-TURN ODDS: " + _metric_text(best.metrics))
            for name, probability in best.metrics.player_survival_probabilities:
                lines.append(
                    f"SURVIVAL: {name} {probability:.1%} at turn end "
                    "(including on the bench)."
                )
            for name, probability in best.metrics.opponent_faint_probabilities:
                lines.append(f"KO: {name} {probability:.1%} faint by turn end.")
    if report.likely_response is not None:
        lines.append(
            f"WATCH: modeled {report.likely_response_probability:.1%} | "
            f"{_format_plan(report.likely_response, snapshot, opponent=True, gen=gen)}"
        )
    lines.extend("WHY: " + reason for reason in report.reasons)
    if len(report.response_comparisons) > 1:
        for comparison in report.response_comparisons[:2]:
            values = f"best plan {comparison.best_plan_value:+.1f} points"
            if comparison.proposed_plan_value is not None:
                values += f"; your plan {comparison.proposed_plan_value:+.1f} points"
            response = _format_plan(
                comparison.response, snapshot, opponent=True, gen=gen
            )
            lines.append(
                f"IF OPPONENT (modeled {comparison.probability:.1%}): "
                f"{response} "
                f"| {values} over the search horizon."
            )
    lines.extend("BOARD NOTE: " + note for note in report.board_notes)
    visible = [o for o in report.options if o.expected_value is not None]
    visible.sort(key=lambda o: (-o.expected_value, o.action.label))
    for option in visible[:max_options]:
        label = "excluded by guards" if option.blocked else f"rank {option.rank}"
        lines.append(
            f"OPTION ({label}): {_grade_text(option)} | "
            f"{option.expected_value:+.1f} points | "
            f"{_format_plan(option.action, snapshot, gen=gen)}"
        )
        if option.metrics is not None:
            lines.append("  THIS-TURN ODDS: " + _metric_text(option.metrics))
    for read in report.opponent_reads[:3]:
        behavior = read.behavior.replace("_", " ").replace("+", " + ")
        evidence = "no comparable habit observations; baseline weights only"
        if read.observed_count is not None:
            blending = "used" if read.habit_weight > 0 else "recorded but not used"
            evidence = (
                f"observed {read.observed_count}/{read.comparable_observations} "
                f"comparable choices; habit blend {read.habit_weight:.0%} "
                f"({blending})"
            )
        lines.append(
            f"READ: {behavior} modeled {read.modeled_probability:.1%}; "
            f"{evidence}; sources: {', '.join(read.sources)}."
        )
    lines.extend("MODEL NOTE: " + note for note in report.caveats)
    return "\n".join(lines)

"""Fast live-battle output formatting for Champions AI.

The formatter deliberately keeps the recommendation compact:
- one BEST PLAY headline;
- the two slot actions;
- one WATCH line for the most likely modeled opponent response;
- no more than two short reasons.

Values shown here are position-heuristic scores until the calibrated
match-win-probability model replaces them.
"""

from dataclasses import dataclass

from poke_env.battle import Move

from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    Gimmick,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.decision_pipeline import (
    DecisionOption,
    DecisionReport,
)
from vgc_bench.src.champions_ai.guards import GuardSeverity


GIMMICK_LABELS = {
    Gimmick.MEGA: "Mega",
    Gimmick.Z_MOVE: "Z-Move",
    Gimmick.DYNAMAX: "Dynamax",
    Gimmick.TERA: "Tera",
}


@dataclass(frozen=True)
class LiveBattleOutput:
    """Compact recommendation designed to be read during a live turn."""

    headline: str
    first_action: str | None
    second_action: str | None
    watch: str | None
    reasons: tuple[str, ...]
    model: str = "live_output_v0"

    def __post_init__(self) -> None:
        if len(self.reasons) > 2:
            raise ValueError("live output supports at most two reasons")

    @property
    def text(self) -> str:
        lines = [self.headline]

        if self.first_action is not None:
            lines.append(f"1. {self.first_action}")
        if self.second_action is not None:
            lines.append(f"2. {self.second_action}")
        if self.watch is not None:
            lines.append(f"WATCH: {self.watch}")

        for reason in self.reasons:
            lines.append(f"WHY: {reason}")

        return "\n".join(lines)


def _display_move_name(move_id: str, *, gen: int) -> str:
    try:
        move = Move(move_id, gen)
        name = move.entry.get("name")
        if name:
            return str(name)
    except (KeyError, ValueError):
        pass

    return move_id.replace("_", " ").replace("-", " ").title()


def _target_label(action: SlotAction) -> str | None:
    if action.target:
        return action.target

    position = action.target_position
    if position == 1:
        return "foe 1"
    if position == 2:
        return "foe 2"
    if position == -1:
        return "ally 1"
    if position == -2:
        return "ally 2"
    return None


def format_slot_action(
    action: SlotAction,
    *,
    gen: int = 9,
) -> str:
    """Return one compact, human-readable slot instruction."""

    actor = action.actor or f"Slot {action.slot + 1}"

    if action.kind is ActionKind.MOVE:
        move_name = _display_move_name(action.move or "", gen=gen)
        target = _target_label(action)
        target_text = "" if target is None else f" -> {target}"
        gimmick = GIMMICK_LABELS.get(action.gimmick)
        gimmick_text = "" if gimmick is None else f" [{gimmick}]"
        return f"{actor}: {move_name}{target_text}{gimmick_text}"

    if action.kind is ActionKind.SWITCH:
        return f"{actor}: Switch -> {action.switch_to}"

    if action.kind is ActionKind.PASS:
        return f"{actor}: Pass"

    return f"{actor}: Default"


def format_joint_action(
    action: JointAction,
    *,
    gen: int = 9,
) -> str:
    """Return both slot actions on one compact line."""

    return (
        f"{format_slot_action(action.first, gen=gen)}"
        f" + {format_slot_action(action.second, gen=gen)}"
    )


def _score(value: float) -> str:
    return f"{value:+.1f}"


def _recommendation_reasons(
    report: DecisionReport,
    option: DecisionOption,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if option.expected_value is not None:
        search = report.search_diagnostics
        if search is not None and search.completed_depth > 1:
            reasons.append(
                f"Highest modeled expected change over up to "
                f"{search.completed_depth} turns "
                f"({_score(option.expected_value)})."
            )
        else:
            reasons.append(
                "Highest modeled expected position change "
                f"({_score(option.expected_value)})."
            )
        if search is not None and (search.cutoffs or search.budget_exhausted):
            reasons.append(
                "Search reached limits; some branches use shorter-horizon values."
            )
            return tuple(reasons[:2])

    warnings = tuple(
        finding
        for finding in option.findings
        if finding.severity is GuardSeverity.WARNING
    )
    if warnings:
        reasons.append(warnings[0].message)
        return tuple(reasons[:2])

    if (
        option.worst_case_value is not None
        and option.worst_case_value > 0
    ):
        reasons.append(
            "Worst modeled response still improves the position "
            f"({_score(option.worst_case_value)})."
        )
        return tuple(reasons[:2])

    if len(report.ranked_options) > 1:
        runner_up = report.ranked_options[1]
        if (
            option.expected_value is not None
            and runner_up.expected_value is not None
        ):
            gap = option.expected_value - runner_up.expected_value
            if gap > 1e-9:
                reasons.append(
                    f"Expected edge over #2: {_score(gap)}."
                )

    return tuple(reasons[:2])


def _blocked_reasons(report: DecisionReport) -> tuple[str, ...]:
    messages: list[str] = []
    for option in report.blocked_options:
        for finding in option.findings:
            if finding.severity is not GuardSeverity.BLOCK:
                continue
            if finding.message in messages:
                continue
            messages.append(finding.message)
            if len(messages) == 2:
                return tuple(messages)
    return tuple(messages)


def build_live_output(
    report: DecisionReport,
    *,
    gen: int = 9,
) -> LiveBattleOutput:
    """Convert a DecisionReport into the fast live-battle display."""

    option = report.recommendation
    if option is None:
        return LiveBattleOutput(
            headline="NO SAFE PLAY",
            first_action=None,
            second_action=None,
            watch=None,
            reasons=_blocked_reasons(report),
        )

    watch: str | None = None
    if (
        option.most_likely_response is not None
        and option.most_likely_response_probability is not None
    ):
        watch = (
            f"{option.most_likely_response_probability:.0%} "
            f"{format_joint_action(option.most_likely_response, gen=gen)}"
        )

    return LiveBattleOutput(
        headline="BEST PLAY",
        first_action=format_slot_action(
            option.action.first,
            gen=gen,
        ),
        second_action=format_slot_action(
            option.action.second,
            gen=gen,
        ),
        watch=watch,
        reasons=_recommendation_reasons(report, option),
    )


def render_live_output(
    report: DecisionReport,
    *,
    gen: int = 9,
) -> str:
    """Return the final compact text intended for live display."""

    return build_live_output(report, gen=gen).text

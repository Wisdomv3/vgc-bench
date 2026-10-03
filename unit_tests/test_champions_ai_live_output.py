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
from vgc_bench.src.champions_ai.guards import (
    GuardCode,
    GuardFinding,
    GuardSeverity,
)
from vgc_bench.src.champions_ai.live_output import (
    build_live_output,
    format_slot_action,
    render_live_output,
)


def _move(
    slot: int,
    actor: str,
    move: str,
    *,
    target: str | None = None,
    target_position: int | None = None,
    gimmick: Gimmick = Gimmick.NONE,
):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target=target,
        target_position=target_position,
        gimmick=gimmick,
    )


def _pass(slot: int, actor: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.PASS,
        actor=actor,
    )


def _joint(first, second):
    return JointAction(first=first, second=second)


def _best_action():
    return _joint(
        _move(
            0,
            "garchomp",
            "bodyslam",
            target="salamence",
            target_position=1,
        ),
        _move(
            1,
            "whimsicott",
            "tailwind",
        ),
    )


def _opponent_response():
    return _joint(
        _move(0, "salamence", "protect"),
        _move(
            1,
            "sneasler",
            "closecombat",
            target="garchomp",
            target_position=1,
        ),
    )


def _report(
    *,
    findings=(),
    second_option: DecisionOption | None = None,
):
    best = DecisionOption(
        action=_best_action(),
        rank=1,
        expected_value=18.5,
        worst_case_value=2.0,
        best_case_value=40.0,
        blocked=False,
        findings=tuple(findings),
        most_likely_response=_opponent_response(),
        most_likely_response_probability=0.625,
    )

    ranked = (best,) if second_option is None else (best, second_option)

    return DecisionReport(
        ranked_options=ranked,
        blocked_options=(),
        opponent_estimates=(),
        response_summaries=(),
        findings=tuple(findings),
    )


def test_live_output_has_best_play_two_actions_watch_and_two_reasons_max() -> None:
    output = build_live_output(_report())

    assert output.headline == "BEST PLAY"
    assert output.first_action == "garchomp: Body Slam -> salamence"
    assert output.second_action == "whimsicott: Tailwind"
    assert output.watch is not None
    assert output.watch.startswith("62% ")
    assert "salamence: Protect" in output.watch
    assert "sneasler: Close Combat -> garchomp" in output.watch
    assert len(output.reasons) <= 2


def test_rendered_live_output_is_compact() -> None:
    text = render_live_output(_report())
    lines = text.splitlines()

    assert lines[0] == "BEST PLAY"
    assert lines[1].startswith("1. garchomp:")
    assert lines[2].startswith("2. whimsicott:")
    assert lines[3].startswith("WATCH:")
    assert sum(line.startswith("WHY:") for line in lines) <= 2
    assert len(lines) <= 6


def test_warning_is_used_as_second_reason() -> None:
    warning = GuardFinding(
        code=GuardCode.REPEATED_PROTECT_RISK,
        severity=GuardSeverity.WARNING,
        action=_best_action(),
        message="Consecutive protection succeeds only 33.333%.",
    )

    output = build_live_output(
        _report(findings=(warning,))
    )

    assert output.reasons == (
        "Highest modeled expected position change (+18.5).",
        "Consecutive protection succeeds only 33.333%.",
    )


def test_runner_up_gap_is_used_when_worst_case_is_not_positive() -> None:
    best = DecisionOption(
        action=_best_action(),
        rank=1,
        expected_value=12.0,
        worst_case_value=-4.0,
        best_case_value=30.0,
        blocked=False,
        findings=(),
        most_likely_response=_opponent_response(),
        most_likely_response_probability=0.5,
    )
    runner_up_action = _joint(
        _move(
            0,
            "garchomp",
            "protect",
        ),
        _pass(1, "whimsicott"),
    )
    runner_up = DecisionOption(
        action=runner_up_action,
        rank=2,
        expected_value=7.0,
        worst_case_value=-1.0,
        best_case_value=20.0,
        blocked=False,
        findings=(),
    )
    report = DecisionReport(
        ranked_options=(best, runner_up),
        blocked_options=(),
        opponent_estimates=(),
        response_summaries=(),
        findings=(),
    )

    output = build_live_output(report)

    assert output.reasons == (
        "Highest modeled expected position change (+12.0).",
        "Expected edge over #2: +5.0.",
    )


def test_no_safe_play_reports_at_most_two_block_reasons() -> None:
    action = _best_action()
    findings = (
        GuardFinding(
            code=GuardCode.GUARANTEED_MATCH_LOSS,
            severity=GuardSeverity.BLOCK,
            action=action,
            message="Every modeled branch loses immediately.",
        ),
        GuardFinding(
            code=GuardCode.STRICTLY_DOMINATED,
            severity=GuardSeverity.BLOCK,
            action=action,
            message="Another legal action strictly dominates this line.",
        ),
        GuardFinding(
            code=GuardCode.FAKE_OUT_NOT_FIRST_TURN,
            severity=GuardSeverity.BLOCK,
            action=action,
            message="Fake Out is no longer eligible.",
        ),
    )
    blocked = DecisionOption(
        action=action,
        rank=None,
        expected_value=None,
        worst_case_value=None,
        best_case_value=None,
        blocked=True,
        findings=findings,
    )
    report = DecisionReport(
        ranked_options=(),
        blocked_options=(blocked,),
        opponent_estimates=(),
        response_summaries=(),
        findings=findings,
    )

    output = build_live_output(report)

    assert output.headline == "NO SAFE PLAY"
    assert output.first_action is None
    assert output.second_action is None
    assert len(output.reasons) == 2
    assert len(output.text.splitlines()) == 3


def test_slot_formatter_handles_switch_target_and_gimmick() -> None:
    switch = SlotAction(
        slot=0,
        kind=ActionKind.SWITCH,
        actor="garchomp",
        switch_to="kingambit",
    )
    mega_move = _move(
        1,
        "farfetchd",
        "bravebird",
        target_position=2,
        gimmick=Gimmick.MEGA,
    )

    assert format_slot_action(switch) == (
        "garchomp: Switch -> kingambit"
    )
    assert format_slot_action(mega_move) == (
        "farfetchd: Brave Bird -> foe 2 [Mega]"
    )

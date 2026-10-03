"""High-confidence catastrophic-error guards for Champions AI.

These guards are deliberately conservative. They only block a line when the
engine can identify a concrete rules error or mathematical domination. Lower
confidence concerns remain warnings instead of hard rejections.
"""

from dataclasses import dataclass
from enum import Enum

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.mechanics_evaluator import ActionPairEvaluation
from vgc_bench.src.champions_ai.response_matrix import ActionResponseSummary
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot, PokemonSnapshot
from vgc_bench.src.champions_ai.spread import normalize_move_id
from vgc_bench.src.champions_ai.turn_order import TurnSide


PROTECT_LIKE_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "burningbulwark",
    "obstruct",
    "maxguard",
    "silktrap",
}


class GuardSeverity(str, Enum):
    BLOCK = "block"
    WARNING = "warning"


class GuardCode(str, Enum):
    FAKE_OUT_NOT_FIRST_TURN = "fake_out_not_first_turn"
    REPEATED_PROTECT_RISK = "repeated_protect_risk"
    STRICTLY_DOMINATED = "strictly_dominated"
    GUARANTEED_MATCH_LOSS = "guaranteed_match_loss"


@dataclass(frozen=True)
class GuardFinding:
    code: GuardCode
    severity: GuardSeverity
    action: JointAction
    message: str
    alternative: JointAction | None = None


def _active_pokemon(
    snapshot: DecisionSnapshot,
    slot: int,
) -> PokemonSnapshot | None:
    active = dict(snapshot.player.active_slots)
    name = active.get(slot)
    if name is None:
        return None
    return next(
        (
            pokemon
            for pokemon in snapshot.player.pokemon
            if pokemon.name == name
        ),
        None,
    )


def _protect_success_probability(streak: int) -> float:
    if streak <= 0:
        return 1.0
    return 1.0 / min(3 ** streak, 729)


def inspect_action_rules(
    snapshot: DecisionSnapshot,
    action: JointAction,
) -> tuple[GuardFinding, ...]:
    """Inspect one selected joint action for deterministic/high-confidence risks."""

    findings: list[GuardFinding] = []

    for slot_action in (action.first, action.second):
        if slot_action.kind is not ActionKind.MOVE or not slot_action.move:
            continue

        pokemon = _active_pokemon(snapshot, slot_action.slot)
        move_id = normalize_move_id(slot_action.move)

        if (
            move_id == "fakeout"
            and pokemon is not None
            and pokemon.first_turn is False
        ):
            findings.append(
                GuardFinding(
                    code=GuardCode.FAKE_OUT_NOT_FIRST_TURN,
                    severity=GuardSeverity.BLOCK,
                    action=action,
                    message=(
                        f"{pokemon.name} cannot use Fake Out successfully because "
                        "it is no longer on its first turn after switching in."
                    ),
                )
            )

        if (
            move_id in PROTECT_LIKE_MOVES
            and pokemon is not None
            and pokemon.protect_streak > 0
        ):
            probability = _protect_success_probability(pokemon.protect_streak)
            findings.append(
                GuardFinding(
                    code=GuardCode.REPEATED_PROTECT_RISK,
                    severity=GuardSeverity.WARNING,
                    action=action,
                    message=(
                        f"{pokemon.name} is attempting consecutive protection; "
                        f"current success probability is {probability:.3%}."
                    ),
                )
            )

    return tuple(findings)


def _cell_values(summary: ActionResponseSummary) -> dict[tuple, float]:
    return {
        cell.opponent_action.key: cell.outcome_value
        for cell in summary.cells
    }


def strictly_dominated_findings(
    summaries: tuple[ActionResponseSummary, ...],
    *,
    tolerance: float = 1e-9,
) -> tuple[GuardFinding, ...]:
    """Block actions that are no better against any modeled opponent response.

    Action A is strictly dominated when another action B is at least as good
    against every modeled opponent response and strictly better against at least
    one of them.
    """

    findings: list[GuardFinding] = []

    for candidate in summaries:
        candidate_values = _cell_values(candidate)

        for alternative in summaries:
            if alternative.our_action == candidate.our_action:
                continue

            alternative_values = _cell_values(alternative)
            if candidate_values.keys() != alternative_values.keys():
                continue

            never_worse = all(
                alternative_values[key] >= candidate_values[key] - tolerance
                for key in candidate_values
            )
            better_somewhere = any(
                alternative_values[key] > candidate_values[key] + tolerance
                for key in candidate_values
            )

            if never_worse and better_somewhere:
                findings.append(
                    GuardFinding(
                        code=GuardCode.STRICTLY_DOMINATED,
                        severity=GuardSeverity.BLOCK,
                        action=candidate.our_action,
                        alternative=alternative.our_action,
                        message=(
                            "Another legal joint action is at least as strong "
                            "against every modeled opponent response and strictly "
                            "better against at least one."
                        ),
                    )
                )
                break

    return tuple(findings)


def is_guaranteed_match_loss(
    evaluation: ActionPairEvaluation,
) -> bool:
    """Return True only when every simulated branch ends in an immediate loss."""

    if not evaluation.distribution.outcomes:
        return False

    for outcome in evaluation.distribution.outcomes:
        state = outcome.result.state

        player_alive = sum(
            profile.current_hp > 0
            for (side, _name), profile in state.profiles.items()
            if side is TurnSide.PLAYER
        )
        opponent_alive = sum(
            profile.current_hp > 0
            for (side, _name), profile in state.profiles.items()
            if side is TurnSide.OPPONENT
        )

        if player_alive > 0 or opponent_alive == 0:
            return False

    return True


def guaranteed_match_loss_finding(
    evaluation: ActionPairEvaluation,
) -> GuardFinding | None:
    if not is_guaranteed_match_loss(evaluation):
        return None

    return GuardFinding(
        code=GuardCode.GUARANTEED_MATCH_LOSS,
        severity=GuardSeverity.BLOCK,
        action=evaluation.our_action,
        message=(
            "Every simulated outcome for this action pairing ends with all of "
            "our remaining Pokemon fainted while the opponent still has one alive."
        ),
    )


def blocked_actions(
    findings: tuple[GuardFinding, ...],
) -> frozenset[JointAction]:
    return frozenset(
        finding.action
        for finding in findings
        if finding.severity is GuardSeverity.BLOCK
    )

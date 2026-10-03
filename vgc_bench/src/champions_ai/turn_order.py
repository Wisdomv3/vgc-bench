"""Turn action ordering for Champions AI.

This module mirrors the high-level queue ordering used by the pinned Pokemon
Showdown simulator. Move-specific and ability-specific priority changes can be
supplied through explicit priority overrides:

1. lower action order first (normal switches happen before normal moves),
2. higher move priority first,
3. effective Speed within the same order/priority bracket,
4. exact Speed ties remain ties so a later simulator can branch them fairly.

It deliberately returns tie groups instead of inventing a deterministic winner.
"""

from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key

from poke_env.battle import Move

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.speed import SpeedState, effective_speed
from vgc_bench.src.champions_ai.spread import normalize_move_id


class TurnSide(str, Enum):
    PLAYER = "player"
    OPPONENT = "opponent"


@dataclass(frozen=True)
class ScheduledAction:
    """One executable action with the ordering information needed this turn."""

    side: TurnSide
    slot: int
    action: SlotAction
    order: int
    priority: int
    speed: int

    @property
    def actor(self) -> str:
        if not self.action.actor:
            raise ValueError("scheduled actions require an actor")
        return self.action.actor


@dataclass(frozen=True)
class TurnOrderGroup:
    """Actions that are exactly tied before the simulator's random tie-break."""

    actions: tuple[ScheduledAction, ...]

    @property
    def is_speed_tie(self) -> bool:
        return len(self.actions) > 1


def _action_order(action: SlotAction) -> int:
    # Pokemon Showdown queue order: normal switch 103, normal move 200.
    if action.kind is ActionKind.SWITCH:
        return 103
    if action.kind is ActionKind.MOVE:
        return 200
    raise ValueError(f"{action.kind.value} is not an executable turn action")


def _move_priority(
    side: TurnSide,
    action: SlotAction,
    *,
    gen: int,
    priority_overrides: dict[tuple[TurnSide, str, str], int] | None = None,
) -> int:
    if action.kind is not ActionKind.MOVE:
        return 0
    if not action.move:
        raise ValueError("move action requires move")
    if not action.actor:
        raise ValueError("move action requires actor")

    move_id = normalize_move_id(action.move)
    key = (side, action.actor, move_id)
    if priority_overrides and key in priority_overrides:
        return priority_overrides[key]

    return Move(move_id, gen).priority


def _scheduled_actions(
    side: TurnSide,
    joint_action: JointAction,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    *,
    gen: int,
    priority_overrides: dict[tuple[TurnSide, str, str], int] | None = None,
) -> list[ScheduledAction]:
    scheduled: list[ScheduledAction] = []

    for action in (joint_action.first, joint_action.second):
        if action.kind in {ActionKind.PASS, ActionKind.DEFAULT}:
            continue
        if not action.actor:
            raise ValueError("move and switch actions require actor names")

        key = (side, action.actor)
        if key not in speed_states:
            raise ValueError(
                f"missing SpeedState for {side.value} actor {action.actor}"
            )

        scheduled.append(
            ScheduledAction(
                side=side,
                slot=action.slot,
                action=action,
                order=_action_order(action),
                priority=_move_priority(
                    side,
                    action,
                    gen=gen,
                    priority_overrides=priority_overrides,
                ),
                speed=effective_speed(speed_states[key]),
            )
        )

    return scheduled


def _compare_actions(
    first: ScheduledAction,
    second: ScheduledAction,
    *,
    trick_room: bool,
) -> int:
    if first.order != second.order:
        return -1 if first.order < second.order else 1

    if first.priority != second.priority:
        return -1 if first.priority > second.priority else 1

    if first.speed == second.speed:
        return 0

    if trick_room:
        return -1 if first.speed < second.speed else 1
    return -1 if first.speed > second.speed else 1


def _same_order_bracket(
    first: ScheduledAction,
    second: ScheduledAction,
) -> bool:
    return (
        first.order == second.order
        and first.priority == second.priority
        and first.speed == second.speed
    )


def build_turn_order(
    our_action: JointAction,
    opponent_action: JointAction,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    *,
    trick_room: bool = False,
    gen: int = 9,
    priority_overrides: dict[tuple[TurnSide, str, str], int] | None = None,
) -> tuple[TurnOrderGroup, ...]:
    """Return ordered action groups for one doubles turn.

    An exact Speed tie is returned as one TurnOrderGroup containing multiple
    actions. The future simulator will branch all legal tie orders instead of
    silently choosing one.
    """

    scheduled = _scheduled_actions(
        TurnSide.PLAYER,
        our_action,
        speed_states,
        gen=gen,
        priority_overrides=priority_overrides,
    )
    scheduled.extend(
        _scheduled_actions(
            TurnSide.OPPONENT,
            opponent_action,
            speed_states,
            gen=gen,
            priority_overrides=priority_overrides,
        )
    )

    ordered = sorted(
        scheduled,
        key=cmp_to_key(
            lambda first, second: _compare_actions(
                first,
                second,
                trick_room=trick_room,
            )
        ),
    )

    groups: list[TurnOrderGroup] = []
    for action in ordered:
        if groups and _same_order_bracket(groups[-1].actions[0], action):
            groups[-1] = TurnOrderGroup(groups[-1].actions + (action,))
        else:
            groups.append(TurnOrderGroup((action,)))

    return tuple(groups)

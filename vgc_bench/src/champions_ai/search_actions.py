"""Joint-action generation from a simulated board, without a live request.

Only supplied moves are considered. Missing move information and replacement
phases return None so search can stop at a labelled leaf instead of inventing
an action. Format-specific trapping/gimmicks and entry effects still require
the full simulator; this generator covers the current mechanics model.
"""

from itertools import product

from poke_env.battle import Move, MoveCategory

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.spread import normalize_move_id
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


def _other_side(side: TurnSide) -> TurnSide:
    return TurnSide.OPPONENT if side is TurnSide.PLAYER else TurnSide.PLAYER


def _move_selectable(
    state: ExactTurnState, side: TurnSide, actor: str, move: Move
) -> bool:
    key = (side, actor)
    move_id = move.id
    # A recharge turn needs a scheduled move to consume the recharge counter.
    if key in state.must_recharge:
        return True
    if move_id == "fakeout" and state.first_turn.get(key) is not True:
        return False
    if state.move_pp.get((side, actor, move_id), 1) <= 0:
        return False
    if (disabled := state.disabled_moves.get(key)) and disabled[0] == move_id:
        return False
    if (encore := state.encore_locks.get(key)) and encore[0] != move_id:
        return False
    if key in state.taunt_turns and move.category is MoveCategory.STATUS:
        return False
    last = state.last_moves.get(key)
    if key in state.tormented and last == move_id and move_id != "struggle":
        return False
    if move.entry.get("flags", {}).get("cantusetwice", False) and last == move_id:
        return False
    profile = state.profile(side, actor)
    locked = (
        normalize_move_id(profile.item or "")
        in {"choiceband", "choicescarf", "choicespecs"}
        or normalize_move_id(profile.ability or "") == "gorillatactics"
    )
    lock = state.choice_locks.get(key, last if locked else None)
    if locked and lock not in {None, "struggle", move_id}:
        return False
    for user_side, user in state.imprison_users:
        if (
            user_side is not side
            and state.is_active(user_side, user)
            and state.profile(user_side, user).current_hp > 0
            and move_id in state.known_moves.get((user_side, user), set())
        ):
            return False
    return True


def _targets(
    state: ExactTurnState, side: TurnSide, slot: int, move: Move
) -> tuple[int | None, ...]:
    target = move.target.name if move.target is not None else "SCRIPTED"
    foes = tuple(
        s + 1
        for (active_side, s), name in state.active_slots.items()
        if active_side is _other_side(side)
        and state.profile(active_side, name).current_hp > 0
    )
    ally = state.active_name(side, 1 - slot)
    allies = (
        ()
        if ally is None or state.profile(side, ally).current_hp <= 0
        else (-(2 - slot),)
    )
    if target == "ADJACENT_ALLY":
        return allies
    if target == "ADJACENT_ALLY_OR_SELF":
        return allies + (-(slot + 1),)
    if target in {"ADJACENT_FOE", "RANDOM_NORMAL"}:
        return foes
    if target in {"ANY", "NORMAL"}:
        return allies + foes
    return (None,)


def generate_search_actions(
    state: ExactTurnState,
    side: TurnSide,
    *,
    include_switches: bool = True,
    gen: int = 9,
) -> tuple[JointAction, ...] | None:
    """Enumerate future choices for the current modeled roster and move sets."""
    bench = sorted(
        name
        for (profile_side, name), profile in state.profiles.items()
        if profile_side is side
        and profile.current_hp > 0
        and not state.is_active(side, name)
    )
    groups: list[tuple[SlotAction, ...]] = []
    for slot in (0, 1):
        actor = state.active_name(side, slot)
        if actor is None or state.profile(side, actor).current_hp <= 0:
            if bench:
                return None  # Replacement is a separate decision, not a turn.
            groups.append((SlotAction(slot, ActionKind.PASS, actor=actor),))
            continue
        moves = state.known_moves.get((side, actor), set())
        if not moves:
            return None
        options: list[SlotAction] = []
        selected_moves = tuple(
            move
            for move_id in sorted(moves)
            if _move_selectable(state, side, actor, move := Move(move_id, gen))
        )
        if not selected_moves:
            selected_moves = (Move("struggle", gen),)
        for move in selected_moves:
            for position in _targets(state, side, slot, move):
                options.append(
                    SlotAction(
                        slot,
                        ActionKind.MOVE,
                        actor=actor,
                        move=move.id,
                        target_position=position,
                    )
                )
        if include_switches:
            options.extend(
                SlotAction(slot, ActionKind.SWITCH, actor=actor, switch_to=name)
                for name in bench
            )
        if not options:
            return None
        groups.append(tuple(options))
    return tuple(
        JointAction(first, second)
        for first, second in product(*groups)
        if not (
            first.kind is ActionKind.SWITCH
            and second.kind is ActionKind.SWITCH
            and first.switch_to == second.switch_to
        )
    )

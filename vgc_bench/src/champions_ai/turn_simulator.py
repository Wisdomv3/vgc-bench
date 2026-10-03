"""Deterministic first-pass turn executor for Champions AI.

This module executes one chosen joint action against one opponent joint action.
It is intentionally a deterministic building block: callers must choose one of
the 16 damage rolls explicitly. A later branch engine will expand accuracy,
damage rolls, Speed ties, Protect odds, critical hits, and secondary effects
into weighted outcomes.

Implemented in this first pass:
- voluntary switches,
- Protect-like blocking,
- Feint/breaks-Protect handling,
- targeted damage,
- spread damage,
- ally damage from all-adjacent spread moves,
- fainting and cancellation of fainted Pokemon's queued moves,
- Tailwind,
- Trick Room,
- Gen 8+ dynamic Speed reordering after each action,
- Prankster priority for status moves,
- Grassy Glide priority while Grassy Terrain is active.

Unsupported mechanics raise instead of being silently guessed.
"""

from dataclasses import dataclass, field, replace
from enum import Enum

from poke_env.battle import Move, MoveCategory

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.matchup import (
    CombatantProfile,
    MoveProfile,
    calculate_matchup_damage,
)
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.spread import (
    SpreadTarget,
    get_spread_profile,
    normalize_move_id,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide, build_turn_order


PROTECT_LIKE_MOVES = {
    "protect",
    "detect",
    "endure",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "burningbulwark",
    "obstruct",
    "maxguard",
    "silktrap",
}


class SimulationEventType(str, Enum):
    SWITCH = "switch"
    PROTECT = "protect"
    PROTECT_FAILED = "protect_failed"
    PROTECT_BROKEN = "protect_broken"
    FIELD = "field"
    DAMAGE = "damage"
    BLOCKED = "blocked"
    FAINT = "faint"
    SKIPPED = "skipped"


class UnsupportedTurnMechanic(RuntimeError):
    """Raised when v0 would otherwise have to guess an unsupported mechanic."""


class UnresolvedSpeedTie(RuntimeError):
    """Raised because the deterministic executor must not invent a tie winner."""


@dataclass
class ExactTurnState:
    """Exact state used by the deterministic turn executor."""

    profiles: dict[tuple[TurnSide, str], CombatantProfile]
    active_slots: dict[tuple[TurnSide, int], str]
    weather: str | None = None
    terrain: str | None = None
    trick_room: bool = False
    tailwind_sides: set[TurnSide] = field(default_factory=set)

    def copy(self) -> "ExactTurnState":
        return ExactTurnState(
            profiles=dict(self.profiles),
            active_slots=dict(self.active_slots),
            weather=self.weather,
            terrain=self.terrain,
            trick_room=self.trick_room,
            tailwind_sides=set(self.tailwind_sides),
        )

    def profile(self, side: TurnSide, name: str) -> CombatantProfile:
        key = (side, name)
        if key not in self.profiles:
            raise ValueError(f"missing profile for {side.value} Pokemon {name}")
        return self.profiles[key]

    def active_name(self, side: TurnSide, slot: int) -> str | None:
        return self.active_slots.get((side, slot))

    def is_active(self, side: TurnSide, name: str) -> bool:
        return name in {
            active_name
            for (active_side, _slot), active_name in self.active_slots.items()
            if active_side is side
        }


@dataclass(frozen=True)
class TurnSimulationConfig:
    """Explicit assumptions for one deterministic branch."""

    damage_roll_index: int
    protect_success: dict[tuple[TurnSide, str], bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.damage_roll_index <= 15:
            raise ValueError("damage_roll_index must be between 0 and 15")


@dataclass(frozen=True)
class SimulationEvent:
    type: SimulationEventType
    side: TurnSide
    actor: str
    move: str | None = None
    target: str | None = None
    damage: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class TurnSimulationResult:
    state: ExactTurnState
    events: tuple[SimulationEvent, ...]


def _other_side(side: TurnSide) -> TurnSide:
    return TurnSide.OPPONENT if side is TurnSide.PLAYER else TurnSide.PLAYER


def _pass_action(slot: int) -> SlotAction:
    return SlotAction(slot=slot, kind=ActionKind.PASS)


def _joint_from_pending(pending: list[SlotAction | None]) -> JointAction:
    return JointAction(
        first=pending[0] if pending[0] is not None else _pass_action(0),
        second=pending[1] if pending[1] is not None else _pass_action(1),
    )


def _move_key(side: TurnSide, actor: str, move: str) -> tuple[TurnSide, str, str]:
    return (side, actor, normalize_move_id(move))


def _derive_priority_overrides(
    state: ExactTurnState,
    our_pending: list[SlotAction | None],
    opponent_pending: list[SlotAction | None],
    *,
    gen: int,
) -> dict[tuple[TurnSide, str, str], int]:
    overrides: dict[tuple[TurnSide, str, str], int] = {}

    for side, pending in (
        (TurnSide.PLAYER, our_pending),
        (TurnSide.OPPONENT, opponent_pending),
    ):
        for action in pending:
            if (
                action is None
                or action.kind is not ActionKind.MOVE
                or not action.move
                or not action.actor
            ):
                continue

            move_id = normalize_move_id(action.move)
            move = Move(move_id, gen)
            priority = move.priority
            profile = state.profile(side, action.actor)
            ability = normalize_move_id(profile.ability or "")

            if ability == "prankster" and move.category is MoveCategory.STATUS:
                priority += 1

            if (
                move_id == "grassyglide"
                and normalize_move_id(state.terrain or "") == "grassyterrain"
            ):
                priority += 1

            if priority != move.priority:
                overrides[(side, action.actor, move_id)] = priority

    return overrides


def _sync_tailwind_speed_states(
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    side: TurnSide,
    active: bool,
) -> None:
    for key, speed_state in list(speed_states.items()):
        if key[0] is side:
            speed_states[key] = replace(speed_state, tailwind=active)


def _resolve_target(
    state: ExactTurnState,
    side: TurnSide,
    action: SlotAction,
) -> tuple[TurnSide, str] | None:
    if action.target_position is not None and action.target_position != 0:
        if action.target_position > 0:
            target_side = _other_side(side)
            target_slot = action.target_position - 1
        else:
            target_side = side
            target_slot = abs(action.target_position) - 1

        target_name = state.active_name(target_side, target_slot)
        if target_name is not None:
            return target_side, target_name

    if action.target:
        normalized_target = normalize_move_id(action.target)
        for (candidate_side, _slot), candidate_name in state.active_slots.items():
            if normalize_move_id(candidate_name) == normalized_target:
                return candidate_side, candidate_name

    return None


def _spread_targets(
    state: ExactTurnState,
    side: TurnSide,
    actor: str,
    move_id: str,
) -> list[tuple[TurnSide, str]]:
    profile = get_spread_profile(move_id)

    # Expanding Force is only spread in Psychic Terrain.
    if (
        normalize_move_id(move_id) == "expandingforce"
        and normalize_move_id(state.terrain or "") != "psychicterrain"
    ):
        return []

    target_mode = profile.target if profile is not None else SpreadTarget.ALL_FOES
    targets: list[tuple[TurnSide, str]] = []

    for (candidate_side, _slot), candidate_name in state.active_slots.items():
        if candidate_name == actor and candidate_side is side:
            continue

        if target_mode is SpreadTarget.ALL_FOES:
            if candidate_side is _other_side(side):
                targets.append((candidate_side, candidate_name))
        elif target_mode is SpreadTarget.ALL_ADJACENT:
            targets.append((candidate_side, candidate_name))

    return targets


def _apply_damage(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move_profile: MoveProfile,
    target_side: TurnSide,
    target_name: str,
    protected: set[tuple[TurnSide, str]],
    damage_roll_index: int,
) -> None:
    attacker = state.profile(side, actor)
    defender = state.profile(target_side, target_name)
    move = Move(normalize_move_id(move_profile.move_id), 9)

    protected_key = (target_side, target_name)
    if protected_key in protected:
        if move.breaks_protect:
            protected.discard(protected_key)
            events.append(
                SimulationEvent(
                    type=SimulationEventType.PROTECT_BROKEN,
                    side=side,
                    actor=actor,
                    move=move_profile.move_id,
                    target=target_name,
                    detail="The move breaks Protect-like protection.",
                )
            )
        else:
            events.append(
                SimulationEvent(
                    type=SimulationEventType.BLOCKED,
                    side=side,
                    actor=actor,
                    move=move_profile.move_id,
                    target=target_name,
                    detail="Blocked by Protect-like protection.",
                )
            )
            return

    damage_result = calculate_matchup_damage(
        attacker,
        defender,
        move_profile,
        weather=state.weather,
    )
    damage = damage_result.rolls[damage_roll_index]
    new_hp = max(0, defender.current_hp - damage)

    state.profiles[(target_side, target_name)] = replace(
        defender,
        current_hp=new_hp,
    )
    events.append(
        SimulationEvent(
            type=SimulationEventType.DAMAGE,
            side=side,
            actor=actor,
            move=move_profile.move_id,
            target=target_name,
            damage=damage,
            detail=f"{target_name}: {defender.current_hp} -> {new_hp} HP",
        )
    )

    if defender.current_hp > 0 and new_hp == 0:
        events.append(
            SimulationEvent(
                type=SimulationEventType.FAINT,
                side=target_side,
                actor=target_name,
                detail=f"{target_name} fainted.",
            )
        )


def _execute_switch(
    state: ExactTurnState,
    events: list[SimulationEvent],
    side: TurnSide,
    action: SlotAction,
) -> None:
    actor = action.actor
    target = action.switch_to
    if actor is None or target is None:
        raise ValueError("switch action requires actor and switch target")

    if state.active_name(side, action.slot) != actor:
        events.append(
            SimulationEvent(
                type=SimulationEventType.SKIPPED,
                side=side,
                actor=actor,
                detail="Actor is no longer active in that slot.",
            )
        )
        return

    target_profile = state.profile(side, target)
    if target_profile.current_hp <= 0:
        raise UnsupportedTurnMechanic("cannot switch to a fainted Pokemon")
    if state.is_active(side, target):
        raise UnsupportedTurnMechanic("cannot switch to an already active Pokemon")

    state.active_slots[(side, action.slot)] = target
    events.append(
        SimulationEvent(
            type=SimulationEventType.SWITCH,
            side=side,
            actor=actor,
            target=target,
            detail=f"{actor} switched to {target}.",
        )
    )


def _execute_move(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    protected: set[tuple[TurnSide, str]],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    config: TurnSimulationConfig,
    side: TurnSide,
    action: SlotAction,
    *,
    gen: int,
) -> None:
    actor = action.actor
    if actor is None or action.move is None:
        raise ValueError("move action requires actor and move")

    if not state.is_active(side, actor) or state.profile(side, actor).current_hp <= 0:
        events.append(
            SimulationEvent(
                type=SimulationEventType.SKIPPED,
                side=side,
                actor=actor,
                move=action.move,
                detail="Pokemon is fainted or no longer active.",
            )
        )
        return

    move_id = normalize_move_id(action.move)
    move = Move(move_id, gen)

    if move_id in PROTECT_LIKE_MOVES:
        success = config.protect_success.get((side, actor), True)
        if success:
            protected.add((side, actor))
            event_type = SimulationEventType.PROTECT
            detail = "Protect-like protection is active."
        else:
            event_type = SimulationEventType.PROTECT_FAILED
            detail = "Protect-like move failed in this deterministic branch."

        events.append(
            SimulationEvent(
                type=event_type,
                side=side,
                actor=actor,
                move=move_id,
                detail=detail,
            )
        )
        return

    if move_id == "tailwind":
        state.tailwind_sides.add(side)
        _sync_tailwind_speed_states(speed_states, side, True)
        events.append(
            SimulationEvent(
                type=SimulationEventType.FIELD,
                side=side,
                actor=actor,
                move=move_id,
                detail=f"Tailwind became active for {side.value}.",
            )
        )
        return

    if move_id == "trickroom":
        state.trick_room = not state.trick_room
        events.append(
            SimulationEvent(
                type=SimulationEventType.FIELD,
                side=side,
                actor=actor,
                move=move_id,
                detail=f"Trick Room set to {state.trick_room}.",
            )
        )
        return

    key = _move_key(side, actor, move_id)
    if key not in move_profiles:
        if move.category is MoveCategory.STATUS:
            raise UnsupportedTurnMechanic(
                f"status move {move_id} is not implemented in turn simulator v0"
            )
        raise ValueError(f"missing MoveProfile for {side.value} {actor} {move_id}")

    move_profile = move_profiles[key]

    targets: list[tuple[TurnSide, str]]
    if move_profile.spread:
        targets = _spread_targets(state, side, actor, move_id)
        if not targets:
            resolved = _resolve_target(state, side, action)
            targets = [] if resolved is None else [resolved]
    else:
        resolved = _resolve_target(state, side, action)
        targets = [] if resolved is None else [resolved]

    if not targets:
        events.append(
            SimulationEvent(
                type=SimulationEventType.SKIPPED,
                side=side,
                actor=actor,
                move=move_id,
                detail="No valid target remained.",
            )
        )
        return

    # Showdown's 0.75 spread modifier applies only when multiple targets exist.
    effective_move_profile = replace(
        move_profile,
        spread=move_profile.spread and len(targets) > 1,
    )

    for target_side, target_name in targets:
        if state.profile(target_side, target_name).current_hp <= 0:
            continue
        _apply_damage(
            state,
            events,
            side=side,
            actor=actor,
            move_profile=effective_move_profile,
            target_side=target_side,
            target_name=target_name,
            protected=protected,
            damage_roll_index=config.damage_roll_index,
        )


def simulate_turn(
    initial_state: ExactTurnState,
    our_action: JointAction,
    opponent_action: JointAction,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    config: TurnSimulationConfig,
    *,
    gen: int = 9,
) -> TurnSimulationResult:
    """Execute one deterministic branch of a doubles turn.

    Gen 8+ action Speed is refreshed after each action, matching Showdown's
    dynamic Speed reordering behavior. Exact Speed ties are deliberately rejected
    here so the future probability brancher can split them rather than guess.
    """

    state = initial_state.copy()
    speed_states = dict(speed_states)

    for side in state.tailwind_sides:
        _sync_tailwind_speed_states(speed_states, side, True)

    our_pending: list[SlotAction | None] = [
        our_action.first,
        our_action.second,
    ]
    opponent_pending: list[SlotAction | None] = [
        opponent_action.first,
        opponent_action.second,
    ]

    protected: set[tuple[TurnSide, str]] = set()
    events: list[SimulationEvent] = []

    while any(action is not None for action in our_pending + opponent_pending):
        priority_overrides = _derive_priority_overrides(
            state,
            our_pending,
            opponent_pending,
            gen=gen,
        )
        groups = build_turn_order(
            _joint_from_pending(our_pending),
            _joint_from_pending(opponent_pending),
            speed_states,
            trick_room=state.trick_room,
            gen=gen,
            priority_overrides=priority_overrides,
        )

        if not groups:
            break

        next_group = groups[0]
        if next_group.is_speed_tie:
            actors = ", ".join(action.actor for action in next_group.actions)
            raise UnresolvedSpeedTie(
                f"deterministic simulator cannot choose Speed tie between {actors}"
            )

        scheduled = next_group.actions[0]
        pending = (
            our_pending
            if scheduled.side is TurnSide.PLAYER
            else opponent_pending
        )
        action = pending[scheduled.slot]
        pending[scheduled.slot] = None

        if action is None:
            continue

        if action.kind is ActionKind.SWITCH:
            _execute_switch(state, events, scheduled.side, action)
        elif action.kind is ActionKind.MOVE:
            _execute_move(
                state,
                speed_states,
                events,
                protected,
                move_profiles,
                config,
                scheduled.side,
                action,
                gen=gen,
            )
        else:
            events.append(
                SimulationEvent(
                    type=SimulationEventType.SKIPPED,
                    side=scheduled.side,
                    actor=action.actor or f"slot {action.slot + 1}",
                    detail=f"{action.kind.value} has no executable effect.",
                )
            )

        # Showdown dynamically refreshes remaining action Speed in Gen 8+.
        # Rebuilding the order on the next loop iteration provides that behavior.

    return TurnSimulationResult(
        state=state,
        events=tuple(events),
    )

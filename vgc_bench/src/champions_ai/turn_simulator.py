"""Deterministic turn executor used by the probabilistic branch engine.

The executor can run in two modes:

- fixed mode, where the caller supplies deterministic assumptions;
- branch-request mode, where unresolved random events raise
  RandomDecisionRequired with their exact options and probabilities.

The separate turn_branching module repeatedly answers those requests and
re-runs the turn, producing a weighted distribution of outcomes without the
deterministic core ever making up a random result.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from poke_env.battle import Move, MoveCategory

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.field_mechanics import (
    bypasses_redirection,
    is_grounded,
    move_can_be_redirected,
    powder_redirection_immune,
    psychic_terrain_blocks_priority,
    wide_guard_blocks,
)
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


# These moves block attacks in the same broad way as Protect for the mechanics
# currently implemented here. Endure is deliberately excluded because it does
# not block damage.
PROTECT_BLOCK_MOVES = {
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

STALL_MOVES = PROTECT_BLOCK_MOVES | {
    "wideguard",
}

RandomChoiceValue: TypeAlias = bool | int | str
RandomChoiceKey: TypeAlias = tuple[str, ...]


class SimulationEventType(str, Enum):
    SWITCH = "switch"
    PROTECT = "protect"
    PROTECT_FAILED = "protect_failed"
    PROTECT_BROKEN = "protect_broken"
    WIDE_GUARD = "wide_guard"
    REDIRECT = "redirect"
    PRIORITY_BLOCKED = "priority_blocked"
    FIELD = "field"
    DAMAGE = "damage"
    MISS = "miss"
    CRITICAL = "critical"
    STATUS = "status"
    BOOST = "boost"
    FLINCH = "flinch"
    FLINCHED = "flinched"
    CANNOT_MOVE = "cannot_move"
    STATUS_CURED = "status_cured"
    CONFUSION = "confusion"
    RECHARGE = "recharge"
    BLOCKED = "blocked"
    ITEM = "item"
    RECOIL = "recoil"
    CONTACT_DAMAGE = "contact_damage"
    HEAL = "heal"
    RESIDUAL = "residual"
    FAINT = "faint"
    SKIPPED = "skipped"


class UnsupportedTurnMechanic(RuntimeError):
    """Raised when the simulator would otherwise have to guess a mechanic."""


class UnresolvedSpeedTie(RuntimeError):
    """Raised in fixed mode when the executor is not allowed to break a tie."""


class RandomDecisionRequired(RuntimeError):
    """Request one unresolved random decision from the branch engine."""

    def __init__(
        self,
        key: RandomChoiceKey,
        options: tuple[tuple[RandomChoiceValue, float], ...],
    ) -> None:
        if not options:
            raise ValueError("random decision requires at least one option")
        total = sum(probability for _value, probability in options)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("random decision probabilities must sum to 1")
        if any(probability < 0 for _value, probability in options):
            raise ValueError("random decision probabilities cannot be negative")

        self.key = key
        self.options = options
        super().__init__(f"random decision required: {key[0]} {key[1:]}")


@dataclass
class ExactTurnState:
    """Exact state used while executing one candidate turn."""

    profiles: dict[tuple[TurnSide, str], CombatantProfile]
    active_slots: dict[tuple[TurnSide, int], str]
    weather: str | None = None
    terrain: str | None = None
    trick_room: bool = False
    tailwind_sides: set[TurnSide] = field(default_factory=set)
    protect_streaks: dict[tuple[TurnSide, str], int] = field(default_factory=dict)
    toxic_stages: dict[tuple[TurnSide, str], int] = field(default_factory=dict)
    sleep_turns: dict[tuple[TurnSide, str], int] = field(default_factory=dict)
    confusion_turns: dict[tuple[TurnSide, str], int] = field(default_factory=dict)
    must_recharge: set[tuple[TurnSide, str]] = field(default_factory=set)
    field_conditions: set[str] = field(default_factory=set)

    def copy(self) -> "ExactTurnState":
        return ExactTurnState(
            profiles=dict(self.profiles),
            active_slots=dict(self.active_slots),
            weather=self.weather,
            terrain=self.terrain,
            trick_room=self.trick_room,
            tailwind_sides=set(self.tailwind_sides),
            protect_streaks=dict(self.protect_streaks),
            toxic_stages=dict(self.toxic_stages),
            sleep_turns=dict(self.sleep_turns),
            confusion_turns=dict(self.confusion_turns),
            must_recharge=set(self.must_recharge),
            field_conditions=set(self.field_conditions),
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
    """Assumptions and branching switches for one deterministic replay."""

    damage_roll_index: int | None = None
    protect_success: dict[tuple[TurnSide, str], bool] = field(default_factory=dict)
    random_choices: dict[RandomChoiceKey, RandomChoiceValue] = field(
        default_factory=dict
    )
    branch_damage_rolls: bool = False
    branch_accuracy: bool = False
    branch_critical_hits: bool = False
    branch_secondary_effects: bool = False
    branch_before_move_status: bool = False
    branch_protect: bool = False
    branch_speed_ties: bool = False

    def __post_init__(self) -> None:
        if self.damage_roll_index is not None and not 0 <= self.damage_roll_index <= 15:
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


def _choice(
    config: TurnSimulationConfig,
    key: RandomChoiceKey,
    options: tuple[tuple[RandomChoiceValue, float], ...],
) -> RandomChoiceValue:
    if key in config.random_choices:
        selected = config.random_choices[key]
        valid_values = {value for value, probability in options if probability > 0}
        if selected not in valid_values:
            raise ValueError(f"invalid random choice {selected!r} for {key}")
        return selected

    raise RandomDecisionRequired(key, options)


def _boolean_choice(
    config: TurnSimulationConfig,
    key: RandomChoiceKey,
    success_probability: float,
) -> bool:
    probability = max(0.0, min(1.0, success_probability))
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return bool(
        _choice(
            config,
            key,
            (
                (True, probability),
                (False, 1.0 - probability),
            ),
        )
    )


def _protect_success_probability(streak: int) -> float:
    """Match Showdown's modern stall counter: 1, 1/3, 1/9 ... capped 1/729."""

    if streak <= 0:
        return 1.0
    denominator = min(3 ** streak, 729)
    return 1.0 / denominator


def _critical_probability(
    attacker: CombatantProfile,
    defender: CombatantProfile,
    move: Move,
) -> float:
    """Base modern critical-hit probability plus a few directly known modifiers."""

    defender_ability = normalize_move_id(defender.ability or "")
    if defender_ability in {"battlearmor", "shellarmor"}:
        return 0.0

    attacker_ability = normalize_move_id(attacker.ability or "")
    if (
        attacker_ability == "merciless"
        and normalize_move_id(defender.status or "") in {"psn", "tox"}
    ):
        return 1.0

    raw_ratio = move.crit_ratio
    if raw_ratio >= 6:
        return 1.0

    # Showdown's active move defaults to crit stage 1 even though poke-env's
    # raw move entry reports 0 when no explicit critRatio field exists.
    stage = raw_ratio if raw_ratio > 0 else 1

    if attacker_ability == "superluck":
        stage += 1
    if normalize_move_id(attacker.item or "") in {"scopelens", "razorclaw"}:
        stage += 1

    stage = max(1, min(stage, 4))
    denominators = {
        1: 24,
        2: 8,
        3: 2,
        4: 1,
    }
    return 1.0 / denominators[stage]


def _effective_move_priority(
    state: ExactTurnState,
    side: TurnSide,
    actor: str,
    move: Move,
) -> int:
    priority = move.priority
    profile = state.profile(side, actor)
    ability = normalize_move_id(profile.ability or "")

    if ability == "prankster" and move.category is MoveCategory.STATUS:
        priority += 1

    if (
        move.id == "grassyglide"
        and normalize_move_id(state.terrain or "") == "grassyterrain"
    ):
        priority += 1

    return priority


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
            priority = _effective_move_priority(
                state,
                side,
                action.actor,
                move,
            )

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


def _redirect_target(
    state: ExactTurnState,
    side: TurnSide,
    actor: str,
    move: Move,
    target: tuple[TurnSide, str] | None,
    redirections: dict[TurnSide, tuple[str, str]],
) -> tuple[tuple[TurnSide, str] | None, str | None]:
    if target is None:
        return None, None

    target_side, target_name = target
    if target_side is side or not move_can_be_redirected(move):
        return target, None

    attacker = state.profile(side, actor)
    if bypasses_redirection(attacker, move):
        return target, None

    redirect = redirections.get(target_side)
    if redirect is None:
        return target, None

    redirect_name, redirect_move = redirect
    if (
        not state.is_active(target_side, redirect_name)
        or state.profile(target_side, redirect_name).current_hp <= 0
    ):
        return target, None

    if (
        redirect_move == "ragepowder"
        and powder_redirection_immune(attacker)
    ):
        return target, None

    if redirect_name == target_name:
        return target, None

    return (target_side, redirect_name), target_name


def _spread_targets(
    state: ExactTurnState,
    side: TurnSide,
    actor: str,
    move_id: str,
) -> list[tuple[TurnSide, str]]:
    profile = get_spread_profile(move_id)

    # Expanding Force is spread only while Psychic Terrain is active.
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


def _accuracy_probability(
    attacker: CombatantProfile,
    defender: CombatantProfile,
    move: Move,
) -> float:
    """Initial accuracy model using move accuracy and directly known No Guard."""

    if normalize_move_id(attacker.ability or "") == "noguard":
        return 1.0
    if normalize_move_id(defender.ability or "") == "noguard":
        return 1.0
    return move.accuracy


def _damage_amount(
    config: TurnSimulationConfig,
    damage_rolls: tuple[int, ...],
    *,
    side: TurnSide,
    actor: str,
    move_id: str,
    target_side: TurnSide,
    target_name: str,
) -> int:
    if config.branch_damage_rolls:
        key = (
            "damage_roll",
            side.value,
            actor,
            move_id,
            target_side.value,
            target_name,
        )

        counts: dict[int, int] = {}
        for damage in damage_rolls:
            counts[damage] = counts.get(damage, 0) + 1

        selected = _choice(
            config,
            key,
            tuple(
                (damage, count / len(damage_rolls))
                for damage, count in sorted(counts.items())
            ),
        )
        return int(selected)

    if config.damage_roll_index is None:
        raise ValueError(
            "fixed simulation requires damage_roll_index when damage branching is off"
        )
    return damage_rolls[config.damage_roll_index]


def _max_hp_fraction(
    max_hp: int,
    numerator: int,
    denominator: int,
) -> int:
    return max(1, max_hp * numerator // denominator)


def _round_fraction(
    value: int,
    numerator: int,
    denominator: int,
) -> int:
    """Positive integer Math.round-style fraction used by Showdown recoil."""

    return max(
        1,
        (value * numerator * 2 + denominator) // (2 * denominator),
    )


def _apply_indirect_damage(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    name: str,
    amount: int,
    event_type: SimulationEventType,
    detail: str,
    source_side: TurnSide | None = None,
    source_name: str | None = None,
) -> int:
    profile = state.profile(side, name)
    if profile.current_hp <= 0:
        return 0

    if normalize_move_id(profile.ability or "") == "magicguard":
        return 0

    dealt = min(max(0, amount), profile.current_hp)
    if dealt <= 0:
        return 0

    updated = replace(
        profile,
        current_hp=profile.current_hp - dealt,
    )
    state.profiles[(side, name)] = updated

    events.append(
        SimulationEvent(
            type=event_type,
            side=source_side or side,
            actor=source_name or name,
            target=name,
            damage=dealt,
            detail=detail,
        )
    )

    if updated.current_hp == 0:
        events.append(
            SimulationEvent(
                type=SimulationEventType.FAINT,
                side=side,
                actor=name,
                detail=f"{name} fainted.",
            )
        )

    return dealt


def _heal_profile(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    name: str,
    amount: int,
    detail: str,
) -> int:
    profile = state.profile(side, name)
    if profile.current_hp <= 0 or profile.current_hp >= profile.max_hp:
        return 0

    healed = min(amount, profile.max_hp - profile.current_hp)
    if healed <= 0:
        return 0

    state.profiles[(side, name)] = replace(
        profile,
        current_hp=profile.current_hp + healed,
    )
    events.append(
        SimulationEvent(
            type=SimulationEventType.HEAL,
            side=side,
            actor=name,
            target=name,
            damage=-healed,
            detail=detail,
        )
    )
    return healed


def _move_makes_contact(
    attacker: CombatantProfile,
    move: Move,
) -> bool:
    if not move.entry.get("flags", {}).get("contact"):
        return False

    if normalize_move_id(attacker.ability or "") == "longreach":
        return False

    if normalize_move_id(attacker.item or "") == "protectivepads":
        return False

    return True


def _apply_contact_punishment(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    attacker_side: TurnSide,
    attacker_name: str,
    defender_side: TurnSide,
    defender_name: str,
    move: Move,
) -> None:
    attacker = state.profile(attacker_side, attacker_name)
    defender = state.profile(defender_side, defender_name)

    if (
        attacker.current_hp <= 0
        or not _move_makes_contact(attacker, move)
    ):
        return

    defender_ability = normalize_move_id(defender.ability or "")
    if defender_ability in {"roughskin", "ironbarbs"}:
        attacker = state.profile(attacker_side, attacker_name)
        _apply_indirect_damage(
            state,
            events,
            side=attacker_side,
            name=attacker_name,
            amount=_max_hp_fraction(attacker.max_hp, 1, 8),
            event_type=SimulationEventType.CONTACT_DAMAGE,
            detail=(
                f"{attacker_name} took contact damage from "
                f"{defender_ability}."
            ),
            source_side=defender_side,
            source_name=defender_name,
        )

    attacker = state.profile(attacker_side, attacker_name)
    if (
        attacker.current_hp > 0
        and normalize_move_id(defender.item or "") == "rockyhelmet"
    ):
        _apply_indirect_damage(
            state,
            events,
            side=attacker_side,
            name=attacker_name,
            amount=_max_hp_fraction(attacker.max_hp, 1, 6),
            event_type=SimulationEventType.CONTACT_DAMAGE,
            detail=f"{attacker_name} took Rocky Helmet damage.",
            source_side=defender_side,
            source_name=defender_name,
        )


def _apply_move_recoil(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    total_damage_dealt: int,
) -> None:
    if total_damage_dealt <= 0 or not move.entry.get("recoil"):
        return

    profile = state.profile(side, actor)
    if profile.current_hp <= 0:
        return

    ability = normalize_move_id(profile.ability or "")
    if ability in {"magicguard", "rockhead"}:
        return

    numerator, denominator = move.entry["recoil"]
    amount = _round_fraction(
        total_damage_dealt,
        int(numerator),
        int(denominator),
    )
    _apply_indirect_damage(
        state,
        events,
        side=side,
        name=actor,
        amount=amount,
        event_type=SimulationEventType.RECOIL,
        detail=f"{actor} took recoil from {move.id}.",
    )


def _apply_life_orb_recoil(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    total_damage_dealt: int,
) -> None:
    if total_damage_dealt <= 0 or move.category is MoveCategory.STATUS:
        return

    profile = state.profile(side, actor)
    if (
        profile.current_hp <= 0
        or normalize_move_id(profile.item or "") != "lifeorb"
    ):
        return

    _apply_indirect_damage(
        state,
        events,
        side=side,
        name=actor,
        amount=_max_hp_fraction(profile.max_hp, 1, 10),
        event_type=SimulationEventType.RECOIL,
        detail=f"{actor} took Life Orb recoil.",
    )


def _sync_speed_profile(
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    side: TurnSide,
    name: str,
    profile: CombatantProfile,
) -> None:
    key = (side, name)
    if key not in speed_states:
        return

    speed_states[key] = replace(
        speed_states[key],
        stage=int(profile.boosts.get("spe", 0)),
        paralyzed=normalize_move_id(profile.status or "") == "par",
    )


def _status_allowed(profile: CombatantProfile, status: str) -> bool:
    if profile.current_hp <= 0 or profile.status is not None:
        return False

    status = normalize_move_id(status)
    types = {normalize_move_id(type_name) for type_name in profile.types}
    ability = normalize_move_id(profile.ability or "")

    if ability == "purifyingsalt":
        return False

    if status in {"psn", "tox"}:
        if types.intersection({"poison", "steel"}):
            return False
        if ability in {"immunity", "pastelveil"}:
            return False

    if status == "par":
        if "electric" in types:
            return False
        if ability == "limber":
            return False

    if status == "brn":
        if "fire" in types:
            return False
        if ability in {"waterveil", "waterbubble", "thermalexchange"}:
            return False

    if status in {"frz", "frostbite"}:
        if "ice" in types:
            return False
        if ability == "magmaarmor":
            return False

    if status == "slp" and ability in {"insomnia", "vitalspirit"}:
        return False

    return True


def _apply_status(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    *,
    source_side: TurnSide,
    source_name: str,
    target_side: TurnSide,
    target_name: str,
    move_id: str,
    status: str,
) -> bool:
    profile = state.profile(target_side, target_name)
    status_id = normalize_move_id(status)
    if not _status_allowed(profile, status_id):
        return False

    updated = replace(profile, status=status_id)
    state.profiles[(target_side, target_name)] = updated

    status_key = (target_side, target_name)
    if status_id == "slp":
        # Zero means the 1-3-turn sleep duration has not been sampled yet.
        state.sleep_turns[status_key] = 0
    if status_id == "tox":
        state.toxic_stages[status_key] = 0

    _sync_speed_profile(speed_states, target_side, target_name, updated)

    events.append(
        SimulationEvent(
            type=SimulationEventType.STATUS,
            side=source_side,
            actor=source_name,
            move=move_id,
            target=target_name,
            detail=f"{target_name} gained status {status_id}.",
        )
    )
    return True


def _clear_status(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    name: str,
    reason: str,
) -> None:
    profile = state.profile(side, name)
    old_status = normalize_move_id(profile.status or "")
    if not old_status:
        return

    updated = replace(profile, status=None)
    state.profiles[(side, name)] = updated
    state.sleep_turns.pop((side, name), None)
    state.toxic_stages.pop((side, name), None)
    _sync_speed_profile(speed_states, side, name, updated)

    events.append(
        SimulationEvent(
            type=SimulationEventType.STATUS_CURED,
            side=side,
            actor=name,
            target=name,
            detail=f"{name} was cured of {old_status}: {reason}.",
        )
    )


def _sleep_or_freeze_allows_move(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    config: TurnSimulationConfig,
) -> bool:
    profile = state.profile(side, actor)
    status = normalize_move_id(profile.status or "")
    status_key = (side, actor)

    if status == "slp":
        if status_key not in state.sleep_turns:
            raise UnsupportedTurnMechanic(
                f"sleep duration is unknown for {side.value} {actor}"
            )

        remaining = state.sleep_turns[status_key]
        if remaining == 0:
            if not config.branch_before_move_status:
                raise UnsupportedTurnMechanic(
                    "sleep duration requires before-move status branching"
                )
            remaining = int(
                _choice(
                    config,
                    (
                        "sleep_duration",
                        side.value,
                        actor,
                    ),
                    (
                        (2, 1.0 / 3.0),
                        (3, 1.0 / 3.0),
                        (4, 1.0 / 3.0),
                    ),
                )
            )

        decrement = (
            2
            if normalize_move_id(profile.ability or "") == "earlybird"
            else 1
        )
        remaining -= decrement

        if remaining <= 0:
            state.sleep_turns[status_key] = 0
            _clear_status(
                state,
                speed_states,
                events,
                side=side,
                name=actor,
                reason="woke up",
            )
            return True

        state.sleep_turns[status_key] = remaining
        if bool(move.entry.get("sleepUsable", False)):
            return True

        events.append(
            SimulationEvent(
                type=SimulationEventType.CANNOT_MOVE,
                side=side,
                actor=actor,
                move=move.id,
                detail=f"{actor} is asleep and cannot move.",
            )
        )
        return False

    if status == "frz":
        flags = move.entry.get("flags", {})
        if bool(flags.get("defrost", False)):
            _clear_status(
                state,
                speed_states,
                events,
                side=side,
                name=actor,
                reason=f"{move.id} defrosted its user",
            )
            return True

        if not config.branch_before_move_status:
            raise UnsupportedTurnMechanic(
                "freeze requires before-move status branching"
            )

        thawed = _boolean_choice(
            config,
            (
                "freeze_thaw",
                side.value,
                actor,
                move.id,
            ),
            1.0 / 5.0,
        )
        if thawed:
            _clear_status(
                state,
                speed_states,
                events,
                side=side,
                name=actor,
                reason="random thaw",
            )
            return True

        events.append(
            SimulationEvent(
                type=SimulationEventType.CANNOT_MOVE,
                side=side,
                actor=actor,
                move=move.id,
                detail=f"{actor} is frozen solid and cannot move.",
            )
        )
        return False

    return True


def _paralysis_allows_move(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    config: TurnSimulationConfig,
) -> bool:
    profile = state.profile(side, actor)
    if normalize_move_id(profile.status or "") != "par":
        return True

    if not config.branch_before_move_status:
        raise UnsupportedTurnMechanic(
            "paralysis requires before-move status branching"
        )

    fully_paralyzed = _boolean_choice(
        config,
        (
            "full_paralysis",
            side.value,
            actor,
            move.id,
        ),
        1.0 / 4.0,
    )
    if not fully_paralyzed:
        return True

    events.append(
        SimulationEvent(
            type=SimulationEventType.CANNOT_MOVE,
            side=side,
            actor=actor,
            move=move.id,
            detail=f"{actor} is fully paralyzed and cannot move.",
        )
    )
    return False


def _stage_stat(stat: int, stage: int) -> int:
    stage = max(-6, min(6, stage))
    if stage >= 0:
        return stat * (2 + stage) // 2
    return stat * 2 // (2 - stage)


def _confusion_damage_rolls(profile: CombatantProfile) -> tuple[int, ...]:
    attack = _stage_stat(
        profile.stats["atk"],
        int(profile.boosts.get("atk", 0)),
    )
    defense = _stage_stat(
        profile.stats["def"],
        int(profile.boosts.get("def", 0)),
    )
    level_factor = 2 * profile.level // 5 + 2
    base_damage = level_factor * 40 * attack
    base_damage //= defense
    base_damage //= 50
    base_damage += 2

    return tuple(
        max(1, base_damage * random_factor // 100)
        for random_factor in range(85, 101)
    )


def _apply_confusion_self_hit(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    config: TurnSimulationConfig,
) -> int:
    profile = state.profile(side, actor)
    damage = _damage_amount(
        config,
        _confusion_damage_rolls(profile),
        side=side,
        actor=actor,
        move_id="confused",
        target_side=side,
        target_name=actor,
    )

    if (
        normalize_move_id(profile.item or "") == "focussash"
        and profile.current_hp == profile.max_hp
        and damage >= profile.current_hp
    ):
        damage = max(0, profile.current_hp - 1)
        profile = replace(profile, item=None)
        events.append(
            SimulationEvent(
                type=SimulationEventType.ITEM,
                side=side,
                actor=actor,
                move="focussash",
                target=actor,
                detail=f"{actor} survived confusion damage with Focus Sash.",
            )
        )

    actual_damage = min(damage, profile.current_hp)
    new_hp = max(0, profile.current_hp - actual_damage)
    state.profiles[(side, actor)] = replace(
        profile,
        current_hp=new_hp,
    )
    events.append(
        SimulationEvent(
            type=SimulationEventType.DAMAGE,
            side=side,
            actor=actor,
            move="confused",
            target=actor,
            damage=actual_damage,
            detail=f"{actor} hurt itself in confusion.",
        )
    )

    if profile.current_hp > 0 and new_hp == 0:
        events.append(
            SimulationEvent(
                type=SimulationEventType.FAINT,
                side=side,
                actor=actor,
                detail=f"{actor} fainted.",
            )
        )

    return actual_damage


def _confusion_allows_move(
    state: ExactTurnState,
    events: list[SimulationEvent],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    config: TurnSimulationConfig,
) -> bool:
    key = (side, actor)
    if key not in state.confusion_turns:
        return True

    profile = state.profile(side, actor)
    if normalize_move_id(profile.ability or "") == "owntempo":
        state.confusion_turns.pop(key, None)
        events.append(
            SimulationEvent(
                type=SimulationEventType.CONFUSION,
                side=side,
                actor=actor,
                move=move.id,
                target=actor,
                detail=f"{actor}'s Own Tempo ended confusion.",
            )
        )
        return True

    remaining = state.confusion_turns[key]
    if remaining == 0:
        if not config.branch_before_move_status:
            raise UnsupportedTurnMechanic(
                "confusion duration requires before-move status branching"
            )
        remaining = int(
            _choice(
                config,
                (
                    "confusion_duration",
                    side.value,
                    actor,
                ),
                (
                    (2, 0.25),
                    (3, 0.25),
                    (4, 0.25),
                    (5, 0.25),
                ),
            )
        )

    remaining -= 1
    if remaining <= 0:
        state.confusion_turns.pop(key, None)
        events.append(
            SimulationEvent(
                type=SimulationEventType.CONFUSION,
                side=side,
                actor=actor,
                move=move.id,
                target=actor,
                detail=f"{actor} snapped out of confusion.",
            )
        )
        return True

    state.confusion_turns[key] = remaining
    if not config.branch_before_move_status:
        raise UnsupportedTurnMechanic(
            "confusion self-hit requires before-move status branching"
        )

    self_hit = _boolean_choice(
        config,
        (
            "confusion_self_hit",
            side.value,
            actor,
            move.id,
            str(remaining),
        ),
        33.0 / 100.0,
    )
    if not self_hit:
        return True

    _apply_confusion_self_hit(
        state,
        events,
        side=side,
        actor=actor,
        config=config,
    )
    events.append(
        SimulationEvent(
            type=SimulationEventType.CANNOT_MOVE,
            side=side,
            actor=actor,
            move=move.id,
            detail=f"{actor} hurt itself in confusion and could not move.",
        )
    )
    return False


def _apply_boosts(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    *,
    source_side: TurnSide,
    source_name: str,
    target_side: TurnSide,
    target_name: str,
    move_id: str,
    boosts: dict[str, int],
) -> bool:
    profile = state.profile(target_side, target_name)
    ability = normalize_move_id(profile.ability or "")
    from_opponent = source_side is not target_side

    updated_boosts = dict(profile.boosts)
    applied: dict[str, int] = {}
    lowered_by_opponent = False

    for stat, delta in boosts.items():
        if stat not in {"atk", "def", "spa", "spd", "spe", "accuracy", "evasion"}:
            continue
        if not isinstance(delta, int) or delta == 0:
            continue

        if (
            delta < 0
            and from_opponent
            and ability in {"clearbody", "whitesmoke", "fullmetalbody"}
        ):
            continue

        old_stage = int(updated_boosts.get(stat, 0))
        new_stage = max(-6, min(6, old_stage + delta))
        actual_delta = new_stage - old_stage
        if actual_delta == 0:
            continue

        updated_boosts[stat] = new_stage
        applied[stat] = actual_delta
        if actual_delta < 0 and from_opponent:
            lowered_by_opponent = True

    if not applied:
        return False

    updated = replace(profile, boosts=updated_boosts)
    state.profiles[(target_side, target_name)] = updated
    _sync_speed_profile(speed_states, target_side, target_name, updated)

    events.append(
        SimulationEvent(
            type=SimulationEventType.BOOST,
            side=source_side,
            actor=source_name,
            move=move_id,
            target=target_name,
            detail=f"{target_name} stat changes: {applied}.",
        )
    )

    if lowered_by_opponent:
        reaction_stat = None
        if ability == "defiant":
            reaction_stat = "atk"
        elif ability == "competitive":
            reaction_stat = "spa"

        if reaction_stat is not None:
            reacted_profile = state.profile(target_side, target_name)
            reacted_boosts = dict(reacted_profile.boosts)
            old_stage = int(reacted_boosts.get(reaction_stat, 0))
            new_stage = min(6, old_stage + 2)
            if new_stage != old_stage:
                reacted_boosts[reaction_stat] = new_stage
                reacted_profile = replace(
                    reacted_profile,
                    boosts=reacted_boosts,
                )
                state.profiles[(target_side, target_name)] = reacted_profile
                _sync_speed_profile(
                    speed_states,
                    target_side,
                    target_name,
                    reacted_profile,
                )
                events.append(
                    SimulationEvent(
                        type=SimulationEventType.BOOST,
                        side=target_side,
                        actor=target_name,
                        move=ability,
                        target=target_name,
                        detail=(
                            f"{ability} raised {target_name}'s "
                            f"{reaction_stat} by {new_stage - old_stage}."
                        ),
                    )
                )

    return True


def _secondary_blocked(
    source_side: TurnSide,
    target_side: TurnSide,
    target: CombatantProfile,
) -> bool:
    if source_side is target_side:
        return False
    if normalize_move_id(target.item or "") == "covertcloak":
        return True
    return normalize_move_id(target.ability or "") == "shielddust"


def _secondary_probability(
    attacker: CombatantProfile,
    secondary: dict,
) -> float:
    probability = float(secondary.get("chance", 100)) / 100.0
    if normalize_move_id(attacker.ability or "") == "serenegrace":
        probability = min(1.0, probability * 2.0)
    return probability


def _apply_secondary_effects(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    flinched: set[tuple[TurnSide, str]],
    *,
    side: TurnSide,
    actor: str,
    move: Move,
    target_side: TurnSide,
    target_name: str,
    config: TurnSimulationConfig,
) -> None:
    attacker = state.profile(side, actor)
    target = state.profile(target_side, target_name)

    if target.current_hp <= 0 or _secondary_blocked(side, target_side, target):
        return

    for index, secondary in enumerate(move.secondary):
        probability = _secondary_probability(attacker, secondary)

        if probability < 1.0:
            if not config.branch_secondary_effects:
                continue
            trigger_key = (
                "secondary",
                side.value,
                actor,
                move.id,
                target_side.value,
                target_name,
                str(index),
            )
            triggered = _boolean_choice(config, trigger_key, probability)
            if not triggered:
                continue

        volatile_status = normalize_move_id(secondary.get("volatileStatus", ""))
        if volatile_status == "flinch":
            target = state.profile(target_side, target_name)
            if normalize_move_id(target.ability or "") != "innerfocus":
                flinched.add((target_side, target_name))
                events.append(
                    SimulationEvent(
                        type=SimulationEventType.FLINCH,
                        side=side,
                        actor=actor,
                        move=move.id,
                        target=target_name,
                        detail=f"{target_name} was flinched.",
                    )
                )

        status = secondary.get("status")
        if isinstance(status, str):
            _apply_status(
                state,
                speed_states,
                events,
                source_side=side,
                source_name=actor,
                target_side=target_side,
                target_name=target_name,
                move_id=move.id,
                status=status,
            )

        boosts = secondary.get("boosts")
        if isinstance(boosts, dict):
            _apply_boosts(
                state,
                speed_states,
                events,
                source_side=side,
                source_name=actor,
                target_side=target_side,
                target_name=target_name,
                move_id=move.id,
                boosts=boosts,
            )

        self_effect = secondary.get("self")
        if isinstance(self_effect, dict):
            self_boosts = self_effect.get("boosts")
            if isinstance(self_boosts, dict):
                _apply_boosts(
                    state,
                    speed_states,
                    events,
                    source_side=side,
                    source_name=actor,
                    target_side=side,
                    target_name=actor,
                    move_id=move.id,
                    boosts=self_boosts,
                )

        # Dire Claw's data uses a custom onHit callback: if its 50% secondary
        # triggers, poison/paralysis/sleep are selected uniformly.
        if move.id == "direclaw" and "onHit" in secondary:
            choice_key = (
                "dire_claw_status",
                side.value,
                actor,
                target_side.value,
                target_name,
            )
            if config.branch_secondary_effects:
                chosen_status = str(
                    _choice(
                        config,
                        choice_key,
                        (
                            ("psn", 1.0 / 3.0),
                            ("par", 1.0 / 3.0),
                            ("slp", 1.0 / 3.0),
                        ),
                    )
                )
            else:
                continue

            _apply_status(
                state,
                speed_states,
                events,
                source_side=side,
                source_name=actor,
                target_side=target_side,
                target_name=target_name,
                move_id=move.id,
                status=chosen_status,
            )


def _apply_damage(
    state: ExactTurnState,
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    events: list[SimulationEvent],
    flinched: set[tuple[TurnSide, str]],
    *,
    side: TurnSide,
    actor: str,
    move_profile: MoveProfile,
    target_side: TurnSide,
    target_name: str,
    protected: set[tuple[TurnSide, str]],
    wide_guard_sides: set[TurnSide],
    config: TurnSimulationConfig,
    gen: int,
) -> int:
    attacker = state.profile(side, actor)
    defender = state.profile(target_side, target_name)
    move_id = normalize_move_id(move_profile.move_id)
    move = Move(move_id, gen)

    priority = _effective_move_priority(
        state,
        side,
        actor,
        move,
    )
    if psychic_terrain_blocks_priority(
        defender,
        priority=priority,
        source_is_ally=side is target_side,
        terrain=state.terrain,
        field_conditions=state.field_conditions,
    ):
        events.append(
            SimulationEvent(
                type=SimulationEventType.PRIORITY_BLOCKED,
                side=side,
                actor=actor,
                move=move_id,
                target=target_name,
                detail="Blocked by Psychic Terrain.",
            )
        )
        return False

    if (
        target_side in wide_guard_sides
        and wide_guard_blocks(move)
        and not move.breaks_protect
    ):
        events.append(
            SimulationEvent(
                type=SimulationEventType.BLOCKED,
                side=side,
                actor=actor,
                move=move_id,
                target=target_name,
                detail="Blocked by Wide Guard.",
            )
        )
        return False

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
            return False

    if config.branch_accuracy:
        hit_key = (
            "accuracy",
            side.value,
            actor,
            move_id,
            target_side.value,
            target_name,
        )
        hit = _boolean_choice(
            config,
            hit_key,
            _accuracy_probability(attacker, defender, move),
        )
    else:
        hit = True

    if not hit:
        events.append(
            SimulationEvent(
                type=SimulationEventType.MISS,
                side=side,
                actor=actor,
                move=move_id,
                target=target_name,
                detail=f"{move_id} missed {target_name}.",
            )
        )
        return False

    if config.branch_critical_hits:
        crit_key = (
            "critical",
            side.value,
            actor,
            move_id,
            target_side.value,
            target_name,
        )
        critical = _boolean_choice(
            config,
            crit_key,
            _critical_probability(attacker, defender, move),
        )
    else:
        critical = False

    if critical:
        events.append(
            SimulationEvent(
                type=SimulationEventType.CRITICAL,
                side=side,
                actor=actor,
                move=move_id,
                target=target_name,
                detail=f"{move_id} critically hit {target_name}.",
            )
        )

    damage_result = calculate_matchup_damage(
        attacker,
        defender,
        move_profile,
        weather=state.weather,
        critical=critical,
    )
    damage = _damage_amount(
        config,
        damage_result.rolls,
        side=side,
        actor=actor,
        move_id=move_id,
        target_side=target_side,
        target_name=target_name,
    )

    if (
        normalize_move_id(defender.item or "") == "focussash"
        and defender.current_hp == defender.max_hp
        and damage >= defender.current_hp
    ):
        damage = max(0, defender.current_hp - 1)
        defender = replace(defender, item=None)
        events.append(
            SimulationEvent(
                type=SimulationEventType.ITEM,
                side=target_side,
                actor=target_name,
                move="focussash",
                target=target_name,
                detail=f"{target_name} survived with Focus Sash.",
            )
        )

    actual_damage = min(damage, defender.current_hp)
    new_hp = max(0, defender.current_hp - actual_damage)

    state.profiles[(target_side, target_name)] = replace(
        defender,
        current_hp=new_hp,
    )
    events.append(
        SimulationEvent(
            type=SimulationEventType.DAMAGE,
            side=side,
            actor=actor,
            move=move_id,
            target=target_name,
            damage=actual_damage,
            detail=f"{target_name}: {defender.current_hp} -> {new_hp} HP",
        )
    )

    if actual_damage > 0:
        current_defender = state.profile(target_side, target_name)
        move_type = (
            move.type.name.lower()
            if move.type is not None
            else ""
        )
        if (
            current_defender.current_hp > 0
            and normalize_move_id(current_defender.status or "") == "frz"
            and move_type == "fire"
            and move.category is not MoveCategory.STATUS
            and move.id != "polarflare"
        ):
            _clear_status(
                state,
                speed_states,
                events,
                side=target_side,
                name=target_name,
                reason=f"hit by {move.id}",
            )

        _apply_contact_punishment(
            state,
            events,
            attacker_side=side,
            attacker_name=actor,
            defender_side=target_side,
            defender_name=target_name,
            move=move,
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
        return actual_damage

    if actual_damage > 0:
        _apply_secondary_effects(
            state,
            speed_states,
            events,
            flinched,
            side=side,
            actor=actor,
            move=move,
            target_side=target_side,
            target_name=target_name,
            config=config,
        )

        current_defender = state.profile(target_side, target_name)
        if (
            current_defender.current_hp > 0
            and normalize_move_id(current_defender.status or "") == "frz"
            and bool(move.entry.get("thawsTarget", False))
        ):
            _clear_status(
                state,
                speed_states,
                events,
                side=target_side,
                name=target_name,
                reason=f"{move.id} thawed the target",
            )

    return actual_damage


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
    state.protect_streaks[(side, actor)] = 0
    state.protect_streaks[(side, target)] = 0
    state.toxic_stages[(side, actor)] = 0
    state.toxic_stages[(side, target)] = 0
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
    wide_guard_sides: set[TurnSide],
    redirections: dict[TurnSide, tuple[str, str]],
    flinched: set[tuple[TurnSide, str]],
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

    if not _sleep_or_freeze_allows_move(
        state,
        speed_states,
        events,
        side=side,
        actor=actor,
        move=move,
        config=config,
    ):
        state.protect_streaks[(side, actor)] = 0
        return

    flinch_key = (side, actor)
    if flinch_key in flinched:
        flinched.discard(flinch_key)
        state.protect_streaks[flinch_key] = 0
        events.append(
            SimulationEvent(
                type=SimulationEventType.FLINCHED,
                side=side,
                actor=actor,
                move=action.move,
                detail=f"{actor} flinched and could not move.",
            )
        )
        return

    if not _paralysis_allows_move(
        state,
        events,
        side=side,
        actor=actor,
        move=move,
        config=config,
    ):
        state.protect_streaks[(side, actor)] = 0
        return

    if move_id in STALL_MOVES:
        streak_key = (side, actor)
        previous_streak = state.protect_streaks.get(streak_key, 0)

        if streak_key in config.protect_success:
            success = config.protect_success[streak_key]
        elif config.branch_protect:
            random_key = (
                "protect",
                side.value,
                actor,
                move_id,
                str(previous_streak),
            )
            success = _boolean_choice(
                config,
                random_key,
                _protect_success_probability(previous_streak),
            )
        else:
            success = True

        if success:
            state.protect_streaks[streak_key] = min(previous_streak + 1, 6)
            if move_id == "wideguard":
                wide_guard_sides.add(side)
                event_type = SimulationEventType.WIDE_GUARD
                detail = "Wide Guard is active for this side."
            else:
                protected.add(streak_key)
                event_type = SimulationEventType.PROTECT
                detail = "Protect-like protection is active."
        else:
            state.protect_streaks[streak_key] = 0
            event_type = SimulationEventType.PROTECT_FAILED
            detail = "Protect-like move failed in this branch."

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

    # Using a non-stalling move resets the shared protection stall chain.
    state.protect_streaks[(side, actor)] = 0

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

    if move_id in {"followme", "ragepowder"}:
        redirections[side] = (actor, move_id)
        events.append(
            SimulationEvent(
                type=SimulationEventType.REDIRECT,
                side=side,
                actor=actor,
                move=move_id,
                detail=f"{move_id} is redirecting eligible attacks.",
            )
        )
        return

    if move.terrain is not None:
        state.terrain = move.terrain.name.lower()
        events.append(
            SimulationEvent(
                type=SimulationEventType.FIELD,
                side=side,
                actor=actor,
                move=move_id,
                detail=f"Terrain changed to {state.terrain}.",
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
        redirected, original_target = _redirect_target(
            state,
            side,
            actor,
            move,
            resolved,
            redirections,
        )
        if original_target is not None and redirected is not None:
            events.append(
                SimulationEvent(
                    type=SimulationEventType.REDIRECT,
                    side=side,
                    actor=actor,
                    move=move_id,
                    target=redirected[1],
                    detail=(
                        f"{move_id} was redirected from "
                        f"{original_target} to {redirected[1]}."
                    ),
                )
            )
        targets = [] if redirected is None else [redirected]

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

    # The 0.75 doubles spread modifier applies only while multiple targets exist.
    effective_move_profile = replace(
        move_profile,
        spread=move_profile.spread and len(targets) > 1,
    )

    total_damage_dealt = 0
    for target_side, target_name in targets:
        if state.profile(target_side, target_name).current_hp <= 0:
            continue

        total_damage_dealt += _apply_damage(
            state,
            speed_states,
            events,
            flinched,
            side=side,
            actor=actor,
            move_profile=effective_move_profile,
            target_side=target_side,
            target_name=target_name,
            protected=protected,
            wide_guard_sides=wide_guard_sides,
            config=config,
            gen=gen,
        )

    if (
        total_damage_dealt > 0
        and state.profile(side, actor).current_hp > 0
        and move.self_boost
    ):
        _apply_boosts(
            state,
            speed_states,
            events,
            source_side=side,
            source_name=actor,
            target_side=side,
            target_name=actor,
            move_id=move_id,
            boosts=move.self_boost,
        )

    _apply_move_recoil(
        state,
        events,
        side=side,
        actor=actor,
        move=move,
        total_damage_dealt=total_damage_dealt,
    )
    _apply_life_orb_recoil(
        state,
        events,
        side=side,
        actor=actor,
        move=move,
        total_damage_dealt=total_damage_dealt,
    )


def _sandstorm_immune(profile: CombatantProfile) -> bool:
    types = {
        normalize_move_id(type_name)
        for type_name in profile.types
    }
    if types.intersection({"rock", "ground", "steel"}):
        return True

    ability = normalize_move_id(profile.ability or "")
    if ability in {"sandforce", "sandrush", "sandveil", "overcoat"}:
        return True

    return normalize_move_id(profile.item or "") == "safetygoggles"


def _hail_immune(profile: CombatantProfile) -> bool:
    types = {
        normalize_move_id(type_name)
        for type_name in profile.types
    }
    if "ice" in types:
        return True

    ability = normalize_move_id(profile.ability or "")
    if ability == "overcoat":
        return True

    return normalize_move_id(profile.item or "") == "safetygoggles"


def _apply_end_of_turn_residuals(
    state: ExactTurnState,
    events: list[SimulationEvent],
) -> None:
    """Apply the supported Gen 9 end-of-turn residual sequence."""

    active = tuple(
        sorted(
            state.active_slots.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        )
    )

    weather = normalize_move_id(state.weather or "")
    if weather in {"sandstorm", "hail"}:
        for (side, _slot), name in active:
            profile = state.profile(side, name)
            if profile.current_hp <= 0:
                continue

            immune = (
                _sandstorm_immune(profile)
                if weather == "sandstorm"
                else _hail_immune(profile)
            )
            if immune:
                continue

            _apply_indirect_damage(
                state,
                events,
                side=side,
                name=name,
                amount=_max_hp_fraction(profile.max_hp, 1, 16),
                event_type=SimulationEventType.RESIDUAL,
                detail=f"{name} took {weather} damage.",
            )

    if normalize_move_id(state.terrain or "") == "grassyterrain":
        for (side, _slot), name in active:
            profile = state.profile(side, name)
            if (
                profile.current_hp <= 0
                or not is_grounded(
                    profile,
                    field_conditions=state.field_conditions,
                )
            ):
                continue

            _heal_profile(
                state,
                events,
                side=side,
                name=name,
                amount=_max_hp_fraction(profile.max_hp, 1, 16),
                detail=f"{name} recovered HP from Grassy Terrain.",
            )

    for (side, _slot), name in active:
        profile = state.profile(side, name)
        if (
            profile.current_hp > 0
            and normalize_move_id(profile.item or "") == "leftovers"
        ):
            _heal_profile(
                state,
                events,
                side=side,
                name=name,
                amount=_max_hp_fraction(profile.max_hp, 1, 16),
                detail=f"{name} recovered HP from Leftovers.",
            )

    for (side, _slot), name in active:
        profile = state.profile(side, name)
        if profile.current_hp <= 0:
            continue

        status = normalize_move_id(profile.status or "")
        ability = normalize_move_id(profile.ability or "")

        if status == "tox":
            toxic_key = (side, name)
            stage = min(state.toxic_stages.get(toxic_key, 0) + 1, 15)
            state.toxic_stages[toxic_key] = stage

            if ability == "poisonheal":
                _heal_profile(
                    state,
                    events,
                    side=side,
                    name=name,
                    amount=_max_hp_fraction(profile.max_hp, 1, 8),
                    detail=f"{name} recovered HP from Poison Heal.",
                )
            else:
                _apply_indirect_damage(
                    state,
                    events,
                    side=side,
                    name=name,
                    amount=_max_hp_fraction(profile.max_hp, 1, 16) * stage,
                    event_type=SimulationEventType.RESIDUAL,
                    detail=f"{name} took toxic poison damage.",
                )

        elif status == "psn":
            if ability == "poisonheal":
                _heal_profile(
                    state,
                    events,
                    side=side,
                    name=name,
                    amount=_max_hp_fraction(profile.max_hp, 1, 8),
                    detail=f"{name} recovered HP from Poison Heal.",
                )
            else:
                _apply_indirect_damage(
                    state,
                    events,
                    side=side,
                    name=name,
                    amount=_max_hp_fraction(profile.max_hp, 1, 8),
                    event_type=SimulationEventType.RESIDUAL,
                    detail=f"{name} took poison damage.",
                )

    for (side, _slot), name in active:
        profile = state.profile(side, name)
        if (
            profile.current_hp > 0
            and normalize_move_id(profile.status or "") == "brn"
        ):
            _apply_indirect_damage(
                state,
                events,
                side=side,
                name=name,
                amount=_max_hp_fraction(profile.max_hp, 1, 16),
                event_type=SimulationEventType.RESIDUAL,
                detail=f"{name} took burn damage.",
            )


def _choose_speed_tie(
    config: TurnSimulationConfig,
    tied_actions,
):
    actors = tuple(
        sorted(
            (
                f"{scheduled.side.value}:{scheduled.slot}:{scheduled.actor}",
                scheduled,
            )
            for scheduled in tied_actions
        )
    )

    if not config.branch_speed_ties:
        names = ", ".join(scheduled.actor for _token, scheduled in actors)
        raise UnresolvedSpeedTie(
            f"deterministic simulator cannot choose Speed tie between {names}"
        )

    key: RandomChoiceKey = (
        "speed_tie",
        *(token for token, _scheduled in actors),
    )
    probability = 1.0 / len(actors)
    selected_token = str(
        _choice(
            config,
            key,
            tuple((token, probability) for token, _scheduled in actors),
        )
    )
    return next(
        scheduled
        for token, scheduled in actors
        if token == selected_token
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
    """Execute one fixed branch of a doubles turn."""

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
    wide_guard_sides: set[TurnSide] = set()
    redirections: dict[TurnSide, tuple[str, str]] = {}
    flinched: set[tuple[TurnSide, str]] = set()
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
            scheduled = _choose_speed_tie(config, next_group.actions)
        else:
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
                wide_guard_sides,
                redirections,
                flinched,
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
        # Rebuilding the queue on each loop iteration reproduces that behavior.

    _apply_end_of_turn_residuals(state, events)

    return TurnSimulationResult(
        state=state,
        events=tuple(events),
    )

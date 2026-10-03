"""Transparent v0 heuristic scorer for Champions AI.

This is scaffolding for ranking legal joint actions before the exact damage,
opponent-model, and simulation layers are complete. It deliberately does NOT
claim to output a true match win probability yet.
"""

from dataclasses import dataclass

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot, PokemonSnapshot
from vgc_bench.src.champions_ai.spread import is_spread_move, normalize_move_id


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


@dataclass(frozen=True)
class ActionSignals:
    """Human-readable features used by the v0 heuristic scorer."""

    revealed_spread_moves: tuple[str, ...]
    protected_slots: int
    offensive_slots: int
    switch_slots: int
    passive_slots: int
    repeated_protect_slots: int
    own_spread_attacks: int


@dataclass(frozen=True)
class ActionEvaluation:
    """Evaluation for one legal joint action.

    estimated_win_probability stays None until the statistical evaluator exists.
    """

    action: JointAction
    heuristic_score: float
    signals: ActionSignals
    reasons: tuple[str, ...]
    estimated_win_probability: float | None = None
    model: str = "heuristic_v0"


def _active_pokemon(side, slot: int) -> PokemonSnapshot | None:
    slot_map = dict(side.active_slots)
    name = slot_map.get(slot)
    if name is None:
        return None
    return next((pokemon for pokemon in side.pokemon if pokemon.name == name), None)


def _is_protect_like(action: SlotAction) -> bool:
    return (
        action.kind is ActionKind.MOVE
        and action.move is not None
        and normalize_move_id(action.move) in PROTECT_LIKE_MOVES
    )


def _targets_opponent(action: SlotAction, snapshot: DecisionSnapshot) -> bool:
    if action.kind is not ActionKind.MOVE or action.move is None:
        return False

    if is_spread_move(action.move):
        return True

    opponent_names = set(dict(snapshot.opponent.active_slots).values())
    return action.target_position in (1, 2) or action.target in opponent_names


def _revealed_spread_moves(snapshot: DecisionSnapshot) -> tuple[str, ...]:
    active_names = set(dict(snapshot.opponent.active_slots).values())
    spread: set[str] = set()

    for pokemon in snapshot.opponent.pokemon:
        if pokemon.name not in active_names:
            continue
        for move in pokemon.revealed_moves:
            if is_spread_move(move):
                spread.add(normalize_move_id(move))

    return tuple(sorted(spread))


def _signals(snapshot: DecisionSnapshot, action: JointAction) -> ActionSignals:
    slot_actions = (action.first, action.second)
    spread_moves = _revealed_spread_moves(snapshot)

    protected_slots = sum(_is_protect_like(slot_action) for slot_action in slot_actions)
    offensive_slots = sum(
        _targets_opponent(slot_action, snapshot) for slot_action in slot_actions
    )
    switch_slots = sum(
        slot_action.kind is ActionKind.SWITCH for slot_action in slot_actions
    )
    passive_slots = sum(
        slot_action.kind in {ActionKind.PASS, ActionKind.DEFAULT}
        for slot_action in slot_actions
    )
    own_spread_attacks = sum(
        slot_action.kind is ActionKind.MOVE
        and slot_action.move is not None
        and is_spread_move(slot_action.move)
        for slot_action in slot_actions
    )

    repeated_protect_slots = 0
    for slot_action in slot_actions:
        if not _is_protect_like(slot_action):
            continue
        pokemon = _active_pokemon(snapshot.player, slot_action.slot)
        if pokemon is not None and pokemon.protect_streak > 0:
            repeated_protect_slots += 1

    return ActionSignals(
        revealed_spread_moves=spread_moves,
        protected_slots=protected_slots,
        offensive_slots=offensive_slots,
        switch_slots=switch_slots,
        passive_slots=passive_slots,
        repeated_protect_slots=repeated_protect_slots,
        own_spread_attacks=own_spread_attacks,
    )


def evaluate_action(
    snapshot: DecisionSnapshot,
    action: JointAction,
) -> ActionEvaluation:
    """Score one action with transparent, intentionally simple v0 heuristics."""

    signals = _signals(snapshot, action)
    score = 0.0
    reasons: list[str] = []

    if signals.offensive_slots:
        score += 6.0 * signals.offensive_slots
        reasons.append(
            f"{signals.offensive_slots} slot(s) apply immediate offensive pressure."
        )

    if signals.own_spread_attacks:
        score += 2.0 * signals.own_spread_attacks
        reasons.append(
            f"{signals.own_spread_attacks} selected move(s) apply spread pressure."
        )

    if signals.passive_slots:
        score -= 25.0 * signals.passive_slots
        reasons.append(
            f"{signals.passive_slots} pass/default action(s) give up immediate pressure."
        )

    if signals.revealed_spread_moves:
        threat_names = ", ".join(signals.revealed_spread_moves)
        reasons.append(f"Revealed opponent spread threat(s): {threat_names}.")

        if signals.protected_slots == 0:
            score -= 12.0
            reasons.append("Both active slots remain exposed to revealed spread pressure.")
        elif signals.protected_slots == 1:
            score += 6.0
            reasons.append(
                "One slot protects while the partner can preserve pressure or reposition."
            )
        else:
            score += 2.0
            reasons.append(
                "Both slots protect from spread pressure, but the line gives up pressure."
            )
    elif signals.protected_slots:
        score -= 1.0 * signals.protected_slots
        reasons.append(
            "Protect is not currently supported by a revealed spread threat in this "
            "baseline model."
        )

    if signals.protected_slots == 2:
        score -= 3.0

    if signals.switch_slots:
        reasons.append(
            f"{signals.switch_slots} switch(es) change positioning; matchup value will "
            "be added by later damage/type modules."
        )

    if signals.repeated_protect_slots:
        score -= 8.0 * signals.repeated_protect_slots
        reasons.append(
            f"{signals.repeated_protect_slots} repeated Protect attempt(s) carry "
            "escalating failure risk."
        )

    return ActionEvaluation(
        action=action,
        heuristic_score=score,
        signals=signals,
        reasons=tuple(reasons),
    )


def rank_actions(
    snapshot: DecisionSnapshot,
    actions: list[JointAction],
) -> list[ActionEvaluation]:
    """Return all candidate actions ranked by the current baseline score."""

    evaluations = [evaluate_action(snapshot, action) for action in actions]
    return sorted(
        evaluations,
        key=lambda evaluation: (
            -evaluation.heuristic_score,
            evaluation.action.label,
        ),
    )

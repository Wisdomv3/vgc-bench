"""Legal joint-action generation from poke-env Showdown battles."""

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    Gimmick,
    JointAction,
    SlotAction,
)


def _pokemon_label(pokemon: Pokemon | None) -> str | None:
    if pokemon is None:
        return None
    return pokemon.species or pokemon.name


def _gimmick(order: SingleBattleOrder) -> Gimmick:
    if order.mega:
        return Gimmick.MEGA
    if order.z_move:
        return Gimmick.Z_MOVE
    if order.dynamax:
        return Gimmick.DYNAMAX
    if order.terastallize:
        return Gimmick.TERA
    return Gimmick.NONE


def _resolve_target_name(battle, position: int) -> str | None:
    if position == -1:
        return _pokemon_label(battle.active_pokemon[0])
    if position == -2:
        return _pokemon_label(battle.active_pokemon[1])
    if position == 1:
        return _pokemon_label(battle.opponent_active_pokemon[0])
    if position == 2:
        return _pokemon_label(battle.opponent_active_pokemon[1])
    return None


def slot_action_from_showdown(
    battle,
    slot: int,
    order: SingleBattleOrder,
) -> SlotAction:
    """Convert one poke-env order into an input-independent SlotAction."""

    actor = _pokemon_label(battle.active_pokemon[slot])

    if isinstance(order, DefaultBattleOrder):
        return SlotAction(slot=slot, kind=ActionKind.DEFAULT, actor=actor)

    if isinstance(order, PassBattleOrder):
        return SlotAction(slot=slot, kind=ActionKind.PASS, actor=actor)

    if isinstance(order.order, Pokemon):
        return SlotAction(
            slot=slot,
            kind=ActionKind.SWITCH,
            actor=actor,
            switch_to=_pokemon_label(order.order),
        )

    if isinstance(order.order, Move):
        return SlotAction(
            slot=slot,
            kind=ActionKind.MOVE,
            actor=actor,
            move=order.order.id,
            target=_resolve_target_name(battle, order.move_target),
            target_position=order.move_target,
            gimmick=_gimmick(order),
        )

    raise TypeError(f"Unsupported Showdown order: {order}")


def legal_joint_actions_from_showdown(battle) -> list[JointAction]:
    """Enumerate every legal compatible doubles action for the current request.

    poke-env already knows the per-slot legal orders. DoubleBattleOrder.join_orders
    applies cross-slot legality rules such as preventing both slots from switching
    to the same Pokemon or using the same one-per-turn gimmick twice.
    """

    first_orders, second_orders = battle.valid_orders
    joined = DoubleBattleOrder.join_orders(first_orders, second_orders)

    return [
        JointAction(
            first=slot_action_from_showdown(battle, 0, order.first_order),
            second=slot_action_from_showdown(battle, 1, order.second_order),
        )
        for order in joined
    ]

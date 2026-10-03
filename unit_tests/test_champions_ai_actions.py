from types import SimpleNamespace

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from vgc_bench.src.champions_ai.actions import ActionKind, Gimmick
from vgc_bench.src.champions_ai.inputs.showdown_actions import (
    legal_joint_actions_from_showdown,
)


def _battle(first_orders, second_orders):
    garchomp = Pokemon(gen=9, species="garchomp")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    salamence = Pokemon(gen=9, species="salamence")
    sneasler = Pokemon(gen=9, species="sneasler")

    return SimpleNamespace(
        active_pokemon=[garchomp, whimsicott],
        opponent_active_pokemon=[salamence, sneasler],
        valid_orders=[first_orders, second_orders],
    )


def test_enumerates_cartesian_product_of_compatible_actions() -> None:
    protect = SingleBattleOrder(Move("protect", 9))
    dragon_claw = SingleBattleOrder(Move("dragonclaw", 9), move_target=1)
    tailwind = SingleBattleOrder(Move("tailwind", 9))
    moonblast = SingleBattleOrder(Move("moonblast", 9), move_target=1)

    battle = _battle(
        [protect, dragon_claw],
        [tailwind, moonblast],
    )

    actions = legal_joint_actions_from_showdown(battle)

    assert len(actions) == 4


def test_target_name_is_resolved_for_explanations() -> None:
    dragon_claw = SingleBattleOrder(Move("dragonclaw", 9), move_target=1)
    moonblast = SingleBattleOrder(Move("moonblast", 9), move_target=2)

    battle = _battle([dragon_claw], [moonblast])
    action = legal_joint_actions_from_showdown(battle)[0]

    assert action.first.target == "salamence"
    assert action.second.target == "sneasler"
    assert "dragonclaw -> salamence" in action.label


def test_duplicate_switch_to_same_pokemon_is_removed() -> None:
    kingambit = Pokemon(gen=9, species="kingambit")
    first_switch = SingleBattleOrder(kingambit)
    second_switch = SingleBattleOrder(kingambit)

    battle = _battle([first_switch], [second_switch])

    assert legal_joint_actions_from_showdown(battle) == []


def test_two_mega_actions_cannot_be_selected_together() -> None:
    first_mega = SingleBattleOrder(Move("dragonclaw", 9), move_target=1, mega=True)
    second_mega = SingleBattleOrder(Move("moonblast", 9), move_target=1, mega=True)

    battle = _battle([first_mega], [second_mega])

    assert legal_joint_actions_from_showdown(battle) == []


def test_move_switch_and_gimmick_metadata_are_preserved() -> None:
    kingambit = Pokemon(gen=9, species="kingambit")
    mega_move = SingleBattleOrder(
        Move("dragonclaw", 9),
        move_target=1,
        mega=True,
    )
    switch = SingleBattleOrder(kingambit)

    battle = _battle([mega_move], [switch])
    action = legal_joint_actions_from_showdown(battle)[0]

    assert action.first.kind is ActionKind.MOVE
    assert action.first.gimmick is Gimmick.MEGA
    assert action.second.kind is ActionKind.SWITCH
    assert action.second.switch_to == "kingambit"

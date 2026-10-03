from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_order import (
    TurnSide,
    build_turn_order,
)


def _move(slot: int, actor: str, move: str, target: str | None = None):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target=target,
    )


def _switch(slot: int, actor: str, switch_to: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.SWITCH,
        actor=actor,
        switch_to=switch_to,
    )


def _joint(first: SlotAction, second: SlotAction):
    return JointAction(first=first, second=second)


def _speeds(**values: int):
    result = {}
    for key, speed in values.items():
        side_name, actor = key.split("__", 1)
        result[(TurnSide(side_name), actor)] = SpeedState(speed_stat=speed)
    return result


def test_switches_happen_before_moves_even_when_move_has_priority() -> None:
    ours = _joint(
        _switch(0, "garchomp", "kingambit"),
        _move(1, "whimsicott", "Protect"),
    )
    theirs = _joint(
        _move(0, "salamence", "Extreme Speed", "garchomp"),
        _move(1, "sneasler", "Fake Out", "whimsicott"),
    )

    groups = build_turn_order(
        ours,
        theirs,
        _speeds(
            player__garchomp=120,
            player__whimsicott=180,
            opponent__salamence=200,
            opponent__sneasler=150,
        ),
    )

    assert groups[0].actions[0].action.kind is ActionKind.SWITCH
    assert groups[0].actions[0].actor == "garchomp"


def test_move_priority_beats_speed() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dragon Claw", "salamence"),
        _move(1, "whimsicott", "Protect"),
    )
    theirs = _joint(
        _move(0, "salamence", "Extreme Speed", "garchomp"),
        _move(1, "sneasler", "Fake Out", "whimsicott"),
    )

    groups = build_turn_order(
        ours,
        theirs,
        _speeds(
            player__garchomp=300,
            player__whimsicott=80,
            opponent__salamence=100,
            opponent__sneasler=90,
        ),
    )

    ordered_moves = [
        group.actions[0].action.move
        for group in groups
        if len(group.actions) == 1
    ]

    assert ordered_moves[:3] == ["Protect", "Fake Out", "Extreme Speed"]


def test_speed_orders_actions_inside_same_priority_bracket() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dragon Claw", "salamence"),
        _move(1, "whimsicott", "Moonblast", "sneasler"),
    )
    theirs = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "whimsicott"),
    )

    groups = build_turn_order(
        ours,
        theirs,
        _speeds(
            player__garchomp=102,
            player__whimsicott=184,
            opponent__salamence=167,
            opponent__sneasler=189,
        ),
    )

    assert [group.actions[0].actor for group in groups] == [
        "sneasler",
        "whimsicott",
        "salamence",
        "garchomp",
    ]


def test_trick_room_reverses_speed_inside_same_priority_bracket() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dragon Claw", "salamence"),
        _move(1, "whimsicott", "Moonblast", "sneasler"),
    )
    theirs = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "whimsicott"),
    )

    groups = build_turn_order(
        ours,
        theirs,
        _speeds(
            player__garchomp=102,
            player__whimsicott=184,
            opponent__salamence=167,
            opponent__sneasler=189,
        ),
        trick_room=True,
    )

    assert [group.actions[0].actor for group in groups] == [
        "garchomp",
        "salamence",
        "whimsicott",
        "sneasler",
    ]


def test_exact_speed_tie_is_preserved_as_a_group() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dragon Claw", "salamence"),
        _move(1, "whimsicott", "Moonblast", "sneasler"),
    )
    theirs = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "whimsicott"),
    )

    groups = build_turn_order(
        ours,
        theirs,
        _speeds(
            player__garchomp=150,
            player__whimsicott=180,
            opponent__salamence=150,
            opponent__sneasler=200,
        ),
    )

    tie_group = next(group for group in groups if group.is_speed_tie)

    assert {action.actor for action in tie_group.actions} == {
        "garchomp",
        "salamence",
    }


def test_tailwind_is_reflected_through_speed_state() -> None:
    ours = _joint(
        _move(0, "garchomp", "Dragon Claw", "salamence"),
        _move(1, "whimsicott", "Moonblast", "sneasler"),
    )
    theirs = _joint(
        _move(0, "salamence", "Draco Meteor", "garchomp"),
        _move(1, "sneasler", "Close Combat", "whimsicott"),
    )

    speed_states = _speeds(
        player__garchomp=100,
        player__whimsicott=100,
        opponent__salamence=150,
        opponent__sneasler=140,
    )
    speed_states[(TurnSide.PLAYER, "garchomp")] = SpeedState(
        speed_stat=100,
        tailwind=True,
    )

    groups = build_turn_order(ours, theirs, speed_states)

    assert groups[0].actions[0].actor == "garchomp"

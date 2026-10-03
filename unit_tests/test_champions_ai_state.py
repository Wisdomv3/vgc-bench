from vgc_bench.src.champions_ai.events import BattleEvent
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.state import BattleState


def test_event_accepts_string_names_for_manual_input() -> None:
    event = BattleEvent(
        type="move_used",
        side="opponent",
        pokemon="Salamence",
        move="Hyper Voice",
    )
    assert event.type.value == "move_used"
    assert event.side is not None
    assert event.side.value == "opponent"


def test_move_used_is_recorded() -> None:
    state = BattleState()
    state.apply(
        BattleEvent(
            type="move_used",
            side="opponent",
            pokemon="Salamence",
            move="Hyper Voice",
        )
    )
    assert "Hyper Voice" in state.opponent.pokemon["Salamence"].revealed_moves


def test_switch_updates_active_slot() -> None:
    state = BattleState()
    state.apply(
        BattleEvent(
            type="switch_in",
            side="player",
            slot=0,
            pokemon="Garchomp",
        )
    )
    state.apply(
        BattleEvent(
            type="switch_in",
            side="player",
            slot=0,
            pokemon="Kingambit",
        )
    )

    assert state.player.active_slots[0] == "Kingambit"
    assert state.player.pokemon["Garchomp"].active_slot is None
    assert state.player.pokemon["Kingambit"].active_slot == 0


def test_hp_and_field_conditions_are_tracked() -> None:
    state = BattleState()
    state.apply(
        BattleEvent(
            type="hp_updated",
            side="player",
            pokemon="Garchomp",
            hp_percent=71,
        )
    )
    state.apply(BattleEvent(type="tailwind_changed", side="player", value=3))
    state.apply(BattleEvent(type="trick_room_changed", value=4))
    state.apply(BattleEvent(type="terrain_changed", value="Psychic Terrain"))

    assert state.player.pokemon["Garchomp"].hp_percent == 71
    assert state.player.tailwind_turns == 3
    assert state.trick_room_turns == 4
    assert state.terrain == "Psychic Terrain"


def test_obs_events_can_carry_confidence() -> None:
    event = BattleEvent(
        type="hp_updated",
        side="opponent",
        pokemon="Salamence",
        hp_percent=63,
        source="obs",
        confidence=0.84,
    )
    assert event.source == "obs"
    assert event.confidence == 0.84


def test_snapshot_does_not_learn_future_information() -> None:
    state = BattleState()
    state.apply(BattleEvent(type="turn_started", turn=6))
    state.apply(
        BattleEvent(
            type="move_used",
            side="opponent",
            pokemon="Salamence",
            move="Hyper Voice",
        )
    )

    snapshot = DecisionSnapshot.from_state(state)

    state.apply(
        BattleEvent(
            type="item_revealed",
            side="opponent",
            pokemon="Salamence",
            item="Choice Scarf",
        )
    )

    snap_salamence = next(
        pokemon
        for pokemon in snapshot.opponent.pokemon
        if pokemon.name == "Salamence"
    )
    assert snap_salamence.item is None
    assert state.opponent.pokemon["Salamence"].item == "Choice Scarf"


def test_same_events_give_same_state_regardless_of_input_source() -> None:
    manual = BattleState()
    showdown = BattleState()

    for state, source in [(manual, "manual"), (showdown, "showdown")]:
        state.apply(
            BattleEvent(
                type="switch_in",
                side="opponent",
                slot=0,
                pokemon="Salamence",
                source=source,
            )
        )
        state.apply(
            BattleEvent(
                type="hp_updated",
                side="opponent",
                pokemon="Salamence",
                hp_percent=80,
                source=source,
            )
        )

    manual_snapshot = DecisionSnapshot.from_state(manual)
    showdown_snapshot = DecisionSnapshot.from_state(showdown)

    assert manual_snapshot.opponent == showdown_snapshot.opponent


def test_first_turn_eligibility_is_tracked_in_snapshot() -> None:
    state = BattleState()
    state.apply(
        BattleEvent(
            type="switch_in",
            side="player",
            slot=0,
            pokemon="Rillaboom",
        )
    )
    state.apply(
        BattleEvent(
            type="first_turn_changed",
            side="player",
            pokemon="Rillaboom",
            value=False,
        )
    )

    snapshot = DecisionSnapshot.from_state(state)
    rillaboom = next(
        pokemon
        for pokemon in snapshot.player.pokemon
        if pokemon.name == "Rillaboom"
    )

    assert rillaboom.first_turn is False

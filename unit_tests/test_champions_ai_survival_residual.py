from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    SimulationEventType,
    TurnSimulationConfig,
    simulate_turn,
)


BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=85,
    category="physical",
    move_type="normal",
)

LETHAL_BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=500,
    category="physical",
    move_type="normal",
)

BRAVE_BIRD = MoveProfile(
    move_id="bravebird",
    base_power=120,
    category="physical",
    move_type="flying",
)


def _profile(
    name: str,
    *,
    hp: int = 160,
    item: str | None = None,
    ability: str | None = None,
    status: str | None = None,
    types: tuple[str, ...] = ("normal",),
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=160,
        types=types,
        stats={
            "atk": 150,
            "def": 100,
            "spa": 120,
            "spd": 100,
            "spe": 100,
        },
        item=item,
        ability=ability,
        status=status,
    )


def _state(
    *,
    attacker_item: str | None = None,
    attacker_ability: str | None = None,
    attacker_status: str | None = None,
    attacker_hp: int = 160,
    target_item: str | None = None,
    target_ability: str | None = None,
    target_types: tuple[str, ...] = ("normal",),
    weather: str | None = None,
    terrain: str | None = None,
):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "attacker"): _profile(
                "attacker",
                hp=attacker_hp,
                item=attacker_item,
                ability=attacker_ability,
                status=attacker_status,
            ),
            (TurnSide.PLAYER, "partner"): _profile("partner"),
            (TurnSide.OPPONENT, "target"): _profile(
                "target",
                item=target_item,
                ability=target_ability,
                types=target_types,
            ),
            (TurnSide.OPPONENT, "partner2"): _profile("partner2"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "attacker",
            (TurnSide.PLAYER, 1): "partner",
            (TurnSide.OPPONENT, 0): "target",
            (TurnSide.OPPONENT, 1): "partner2",
        },
        weather=weather,
        terrain=terrain,
    )


def _move(slot: int, actor: str, move: str, target_position: int | None = None):
    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target_position=target_position,
    )


def _pass(slot: int, actor: str):
    return SlotAction(
        slot=slot,
        kind=ActionKind.PASS,
        actor=actor,
    )


def _joint(first, second):
    return JointAction(first=first, second=second)


def _ours(move: str | None = None):
    first = (
        _move(0, "attacker", move, 1)
        if move is not None
        else _pass(0, "attacker")
    )
    return _joint(first, _pass(1, "partner"))


def _theirs():
    return _joint(
        _pass(0, "target"),
        _pass(1, "partner2"),
    )


def _speeds():
    return {
        (TurnSide.PLAYER, "attacker"): SpeedState(200),
        (TurnSide.PLAYER, "partner"): SpeedState(90),
        (TurnSide.OPPONENT, "target"): SpeedState(100),
        (TurnSide.OPPONENT, "partner2"): SpeedState(80),
    }


def _config():
    return TurnSimulationConfig(
        damage_roll_index=7,
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_protect=False,
        branch_speed_ties=False,
    )


def _moves(profile: MoveProfile):
    return {
        (TurnSide.PLAYER, "attacker", profile.move_id): profile,
    }


def test_brave_bird_recoil_uses_actual_damage_dealt() -> None:
    result = simulate_turn(
        _state(),
        _ours("Brave Bird"),
        _theirs(),
        _speeds(),
        _moves(BRAVE_BIRD),
        _config(),
    )

    damage = next(
        event.damage
        for event in result.events
        if event.type is SimulationEventType.DAMAGE
        and event.target == "target"
    )
    recoil = next(
        event.damage
        for event in result.events
        if event.type is SimulationEventType.RECOIL
        and "bravebird" in event.detail
    )

    assert damage is not None
    assert recoil == max(1, (damage * 33 * 2 + 100) // 200)


def test_rock_head_blocks_move_recoil() -> None:
    result = simulate_turn(
        _state(attacker_ability="rockhead"),
        _ours("Brave Bird"),
        _theirs(),
        _speeds(),
        _moves(BRAVE_BIRD),
        _config(),
    )

    assert not any(
        event.type is SimulationEventType.RECOIL
        and "bravebird" in event.detail
        for event in result.events
    )


def test_rough_skin_and_rocky_helmet_stack_on_contact() -> None:
    result = simulate_turn(
        _state(
            target_ability="roughskin",
            target_item="rockyhelmet",
        ),
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    attacker = result.state.profile(TurnSide.PLAYER, "attacker")
    assert attacker.current_hp == 160 - 20 - 26

    contact_events = [
        event
        for event in result.events
        if event.type is SimulationEventType.CONTACT_DAMAGE
    ]
    assert sorted(event.damage for event in contact_events) == [20, 26]


def test_protective_pads_block_contact_punishment() -> None:
    result = simulate_turn(
        _state(
            attacker_item="protectivepads",
            target_ability="roughskin",
            target_item="rockyhelmet",
        ),
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER, "attacker"
    ).current_hp == 160


def test_life_orb_recoil_is_one_tenth_max_hp() -> None:
    result = simulate_turn(
        _state(attacker_item="lifeorb"),
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    life_orb = next(
        event
        for event in result.events
        if event.type is SimulationEventType.RECOIL
        and "Life Orb" in event.detail
    )
    assert life_orb.damage == 16


def test_magic_guard_blocks_life_orb_recoil() -> None:
    result = simulate_turn(
        _state(
            attacker_item="lifeorb",
            attacker_ability="magicguard",
        ),
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(BODY_SLAM),
        _config(),
    )

    assert not any(
        event.type is SimulationEventType.RECOIL
        for event in result.events
    )


def test_focus_sash_survives_lethal_move_and_is_consumed() -> None:
    result = simulate_turn(
        _state(target_item="focussash"),
        _ours("Body Slam"),
        _theirs(),
        _speeds(),
        _moves(LETHAL_BODY_SLAM),
        _config(),
    )

    target = result.state.profile(TurnSide.OPPONENT, "target")
    assert target.current_hp == 1
    assert target.item is None
    assert any(
        event.type is SimulationEventType.ITEM
        and event.move == "focussash"
        for event in result.events
    )


def test_leftovers_heal_before_burn_damage() -> None:
    state = _state(
        attacker_item="leftovers",
        attacker_status="brn",
        attacker_hp=140,
    )

    result = simulate_turn(
        state,
        _ours(),
        _theirs(),
        _speeds(),
        {},
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER, "attacker"
    ).current_hp == 140

    relevant = [
        event.type
        for event in result.events
        if event.actor == "attacker"
        and event.type in {
            SimulationEventType.HEAL,
            SimulationEventType.RESIDUAL,
        }
    ]
    assert relevant == [
        SimulationEventType.HEAL,
        SimulationEventType.RESIDUAL,
    ]


def test_toxic_damage_escalates_each_turn() -> None:
    state = _state(attacker_status="tox")

    first = simulate_turn(
        state,
        _ours(),
        _theirs(),
        _speeds(),
        {},
        _config(),
    )
    second = simulate_turn(
        first.state,
        _ours(),
        _theirs(),
        _speeds(),
        {},
        _config(),
    )

    assert first.state.toxic_stages[(TurnSide.PLAYER, "attacker")] == 1
    assert second.state.toxic_stages[(TurnSide.PLAYER, "attacker")] == 2
    assert first.state.profile(
        TurnSide.PLAYER, "attacker"
    ).current_hp == 150
    assert second.state.profile(
        TurnSide.PLAYER, "attacker"
    ).current_hp == 130


def test_sandstorm_chips_normal_type_but_not_rock_type() -> None:
    state = _state(
        weather="sandstorm",
        target_types=("rock",),
    )

    result = simulate_turn(
        state,
        _ours(),
        _theirs(),
        _speeds(),
        {},
        _config(),
    )

    assert result.state.profile(
        TurnSide.PLAYER, "attacker"
    ).current_hp == 150
    assert result.state.profile(
        TurnSide.OPPONENT, "target"
    ).current_hp == 160

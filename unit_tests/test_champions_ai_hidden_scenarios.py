import pytest

from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.hidden_scenarios import (
    hidden_state_scenarios,
)
from vgc_bench.src.champions_ai.hidden_sets import SetHypothesis
from vgc_bench.src.champions_ai.matchup import (
    CombatantProfile,
    MoveProfile,
)
from vgc_bench.src.champions_ai.mechanics_evaluator import (
    evaluate_action_pair_scenarios,
)
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
    SideSnapshot,
)
from vgc_bench.src.champions_ai.speed_context import (
    speed_states_from_exact_state,
)
from vgc_bench.src.champions_ai.turn_branching import BranchingPolicy
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


BODY_SLAM = MoveProfile(
    move_id="bodyslam",
    base_power=200,
    category="physical",
    move_type="normal",
)


def _profile(
    name: str,
    *,
    hp: int = 100,
    max_hp: int = 100,
    attack: int = 200,
    defense: int = 100,
    speed: int = 80,
    item: str | None = None,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=max_hp,
        types=("normal",),
        stats={
            "atk": attack,
            "def": defense,
            "spa": 100,
            "spd": 100,
            "spe": speed,
        },
        item=item,
        ability=ability,
    )


def _pokemon_snapshot(
    name: str,
    *,
    hp_percent: float = 100.0,
    active_slot: int | None = None,
    moves=("bodyslam",),
    item: str | None = None,
    ability: str | None = None,
):
    return PokemonSnapshot(
        name=name,
        hp_percent=hp_percent,
        status=None,
        fainted=hp_percent <= 0,
        active_slot=active_slot,
        revealed_moves=tuple(moves),
        item=item,
        ability=ability,
        stat_stages=(),
        protect_streak=0,
        first_turn=True,
    )


def _side(pokemon, active_slots):
    return SideSnapshot(
        pokemon=tuple(pokemon),
        active_slots=tuple(active_slots),
        tailwind_turns=0,
        side_conditions=(),
    )


def _snapshot(
    *,
    foe_hp_percent: float = 100.0,
    foe_moves=("bodyslam",),
):
    return DecisionSnapshot(
        turn=1,
        player=_side(
            (
                _pokemon_snapshot(
                    "hero",
                    active_slot=0,
                ),
                _pokemon_snapshot(
                    "ally",
                    active_slot=1,
                ),
            ),
            ((0, "hero"), (1, "ally")),
        ),
        opponent=_side(
            (
                _pokemon_snapshot(
                    "foe",
                    hp_percent=foe_hp_percent,
                    active_slot=0,
                    moves=foe_moves,
                ),
                _pokemon_snapshot(
                    "foe2",
                    active_slot=1,
                    moves=("protect",),
                ),
            ),
            ((0, "foe"), (1, "foe2")),
        ),
        weather=None,
        terrain=None,
        trick_room_turns=0,
        field_conditions=(),
        history_size=0,
    )


def _state():
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "hero"): _profile(
                "hero",
                speed=100,
            ),
            (TurnSide.PLAYER, "ally"): _profile(
                "ally",
                attack=100,
                speed=60,
            ),
            (TurnSide.OPPONENT, "foe"): _profile(
                "foe",
                speed=80,
            ),
            (TurnSide.OPPONENT, "foe2"): _profile(
                "foe2",
                attack=100,
                speed=50,
            ),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "hero",
            (TurnSide.PLAYER, 1): "ally",
            (TurnSide.OPPONENT, 0): "foe",
            (TurnSide.OPPONENT, 1): "foe2",
        },
    )


def _move(
    slot: int,
    actor: str,
    move: str,
    target_position: int | None = None,
):
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


def _ours():
    return _joint(
        _move(0, "hero", "Body Slam", 1),
        _pass(1, "ally"),
    )


def _theirs(move: str = "Body Slam"):
    return _joint(
        _move(0, "foe", move, 1),
        _pass(1, "foe2"),
    )


def _policy():
    return BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=False,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=15,
        merge_equivalent_states=False,
    )


def test_hidden_scenarios_preserve_posterior_weights_and_exact_stats() -> None:
    state = _state()
    hypotheses = (
        SetHypothesis(
            label="bulky",
            profile=_profile(
                "foe",
                hp=200,
                max_hp=200,
                attack=120,
                speed=70,
                item="leftovers",
            ),
            moves=("bodyslam",),
            prior_weight=3,
        ),
        SetHypothesis(
            label="fast",
            profile=_profile(
                "foe",
                hp=120,
                max_hp=120,
                attack=200,
                speed=140,
                item="choicescarf",
            ),
            moves=("bodyslam",),
            prior_weight=1,
        ),
    )

    scenarios = hidden_state_scenarios(
        _snapshot(foe_hp_percent=50.0),
        state,
        _theirs(),
        {"foe": hypotheses},
        speed_states_from_exact_state(state),
    )

    assert [scenario.probability for scenario in scenarios] == pytest.approx(
        [0.75, 0.25]
    )

    bulky = scenarios[0]
    fast = scenarios[1]

    bulky_profile = bulky.state.profile(
        TurnSide.OPPONENT,
        "foe",
    )
    fast_profile = fast.state.profile(
        TurnSide.OPPONENT,
        "foe",
    )

    assert bulky_profile.current_hp == 100
    assert bulky_profile.max_hp == 200
    assert bulky_profile.stats["atk"] == 120
    assert fast_profile.current_hp == 60
    assert fast_profile.max_hp == 120
    assert fast_profile.stats["atk"] == 200

    assert bulky.speed_states[
        (TurnSide.OPPONENT, "foe")
    ].speed_stat == 70
    assert fast.speed_states[
        (TurnSide.OPPONENT, "foe")
    ].speed_stat == 140
    assert fast.speed_states[
        (TurnSide.OPPONENT, "foe")
    ].choice_scarf is True


def test_modeled_move_conditions_hidden_posterior() -> None:
    state = _state()
    hypotheses = (
        SetHypothesis(
            label="body",
            profile=_profile("foe"),
            moves=("bodyslam", "protect"),
            prior_weight=3,
        ),
        SetHypothesis(
            label="rock",
            profile=_profile("foe"),
            moves=("rockslide", "protect"),
            prior_weight=1,
        ),
    )

    scenarios = hidden_state_scenarios(
        _snapshot(foe_moves=("protect",)),
        state,
        _theirs("Rock Slide"),
        {"foe": hypotheses},
        speed_states_from_exact_state(state),
    )

    assert len(scenarios) == 1
    assert scenarios[0].probability == pytest.approx(1.0)
    assert scenarios[0].labels == ("foe:rock",)


def test_revealed_item_overrides_hidden_profile_item() -> None:
    snapshot = _snapshot()
    opponent = list(snapshot.opponent.pokemon)
    opponent[0] = PokemonSnapshot(
        **{
            **opponent[0].__dict__,
            "item": "leftovers",
        }
    )
    snapshot = DecisionSnapshot(
        **{
            **snapshot.__dict__,
            "opponent": SideSnapshot(
                pokemon=tuple(opponent),
                active_slots=snapshot.opponent.active_slots,
                tailwind_turns=0,
                side_conditions=(),
            ),
        }
    )
    state = _state()
    hypotheses = (
        SetHypothesis(
            label="revealed",
            profile=_profile(
                "foe",
                item="leftovers",
            ),
            moves=("bodyslam",),
        ),
    )

    scenario = hidden_state_scenarios(
        snapshot,
        state,
        _theirs(),
        {"foe": hypotheses},
        speed_states_from_exact_state(state),
    )[0]

    assert scenario.state.profile(
        TurnSide.OPPONENT,
        "foe",
    ).item == "leftovers"


def test_hidden_choice_scarf_speed_changes_which_side_moves_first() -> None:
    state = _state()
    hypotheses = (
        SetHypothesis(
            label="slow",
            profile=_profile(
                "foe",
                speed=80,
                item=None,
            ),
            moves=("bodyslam",),
            prior_weight=7,
        ),
        SetHypothesis(
            label="scarf",
            profile=_profile(
                "foe",
                speed=80,
                item="choicescarf",
            ),
            moves=("bodyslam",),
            prior_weight=3,
        ),
    )
    scenarios = hidden_state_scenarios(
        _snapshot(),
        state,
        _theirs(),
        {"foe": hypotheses},
        speed_states_from_exact_state(state),
    )

    evaluation = evaluate_action_pair_scenarios(
        scenarios,
        _ours(),
        _theirs(),
        {
            (TurnSide.PLAYER, "hero", "bodyslam"): BODY_SLAM,
            (TurnSide.OPPONENT, "foe", "bodyslam"): BODY_SLAM,
        },
        branching_policy=_policy(),
    )

    assert evaluation.model == "mechanics_position_hidden_v0"
    assert evaluation.distribution.total_probability == pytest.approx(1.0)
    assert len(evaluation.distribution.outcomes) == 2

    player_wins = 0.0
    opponent_wins = 0.0
    for outcome in evaluation.distribution.outcomes:
        result = outcome.result.state
        hero_hp = result.profile(
            TurnSide.PLAYER,
            "hero",
        ).current_hp
        foe_hp = result.profile(
            TurnSide.OPPONENT,
            "foe",
        ).current_hp

        if hero_hp > 0 and foe_hp == 0:
            player_wins += outcome.probability
        if hero_hp == 0 and foe_hp > 0:
            opponent_wins += outcome.probability

    assert player_wins == pytest.approx(0.7)
    assert opponent_wins == pytest.approx(0.3)


def test_hidden_scenario_cap_renormalizes_retained_probability() -> None:
    state = _state()
    hypotheses = tuple(
        SetHypothesis(
            label=f"set-{index}",
            profile=_profile(
                "foe",
                speed=70 + index,
            ),
            moves=("bodyslam",),
            prior_weight=weight,
        )
        for index, weight in enumerate((4, 3, 2, 1))
    )

    scenarios = hidden_state_scenarios(
        _snapshot(),
        state,
        _theirs(),
        {"foe": hypotheses},
        speed_states_from_exact_state(state),
        max_scenarios=2,
    )

    assert len(scenarios) == 2
    assert sum(
        scenario.probability
        for scenario in scenarios
    ) == pytest.approx(1.0)
    assert [scenario.labels for scenario in scenarios] == [
        ("foe:set-0",),
        ("foe:set-1",),
    ]

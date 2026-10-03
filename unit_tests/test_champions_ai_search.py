from dataclasses import replace

import pytest

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.decision_pipeline import rank_decision
from vgc_bench.src.champions_ai.hidden_scenarios import HiddenStateScenario
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.opponent_model import OpponentActionCandidate
from vgc_bench.src.champions_ai.search import (
    SearchConfig,
    _observation_key,
    _Row,
    _Search,
    _World,
)
from vgc_bench.src.champions_ai.search_actions import generate_search_actions
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
    SideSnapshot,
)
from vgc_bench.src.champions_ai.speed_context import speed_states_from_exact_state
from vgc_bench.src.champions_ai.turn_branching import (
    BranchingPolicy,
    TurnOutcomeDistribution,
    WeightedTurnOutcome,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    TurnSimulationConfig,
    TurnSimulationResult,
    simulate_turn,
)

P = TurnSide.PLAYER
FOE_SIDE = TurnSide.OPPONENT


def _profile(name, *, attack=200, defense=100, speed=80, hp=100, item=None):
    return CombatantProfile(
        name,
        50,
        hp,
        100,
        ("normal",),
        {"atk": attack, "def": defense, "spa": attack, "spd": defense, "spe": speed},
        item=item,
    )


def _move(slot, actor, move, target=None):
    return SlotAction(
        slot, ActionKind.MOVE, actor=actor, move=move, target_position=target
    )


def _pass(slot, actor=None):
    return SlotAction(slot, ActionKind.PASS, actor=actor)


def _joint(first, second=None):
    return JointAction(first, second or _pass(1))


def _policy(**kwargs):
    values = dict(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=False,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=15,
    )
    return BranchingPolicy(**{**values, **kwargs})


def _snapshot(state):
    def side(s):
        return SideSnapshot(
            tuple(
                PokemonSnapshot(
                    name,
                    100 * p.current_hp / p.max_hp,
                    p.status,
                    p.current_hp == 0,
                    next(
                        (
                            slot
                            for (side, slot), n in state.active_slots.items()
                            if side is s and n == name
                        ),
                        None,
                    ),
                    tuple(state.known_moves.get((s, name), ())),
                    p.item,
                    p.ability,
                    tuple(p.boosts.items()),
                    0,
                    state.first_turn.get((s, name), False),
                )
                for (ps, name), p in state.profiles.items()
                if ps is s
            ),
            tuple((slot, n) for (ps, slot), n in state.active_slots.items() if ps is s),
            state.tailwind_turns.get(s, 0),
            (),
        )

    return DecisionSnapshot(
        1,
        side(P),
        side(FOE_SIDE),
        state.weather,
        state.terrain,
        state.trick_room_turns or 0,
        (),
        0,
    )


def _setup_position():
    state = ExactTurnState(
        profiles={
            (P, "hero"): _profile("hero", attack=600, speed=80),
            (P, "ally"): _profile("ally", attack=300, speed=70),
            (FOE_SIDE, "threat"): _profile("threat", defense=400, speed=120),
            (FOE_SIDE, "weak"): _profile("weak", attack=1, speed=30),
        },
        active_slots={
            (P, 0): "hero",
            (P, 1): "ally",
            (FOE_SIDE, 0): "threat",
            (FOE_SIDE, 1): "weak",
        },
        must_recharge={(FOE_SIDE, "threat")},
        known_moves={
            (P, "hero"): {"tackle", "protect"},
            (P, "ally"): {"tackle", "protect", "tailwind"},
            (FOE_SIDE, "threat"): {"bodyslam"},
            (FOE_SIDE, "weak"): {"tackle"},
        },
    )
    profiles = {
        (P, "hero", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
        (P, "ally", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
        (FOE_SIDE, "threat", "bodyslam"): MoveProfile(
            "bodyslam", 200, "physical", "normal"
        ),
        (FOE_SIDE, "weak", "tackle"): MoveProfile("tackle", 1, "physical", "normal"),
    }
    immediate = _joint(_move(0, "hero", "tackle", 2), _move(1, "ally", "protect"))
    setup = _joint(_move(0, "hero", "protect"), _move(1, "ally", "tailwind"))
    theirs = _joint(_move(0, "threat", "bodyslam", 1), _move(1, "weak", "tackle", 2))
    return state, profiles, immediate, setup, theirs


def _rank(config=None, *, policy=None):
    state, profiles, immediate, setup, theirs = _setup_position()
    return rank_decision(
        _snapshot(state),
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=policy or _policy(),
        search_config=config,
    )


def test_depth_one_preserves_original_ranking_and_scores():
    baseline = _rank()
    shallow = _rank(SearchConfig(depth=1))
    assert shallow.ranked_options == baseline.ranked_options
    assert shallow.blocked_options == baseline.blocked_options
    assert shallow.response_summaries == baseline.response_summaries
    assert shallow.search_diagnostics.completed_depth == 1
    assert shallow.search_diagnostics.simulations == 0


def test_two_turns_prefer_setup_over_immediate_ko():
    state, _, immediate, setup, _ = _setup_position()
    baseline = _rank()
    deeper = _rank(SearchConfig(depth=2, max_observation_branches=None))
    assert baseline.best_play == immediate
    assert deeper.best_play == setup
    assert deeper.search_diagnostics.completed_depth == 2
    assert deeper.search_diagnostics.simulations > 0
    assert deeper.recommendation.expected_value > baseline.recommendation.expected_value
    assert state.must_recharge == {(FOE_SIDE, "threat")}
    assert not state.tailwind_sides


def test_simulation_budget_returns_last_complete_depth_without_partial_ranking():
    report = _rank(SearchConfig(depth=3, max_simulations=1))
    assert report.ranked_options == _rank().ranked_options
    assert report.search_diagnostics.completed_depth == 1
    assert report.search_diagnostics.budget_exhausted
    assert dict(report.search_diagnostics.cutoffs)["simulation_limit"] == 1


def test_three_turns_complete_and_reuse_cached_continuations():
    report = _rank(
        SearchConfig(depth=3, max_simulations=10000, max_observation_branches=None)
    )
    assert report.search_diagnostics.completed_depth == 3
    assert report.search_diagnostics.cache_hits > 0


def test_remaining_field_turns_expire_after_residuals_without_mutating_input():
    state, profiles, _, _, _ = _setup_position()
    state.tailwind_sides = {P}
    state.tailwind_turns = {P: 1}
    state.trick_room = True
    state.trick_room_turns = 1
    state.terrain = "grassyterrain"
    state.terrain_turns = 1
    state.weather = "raindance"
    state.weather_turns = 1
    passes = _joint(_pass(0), _pass(1))
    result = simulate_turn(
        state,
        passes,
        passes,
        speed_states_from_exact_state(state),
        profiles,
        TurnSimulationConfig(damage_roll_index=15),
    )
    assert not result.state.tailwind_sides
    assert result.state.trick_room is False
    assert result.state.terrain is None
    assert result.state.weather is None
    assert state.tailwind_sides == {P}
    assert state.trick_room is True


def test_new_tailwind_has_three_future_turns_after_setup():
    state, profiles, _, setup, theirs = _setup_position()
    result = simulate_turn(
        state,
        setup,
        theirs,
        speed_states_from_exact_state(state),
        profiles,
        TurnSimulationConfig(damage_roll_index=15),
    )
    assert result.state.tailwind_turns[P] == 3
    assert P in result.state.tailwind_sides


def test_future_move_generator_respects_fake_out_pp_and_choice_lock():
    state = ExactTurnState(
        {
            (P, "hero"): _profile("hero", item="choicescarf"),
            (FOE_SIDE, "foe"): _profile("foe"),
        },
        {(P, 0): "hero", (FOE_SIDE, 0): "foe"},
        known_moves={(P, "hero"): {"fakeout", "tackle", "bodyslam"}},
        first_turn={(P, "hero"): False},
        last_moves={(P, "hero"): "bodyslam"},
        move_pp={(P, "hero", "tackle"): 0},
    )
    actions = generate_search_actions(state, P)
    assert {a.first.move for a in actions} == {"bodyslam"}
    state.move_pp[(P, "hero", "bodyslam")] = 0
    assert {a.first.move for a in generate_search_actions(state, P)} == {"struggle"}


def test_replacement_phase_is_labelled_cutoff_instead_of_free_switch():
    state, profiles, immediate, setup, theirs = _setup_position()
    state.profiles[(P, "reserve")] = _profile("reserve")
    state.profiles[(P, "hero")] = replace(state.profiles[(P, "hero")], current_hp=0)
    assert generate_search_actions(state, P) is None


def test_unknown_field_duration_is_reported_as_leaf_cutoff():
    state, profiles, immediate, setup, theirs = _setup_position()
    state.weather = "raindance"
    report = rank_decision(
        _snapshot(state),
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=_policy(),
        search_config=SearchConfig(depth=2),
    )
    assert dict(report.search_diagnostics.cutoffs)["unknown_field_duration"] > 0


def test_hidden_sets_share_one_continuation_not_a_per_set_oracle(monkeypatch):
    import vgc_bench.src.champions_ai.search as search_module

    states = tuple(
        ExactTurnState(
            {
                (P, "hero"): _profile("hero"),
                (FOE_SIDE, "foe"): _profile("foe", defense=defense),
            },
            {(P, 0): "hero", (FOE_SIDE, 0): "foe"},
            known_moves={
                (P, "hero"): {"tackle", "bodyslam"},
                (FOE_SIDE, "foe"): {"tackle"},
            },
        )
        for defense in (100, 200)
    )
    worlds = tuple(
        _World(
            0.5,
            state,
            speed_states_from_exact_state(state),
            {k: frozenset(v) for k, v in state.known_moves.items()},
            {},
        )
        for state in states
    )
    assert _observation_key(worlds[0]) == _observation_key(worlds[1])
    our_actions = (
        _joint(_move(0, "hero", "tackle", 1)),
        _joint(_move(0, "hero", "bodyslam", 1)),
    )
    theirs = _joint(_move(0, "foe", "tackle", 1))
    monkeypatch.setattr(
        search_module,
        "generate_search_actions",
        lambda state, side, **kwargs: our_actions if side is P else (theirs,),
    )

    def outcomes(state, ours, theirs, speeds, profiles, **kwargs):
        state = state.copy()
        correct = (ours.first.move == "tackle") == (
            state.profile(FOE_SIDE, "foe").stats["def"] == 100
        )
        state.profiles[(P, "hero")] = replace(
            state.profile(P, "hero"), current_hp=100 if correct else 60
        )
        return TurnOutcomeDistribution(
            (WeightedTurnOutcome(1.0, TurnSimulationResult(state, ()), ()),)
        )

    monkeypatch.setattr(search_module, "simulate_turn_distribution", outcomes)
    profiles = {
        (P, "hero", "tackle"): MoveProfile("tackle", 1, "physical", "normal"),
        (P, "hero", "bodyslam"): MoveProfile("bodyslam", 1, "physical", "normal"),
        (FOE_SIDE, "foe", "tackle"): MoveProfile("tackle", 1, "physical", "normal"),
    }
    engine = _Search(SearchConfig(depth=2), profiles, _policy(), None, 9)
    rows = tuple(_Row(i, 1.0, w) for i, w in enumerate(worlds))
    values = engine.continuation_values(rows, 1)
    # A clairvoyant per-set maximization would incorrectly report 0 in both.
    assert sorted(values) == pytest.approx([-10, 0])
    assert sum(w.probability * v for w, v in zip(worlds, values)) == pytest.approx(-5)


def test_observation_cap_retains_rare_branch_probability(monkeypatch):
    state = ExactTurnState(
        {(P, "hero"): _profile("hero"), (FOE_SIDE, "foe"): _profile("foe")},
        {(P, 0): "hero", (FOE_SIDE, 0): "foe"},
    )
    bad = state.copy()
    bad.profiles[(P, "hero")] = replace(bad.profile(P, "hero"), current_hp=0)
    worlds = (_World(0.99, state, {}, {}, {}), _World(0.01, bad, {}, {}, {}))
    engine = _Search(SearchConfig(max_observation_branches=1), {}, _policy(), None, 9)
    monkeypatch.setattr(
        engine, "value", lambda belief, remaining: (10.0,) * len(belief)
    )
    values = engine.continuation_values(
        tuple(_Row(0, w.probability, w) for w in worlds), 1
    )
    assert values == pytest.approx([10, -125])
    assert sum(w.probability * v for w, v in zip(worlds, values)) == pytest.approx(8.65)
    assert engine.cutoffs["observation_branch_limit"] == 1


def test_hidden_speed_worlds_and_accuracy_combine_across_future_turn():
    state = ExactTurnState(
        {
            (P, "hero"): _profile("hero", speed=100),
            (FOE_SIDE, "foe"): _profile("foe", speed=80),
        },
        {(P, 0): "hero", (FOE_SIDE, 0): "foe"},
        known_moves={(P, "hero"): {"focusblast"}, (FOE_SIDE, "foe"): {"tackle"}},
    )
    scarf = state.copy()
    scarf.profiles[(FOE_SIDE, "foe")] = replace(
        scarf.profile(FOE_SIDE, "foe"), item="choicescarf"
    )
    scenarios = tuple(
        HiddenStateScenario(
            weight,
            world,
            speed_states_from_exact_state(world),
            (label,),
            {(FOE_SIDE, "foe"): frozenset({"tackle"})},
        )
        for weight, world, label in ((0.75, state, "slow"), (0.25, scarf, "scarf"))
    )
    passes = _joint(_pass(0, "hero"))
    foe_passes = _joint(_pass(0, "foe"))
    profiles = {
        (P, "hero", "focusblast"): MoveProfile("focusblast", 1000, "special", "normal"),
        (FOE_SIDE, "foe", "tackle"): MoveProfile("tackle", 1000, "physical", "normal"),
    }
    report = rank_decision(
        _snapshot(state),
        state,
        (passes,),
        (OpponentActionCandidate(foe_passes),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=_policy(branch_accuracy=True),
        search_config=SearchConfig(depth=2, max_observation_branches=None),
        scenario_provider=lambda action: scenarios,
    )
    # 0.75 * 0.70 = 52.5% wins; the remaining 47.5% lose the modeled match.
    assert report.recommendation.expected_value == pytest.approx(6.25)
    assert report.search_diagnostics.completed_depth == 2
    assert state.profile(FOE_SIDE, "foe").item is None


def test_unsupported_future_move_stops_whole_node_without_dropping_choice():
    state = ExactTurnState(
        {
            (P, "hero"): _profile("hero", speed=100),
            (FOE_SIDE, "foe"): _profile("foe", speed=80),
        },
        {(P, 0): "hero", (FOE_SIDE, 0): "foe"},
        known_moves={
            (P, "hero"): {"tackle", "swordsdance"},
            (FOE_SIDE, "foe"): {"tackle"},
        },
    )
    passes = _joint(_pass(0, "hero"))
    profiles = {
        (P, "hero", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
        (FOE_SIDE, "foe", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
    }
    report = rank_decision(
        _snapshot(state),
        state,
        (passes,),
        (OpponentActionCandidate(_joint(_pass(0, "foe"))),),
        speed_states_from_exact_state(state),
        profiles,
        branching_policy=_policy(),
        search_config=SearchConfig(depth=2),
    )
    assert report.recommendation.expected_value == pytest.approx(0)
    assert dict(report.search_diagnostics.cutoffs)["unsupported_mechanic"] > 0


def test_active_tailwind_cannot_refresh_its_duration():
    state, profiles, _, setup, theirs = _setup_position()
    state.tailwind_sides = {P}
    state.tailwind_turns = {P: 2}
    result = simulate_turn(
        state,
        setup,
        theirs,
        speed_states_from_exact_state(state),
        profiles,
        TurnSimulationConfig(damage_roll_index=15),
    )
    assert result.state.tailwind_turns[P] == 1


def test_switch_resets_boosts_and_enables_next_turn_fake_out():
    state = ExactTurnState(
        {
            (P, "hero"): replace(_profile("hero"), boosts={"atk": 2, "spe": 2}),
            (P, "reserve"): _profile("reserve"),
            (P, "ally"): _profile("ally"),
            (FOE_SIDE, "foe"): _profile("foe"),
        },
        {(P, 0): "hero", (P, 1): "ally", (FOE_SIDE, 0): "foe"},
        first_turn={(P, "hero"): False},
        known_moves={(P, "reserve"): {"fakeout"}, (P, "ally"): {"protect"}},
    )
    switch = _joint(
        SlotAction(0, ActionKind.SWITCH, actor="hero", switch_to="reserve"),
        _move(1, "ally", "protect"),
    )
    passes = _joint(_pass(0))
    result = simulate_turn(
        state,
        switch,
        passes,
        speed_states_from_exact_state(state),
        {},
        TurnSimulationConfig(damage_roll_index=15),
    )
    assert result.state.profile(P, "hero").boosts == {}
    assert result.state.first_turn[(P, "reserve")] is True
    assert {
        a.first.move
        for a in generate_search_actions(result.state, P, include_switches=False)
    } == {"fakeout"}


def test_live_output_reports_search_horizon_and_shortened_branches():
    from vgc_bench.src.champions_ai.live_output import render_live_output

    output = render_live_output(_rank(SearchConfig(depth=2)))
    assert "up to 2 turns" in output
    limited = render_live_output(_rank(SearchConfig(depth=2, max_simulations=1)))
    assert "some branches use shorter-horizon values" in limited


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0},
        {"depth": 1.5},
        {"max_simulations": 0},
        {"max_seconds": float("nan")},
        {"discount": float("inf")},
        {"max_observation_branches": 0},
        {"opponent_switch_weight": -1},
    ],
)
def test_invalid_search_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)

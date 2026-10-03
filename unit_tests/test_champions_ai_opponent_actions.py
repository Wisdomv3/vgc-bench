from types import SimpleNamespace

import pytest

from vgc_bench.src.champions_ai.actions import ActionKind
from vgc_bench.src.champions_ai.hidden_sets import SetHypothesis
from vgc_bench.src.champions_ai.matchup import CombatantProfile
from vgc_bench.src.champions_ai.opponent_actions import (
    generate_opponent_action_candidates,
)
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
    SideSnapshot,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


def _mon(name: str, *, fainted: bool = False):
    return SimpleNamespace(
        species=name,
        name=name,
        fainted=fainted,
    )


def _battle(
    *,
    opponent_active=("salamence", "sneasler"),
    player_active=("garchomp", "whimsicott"),
    bench=(),
):
    opponent_mons = [
        None if name is None else _mon(name)
        for name in opponent_active
    ]
    player_mons = [
        None if name is None else _mon(name)
        for name in player_active
    ]
    team = {
        f"p2:{index}": pokemon
        for index, pokemon in enumerate(
            [
                mon
                for mon in opponent_mons
                if mon is not None
            ]
            + [_mon(name) for name in bench]
        )
    }
    return SimpleNamespace(
        active_pokemon=player_mons,
        opponent_active_pokemon=opponent_mons,
        opponent_team=team,
    )


def _pokemon_snapshot(
    name: str,
    *,
    active_slot: int | None,
    moves=(),
    item: str | None = None,
    ability: str | None = None,
):
    return PokemonSnapshot(
        name=name,
        hp_percent=100.0,
        status=None,
        fainted=False,
        active_slot=active_slot,
        revealed_moves=tuple(moves),
        item=item,
        ability=ability,
        stat_stages=(),
        protect_streak=0,
        first_turn=True,
    )


def _side(
    pokemon,
    *,
    active_slots=(),
):
    return SideSnapshot(
        pokemon=tuple(pokemon),
        active_slots=tuple(active_slots),
        tailwind_turns=0,
        side_conditions=(),
    )


def _snapshot(
    *,
    salamence_moves=("protect",),
    sneasler_moves=("protect",),
    salamence_item=None,
):
    return DecisionSnapshot(
        turn=1,
        player=_side(
            (
                _pokemon_snapshot(
                    "garchomp",
                    active_slot=0,
                    moves=("protect",),
                ),
                _pokemon_snapshot(
                    "whimsicott",
                    active_slot=1,
                    moves=("tailwind",),
                ),
            ),
            active_slots=((0, "garchomp"), (1, "whimsicott")),
        ),
        opponent=_side(
            (
                _pokemon_snapshot(
                    "salamence",
                    active_slot=0,
                    moves=salamence_moves,
                    item=salamence_item,
                ),
                _pokemon_snapshot(
                    "sneasler",
                    active_slot=1,
                    moves=sneasler_moves,
                ),
            ),
            active_slots=((0, "salamence"), (1, "sneasler")),
        ),
        weather=None,
        terrain=None,
        trick_room_turns=0,
        field_conditions=(),
        history_size=0,
    )


def _profile(
    name: str,
    *,
    item: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=100,
        max_hp=100,
        types=("normal",),
        stats={
            "atk": 120,
            "def": 100,
            "spa": 120,
            "spd": 100,
            "spe": 100,
        },
        item=item,
    )


def _state(*, salamence_item: str | None = None):
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "garchomp"): _profile("garchomp"),
            (TurnSide.PLAYER, "whimsicott"): _profile("whimsicott"),
            (
                TurnSide.OPPONENT,
                "salamence",
            ): _profile(
                "salamence",
                item=salamence_item,
            ),
            (TurnSide.OPPONENT, "sneasler"): _profile("sneasler"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "garchomp",
            (TurnSide.PLAYER, 1): "whimsicott",
            (TurnSide.OPPONENT, 0): "salamence",
            (TurnSide.OPPONENT, 1): "sneasler",
        },
    )


def test_revealed_spread_and_protect_generate_one_joint_action() -> None:
    snapshot = _snapshot(
        salamence_moves=("rockslide",),
        sneasler_moves=("protect",),
    )

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        _state(),
        include_switches=False,
    )

    assert len(candidates) == 1
    action = candidates[0].action
    assert action.first.move == "rockslide"
    assert action.first.target is None
    assert action.second.move == "protect"


def test_targeted_move_generates_all_legal_adjacent_targets() -> None:
    snapshot = _snapshot(
        salamence_moves=("tackle",),
        sneasler_moves=("protect",),
    )

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        _state(),
        include_switches=False,
    )

    tackle_targets = {
        candidate.action.first.target
        for candidate in candidates
    }
    assert tackle_targets == {
        "garchomp",
        "whimsicott",
        "sneasler",
    }
    assert all(
        candidate.prior_weight == pytest.approx(1 / 3)
        for candidate in candidates
    )


def test_hidden_set_posterior_adds_unrevealed_moves_with_weights() -> None:
    snapshot = _snapshot(
        salamence_moves=("protect",),
        sneasler_moves=("protect",),
    )
    hypotheses = (
        SetHypothesis(
            label="heat",
            profile=_profile("salamence"),
            moves=("protect", "heatwave"),
            prior_weight=3,
        ),
        SetHypothesis(
            label="rock",
            profile=_profile("salamence"),
            moves=("protect", "rockslide"),
            prior_weight=1,
        ),
    )

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        _state(),
        hidden_hypotheses={"salamence": hypotheses},
        include_switches=False,
    )

    weights = {
        candidate.action.first.move: candidate.prior_weight
        for candidate in candidates
    }
    assert weights["protect"] == pytest.approx(1.0)
    assert weights["heatwave"] == pytest.approx(0.75)
    assert weights["rockslide"] == pytest.approx(0.25)
    assert all(
        "hidden-set" in (candidate.source or "")
        or candidate.action.first.move == "protect"
        for candidate in candidates
    )


def test_switch_generation_excludes_double_switch_to_same_bench() -> None:
    snapshot = _snapshot()

    candidates = generate_opponent_action_candidates(
        _battle(bench=("incineroar", "rillaboom")),
        snapshot,
        _state(),
        include_switches=True,
    )

    double_switches = [
        candidate.action
        for candidate in candidates
        if candidate.action.first.kind is ActionKind.SWITCH
        and candidate.action.second.kind is ActionKind.SWITCH
    ]
    assert double_switches
    assert all(
        action.first.switch_to != action.second.switch_to
        for action in double_switches
    )


def test_missing_move_coverage_fails_instead_of_silently_only_switching() -> None:
    snapshot = _snapshot(
        salamence_moves=(),
        sneasler_moves=("protect",),
    )

    with pytest.raises(
        ValueError,
        match="provide compatible hidden-set hypotheses",
    ):
        generate_opponent_action_candidates(
            _battle(bench=("incineroar",)),
            snapshot,
            _state(),
        )


def test_encore_public_state_filters_to_locked_move() -> None:
    snapshot = _snapshot(
        salamence_moves=("protect", "tackle"),
        sneasler_moves=("protect",),
    )
    state = _state()
    state.encore_locks[
        (TurnSide.OPPONENT, "salamence")
    ] = ("tackle", 2)

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        state,
        include_switches=False,
    )

    assert candidates
    assert {
        candidate.action.first.move
        for candidate in candidates
    } == {"tackle"}


def test_choice_item_public_state_filters_to_last_move() -> None:
    snapshot = _snapshot(
        salamence_moves=("protect", "tackle"),
        sneasler_moves=("protect",),
        salamence_item="choicescarf",
    )
    state = _state(salamence_item="choicescarf")
    state.last_moves[
        (TurnSide.OPPONENT, "salamence")
    ] = "tackle"

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        state,
        include_switches=False,
    )

    assert {
        candidate.action.first.move
        for candidate in candidates
    } == {"tackle"}


def test_forced_recharge_uses_single_pass_action() -> None:
    snapshot = _snapshot(
        salamence_moves=(),
        sneasler_moves=("protect",),
    )
    state = _state()
    state.must_recharge.add(
        (TurnSide.OPPONENT, "salamence")
    )

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        state,
        include_switches=False,
    )

    assert len(candidates) == 1
    assert candidates[0].action.first.kind is ActionKind.PASS
    assert candidates[0].action.second.move == "protect"


def test_max_candidates_prunes_by_prior_weight() -> None:
    snapshot = _snapshot(
        salamence_moves=("protect", "heatwave", "rockslide"),
        sneasler_moves=("protect",),
    )

    candidates = generate_opponent_action_candidates(
        _battle(),
        snapshot,
        _state(),
        include_switches=False,
        max_candidates=2,
    )

    assert len(candidates) == 2
    assert candidates[0].prior_weight >= candidates[1].prior_weight

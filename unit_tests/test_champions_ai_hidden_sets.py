from vgc_bench.src.champions_ai.hidden_sets import (
    SetHypothesis,
    evaluate_hidden_attacker,
    evaluate_hidden_defender,
    posterior_hypotheses,
    sample_hypotheses,
)
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.snapshot import PokemonSnapshot


def _profile(
    name: str,
    *,
    current_hp: int = 150,
    max_hp: int = 150,
    stats: dict[str, int] | None = None,
    item: str | None = None,
    ability: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=current_hp,
        max_hp=max_hp,
        types=("normal",),
        stats=stats
        or {
            "atk": 150,
            "def": 100,
            "spa": 150,
            "spd": 100,
            "spe": 100,
        },
        item=item,
        ability=ability,
    )


def _snapshot(
    *,
    moves: tuple[str, ...] = (),
    item: str | None = None,
    ability: str | None = None,
):
    return PokemonSnapshot(
        name="opponent",
        hp_percent=100.0,
        status=None,
        fainted=False,
        active_slot=0,
        revealed_moves=moves,
        item=item,
        ability=ability,
        stat_stages=(
            ("accuracy", 0),
            ("atk", 0),
            ("def", 0),
            ("evasion", 0),
            ("spa", 0),
            ("spd", 0),
            ("spe", 0),
        ),
        protect_streak=0,
    )


def _hypothesis(
    label: str,
    *,
    moves: tuple[str, ...],
    weight: float = 1.0,
    item: str | None = None,
    ability: str | None = None,
    stats: dict[str, int] | None = None,
    current_hp: int = 150,
    max_hp: int = 150,
):
    return SetHypothesis(
        label=label,
        profile=_profile(
            label,
            current_hp=current_hp,
            max_hp=max_hp,
            stats=stats,
            item=item,
            ability=ability,
        ),
        moves=moves,
        prior_weight=weight,
    )


def test_revealed_move_removes_incompatible_sets() -> None:
    snapshot = _snapshot(moves=("Hyper Voice",))
    specs = (
        _hypothesis("special", moves=("Hyper Voice", "Protect")),
        _hypothesis("physical", moves=("Double-Edge", "Protect")),
    )

    posterior = posterior_hypotheses(snapshot, specs)

    assert len(posterior) == 1
    assert posterior[0].hypothesis.label == "special"


def test_revealed_item_and_ability_filter_hypotheses() -> None:
    snapshot = _snapshot(item="Choice Scarf", ability="Intimidate")
    specs = (
        _hypothesis(
            "match",
            moves=("Hyper Voice",),
            item="choicescarf",
            ability="intimidate",
        ),
        _hypothesis(
            "wrong-item",
            moves=("Hyper Voice",),
            item="lifeorb",
            ability="intimidate",
        ),
        _hypothesis(
            "wrong-ability",
            moves=("Hyper Voice",),
            item="choicescarf",
            ability="moxie",
        ),
    )

    posterior = posterior_hypotheses(snapshot, specs)

    assert [entry.hypothesis.label for entry in posterior] == ["match"]


def test_prior_weights_are_normalized_not_invented() -> None:
    snapshot = _snapshot()
    specs = (
        _hypothesis("common", moves=("Protect",), weight=3),
        _hypothesis("rare", moves=("Protect",), weight=1),
    )

    posterior = posterior_hypotheses(snapshot, specs)

    assert posterior[0].probability == 0.75
    assert posterior[1].probability == 0.25


def test_zero_weight_pool_falls_back_to_equal_probabilities() -> None:
    snapshot = _snapshot()
    specs = (
        _hypothesis("a", moves=("Protect",), weight=0),
        _hypothesis("b", moves=("Protect",), weight=0),
    )

    posterior = posterior_hypotheses(snapshot, specs)

    assert posterior[0].probability == 0.5
    assert posterior[1].probability == 0.5


def test_sampling_is_reproducible_with_seed() -> None:
    snapshot = _snapshot()
    specs = (
        _hypothesis("a", moves=("Protect",), weight=3),
        _hypothesis("b", moves=("Protect",), weight=1),
    )

    first = sample_hypotheses(snapshot, specs, count=20, seed=7)
    second = sample_hypotheses(snapshot, specs, count=20, seed=7)

    assert [sample.label for sample in first] == [
        sample.label for sample in second
    ]


def test_no_compatible_sets_returns_empty_posterior() -> None:
    snapshot = _snapshot(moves=("Hyper Voice",))
    specs = (
        _hypothesis("physical", moves=("Double-Edge", "Protect")),
    )

    assert posterior_hypotheses(snapshot, specs) == ()
    assert sample_hypotheses(snapshot, specs, count=5, seed=1) == ()


def test_hidden_defender_damage_is_weighted_across_sets() -> None:
    attacker = _profile("attacker")
    snapshot = _snapshot()
    move = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    frail = _hypothesis(
        "frail",
        moves=("Protect",),
        weight=3,
        current_hp=80,
        stats={
            "atk": 100,
            "def": 80,
            "spa": 100,
            "spd": 80,
            "spe": 120,
        },
    )
    bulky = _hypothesis(
        "bulky",
        moves=("Protect",),
        weight=1,
        current_hp=160,
        max_hp=180,
        stats={
            "atk": 100,
            "def": 160,
            "spa": 100,
            "spd": 160,
            "spe": 60,
        },
    )

    estimate = evaluate_hidden_defender(
        attacker,
        snapshot,
        (frail, bulky),
        move,
    )

    assert len(estimate.hypotheses) == 2
    expected_ko = (
        0.75 * estimate.hypotheses[0].ko_probability
        + 0.25 * estimate.hypotheses[1].ko_probability
    )
    assert estimate.ko_probability == expected_ko
    assert estimate.minimum <= estimate.maximum


def test_hidden_attacker_damage_is_weighted_across_sets() -> None:
    snapshot = _snapshot(moves=("Body Slam",))
    defender = _profile("defender", current_hp=120)
    move = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    offensive = _hypothesis(
        "offensive",
        moves=("Body Slam",),
        weight=2,
        stats={
            "atk": 180,
            "def": 100,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        },
    )
    defensive = _hypothesis(
        "defensive",
        moves=("Body Slam",),
        weight=1,
        stats={
            "atk": 100,
            "def": 180,
            "spa": 100,
            "spd": 180,
            "spe": 60,
        },
    )

    estimate = evaluate_hidden_attacker(
        snapshot,
        (offensive, defensive),
        defender,
        move,
    )

    assert len(estimate.hypotheses) == 2
    assert estimate.hypotheses[0].probability == 2 / 3
    assert estimate.hypotheses[1].probability == 1 / 3
    assert estimate.maximum >= estimate.minimum

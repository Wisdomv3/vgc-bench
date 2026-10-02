from vgc_bench.src.champions_ai.spread import (
    SpreadTarget,
    get_spread_profile,
    is_spread_move,
    normalize_move_id,
)


def test_normalize_move_id() -> None:
    assert normalize_move_id("Hyper Voice") == "hypervoice"
    assert normalize_move_id("Matcha Gotcha") == "matchagotcha"
    assert normalize_move_id("Rock Slide") == "rockslide"


def test_common_spread_moves_are_detected() -> None:
    for move in [
        "Hyper Voice",
        "Heat Wave",
        "Rock Slide",
        "Earthquake",
        "Water Spout",
        "Eruption",
        "Matcha Gotcha",
        "Dazzling Gleam",
        "Icy Wind",
        "Surf",
        "Discharge",
    ]:
        assert is_spread_move(move), move


def test_single_target_moves_are_not_spread() -> None:
    for move in ["Draco Meteor", "Wood Hammer", "Dire Claw", "Brave Bird"]:
        assert not is_spread_move(move), move


def test_partner_hitting_spread_moves_are_classified() -> None:
    earthquake = get_spread_profile("Earthquake")
    surf = get_spread_profile("Surf")

    assert earthquake is not None
    assert surf is not None
    assert earthquake.target is SpreadTarget.ALL_ADJACENT
    assert surf.target is SpreadTarget.ALL_ADJACENT


def test_foe_only_spread_move_is_classified() -> None:
    hyper_voice = get_spread_profile("Hyper Voice")

    assert hyper_voice is not None
    assert hyper_voice.target is SpreadTarget.ALL_FOES

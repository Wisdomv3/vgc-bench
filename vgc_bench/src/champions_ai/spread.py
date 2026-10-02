"""Spread-move detection for Pokemon Champions doubles."""

from dataclasses import dataclass
from enum import Enum


class SpreadTarget(str, Enum):
    ALL_FOES = "all_foes"
    ALL_ADJACENT = "all_adjacent"


@dataclass(frozen=True)
class SpreadMoveProfile:
    move_id: str
    target: SpreadTarget
    notes: str = ""


SPREAD_MOVES: dict[str, SpreadMoveProfile] = {
    "blizzard": SpreadMoveProfile("blizzard", SpreadTarget.ALL_FOES),
    "dazzlinggleam": SpreadMoveProfile("dazzlinggleam", SpreadTarget.ALL_FOES),
    "discharge": SpreadMoveProfile(
        "discharge",
        SpreadTarget.ALL_ADJACENT,
        "Can hit the user's partner.",
    ),
    "earthquake": SpreadMoveProfile(
        "earthquake",
        SpreadTarget.ALL_ADJACENT,
        "Can hit the user's partner.",
    ),
    "eruption": SpreadMoveProfile(
        "eruption",
        SpreadTarget.ALL_FOES,
        "Power scales with current HP.",
    ),
    "expandingforce": SpreadMoveProfile(
        "expandingforce",
        SpreadTarget.ALL_FOES,
        "Becomes spread under Psychic Terrain.",
    ),
    "heatwave": SpreadMoveProfile("heatwave", SpreadTarget.ALL_FOES),
    "hypervoice": SpreadMoveProfile("hypervoice", SpreadTarget.ALL_FOES),
    "icywind": SpreadMoveProfile("icywind", SpreadTarget.ALL_FOES),
    "matchagotcha": SpreadMoveProfile(
        "matchagotcha",
        SpreadTarget.ALL_FOES,
        "Damages both foes and can heal the user.",
    ),
    "muddywater": SpreadMoveProfile("muddywater", SpreadTarget.ALL_FOES),
    "rockslide": SpreadMoveProfile(
        "rockslide",
        SpreadTarget.ALL_FOES,
        "Can flinch targets when the user moves first.",
    ),
    "snarl": SpreadMoveProfile("snarl", SpreadTarget.ALL_FOES),
    "surf": SpreadMoveProfile(
        "surf",
        SpreadTarget.ALL_ADJACENT,
        "Can hit the user's partner.",
    ),
    "waterspout": SpreadMoveProfile(
        "waterspout",
        SpreadTarget.ALL_FOES,
        "Power scales with current HP.",
    ),
}


def normalize_move_id(move_name: str) -> str:
    """Convert a readable move name to a Showdown-style move id."""
    return "".join(ch for ch in move_name.lower() if ch.isalnum())


def get_spread_profile(move_name: str) -> SpreadMoveProfile | None:
    """Return spread metadata for a move, or None when unknown."""
    return SPREAD_MOVES.get(normalize_move_id(move_name))


def is_spread_move(move_name: str) -> bool:
    """Return True when the move is in the known spread table."""
    return get_spread_profile(move_name) is not None

"""Input-independent battle events for Champions AI.

Manual entry, Pokemon Showdown, and OBS/capture-card adapters should all emit
these same events. The state tracker therefore does not need to know where an
observation came from.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


ScalarValue: TypeAlias = str | int | float | bool | None


class Side(str, Enum):
    PLAYER = "player"
    OPPONENT = "opponent"


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    SWITCH_IN = "switch_in"
    MOVE_USED = "move_used"
    MOVE_REVEALED = "move_revealed"
    HP_UPDATED = "hp_updated"
    FAINTED = "fainted"
    STATUS_CHANGED = "status_changed"
    ITEM_REVEALED = "item_revealed"
    ABILITY_REVEALED = "ability_revealed"
    STAT_STAGE_CHANGED = "stat_stage_changed"
    TAILWIND_CHANGED = "tailwind_changed"
    TRICK_ROOM_CHANGED = "trick_room_changed"
    WEATHER_CHANGED = "weather_changed"
    TERRAIN_CHANGED = "terrain_changed"
    FIELD_CONDITION_CHANGED = "field_condition_changed"
    SIDE_CONDITION_CHANGED = "side_condition_changed"
    PROTECT_USED = "protect_used"
    PROTECT_STREAK_CHANGED = "protect_streak_changed"
    NOTE = "note"


@dataclass(frozen=True)
class BattleEvent:
    """One observed fact about a battle.

    Source identifies where the observation came from. The first supported
    sources will be manual, showdown, and obs.

    Confidence is useful for computer-vision observations. A manually
    confirmed or Showdown-protocol event should normally use 1.0.
    """

    type: EventType | str
    side: Side | str | None = None
    slot: int | None = None
    pokemon: str | None = None
    move: str | None = None
    hp_percent: float | None = None
    status: str | None = None
    item: str | None = None
    ability: str | None = None
    field: str | None = None
    value: ScalarValue = None
    turn: int | None = None
    source: str = "manual"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", EventType(self.type))
        if isinstance(self.side, str):
            object.__setattr__(self, "side", Side(self.side))
        if self.slot is not None and self.slot not in (0, 1):
            raise ValueError("slot must be 0 or 1 for a doubles battle")
        if self.hp_percent is not None and not 0 <= self.hp_percent <= 100:
            raise ValueError("hp_percent must be between 0 and 100")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

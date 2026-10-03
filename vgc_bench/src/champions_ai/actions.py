"""Universal action models for Champions AI doubles decisions."""

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    MOVE = "move"
    SWITCH = "switch"
    PASS = "pass"
    DEFAULT = "default"


class Gimmick(str, Enum):
    NONE = "none"
    MEGA = "mega"
    Z_MOVE = "z_move"
    DYNAMAX = "dynamax"
    TERA = "tera"


@dataclass(frozen=True)
class SlotAction:
    """One action selected for one active slot."""

    slot: int
    kind: ActionKind
    actor: str | None = None
    move: str | None = None
    target: str | None = None
    target_position: int | None = None
    switch_to: str | None = None
    gimmick: Gimmick = Gimmick.NONE

    def __post_init__(self) -> None:
        if self.slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")

        if self.kind is ActionKind.MOVE and not self.move:
            raise ValueError("move actions require move")
        if self.kind is ActionKind.SWITCH and not self.switch_to:
            raise ValueError("switch actions require switch_to")

    @property
    def label(self) -> str:
        actor = self.actor or f"slot {self.slot + 1}"

        if self.kind is ActionKind.MOVE:
            gimmick = "" if self.gimmick is Gimmick.NONE else f" [{self.gimmick.value}]"
            target = "" if self.target is None else f" -> {self.target}"
            return f"{actor}: {self.move}{target}{gimmick}"

        if self.kind is ActionKind.SWITCH:
            return f"{actor}: switch -> {self.switch_to}"

        return f"{actor}: {self.kind.value}"


@dataclass(frozen=True)
class JointAction:
    """The two simultaneous choices that make up one doubles turn."""

    first: SlotAction
    second: SlotAction

    def __post_init__(self) -> None:
        if self.first.slot != 0 or self.second.slot != 1:
            raise ValueError("joint actions must contain slot 0 followed by slot 1")

    @property
    def label(self) -> str:
        return f"{self.first.label} | {self.second.label}"

    @property
    def key(self) -> tuple[SlotAction, SlotAction]:
        return (self.first, self.second)

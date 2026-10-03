"""Universal battle-state tracker for Champions AI."""

from dataclasses import dataclass, field

from vgc_bench.src.champions_ai.events import BattleEvent, EventType, Side


@dataclass
class PokemonState:
    """Information currently known about one Pokemon."""

    name: str
    hp_percent: float = 100.0
    status: str | None = None
    fainted: bool = False
    active_slot: int | None = None
    revealed_moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    stat_stages: dict[str, int] = field(
        default_factory=lambda: {
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "accuracy": 0,
            "evasion": 0,
        }
    )
    protect_streak: int = 0
    first_turn: bool | None = None


@dataclass
class SideState:
    """Known state for one player's side of the battle."""

    pokemon: dict[str, PokemonState] = field(default_factory=dict)
    active_slots: dict[int, str] = field(default_factory=dict)
    tailwind_turns: int = 0
    side_conditions: set[str] = field(default_factory=set)


@dataclass
class BattleState:
    """Current battle state reconstructed from a stream of BattleEvents."""

    turn: int = 0
    player: SideState = field(default_factory=SideState)
    opponent: SideState = field(default_factory=SideState)
    weather: str | None = None
    terrain: str | None = None
    trick_room_turns: int = 0
    field_conditions: set[str] = field(default_factory=set)
    events: list[BattleEvent] = field(default_factory=list)

    def side_state(self, side: Side) -> SideState:
        return self.player if side is Side.PLAYER else self.opponent

    def get_or_create_pokemon(self, side: Side, name: str) -> PokemonState:
        side_state = self.side_state(side)
        if name not in side_state.pokemon:
            side_state.pokemon[name] = PokemonState(name=name)
        return side_state.pokemon[name]

    @staticmethod
    def _require_side(event: BattleEvent) -> Side:
        if event.side is None:
            raise ValueError(f"{event.type.value} requires a side")
        return event.side

    @staticmethod
    def _require_pokemon(event: BattleEvent) -> str:
        if not event.pokemon:
            raise ValueError(f"{event.type.value} requires a pokemon")
        return event.pokemon

    def apply(self, event: BattleEvent) -> None:
        """Apply one observed event to the current state."""

        if event.type is EventType.TURN_STARTED:
            if event.turn is None:
                raise ValueError("turn_started requires turn")
            self.turn = event.turn

        elif event.type is EventType.SWITCH_IN:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if event.slot is None:
                raise ValueError("switch_in requires slot")

            side_state = self.side_state(side)
            previous_name = side_state.active_slots.get(event.slot)
            if previous_name is not None and previous_name in side_state.pokemon:
                side_state.pokemon[previous_name].active_slot = None

            pokemon = self.get_or_create_pokemon(side, name)
            pokemon.active_slot = event.slot
            pokemon.fainted = False
            pokemon.protect_streak = 0
            pokemon.first_turn = True
            side_state.active_slots[event.slot] = name

        elif event.type in {EventType.MOVE_USED, EventType.MOVE_REVEALED}:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if not event.move:
                raise ValueError("move_used requires move")
            pokemon = self.get_or_create_pokemon(side, name)
            pokemon.revealed_moves.add(event.move)

        elif event.type is EventType.HP_UPDATED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if event.hp_percent is None:
                raise ValueError("hp_updated requires hp_percent")
            pokemon = self.get_or_create_pokemon(side, name)
            pokemon.hp_percent = event.hp_percent
            if event.hp_percent == 0:
                pokemon.fainted = True

        elif event.type is EventType.FAINTED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            pokemon = self.get_or_create_pokemon(side, name)
            pokemon.hp_percent = 0.0
            pokemon.fainted = True

        elif event.type is EventType.STATUS_CHANGED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            self.get_or_create_pokemon(side, name).status = event.status

        elif event.type is EventType.ITEM_REVEALED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            self.get_or_create_pokemon(side, name).item = event.item

        elif event.type is EventType.ABILITY_REVEALED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            self.get_or_create_pokemon(side, name).ability = event.ability

        elif event.type is EventType.STAT_STAGE_CHANGED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if event.field not in {
                "atk",
                "def",
                "spa",
                "spd",
                "spe",
                "accuracy",
                "evasion",
            }:
                raise ValueError("stat_stage_changed requires a valid stat in field")
            if not isinstance(event.value, int):
                raise ValueError("stat_stage_changed requires an integer value")
            stage = max(-6, min(6, event.value))
            self.get_or_create_pokemon(side, name).stat_stages[event.field] = stage

        elif event.type is EventType.TAILWIND_CHANGED:
            side = self._require_side(event)
            if not isinstance(event.value, int) or event.value < 0:
                raise ValueError("tailwind_changed requires a non-negative turn count")
            self.side_state(side).tailwind_turns = event.value

        elif event.type is EventType.TRICK_ROOM_CHANGED:
            if not isinstance(event.value, int) or event.value < 0:
                raise ValueError(
                    "trick_room_changed requires a non-negative turn count"
                )
            self.trick_room_turns = event.value

        elif event.type is EventType.WEATHER_CHANGED:
            if event.value is not None and not isinstance(event.value, str):
                raise ValueError("weather_changed value must be a string or None")
            self.weather = event.value

        elif event.type is EventType.TERRAIN_CHANGED:
            if event.value is not None and not isinstance(event.value, str):
                raise ValueError("terrain_changed value must be a string or None")
            self.terrain = event.value

        elif event.type is EventType.FIELD_CONDITION_CHANGED:
            if not event.field or not isinstance(event.value, bool):
                raise ValueError(
                    "field_condition_changed requires field and boolean value"
                )
            if event.value:
                self.field_conditions.add(event.field)
            else:
                self.field_conditions.discard(event.field)

        elif event.type is EventType.SIDE_CONDITION_CHANGED:
            side = self._require_side(event)
            if not event.field or not isinstance(event.value, bool):
                raise ValueError(
                    "side_condition_changed requires field and boolean value"
                )
            side_state = self.side_state(side)
            if event.value:
                side_state.side_conditions.add(event.field)
            else:
                side_state.side_conditions.discard(event.field)

        elif event.type is EventType.PROTECT_USED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            self.get_or_create_pokemon(side, name).protect_streak += 1

        elif event.type is EventType.PROTECT_STREAK_CHANGED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if not isinstance(event.value, int) or event.value < 0:
                raise ValueError(
                    "protect_streak_changed requires a non-negative integer"
                )
            self.get_or_create_pokemon(side, name).protect_streak = event.value

        elif event.type is EventType.FIRST_TURN_CHANGED:
            side = self._require_side(event)
            name = self._require_pokemon(event)
            if not isinstance(event.value, bool):
                raise ValueError("first_turn_changed requires a boolean value")
            self.get_or_create_pokemon(side, name).first_turn = event.value

        elif event.type is EventType.NOTE:
            pass

        else:
            raise ValueError(f"unsupported event type: {event.type}")

        self.events.append(event)

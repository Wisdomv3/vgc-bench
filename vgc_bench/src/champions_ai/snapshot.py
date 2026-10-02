"""Immutable decision snapshots used to prevent hindsight leakage."""

from dataclasses import dataclass

from vgc_bench.src.champions_ai.state import BattleState, PokemonState, SideState


@dataclass(frozen=True)
class PokemonSnapshot:
    name: str
    hp_percent: float
    status: str | None
    fainted: bool
    active_slot: int | None
    revealed_moves: tuple[str, ...]
    item: str | None
    ability: str | None
    stat_stages: tuple[tuple[str, int], ...]
    protect_streak: int

    @classmethod
    def from_state(cls, pokemon: PokemonState) -> "PokemonSnapshot":
        return cls(
            name=pokemon.name,
            hp_percent=pokemon.hp_percent,
            status=pokemon.status,
            fainted=pokemon.fainted,
            active_slot=pokemon.active_slot,
            revealed_moves=tuple(sorted(pokemon.revealed_moves)),
            item=pokemon.item,
            ability=pokemon.ability,
            stat_stages=tuple(sorted(pokemon.stat_stages.items())),
            protect_streak=pokemon.protect_streak,
        )


@dataclass(frozen=True)
class SideSnapshot:
    pokemon: tuple[PokemonSnapshot, ...]
    active_slots: tuple[tuple[int, str], ...]
    tailwind_turns: int
    side_conditions: tuple[str, ...]

    @classmethod
    def from_state(cls, side: SideState) -> "SideSnapshot":
        return cls(
            pokemon=tuple(
                PokemonSnapshot.from_state(side.pokemon[name])
                for name in sorted(side.pokemon)
            ),
            active_slots=tuple(sorted(side.active_slots.items())),
            tailwind_turns=side.tailwind_turns,
            side_conditions=tuple(sorted(side.side_conditions)),
        )


@dataclass(frozen=True)
class DecisionSnapshot:
    """Information the evaluator is allowed to know at one decision point."""

    turn: int
    player: SideSnapshot
    opponent: SideSnapshot
    weather: str | None
    terrain: str | None
    trick_room_turns: int
    field_conditions: tuple[str, ...]
    history_size: int
    legal_actions: tuple[str, ...] = ()

    @classmethod
    def from_state(
        cls, state: BattleState, legal_actions: tuple[str, ...] = ()
    ) -> "DecisionSnapshot":
        return cls(
            turn=state.turn,
            player=SideSnapshot.from_state(state.player),
            opponent=SideSnapshot.from_state(state.opponent),
            weather=state.weather,
            terrain=state.terrain,
            trick_room_turns=state.trick_room_turns,
            field_conditions=tuple(sorted(state.field_conditions)),
            history_size=len(state.events),
            legal_actions=legal_actions,
        )

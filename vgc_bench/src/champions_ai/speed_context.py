"""Supported live SpeedState construction for Champions AI."""

from dataclasses import replace

from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.spread import normalize_move_id
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


WEATHER_SPEED_ABILITIES = {
    "raindance": {"swiftswim"},
    "sunnyday": {"chlorophyll"},
    "desolateland": {"chlorophyll"},
    "primordialsea": {"swiftswim"},
    "sandstorm": {"sandrush"},
    "hail": {"slushrush"},
    "snow": {"slushrush"},
}


def speed_states_from_exact_state(
    state: ExactTurnState,
    *,
    unburden_active: frozenset[tuple[TurnSide, str]] = frozenset(),
    paradox_speed_active: frozenset[tuple[TurnSide, str]] = frozenset(),
) -> dict[tuple[TurnSide, str], SpeedState]:
    """Build supported SpeedState inputs from an ExactTurnState."""

    weather = normalize_move_id(state.weather or "")
    weather_abilities = WEATHER_SPEED_ABILITIES.get(weather, set())

    output: dict[tuple[TurnSide, str], SpeedState] = {}
    for key, profile in state.profiles.items():
        side, _name = key
        ability = normalize_move_id(profile.ability or "")
        item = normalize_move_id(profile.item or "")

        output[key] = SpeedState(
            speed_stat=int(profile.stats["spe"]),
            stage=int(profile.boosts.get("spe", 0)),
            tailwind=side in state.tailwind_sides,
            unburden=key in unburden_active,
            choice_scarf=item == "choicescarf",
            weather_speed_boost=ability in weather_abilities,
            paradox_speed_boost=key in paradox_speed_active,
            paralyzed=normalize_move_id(profile.status or "") == "par",
        )

    return output


def refresh_speed_states_for_exact_state(
    state: ExactTurnState,
    base_speed_states: dict[tuple[TurnSide, str], SpeedState],
) -> dict[tuple[TurnSide, str], SpeedState]:
    """Refresh set-dependent Speed inputs while preserving explicit activations.

    Unburden and Paradox Speed activation are history-dependent, so their
    booleans are preserved from the caller's base SpeedState mapping.
    """

    unburden_active = frozenset(
        key
        for key, speed_state in base_speed_states.items()
        if speed_state.unburden
    )
    paradox_speed_active = frozenset(
        key
        for key, speed_state in base_speed_states.items()
        if speed_state.paradox_speed_boost
    )

    refreshed = speed_states_from_exact_state(
        state,
        unburden_active=unburden_active,
        paradox_speed_active=paradox_speed_active,
    )

    for key, speed_state in tuple(refreshed.items()):
        base = base_speed_states.get(key)
        if base is None:
            continue
        refreshed[key] = replace(
            speed_state,
            unburden=base.unburden,
            paradox_speed_boost=base.paradox_speed_boost,
        )

    return refreshed

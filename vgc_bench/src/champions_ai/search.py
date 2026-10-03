"""Bounded, belief-aware expectimax over complete simultaneous doubles turns.

Player choices maximize the probability-weighted continuation value. Opponent
choices and battle RNG are chance nodes. Hidden worlds persist across turns;
worlds with the same public observation share ONE next player choice. This
avoids the optimistic 'strategy fusion' of searching each hidden set alone.

Values remain position-heuristic scores, not calibrated win probabilities.
Limits are checked between simulations; one exact turn still has its own
BranchingPolicy.max_nodes. Incomplete depth iterations never replace the last
completed result. Low-probability observation branches retain their mass and
receive leaf values rather than being discarded and renormalized.
"""

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from math import ceil, isfinite
from time import monotonic
from typing import Callable

from poke_env.battle import Move, MoveCategory

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction
from vgc_bench.src.champions_ai.hidden_scenarios import HiddenStateScenario
from vgc_bench.src.champions_ai.matchup import MoveProfile
from vgc_bench.src.champions_ai.mechanics_evaluator import MechanicsResponseAnalysis
from vgc_bench.src.champions_ai.opponent_model import (
    ActionProbability,
    OpponentActionCandidate,
    estimate_action_probabilities,
)
from vgc_bench.src.champions_ai.position_value import PositionWeights, evaluate_position
from vgc_bench.src.champions_ai.response_matrix import build_response_matrix
from vgc_bench.src.champions_ai.search_actions import generate_search_actions
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.speed import SpeedState
from vgc_bench.src.champions_ai.speed_context import (
    refresh_speed_states_for_exact_state,
)
from vgc_bench.src.champions_ai.spread import normalize_move_id
from vgc_bench.src.champions_ai.turn_branching import (
    BranchingPolicy,
    BranchLimitExceeded,
    simulate_turn_distribution,
)
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import (
    ExactTurnState,
    UnsupportedTurnMechanic,
)


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 2  # Includes the current turn; 1 keeps the original evaluator.
    max_simulations: int = 2_000
    max_seconds: float | None = None
    max_observation_branches: int | None = 32
    discount: float = 1.0
    include_switches: bool = True
    opponent_switch_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("depth", "max_simulations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_seconds is not None and (
            not isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.max_observation_branches is not None and (
            isinstance(self.max_observation_branches, bool)
            or not isinstance(self.max_observation_branches, int)
            or self.max_observation_branches < 1
        ):
            raise ValueError("max_observation_branches must be a positive integer")
        if not isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("discount must be finite and between 0 and 1")
        if not isfinite(self.opponent_switch_weight) or self.opponent_switch_weight < 0:
            raise ValueError("opponent_switch_weight must be finite and nonnegative")


@dataclass(frozen=True)
class SearchDiagnostics:
    requested_depth: int
    completed_depth: int
    simulations: int
    decision_nodes: int
    cache_hits: int
    cutoffs: tuple[tuple[str, int], ...]
    budget_exhausted: bool


@dataclass(frozen=True)
class SearchAnalysis:
    analysis: MechanicsResponseAnalysis
    diagnostics: SearchDiagnostics


@dataclass(frozen=True)
class _World:
    probability: float
    state: ExactTurnState
    speeds: dict[tuple[TurnSide, str], SpeedState]
    moves: dict[tuple[TurnSide, str], frozenset[str]]
    public_moves: dict[str, frozenset[str]]


@dataclass(frozen=True)
class _Row:
    parent: int
    conditional_probability: float
    world: _World


def _freeze(value):
    """Full-state cache key, including exact stats and all history counters."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return tuple((f.name, _freeze(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
        return tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _public_state(world: _World) -> ExactTurnState:
    """Remove hidden opponent data before generating a player continuation."""
    state = world.state.copy()
    for key, profile in tuple(state.profiles.items()):
        if key[0] is TurnSide.OPPONENT:
            state.profiles[key] = replace(
                profile,
                current_hp=ceil(100 * profile.current_hp / profile.max_hp),
                max_hp=100,
                stats={stat: 1 for stat in profile.stats},
                item=None,
                ability=None,
            )
            state.known_moves[key] = set(world.public_moves.get(key[1], ()))
        else:
            state.known_moves[key] = set(world.moves.get(key, ()))
    return state


def _observation_key(world: _World) -> tuple:
    state = world.state
    player_profiles = {
        key: profile
        for key, profile in state.profiles.items()
        if key[0] is TurnSide.PLAYER
    }
    # Deliberately coarse public HP (Showdown percentage), with no hidden max
    # HP, stats, item, ability, PP, move set, or RNG decision in the key.
    opponent_profiles = tuple(
        sorted(
            (
                name,
                ceil(100 * p.current_hp / p.max_hp),
                p.status,
                tuple(p.types),
                tuple(sorted(p.boosts.items())),
            )
            for (side, name), p in state.profiles.items()
            if side is TurnSide.OPPONENT
        )
    )
    return (
        _freeze(player_profiles),
        opponent_profiles,
        _freeze(state.active_slots),
        _freeze(world.public_moves),
        state.weather,
        state.terrain,
        state.trick_room,
        _freeze(state.tailwind_sides),
        _freeze(state.tailwind_turns),
        state.trick_room_turns,
        state.terrain_turns,
        state.weather_turns,
        _freeze(state.field_conditions),
        # These counters describe observable actions/conditions. Hidden Choice
        # locks, confusion duration, and hidden PP are intentionally omitted.
        _freeze(state.last_moves),
        _freeze(state.first_turn),
        _freeze(state.protect_streaks),
        _freeze(state.disabled_moves),
        _freeze(state.taunt_turns),
        _freeze(state.encore_locks),
        _freeze(state.imprison_users),
        _freeze(state.tormented),
        _freeze(state.attractions),
    )


def _terminal(state: ExactTurnState) -> bool:
    return any(
        not any(p.current_hp > 0 for (s, _), p in state.profiles.items() if s is side)
        for side in TurnSide
    )


def prepare_search_state(
    snapshot: DecisionSnapshot, initial_state: ExactTurnState
) -> ExactTurnState:
    """Copy a state and seed known field durations and entry history."""
    state = initial_state.copy()
    for side, side_snapshot in (
        (TurnSide.PLAYER, snapshot.player),
        (TurnSide.OPPONENT, snapshot.opponent),
    ):
        if side in state.tailwind_sides and side not in state.tailwind_turns:
            if side_snapshot.tailwind_turns > 0:
                state.tailwind_turns[side] = side_snapshot.tailwind_turns
        for pokemon in side_snapshot.pokemon:
            if pokemon.first_turn is not None:
                state.first_turn.setdefault((side, pokemon.name), pokemon.first_turn)
    if state.trick_room and state.trick_room_turns is None:
        if snapshot.trick_room_turns > 0:
            state.trick_room_turns = snapshot.trick_room_turns
    return state


class _BudgetExceeded(RuntimeError):
    pass


class _Search:
    def __init__(self, config, move_profiles, branching_policy, weights, gen):
        self.config = config
        self.move_profiles = move_profiles
        self.branching_policy = branching_policy
        self.weights = weights
        self.gen = gen
        self.started = monotonic()
        self.simulations = 0
        self.nodes = 0
        self.hits = 0
        self.cutoffs: dict[str, int] = defaultdict(int)
        self.cache: dict[tuple, tuple[float, ...]] = {}
        self.turn_cache: dict[tuple, tuple[_Row, ...]] = {}

    def check_budget(self):
        if (
            self.config.max_seconds is not None
            and monotonic() - self.started >= self.config.max_seconds
        ):
            raise _BudgetExceeded("time_limit")

    def score(self, state):
        return evaluate_position(state, weights=self.weights).score

    def transitions(self, world, ours, theirs):
        self.check_budget()
        key = (_freeze(world), ours.key, theirs.key)
        if key in self.turn_cache:
            self.hits += 1
            return self.turn_cache[key]
        if self.simulations >= self.config.max_simulations:
            raise _BudgetExceeded("simulation_limit")
        self.simulations += 1
        state = world.state.copy()
        state.known_moves = {key: set(moves) for key, moves in world.moves.items()}
        profiles = {
            key: profile
            for key, profile in self.move_profiles.items()
            if key[2] in world.moves.get(key[:2], ())
        }
        for side, action in ((TurnSide.PLAYER, ours), (TurnSide.OPPONENT, theirs)):
            for slot in (action.first, action.second):
                if slot.kind is not ActionKind.MOVE:
                    continue
                move = Move(slot.move, self.gen)
                if (
                    move.category is not MoveCategory.STATUS
                    and move.id != "struggle"
                    and (side, slot.actor, move.id) not in profiles
                ):
                    raise UnsupportedTurnMechanic(
                        f"missing future damage model for {side.value} "
                        f"{slot.actor} {move.id}"
                    )
        distribution = simulate_turn_distribution(
            state,
            ours,
            theirs,
            world.speeds,
            profiles,
            policy=self.branching_policy,
            gen=self.gen,
        )
        rows = []
        for outcome in distribution.outcomes:
            if outcome.probability <= 0:
                continue
            public_moves = dict(world.public_moves)
            for (side, name), move in outcome.result.state.last_moves.items():
                if side is TurnSide.OPPONENT:
                    public_moves[name] = public_moves.get(name, frozenset()) | {move}
            child = _World(
                world.probability * outcome.probability,
                outcome.result.state,
                refresh_speed_states_for_exact_state(
                    outcome.result.state, world.speeds
                ),
                world.moves,
                public_moves,
            )
            rows.append(_Row(0, outcome.probability, child))
        result = tuple(rows)
        self.turn_cache[key] = result
        return result

    def continuation_values(self, rows, remaining):
        groups: dict[tuple, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[_observation_key(row.world)].append(index)
        ordered = sorted(
            groups.values(),
            key=lambda indices: -sum(rows[i].world.probability for i in indices),
        )
        values = [self.score(row.world.state) for row in rows]
        cap = self.config.max_observation_branches
        for group_index, indices in enumerate(ordered):
            if remaining == 0:
                continue
            if cap is not None and group_index >= cap:
                self.cutoffs["observation_branch_limit"] += 1
                continue
            mass = sum(rows[i].world.probability for i in indices)
            belief = tuple(
                replace(rows[i].world, probability=rows[i].world.probability / mass)
                for i in indices
            )
            continuation = self.value(belief, remaining)
            for i, continued in zip(indices, continuation):
                values[i] += self.config.discount * (continued - values[i])
        return values

    def value(self, belief, remaining):
        self.check_budget()
        leaf = tuple(self.score(world.state) for world in belief)
        if remaining == 0 or all(_terminal(world.state) for world in belief):
            return leaf
        key = (_freeze(belief), remaining)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.nodes += 1
        # Do not extend a field beyond an unknown expiry date.
        for world in belief:
            state = world.state
            if (
                state.weather
                and state.weather_turns is None
                or state.terrain
                and state.terrain_turns is None
                or state.trick_room
                and state.trick_room_turns is None
                or any(s not in state.tailwind_turns for s in state.tailwind_sides)
                or state.field_conditions
            ):
                self.cutoffs["unknown_field_duration"] += 1
                return leaf
        ours = generate_search_actions(
            _public_state(belief[0]),
            TurnSide.PLAYER,
            include_switches=self.config.include_switches,
            gen=self.gen,
        )
        opponents = []
        for world in belief:
            state = world.state.copy()
            state.known_moves = {key: set(v) for key, v in world.moves.items()}
            actions = generate_search_actions(
                state,
                TurnSide.OPPONENT,
                include_switches=self.config.include_switches,
                gen=self.gen,
            )
            if actions is None or ours is None:
                self.cutoffs["replacement_or_missing_moves"] += 1
                return leaf
            candidates = tuple(
                OpponentActionCandidate(
                    action,
                    prior_weight=(
                        self.config.opponent_switch_weight
                        if any(
                            s.kind is ActionKind.SWITCH
                            for s in (action.first, action.second)
                        )
                        else 1.0
                    ),
                    source="search-neutral-prior",
                )
                for action in actions
            )
            opponents.append(estimate_action_probabilities(candidates))
        best = None
        best_expected = float("-inf")
        # A selected player action is common to the entire information set.
        try:
            for action in sorted(ours, key=lambda a: a.label):
                rows = []
                for parent, (world, estimates) in enumerate(zip(belief, opponents)):
                    for estimate in estimates:
                        if estimate.probability <= 0:
                            continue
                        for row in self.transitions(
                            world, action, estimate.candidate.action
                        ):
                            rows.append(
                                _Row(
                                    parent,
                                    estimate.probability * row.conditional_probability,
                                    replace(
                                        row.world,
                                        probability=(
                                            row.world.probability * estimate.probability
                                        ),
                                    ),
                                )
                            )
                child_values = self.continuation_values(rows, remaining - 1)
                conditional_values = [0.0] * len(belief)
                for row, value in zip(rows, child_values):
                    conditional_values[row.parent] += (
                        row.conditional_probability * value
                    )
                expected = sum(
                    w.probability * v for w, v in zip(belief, conditional_values)
                )
                if expected > best_expected:
                    best_expected = expected
                    best = tuple(conditional_values)
        except (UnsupportedTurnMechanic, BranchLimitExceeded) as exc:
            # Dropping just the unsupported choice would bias the maximization.
            reason = (
                "unsupported_mechanic"
                if isinstance(exc, UnsupportedTurnMechanic)
                else "turn_branch_limit"
            )
            self.cutoffs[reason] += 1
            return leaf
        if best is None:
            return leaf
        self.cache[key] = best
        return best


def build_search_analysis(
    snapshot: DecisionSnapshot,
    initial_state: ExactTurnState,
    baseline: MechanicsResponseAnalysis,
    our_actions: tuple[JointAction, ...],
    opponent_estimates: tuple[ActionProbability, ...],
    speed_states: dict[tuple[TurnSide, str], SpeedState],
    move_profiles: dict[tuple[TurnSide, str, str], MoveProfile],
    *,
    config: SearchConfig,
    branching_policy: BranchingPolicy | None = None,
    position_weights: PositionWeights | None = None,
    scenario_provider: Callable[[JointAction], tuple[HiddenStateScenario, ...]]
    | None = None,
    gen: int = 9,
) -> SearchAnalysis:
    """Deepen a complete one-turn analysis, retaining its immediate-loss evidence."""
    if config.depth == 1:
        return SearchAnalysis(baseline, SearchDiagnostics(1, 1, 0, 0, 0, (), False))
    engine = _Search(config, move_profiles, branching_policy, position_weights, gen)
    catalog = {key: frozenset(v) for key, v in initial_state.known_moves.items()}
    for side, name, move in move_profiles:
        key = (side, name)
        catalog[key] = catalog.get(key, frozenset()) | {normalize_move_id(move)}
    for side, actions in (
        (TurnSide.PLAYER, our_actions),
        (TurnSide.OPPONENT, tuple(e.candidate.action for e in opponent_estimates)),
    ):
        for action in actions:
            for slot in (action.first, action.second):
                if slot.actor and slot.move:
                    key = (side, slot.actor)
                    catalog[key] = catalog.get(key, frozenset()) | {
                        normalize_move_id(slot.move)
                    }
    public_moves = {
        p.name: frozenset(normalize_move_id(m) for m in p.revealed_moves)
        for p in snapshot.opponent.pokemon
    }
    worlds_by_response = []
    for estimate in opponent_estimates:
        if estimate.probability <= 0:
            worlds_by_response.append(())
            continue
        scenarios = (
            scenario_provider(estimate.candidate.action)
            if scenario_provider
            else (HiddenStateScenario(1.0, initial_state, speed_states, ()),)
        )
        total = sum(s.probability for s in scenarios)
        if not scenarios or not isfinite(total) or total <= 0:
            raise ValueError(
                "search scenario probability mass must be positive and finite"
            )
        worlds = []
        for scenario in scenarios:
            if not isfinite(scenario.probability) or scenario.probability < 0:
                raise ValueError(
                    "search scenario probabilities must be finite and nonnegative"
                )
            if scenario.probability == 0:
                continue
            state = prepare_search_state(snapshot, scenario.state)
            worlds.append(
                _World(
                    scenario.probability / total,
                    state,
                    scenario.speed_states,
                    {**catalog, **scenario.move_sets},
                    public_moves,
                )
            )
        worlds_by_response.append(tuple(worlds))
    analysis = baseline
    completed_depth = 1
    exhausted = False
    for depth in range(2, config.depth + 1):
        cells = {}
        try:
            for action in our_actions:
                rows = []
                for parent, (estimate, worlds) in enumerate(
                    zip(opponent_estimates, worlds_by_response)
                ):
                    for world in worlds:
                        for row in engine.transitions(
                            world, action, estimate.candidate.action
                        ):
                            rows.append(
                                _Row(
                                    parent,
                                    world.probability * row.conditional_probability,
                                    replace(
                                        row.world,
                                        probability=(
                                            row.world.probability * estimate.probability
                                        ),
                                    ),
                                )
                            )
                values = engine.continuation_values(rows, depth - 1)
                pair_values = [0.0] * len(opponent_estimates)
                for row, value in zip(rows, values):
                    pair_values[row.parent] += row.conditional_probability * value
                for index, estimate in enumerate(opponent_estimates):
                    pair = next(
                        p
                        for p in baseline.pair_evaluations
                        if p.our_action == action
                        and p.opponent_action == estimate.candidate.action
                    )
                    cells[(action.key, estimate.candidate.action.key)] = (
                        pair_values[index] - pair.initial_score
                        if estimate.probability > 0
                        else pair.expected_score_delta
                    )
            summaries = build_response_matrix(
                our_actions,
                opponent_estimates,
                lambda ours, theirs: cells[(ours.key, theirs.key)],
            )
        except _BudgetExceeded as exc:
            exhausted = True
            engine.cutoffs[str(exc)] += 1
            break
        except (UnsupportedTurnMechanic, BranchLimitExceeded) as exc:
            reason = (
                "unsupported_mechanic"
                if isinstance(exc, UnsupportedTurnMechanic)
                else "turn_branch_limit"
            )
            engine.cutoffs[reason] += 1
            break
        # Keep one-turn distributions for the catastrophic immediate-loss guard.
        # Search values live in summaries; pair_evaluations stay one-turn evidence.
        analysis = MechanicsResponseAnalysis(summaries, baseline.pair_evaluations)
        completed_depth = depth
    return SearchAnalysis(
        analysis,
        SearchDiagnostics(
            config.depth,
            completed_depth,
            engine.simulations,
            engine.nodes,
            engine.hits,
            tuple(sorted(engine.cutoffs.items())),
            exhausted,
        ),
    )

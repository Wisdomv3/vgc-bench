"""Hidden-set hypothesis filtering and weighted sampling for Champions AI.

This module does not invent usage statistics. Each hypothesis carries an explicit
prior weight supplied by the caller. If no sourced prior is available, equal
weights are the safe default.

Revealed moves, items, and abilities are used to remove impossible hypotheses.
The remaining weights are then normalized into a posterior over still-plausible
sets.
"""

from dataclasses import dataclass
from random import Random

from vgc_bench.src.champions_ai.matchup import (
    CombatantProfile,
    MoveProfile,
    calculate_matchup_damage,
)
from vgc_bench.src.champions_ai.snapshot import PokemonSnapshot
from vgc_bench.src.champions_ai.spread import normalize_move_id


@dataclass(frozen=True)
class SetHypothesis:
    """One plausible exact build for an opponent Pokemon."""

    label: str
    profile: CombatantProfile
    moves: tuple[str, ...]
    prior_weight: float = 1.0
    source: str | None = None

    def __post_init__(self) -> None:
        if self.prior_weight < 0:
            raise ValueError("prior_weight cannot be negative")
        if not self.label:
            raise ValueError("label cannot be empty")

    @property
    def normalized_moves(self) -> frozenset[str]:
        return frozenset(normalize_move_id(move) for move in self.moves)


@dataclass(frozen=True)
class WeightedHypothesis:
    """A still-plausible hypothesis with normalized probability."""

    hypothesis: SetHypothesis
    probability: float


@dataclass(frozen=True)
class HypothesisDamage:
    """Damage output for one hidden-set hypothesis."""

    hypothesis: SetHypothesis
    probability: float
    minimum: int
    maximum: int
    expected_damage: float
    ko_probability: float


@dataclass(frozen=True)
class HiddenDamageEstimate:
    """Weighted damage summary over all currently plausible hidden sets."""

    hypotheses: tuple[HypothesisDamage, ...]

    @property
    def minimum(self) -> int:
        if not self.hypotheses:
            raise ValueError("no compatible hypotheses")
        return min(result.minimum for result in self.hypotheses)

    @property
    def maximum(self) -> int:
        if not self.hypotheses:
            raise ValueError("no compatible hypotheses")
        return max(result.maximum for result in self.hypotheses)

    @property
    def expected_damage(self) -> float:
        return sum(
            result.probability * result.expected_damage
            for result in self.hypotheses
        )

    @property
    def ko_probability(self) -> float:
        return sum(
            result.probability * result.ko_probability
            for result in self.hypotheses
        )


def _same_known_value(revealed: str | None, candidate: str | None) -> bool:
    if revealed is None:
        return True
    if candidate is None:
        return False
    return normalize_move_id(revealed) == normalize_move_id(candidate)


def is_compatible(
    snapshot: PokemonSnapshot,
    hypothesis: SetHypothesis,
) -> bool:
    """Return whether a hypothesis matches everything revealed so far."""

    revealed_moves = {
        normalize_move_id(move)
        for move in snapshot.revealed_moves
    }
    if not revealed_moves.issubset(hypothesis.normalized_moves):
        return False

    if not _same_known_value(snapshot.item, hypothesis.profile.item):
        return False

    if not _same_known_value(snapshot.ability, hypothesis.profile.ability):
        return False

    return True


def posterior_hypotheses(
    snapshot: PokemonSnapshot,
    hypotheses: tuple[SetHypothesis, ...],
) -> tuple[WeightedHypothesis, ...]:
    """Filter impossible sets and normalize the remaining prior weights."""

    compatible = tuple(
        hypothesis
        for hypothesis in hypotheses
        if is_compatible(snapshot, hypothesis)
    )
    if not compatible:
        return ()

    total_weight = sum(hypothesis.prior_weight for hypothesis in compatible)
    if total_weight <= 0:
        equal = 1.0 / len(compatible)
        return tuple(
            WeightedHypothesis(hypothesis, equal)
            for hypothesis in compatible
        )

    return tuple(
        WeightedHypothesis(
            hypothesis=hypothesis,
            probability=hypothesis.prior_weight / total_weight,
        )
        for hypothesis in compatible
    )


def sample_hypotheses(
    snapshot: PokemonSnapshot,
    hypotheses: tuple[SetHypothesis, ...],
    *,
    count: int,
    seed: int | None = None,
) -> tuple[SetHypothesis, ...]:
    """Draw reproducible weighted samples from the current posterior."""

    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return ()

    posterior = posterior_hypotheses(snapshot, hypotheses)
    if not posterior:
        return ()

    rng = Random(seed)
    population = [entry.hypothesis for entry in posterior]
    weights = [entry.probability for entry in posterior]

    return tuple(rng.choices(population, weights=weights, k=count))


def evaluate_hidden_defender(
    attacker: CombatantProfile,
    defender_snapshot: PokemonSnapshot,
    defender_hypotheses: tuple[SetHypothesis, ...],
    move: MoveProfile,
    *,
    weather: str | None = None,
    critical: bool = False,
) -> HiddenDamageEstimate:
    """Evaluate damage across every compatible defender build.

    Each hypothesis profile must contain the current HP value to use for its KO
    probability. A later state-sync layer will update candidate profiles from
    observed HP information.
    """

    posterior = posterior_hypotheses(
        defender_snapshot,
        defender_hypotheses,
    )

    results: list[HypothesisDamage] = []
    for entry in posterior:
        defender = entry.hypothesis.profile
        damage = calculate_matchup_damage(
            attacker,
            defender,
            move,
            weather=weather,
            critical=critical,
        )

        results.append(
            HypothesisDamage(
                hypothesis=entry.hypothesis,
                probability=entry.probability,
                minimum=damage.minimum,
                maximum=damage.maximum,
                expected_damage=sum(damage.rolls) / len(damage.rolls),
                ko_probability=damage.ko_probability(defender.current_hp),
            )
        )

    return HiddenDamageEstimate(tuple(results))


def evaluate_hidden_attacker(
    attacker_snapshot: PokemonSnapshot,
    attacker_hypotheses: tuple[SetHypothesis, ...],
    defender: CombatantProfile,
    move: MoveProfile,
    *,
    weather: str | None = None,
    critical: bool = False,
) -> HiddenDamageEstimate:
    """Evaluate incoming damage across plausible opponent attacker builds."""

    posterior = posterior_hypotheses(
        attacker_snapshot,
        attacker_hypotheses,
    )

    results: list[HypothesisDamage] = []
    for entry in posterior:
        attacker = entry.hypothesis.profile
        damage = calculate_matchup_damage(
            attacker,
            defender,
            move,
            weather=weather,
            critical=critical,
        )

        results.append(
            HypothesisDamage(
                hypothesis=entry.hypothesis,
                probability=entry.probability,
                minimum=damage.minimum,
                maximum=damage.maximum,
                expected_damage=sum(damage.rolls) / len(damage.rolls),
                ko_probability=damage.ko_probability(defender.current_hp),
            )
        )

    return HiddenDamageEstimate(tuple(results))

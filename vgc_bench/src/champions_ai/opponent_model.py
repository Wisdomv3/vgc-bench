"""Opponent action priors and habit-aware probability estimates.

This module separates three ideas:

1. Baseline priors supplied by data or by an explicitly neutral prior.
2. Observed opponent habits from comparable decision contexts.
3. A caller-chosen blend between baseline and personal-history evidence.

The blend is intentionally explicit so Champions AI does not manufacture
confidence from a tiny sample. Later calibration can learn the blend strength
from replay data.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.spread import is_spread_move, normalize_move_id


PROTECT_LIKE_MOVES = {
    "protect",
    "detect",
    "endure",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "burningbulwark",
    "obstruct",
    "maxguard",
    "silktrap",
}


@dataclass(frozen=True)
class OpponentActionCandidate:
    """One plausible opponent joint action with an explicit prior weight."""

    action: JointAction
    prior_weight: float = 1.0
    behavior_key: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.prior_weight < 0:
            raise ValueError("prior_weight cannot be negative")

    @property
    def behavior(self) -> str:
        return self.behavior_key or behavior_key_from_action(self.action)


@dataclass(frozen=True)
class ActionProbability:
    """Probability estimate for one candidate opponent action."""

    candidate: OpponentActionCandidate
    probability: float
    baseline_probability: float
    empirical_probability: float | None
    comparable_observations: int


@dataclass
class OpponentHabitTracker:
    """Counts behavior choices in explicitly comparable decision contexts."""

    counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def record(self, context_key: str, behavior_key: str) -> None:
        if not context_key:
            raise ValueError("context_key cannot be empty")
        if not behavior_key:
            raise ValueError("behavior_key cannot be empty")
        self.counts[context_key][behavior_key] += 1

    def record_action(self, context_key: str, action: JointAction) -> None:
        self.record(context_key, behavior_key_from_action(action))

    def behavior_counts(self, context_key: str) -> Counter[str]:
        return Counter(self.counts.get(context_key, Counter()))


def _slot_behavior(action: SlotAction) -> str:
    if action.kind is ActionKind.SWITCH:
        return "switch"

    if action.kind in {ActionKind.PASS, ActionKind.DEFAULT}:
        return "passive"

    if action.kind is not ActionKind.MOVE or action.move is None:
        return "other"

    move_id = normalize_move_id(action.move)
    if move_id in PROTECT_LIKE_MOVES:
        return "protect"

    if is_spread_move(action.move):
        return "spread"

    if action.target is not None or action.target_position in (-2, -1, 1, 2):
        return "targeted_move"

    return "support"


def behavior_key_from_action(action: JointAction) -> str:
    """Return a coarse, order-independent behavior category for a joint action."""

    parts = sorted((_slot_behavior(action.first), _slot_behavior(action.second)))
    return "+".join(parts)


def action_features(action: JointAction) -> frozenset[str]:
    """Return descriptive features that can be displayed as opponent tendencies."""

    features: set[str] = set()
    slot_actions = (action.first, action.second)

    for slot_action in slot_actions:
        behavior = _slot_behavior(slot_action)
        features.add(behavior)

        if slot_action.target:
            features.add(f"target:{normalize_move_id(slot_action.target)}")

        if (
            slot_action.kind is ActionKind.MOVE
            and slot_action.move is not None
            and is_spread_move(slot_action.move)
        ):
            features.add("spread_pressure")

    slot_behaviors = [_slot_behavior(slot_action) for slot_action in slot_actions]
    if slot_behaviors.count("protect") == 2:
        features.add("double_protect")
    if slot_behaviors.count("switch") == 2:
        features.add("double_switch")
    if all(
        behavior in {"targeted_move", "spread"}
        for behavior in slot_behaviors
    ):
        features.add("double_pressure")

    return frozenset(features)


def _normalize_weights(
    candidates: tuple[OpponentActionCandidate, ...],
) -> tuple[float, ...]:
    if not candidates:
        return ()

    total = sum(candidate.prior_weight for candidate in candidates)
    if total <= 0:
        equal = 1.0 / len(candidates)
        return tuple(equal for _ in candidates)

    return tuple(candidate.prior_weight / total for candidate in candidates)


def estimate_action_probabilities(
    candidates: tuple[OpponentActionCandidate, ...],
    *,
    tracker: OpponentHabitTracker | None = None,
    context_key: str | None = None,
    habit_weight: float = 0.0,
) -> tuple[ActionProbability, ...]:
    """Estimate opponent action probabilities without hiding the evidence source.

    habit_weight is the share of the final distribution assigned to this
    opponent's observed choices in the same context. A value of 0 uses only the
    baseline prior. A value of 1 uses only comparable observed behavior.

    If there are no relevant observations, the baseline prior is returned even
    when habit_weight is nonzero.
    """

    if not 0 <= habit_weight <= 1:
        raise ValueError("habit_weight must be between 0 and 1")

    if not candidates:
        return ()

    baseline = _normalize_weights(candidates)
    behavior_keys = tuple(candidate.behavior for candidate in candidates)

    relevant_counts: Counter[str] = Counter()
    if tracker is not None and context_key:
        all_counts = tracker.behavior_counts(context_key)
        current_behaviors = set(behavior_keys)
        relevant_counts = Counter(
            {
                behavior: count
                for behavior, count in all_counts.items()
                if behavior in current_behaviors and count > 0
            }
        )

    observation_count = sum(relevant_counts.values())
    empirical_by_action: list[float] | None = None

    if observation_count > 0:
        empirical_behavior = {
            behavior: count / observation_count
            for behavior, count in relevant_counts.items()
        }

        baseline_mass_by_behavior: dict[str, float] = defaultdict(float)
        for probability, behavior in zip(baseline, behavior_keys):
            baseline_mass_by_behavior[behavior] += probability

        empirical_by_action = []
        for probability, behavior in zip(baseline, behavior_keys):
            behavior_probability = empirical_behavior.get(behavior, 0.0)
            behavior_baseline_mass = baseline_mass_by_behavior[behavior]

            if behavior_baseline_mass <= 0:
                empirical_by_action.append(0.0)
            else:
                empirical_by_action.append(
                    behavior_probability * probability / behavior_baseline_mass
                )

    output: list[ActionProbability] = []
    for index, candidate in enumerate(candidates):
        empirical = (
            None
            if empirical_by_action is None
            else empirical_by_action[index]
        )

        if empirical is None:
            final_probability = baseline[index]
        else:
            final_probability = (
                (1.0 - habit_weight) * baseline[index]
                + habit_weight * empirical
            )

        output.append(
            ActionProbability(
                candidate=candidate,
                probability=final_probability,
                baseline_probability=baseline[index],
                empirical_probability=empirical,
                comparable_observations=observation_count,
            )
        )

    return tuple(
        sorted(
            output,
            key=lambda result: (
                -result.probability,
                result.candidate.action.label,
            ),
        )
    )


def feature_probability(
    estimates: tuple[ActionProbability, ...],
    feature: str,
) -> float:
    """Return the marginal probability of a displayed action feature."""

    return sum(
        estimate.probability
        for estimate in estimates
        if feature in action_features(estimate.candidate.action)
    )

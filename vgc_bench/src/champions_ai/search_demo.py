"""Run with python -m vgc_bench.src.champions_ai.search_demo.

A small controlled mechanics example, with synthetic stats and move powers.
It demonstrates planning; it is not a live battle or a sourced Pokemon team.
"""

from vgc_bench.src.champions_ai.actions import ActionKind, JointAction, SlotAction
from vgc_bench.src.champions_ai.decision_pipeline import rank_decision
from vgc_bench.src.champions_ai.live_output import render_live_output
from vgc_bench.src.champions_ai.matchup import CombatantProfile, MoveProfile
from vgc_bench.src.champions_ai.opponent_model import OpponentActionCandidate
from vgc_bench.src.champions_ai.search import SearchConfig
from vgc_bench.src.champions_ai.snapshot import (
    DecisionSnapshot,
    PokemonSnapshot,
    SideSnapshot,
)
from vgc_bench.src.champions_ai.speed_context import speed_states_from_exact_state
from vgc_bench.src.champions_ai.turn_branching import BranchingPolicy
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


def demo_inputs():
    """Reusable synthetic pre-turn inputs for search and coaching demos."""
    player, opponent = TurnSide.PLAYER, TurnSide.OPPONENT

    def profile(name, attack, defense, speed):
        return CombatantProfile(
            name,
            50,
            100,
            100,
            ("normal",),
            {
                "atk": attack,
                "def": defense,
                "spa": attack,
                "spd": defense,
                "spe": speed,
            },
        )

    def move(slot, actor, name, target=None):
        return SlotAction(
            slot, ActionKind.MOVE, actor=actor, move=name, target_position=target
        )

    state = ExactTurnState(
        profiles={
            (player, "hero"): profile("hero", 600, 100, 80),
            (player, "ally"): profile("ally", 300, 100, 70),
            (opponent, "threat"): profile("threat", 200, 400, 120),
            (opponent, "weak"): profile("weak", 1, 100, 30),
        },
        active_slots={
            (player, 0): "hero",
            (player, 1): "ally",
            (opponent, 0): "threat",
            (opponent, 1): "weak",
        },
        must_recharge={(opponent, "threat")},
        known_moves={
            (player, "hero"): {"tackle", "protect"},
            (player, "ally"): {"tackle", "protect", "tailwind"},
            (opponent, "threat"): {"bodyslam"},
            (opponent, "weak"): {"tackle"},
        },
    )

    def side_snapshot(side):
        return SideSnapshot(
            tuple(
                PokemonSnapshot(
                    name,
                    100,
                    None,
                    False,
                    slot,
                    tuple(sorted(state.known_moves[(side, name)])),
                    None,
                    None,
                    (),
                    0,
                    False,
                )
                for (active_side, slot), name in state.active_slots.items()
                if active_side is side
            ),
            tuple(
                (slot, name)
                for (active_side, slot), name in state.active_slots.items()
                if active_side is side
            ),
            0,
            (),
        )

    snapshot = DecisionSnapshot(
        1, side_snapshot(player), side_snapshot(opponent), None, None, 0, (), 0
    )
    immediate = JointAction(move(0, "hero", "tackle", 2), move(1, "ally", "protect"))
    setup = JointAction(move(0, "hero", "protect"), move(1, "ally", "tailwind"))
    theirs = JointAction(move(0, "threat", "bodyslam", 1), move(1, "weak", "tackle", 2))
    profiles = {
        (player, "hero", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
        (player, "ally", "tackle"): MoveProfile("tackle", 200, "physical", "normal"),
        (opponent, "threat", "bodyslam"): MoveProfile(
            "bodyslam", 200, "physical", "normal"
        ),
        (opponent, "weak", "tackle"): MoveProfile("tackle", 1, "physical", "normal"),
    }
    policy = BranchingPolicy(
        branch_damage_rolls=False,
        branch_accuracy=False,
        branch_critical_hits=False,
        branch_secondary_effects=False,
        branch_before_move_status=False,
        branch_protect=False,
        branch_speed_ties=False,
        fixed_damage_roll_index=15,
    )
    return snapshot, state, immediate, setup, theirs, profiles, policy


def main() -> None:
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    for depth in (1, 2):
        report = rank_decision(
            snapshot,
            state,
            (immediate, setup),
            (OpponentActionCandidate(theirs),),
            speed_states_from_exact_state(state),
            profiles,
            branching_policy=policy,
            search_config=SearchConfig(depth=depth, max_observation_branches=None),
        )
        print(f"\n{depth}-turn search (synthetic fixed-RNG demonstration)")
        print(render_live_output(report))
        print(f"Diagnostics: {report.search_diagnostics}")


if __name__ == "__main__":
    main()

"""Run: python -m vgc_bench.src.champions_ai.coaching_demo.

Synthetic fixed-RNG example, not a live team or calibrated statistical model.
Propose the immediate KO before the turn and see why two-turn search favors
Protect + Tailwind. No battle is played or later observation read by this demo.
"""

from vgc_bench.src.champions_ai.coaching import coach_decision, render_coaching_output
from vgc_bench.src.champions_ai.opponent_model import OpponentActionCandidate
from vgc_bench.src.champions_ai.search import SearchConfig
from vgc_bench.src.champions_ai.search_demo import demo_inputs
from vgc_bench.src.champions_ai.speed_context import speed_states_from_exact_state


def main() -> None:
    snapshot, state, immediate, setup, theirs, profiles, policy = demo_inputs()
    report = coach_decision(
        snapshot,
        state,
        (immediate, setup),
        (OpponentActionCandidate(theirs, source="synthetic single-response demo"),),
        speed_states_from_exact_state(state),
        profiles,
        chosen_action=immediate,
        branching_policy=policy,
        search_config=SearchConfig(depth=2, max_observation_branches=None),
    )
    print("Synthetic fixed-RNG coaching demonstration\n")
    print(render_coaching_output(report))


if __name__ == "__main__":
    main()

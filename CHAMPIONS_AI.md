# Champions AI v1

Goal: adapt VGC-Bench into a fast Pokemon Champions doubles decision engine that ranks legal joint actions by expected battle win probability.

## Hard live-battle rules

Before recommending any action, evaluate in this order:

1. Exact active board state and HP.
2. Potential spread damage and possible double-KO.
3. Effective speed order with Tailwind, Trick Room, Unburden, weather abilities, priority, and boosts.
4. Fake Out eligibility and priority legality.
5. Terrain, weather, redirection, and priority-blocking effects.
6. Protect history.
7. Residual damage, recoil, Rocky Helmet/contact punishment, and status.
8. Exact damage ranges, KO probabilities, and combined-damage lines.
9. Offensive KO lines.

## v1 milestones

- [ ] Run the fork locally with its pinned Pokemon Showdown submodule.
- [ ] Confirm VGC-Bench training/play modules start successfully.
- [ ] Add the user's Pokemon Champions team.
- [ ] Build a battle-state tracker.
- [ ] Build spread-damage threat detection.
- [ ] Build exact damage-calculation layer and verify it against established damage-calculator outputs.
- [ ] Build effective-speed calculation.
- [ ] Build Fake Out / Protect / field-condition trackers.
- [ ] Enumerate legal joint actions.
- [ ] Add rule-based blunder guards.
- [ ] Rank candidate actions with a heuristic scorer.
- [ ] Benchmark against VGC-Bench heuristic agents.
- [ ] Add behavior cloning and self-play.
- [ ] Add hidden-set sampling and deeper search.

## Important design principle

The engine should not choose the move with the highest immediate damage. It should choose the legal joint action with the highest estimated probability of winning the entire battle.

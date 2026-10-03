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

- [x] Run the fork locally with its pinned Pokemon Showdown submodule.
- [x] Confirm VGC-Bench training/play modules start successfully.
- [ ] Add the user's Pokemon Champions team.
- [x] Build a universal battle-state tracker.
- [x] Build initial spread-damage threat detection.
- [x] Add Pokemon Showdown structured-state input adapter.
- [ ] Build exact damage-calculation layer and verify it against established damage-calculator outputs.
- [x] Build initial effective-speed calculation.
- [ ] Build Fake Out / Protect / field-condition trackers.
- [x] Enumerate legal joint actions.
- [ ] Add rule-based catastrophic-error guards.
- [x] Rank candidate actions with an initial transparent heuristic scorer.
- [ ] Benchmark against VGC-Bench heuristic agents.
- [ ] Add behavior cloning and self-play.
- [ ] Add hidden-set sampling and deeper search.

## Important design principle

The engine should not choose the move with the highest immediate damage. It should choose the legal joint action with the highest estimated probability of winning the entire battle.


## Decision grading scale

The coaching/review system grades each joint action using only the information
available at that decision point. It must not use later battle information to
re-grade an earlier turn.

Best to worst:

1. **Stupendous**
2. **Amazing**
3. **Outstanding**
4. **Awesome**
5. **Great**
6. **Good**
7. **Ok**
8. **Mistake**
9. **Miss**
10. **Throwing**

Grades should primarily reflect the estimated loss in match win probability
between the player's chosen joint action and the strongest action the engine
could identify from the same decision snapshot. Exact thresholds should be
calibrated from simulation and benchmark data rather than chosen arbitrarily.

A strong decision that receives an unlucky outcome should retain its strong
decision grade. Outcome variance, such as critical hits, misses, flinches,
damage rolls, and speed ties, should be reported separately from decision
quality.

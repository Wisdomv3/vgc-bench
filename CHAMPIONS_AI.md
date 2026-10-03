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
- [ ] Build full exact damage-calculation layer and verify matchup outputs against established damage calculators.
- [x] Build exact generation-9 core damage arithmetic and KO-probability utilities.
- [x] Add initial exact matchup resolver for stats, stages, STAB, type effectiveness, weather, burn, and Life Orb.
- [x] Build initial effective-speed calculation.
- [ ] Build Fake Out / Protect / field-condition trackers.
- [x] Track first-turn Fake Out eligibility across universal state snapshots and Showdown input.
- [x] Enumerate legal joint actions.
- [x] Add initial catastrophic-error guards for impossible Fake Out, repeated Protect risk, strictly dominated lines, and guaranteed immediate match loss.
- [ ] Expand catastrophic guards for priority blocking, redirection, Wide Guard, recoil/contact punishment, and other format-specific failure modes.
- [x] Add Psychic Terrain priority blocking, grounded checks, Wide Guard spread protection, Follow Me/Rage Powder redirection, and common redirection bypasses to the turn simulator.
- [x] Add move recoil, Life Orb recoil, Rough Skin/Iron Barbs, Rocky Helmet, Protective Pads, Focus Sash survival, and end-of-turn residual/healing mechanics.
- [x] Add before-move paralysis, sleep, and freeze mechanics with exact branching, sleep counters, Early Bird, sleep-usable moves, defrost moves, and thaw-on-hit interactions.
- [x] Add confusion duration/self-hit branching, Own Tempo handling, and Hyper Beam/Giga Impact-style recharge turns.
- [ ] Add remaining volatile action-denial mechanics such as Truant, attraction, locked/restricted moves, and format-relevant edge cases.
- [x] Rank candidate actions with an initial transparent heuristic scorer.
- [ ] Benchmark against VGC-Bench heuristic agents.
- [ ] Add behavior cloning and self-play.
- [ ] Add hidden-set sampling and deeper search.
- [x] Add data-agnostic hidden-set hypothesis filtering, weighted sampling, and weighted damage estimates.
- [ ] Connect hidden-set priors to sourced usage/replay data instead of hand-entered weights.
- [x] Add a transparent opponent-action probability model with context-specific habit tracking.
- [x] Add a turn-response matrix that evaluates each of our legal actions across weighted opponent responses.
- [x] Add initial turn action ordering for switches, priority, Speed, Tailwind, Trick Room, and explicit Speed ties.
- [x] Add deterministic turn simulator core for switches, Protect, targeted/spread damage, faint cancellation, Tailwind, Trick Room, Prankster, and Grassy Glide.
- [x] Add probabilistic turn branching for damage rolls, accuracy, Speed ties, repeated Protect odds, and critical hits.
- [x] Add probabilistic flinches and common damaging-move secondary effects, including status, stat drops, Dire Claw, Inner Focus, Covert Cloak, and Defiant/Competitive reactions.
- [ ] Expand secondary-effect immunities and ability/item interactions to full format coverage.
- [x] Add a mechanics-based turn outcome evaluator that converts weighted simulated outcomes into transparent position-score changes and plugs them into the response matrix.
- [ ] Replace the temporary position heuristic with a calibrated match-value / win-probability model.
- [ ] Learn/calibrate opponent-action priors and habit blending from replay data.

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

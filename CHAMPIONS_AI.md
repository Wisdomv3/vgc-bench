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
- [x] Add Truant loaf turns, Choice-item/Gorilla Tactics move locks, and cant-use-twice restrictions such as Gigaton Hammer/Blood Moon.
- [x] Add Disable, Taunt, Encore action override, and Imprison shared-move restrictions with volatile duration tracking.
- [x] Add Torment repeat-move restriction, Attract gender/Oblivious handling, exact 50% immobilization branching, PP tracking with Pressure, and Struggle fallback/recoil.
- [x] Synchronize live Showdown observations into simulator templates, including HP scaling, boosts, types, gender, PP, last move, recharge, protection streaks, field state, and supported restriction volatiles.
- [ ] Add remaining selection edge cases, exact volatile-source/history synchronization, and other format-relevant move restrictions.
- [x] Rank candidate actions with an initial transparent heuristic scorer.
- [ ] Benchmark against VGC-Bench heuristic agents.
- [ ] Add behavior cloning and self-play.
- [x] Add hidden-set scenarios and bounded multi-turn search over supported mechanics.
- [x] Add data-agnostic hidden-set hypothesis filtering, weighted sampling, and weighted damage estimates.
- [ ] Connect hidden-set priors to sourced usage/replay data instead of hand-entered weights.
- [x] Add a transparent opponent-action probability model with context-specific habit tracking.
- [x] Harden opponent-action probability normalization, including zero-baseline empirical behaviors and non-finite prior rejection.
- [x] Automatically generate plausible opponent joint actions from the live board, revealed moves/PP, revealed switches, public move restrictions, and compatible hidden-set hypotheses.
- [x] Propagate hidden-set uncertainty through mechanics simulation, conditioning on modeled opponent actions and weighting exact stats, HP, item, ability, Speed, damage, and turn order across compatible sets.
- [x] Add a turn-response matrix that evaluates each of our legal actions across weighted opponent responses.
- [x] Build an end-to-end decision pipeline combining live state, legal actions, opponent probabilities, probabilistic mechanics simulation, catastrophic guards, and ranked recommendations.
- [x] Add a compact BEST PLAY live-output formatter with two slot actions, opponent-response watch line, and no more than two concise reasons.
- [x] Add a secondary pre-turn coaching report, provisional ten-label grading, explicit one-turn event probabilities, habit evidence, and a runnable demo.
- [ ] Connect coaching to the live player input/display and calibrate decision grades and match-win probabilities from battle data.
- [x] Add initial turn action ordering for switches, priority, Speed, Tailwind, Trick Room, and explicit Speed ties.
- [x] Add deterministic turn simulator core for switches, Protect, targeted/spread damage, faint cancellation, Tailwind, Trick Room, Prankster, and Grassy Glide.
- [x] Add probabilistic turn branching for damage rolls, accuracy, Speed ties, repeated Protect odds, and critical hits.
- [x] Add probabilistic flinches and common damaging-move secondary effects, including status, stat drops, Dire Claw, Inner Focus, Covert Cloak, and Defiant/Competitive reactions.
- [ ] Expand secondary-effect immunities and ability/item interactions to full format coverage.
- [x] Add a mechanics-based turn outcome evaluator that converts weighted simulated outcomes into transparent position-score changes and plugs them into the response matrix.
- [ ] Replace the temporary position heuristic with a calibrated match-value / win-probability model.
- [ ] Learn/calibrate opponent-action priors and habit blending from replay data.

## Multi-turn search checkpoint

Pass `search_config=SearchConfig(depth=2)` to `rank_decision` or
`rank_showdown_decision` to search this turn plus the next turn. The import is
`from vgc_bench.src.champions_ai.search import SearchConfig`.
Omitting the argument preserves the original one-turn ranking.

Search keeps weighted hidden worlds fixed, combines opponent responses with
enabled battle RNG, groups indistinguishable public observations, and selects
one continuation per information set. It uses iterative deepening and full-state
caches. Limits apply between turn simulations, and incomplete iterations retain
the last completed depth. The optional observation-branch cap keeps rare branch
probability mass and evaluates those branches at their shorter leaf horizon.

`DecisionReport.search_diagnostics` reports requested/completed depth, simulation
count, cache hits, budget exhaustion, and early-cutoff counts. Live output shows
the horizon and any shorter-horizon caveat. Opponent continuation priors are
explicit neutral priors with a configurable switch weight, not learned optimal
opponent play. Root priors and habits continue to use the existing model.

Supported field timers, entry history/Fake Out, PP, status, recharge, restriction
counters, items, and Speed survive the simulated transitions. Active Tailwind
cannot be refreshed, and ordinary switches reset stat boosts. Known Tailwind
and Trick Room durations are copied from the decision snapshot. Weather/terrain
durations must be supplied when known; unknown field durations stop deeper
search. Gravity and other fields without supported timers also stop it.

Forced replacements, missing move information/damage models, and unsupported
mechanics stop at labelled leaves. This version does not implement a complete
replacement phase, full switch-entry effects/trapping, new gimmick use on future
turns, full activation/order-based Bayesian observation inference, or calibrated
match win probabilities. Observation grouping is deliberately conservative.
Terminal detection uses the supplied modeled roster; unseen bench Pokemon are
not invented. Scores are still the existing transparent position heuristic.

Run `python -m vgc_bench.src.champions_ai.search_demo` for the controlled example.
It uses synthetic stats/powers and fixed RNG: one-turn search takes an immediate
KO, while two-turn search chooses Protect plus Tailwind to win more afterward.

Next milestones: model forced-replacement decisions and switch-entry effects,
expand exact mechanic coverage, improve public observation inference, benchmark
search depth/caps, and calibrate the value function and opponent priors.

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

## Pre-turn coaching checkpoint

Run `python -m vgc_bench.src.champions_ai.coaching_demo` for a synthetic
fixed-RNG demonstration. It evaluates a proposed immediate KO before the turn,
then explains why two-turn search favors Protect plus Tailwind. It does not
play a live battle. The same decision engine serves the bot and the coach.

Use `coach_decision` with the same inputs as `rank_decision`, or
`coach_showdown_decision` with the same inputs as `rank_showdown_decision`:

```python
from vgc_bench.src.champions_ai.coaching import (
    coach_showdown_decision,
    render_coaching_output,
)
from vgc_bench.src.champions_ai.search import SearchConfig

coaching = coach_showdown_decision(
    battle,
    profiles,
    move_profiles,
    chosen_action=proposed_joint_action,
    search_config=SearchConfig(depth=2, max_simulations=2000),
    tracker=opponent_habits,
    context_key=current_context,
    habit_weight=0.3,  # Explicit prototype blend, not a learned confidence.
)
print(render_coaching_output(coaching))
```

The wrappers default to bounded depth-two search. Pass `SearchConfig(depth=1)`
for one-turn coaching. Omit `chosen_action` for advice before choosing a plan.
The proposed action must be in the evaluated legal joint-action set. The
comparison concerns the two actions together, not two independent move grades.

Call the decision engine before the current turn resolves, and before recording
the opponent's current response in the habit tracker. Only already revealed
information, prior habit observations and explicit hidden-set hypotheses belong
in the inputs. The coaching builder accepts the saved `DecisionReport`, never
a later battle state or tracker. Frozen snapshots, metrics and habit estimates
preserve the original grade if later results or observations change. Input
collection is still responsible for respecting this timing boundary.

For an existing decision call, pass `include_coaching_metrics=True` to
`rank_decision` / `rank_showdown_decision`, then use
`build_coaching_report(report, proposed_joint_action)`. Reusing that report for
another evaluated plan does not rerun search or obtain later information. Normal
bot calls retain their existing behavior and do not aggregate coaching metrics.

### Probabilities and evidence

The report separates the multi-turn expected position score from **this-turn**
weighted event odds: at least one new opposing KO, at least one new own faint,
survival of the starting active Pokemon, and Tailwind active at turn end. It also
exposes individual opposing KO and starting-player survival probabilities. A
successful switch counts as survival, even if that Pokemon ends on the bench.
Previously fainted Pokemon are excluded from new-KO events. Forced-replacement
and switch-entry limitations of the underlying simulator still apply.

These are conditional model estimates: opponent response probabilities are
multiplied by hidden-world and enabled-RNG outcome probabilities. Missing
simulations or incomplete/non-finite probability mass cause an error. Events
are computed from final states, so merging equivalent states is safe. Hit/miss
or Protect activation probabilities are not inferred from representative logs,
which can lose event history during state merging. Disabled RNG branches are
identified explicitly. There is no undifferentiated "move success rate" and no
position-score-to-win-percent conversion.

Opponent reads group the existing joint behavior categories and report their
modeled probability, baseline weight, observed count / comparable observation
count, explicit habit blend, and candidate sources. No history means baseline
weights only. Future search responses currently use neutral continuation priors;
root habit evidence does not make those future priors learned optimal play.

### Provisional grading

Until calibration, every assigned grade is marked **provisional**. It measures
the expected position-score loss versus the best nonblocked action found among
the evaluated candidates. It does not measure actual battle-win loss or prove
that the strongest possible action was found. The top label means zero modeled
score loss, rather than demonstrated tactical brilliance. Ties receive the same
grade; adding a weaker candidate cannot inflate another plan's grade.

The prototype reference unit is `pokemon_alive + total_hp_fraction` from the
active `PositionWeights`: 125 points with the default weights. The default
`ProvisionalGradeRubric` uses the following explicit tuning bands. These bands
are development settings, not empirically calibrated thresholds; replace them
after replay/simulation benchmarks and value-model calibration.

| Grade | Maximum loss in reference units | Default maximum point loss |
| --- | ---: | ---: |
| Stupendous | 0 | 0 |
| Amazing | 0.02 | 2.5 |
| Outstanding | 0.05 | 6.25 |
| Awesome | 0.10 | 12.5 |
| Great | 0.20 | 25 |
| Good | 0.35 | 43.75 |
| Ok | 0.50 | 62.5 |
| Mistake | 0.75 | 93.75 |
| Miss | 1.00 | 125 |
| Throwing | Greater than 1.00 | Greater than 125 |

An unsimulated rule-blocked action is explained but unrated. A simulated
dominated action retains its comparison score and provisional grade. Fewer
than two simulated plans, or no nonblocked recommendation, leave grades
unavailable. Search budget exhaustion, shorter-horizon leaves and fixed RNG
assumptions are displayed rather than hidden behind a grade. The next work is
live input/display integration, fuller mechanics coverage, value and grade
calibration, and replay-based validation of opponent reads.

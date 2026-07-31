# Phase III assumptions (brief §7, Appendix A)

Appendix A requires every parameter to have a source, a rationale, or an explicit
assumption label. This file is the list of the ones that have no source — the
decisions made while implementing the learner engine that the brief does not
determine.

Hand-written. `docs/learner_equations.md` is generated from the docstrings and is
the companion to this file: it says what each equation *is*, this one says which
parts of it were chosen rather than given.

**The parameter values themselves are all assumptions.** §7.2 says so
explicitly — "the baseline parameter values must be treated as assumptions" — so
`config/learners.yaml` carries a low / medium / high arm for every free numeric
parameter and `--setting` selects which. Nothing below repeats that; it lists the
places where a *structural* choice, not just a value, had to be invented.

Every quantity here describes a **simulated** learner from a declared generative
model. None of it is a measurement of a person, none of it is a causal effect, and
Phase III computes no neural quantity at all.

---

## A note on where these came from

The project brief is not in this repository. Equations 15–20 and 26–27 were
supplied with the Phase III task; equations 21–25 were supplied verbatim by the
supervisor during the build, and are implemented literally. Two things in them are
easy to misread and are worth stating plainly:

- **`M` is memory strength** — a consolidation trace with its own decay
  `delta_M,i`. Not motivation, not metacognition.
- **`R` is independent reasoning**, not retrieval. Retrieval is an *input* to
  eq. (22), not a state.

---

## 1. The difficulty → `b_u` map

**ASSUMPTION.** §5.1 item 5 declares `difficulty` as an integer 1–5. Equations
(17)–(18) need a difficulty on the logit scale. The brief never states what that
scale is.

```text
b_u = b_intercept + b_slope * (difficulty - difficulty_center)
b_intercept = 0.0,  b_slope = 0.6,  difficulty_center = 3.0
```

Rationale: the simplest monotone map. A mid-difficulty unit sits at `b_u = 0`, and
one difficulty step is 0.6 logits — roughly a 14-point swing in accuracy near
p = 0.5. `CurriculumConfig` enforces `b_slope > 0`, which is what makes "a harder
unit is a harder unit" a property of the code rather than of the numbers.

**Open question for the supervisor.** Nothing anchors 0.6. If the corpus ever
carries an empirical difficulty estimate, this map should be fitted rather than
declared.

Implemented in [`curriculum.difficulty_to_b_u`](../src/learners/curriculum.py).

## 2. The `K` → `theta` map

**ASSUMPTION.** The brief puts ability on the logit scale but does not say how it
relates to knowledge.

```text
theta = theta_intercept + theta_slope * (K - theta_center)
theta_intercept = 0.0,  theta_slope = 4.0,  theta_center = 0.5
```

Rationale: linear and monotone, centred so that `K = 0.5` is the origin of the
ability scale. `K` enters eq. (17) through this channel **only** — `M` and `R`
have their own terms — so nothing is double-counted.

## 3. `omega`, the support boost in eq. (18)

**ASSUMPTION**, and a supervisor question. `omega = 2.5` at the medium setting.
Nothing in the brief or in data anchors it, and it directly scales the eq. (26)
support gap, which is one of the study's headline quantities.

`ResponseConfig` refuses `omega <= 0`: a non-positive value would make support
harmful or inert *by construction*, which is a modelling claim rather than a
parameter value.

## 4. The confidence model

**ASSUMPTION.** The brief requires a confidence to score eq. (27) but specifies no
confidence model.

```text
confidence = clip(p_correct + bias_i + eps, 0, 1)
bias_i ~ Normal(0, confidence_bias_sd)   drawn once per learner
eps    ~ Normal(0, confidence_noise_sd)  drawn per response
```

Rationale: `bias_i` makes a learner systematically over- or under-confident;
`eps` makes them inconsistent. Both are needed. Without them confidence is a
deterministic function of `p_correct`, the Brier score carries no calibration
information beyond accuracy, and `C` — implemented as `1 - running Brier` — is
degenerate.

## 5. The support-request model

**ASSUMPTION.** §7.7 requires a dependence outcome — "probability of requesting
assistance before an independent attempt" — so a request process must exist. The
brief gives no equation for one.

```text
P(request) = logistic(s0 + s1*D - s2*(theta - b_u))
```

Rationale: dependence raises requesting, relative ability lowers it. At a
checkpoint the request is *recorded but never granted*, because a checkpoint
removes support by definition.

## 6. `latency_proxy`

**ASSUMPTION.** §4.2 requires the column; the brief gives no latency model.

```text
latency = speed_i * (base + per_hint*hint_depth + attempt*attempt_made)
speed_i ~ LogNormal(0, speed_sigma)
```

This is bookkeeping, **not a reaction-time prediction**. No claim is made about
human response times and no result should be built on this column.

## 7. The operational proxies of eq. (19), (22), (23) and (25)

**ASSUMPTION, one per row.** §7.4 says only that "operational proxies should be
derived from the tutor interaction". `Retrieval`, `CorrectAfterError`,
`TransferSuccess`, `Offloading`, `SupportUsed`, `SupportFaded` and
`IndependentSuccess` appear only inside the equations; nowhere is their scale or
derivation given.

The governing rule: **every one is an observable of the interaction, never a free
parameter.** None of them can be tuned to produce a result.

| input | range | definition |
|---|---|---|
| `Attempt` | {0, 1} | the learner produced an independent attempt |
| `Retrieval` | {0, 0.5, 1} | `Attempt * (1 + unaided_correct) / 2` |
| `Explanation` | [0, 1] | `Attempt * (1 - hint_depth / max_hint_depth)` |
| `AnswerProvided` | {0, 1} | the solution was handed over |
| `Offloading` | [0, 1] | `1 - Explanation` |
| `CorrectAfterError` | {0, 1} | wrong unaided, then right under support, **answer not revealed** |
| `TransferSuccess` | {0, 1} | the unaided transfer question was answered correctly |
| `SupportUsed` | [0, 1] | the realised support level `h` |
| `SupportFaded` | [0, 1] | `clip((h_nominal - h) / h_nominal, 0, 1)`; 1 when no support was offered |
| `IndependentSuccess` | {0, 1} | `Attempt AND unaided_correct` |
| `Adaptation` | [0, 1] | how well the hint targeted the actual error |
| `Mismatch` | [0, 1] | `clip(|b_u - theta| / mismatch_scale, 0, 1)` |

Three of these carry a decision worth stating on its own:

- **`Retrieval` is graded, not binary.** A failed unaided attempt scores 0.5
  because attempting is the effortful part; a successful one scores 1 because
  eq. (22) reads the same quantity as consolidation. It is deliberately *not*
  collinear with `Attempt`, which is what keeps `a1` and `a2` in eq. (19)
  separately identifiable.
- **`CorrectAfterError` excludes revealed answers.** Reproducing a solution one
  was shown is not the same event as working past one's own error. Counting it
  would let the `ai_substitution` condition accumulate memory strength for
  reading.
- **`Explanation` uses hint depth as a stand-in for solution tokens.** §7.4 names
  "proportion of solution tokens generated by the learner". A
  structured-descriptor engine produces no tokens, so hint depth stands in.

## 8. `Coverage_u` and `Correct_u` default to 1.0 — with a warning

**ASSUMPTION, and a temporary one.** Both are eq. (20) inputs produced by the
Phase I §5.4 corpus validation. That output does not exist yet.

When the units table lacks the column, the configured default is used **and a
warning naming the missing column is both raised and written into the run log**.
It is never a silent substitution. While the default is in force the term is a
constant and contributes no between-unit variation, which is stated in the warning
text so it cannot be forgotten when reading a report.

## 9. `delta_M,i` is derived, not drawn

**ASSUMPTION.** Equation (22) subscripts its decay by learner, but equation (16)
draws only `alpha_i` and `delta_i`. So:

```text
delta_M,i = clip(m_decay_scale * delta_i, 0, 1)
```

Rationale: a learner who forgets faster also consolidates less durably.
`m_decay_scale < 1` makes the memory trace outlast expressed knowledge, which is
the ordering the two states are meant to have. Clipped at 1 because `(1 -
delta_M,i)` must stay non-negative — a decay rate above one would flip the sign of
the trace rather than erase it.

## 10. Equation (24) is not implemented literally — **mandated by the brief**

Not an assumption. A deviation the brief requires, in its own words (§7.6):

> "The calibration equation should be implemented more transparently in code as
> one minus a running Brier score, rescaled to [0,1]. All update equations require
> parameter grids and sensitivity analysis."

So `C` is **derived**, not incremented:

```text
C = clip(1 - (brier_sum / brier_count) / brier_max, 0, 1)
```

per-learner running Brier accumulators, `brier_max = 1.0` (the worst attainable
Brier score for a binary outcome with a forecast in [0, 1]). This also disposes of
the undefined `feedback_gain` term that eq. (24) carries.

Two consequences worth recording:

- Calibration is scored on **unaided** responses only. A supported response whose
  answer was revealed is not a forecast, and counting it would reward the
  substitution condition for confidence in a solution it was handed.
- A learner with no forecasts yet keeps their initial `C` from the eq. (15) draw,
  rather than being assigned a calibration they have not demonstrated.

## 11. The retention projection

**ASSUMPTION.** §7.7 requires "retention after a configurable no-practice
interval" but the brief gives no retention equation.

```text
K <- K * (1 - delta_i)^n
M <- M * (1 - delta_M,i)^n
R, C, D unchanged
```

Rationale: this is the literal reading of eq. (21)–(22) with every learning input
set to zero and iterated `n` times. `R` and `D` are untouched because eq. (23) and
eq. (25) have no decay term; `C` is untouched because it is a function of
forecasts already made. Computed on a **copy** — a checkpoint never mutates state.

## 12. Expected calibration error

**ASSUMPTION on two counts.** §7.7 requires an ECE and names no bin count, and
§4.2 gives the checkpoint table one row per learner.

```text
ECE = sum_b (n_b / J) * |accuracy_b - mean_confidence_b|,  10 equal-width bins
```

computed **per learner** over that learner's unaided checkpoint forecasts. With a
few dozen forecasts per learner the estimate is coarse and biased upward. It is
comparable across conditions — which is what the §10.6 contrast needs — but it is
**not** a population calibration estimate and should not be reported as one.

## 13. Near- and far-transfer difficulty

**ASSUMPTION.** §7.7 names near and far transfer but gives no difficulty model.
Both are the trained item with an additive offset on the `b_u` scale
(`near_b_delta = 0.5`, `far_b_delta = 1.2`), with `far >= near` enforced by
`ResponseConfig`.

## 14. The tutor's `adaptation` and `diagnosis_accuracy`

**ASSUMPTION.** The §3.4 condition table describes `ai_scaffolding` as
"graduated, error-diagnosis dependent" but gives no diagnosis model.

The scaffolding tutor diagnoses the learner's actual error with probability
`diagnosis_accuracy`, which raises `adaptation` from the base to `base + gain`. It
does **not** change `h`.

That choice is load-bearing and deliberate: `traditional` and `ai_scaffolding`
walk ladders of equal depth with the same attempt requirement, so a condition
contrast is a contrast in how well instruction is *targeted*, not in how much of
the work the tutor did. `adaptation` feeds effectiveness (eq. 20), not effort
(eq. 19). Meanwhile `ai_substitution` collapses effort instead, by requiring no
attempt and revealing the answer. **That mechanism is the study's hypothesis**, so
it is stated here rather than left to emerge from a coincidence of parameters.

## 15. The §10.3 zero-effort control zeroes `a4` as well as `a1`–`a3`

**DEVIATION from the Phase III task spec**, which writes the control as
`a1 = a2 = a3 = 0`.

With `a4 != 0` the `AnswerProvided` term still separates `ai_substitution` from
the other conditions, so effort-mediated differences do **not** vanish and the
control is not a control. `effort.zero_effort_override` in the config therefore
sets all four to zero. `tests/test_effort.py::test_keeping_a4_would_break_the_control`
demonstrates the failure directly.

## 16. The placeholder corpus

**CONVENTION.** `data/synthetic/units_placeholder.csv` is generated by
`scripts/make_placeholder_units.py`, committed for reproducibility, and stamped
`provenance = "synthetic_placeholder"` on every row.

**No output derived from it is a project result.** Every unit is fictitious; the
domains, concepts, answers and misconceptions are placeholders with no
instructional content behind them. They exist so the learner engine can be
exercised while decision gate 16 is open, and `load_curriculum` waives the gate-16
guard for that provenance and no other. Any report generated from such a run
carries a placeholder banner, and the script prints one.

## 17. Two additions to the §4.2 output schemas

**DEVIATION**, forced by the declared columns not identifying a row.

- `responses.csv` gains **`stage`** and **`episode_id`**. §3.1 scores three
  responses per learner per episode (first unaided, supported, transfer), so
  `learner_id + episode + condition` is not a key. Collapsing the three into one
  row would destroy the supported-minus-unaided information eq. (26) is built
  from.
- `learner_state.parquet` gains **`condition`**. Three conditions share a
  `(learner_id, time)` key.

Nothing was removed or renamed. Both are documented in
`docs/data_dictionary_phase3.md`.

## 18. `prompt`, `response`, `answer` and `token_count` under the logistic engine

**CONVENTION.** The logistic engine has no prompt and generates no text. §4.2
requires the columns, so they carry compact structured descriptors rather than
free text invented to fill a string field:

- `prompt` — `unit|condition|stage|d=<difficulty>|h=<support>`
- `response` — `correct`, or `misconception` where the unit documents one, else
  `incorrect`
- `answer` — the unit's `reference_answer` or `misconception_answer`; `unspecified`
  where the units table has neither. **No answer is invented.**
- `token_count` — counts descriptor fields, **not LLM tokens**. The column exists
  so the schema is identical whichever engine produced the row.

---

## Structural findings, not assumptions

These are properties of the equations as written, discovered while implementing
them. They are not choices, and they cannot be fixed by picking different values.

### F1. Equations (23) and (25) have no interior equilibrium

Neither has a headroom term nor a decay term:

```text
(23)  R <- clip(R + eta_R*E*TransferSuccess - eta_O*Offloading, 0, 1)
(25)  D <- clip(D + eta_D*SupportUsed - eta_F*SupportFaded*IndependentSuccess, 0, 1)
```

Both are therefore **bounded random walks**. Whichever term dominates on average,
the state eventually pins at 0 or 1 and every subsequent update is clipped.
Compare eq. (21), whose `(1 - K)` headroom gives it a genuine fixed point, and
eq. (22), whose geometric decay gives it one at `gain / delta_M,i`.

This is not hypothetical. The first parameterisation of this build clipped `D` on
**47%** of updates and `R` on **34%**. The rates were reduced so that pinning does
not occur inside a run — but that is a workaround, not a fix: at the default 120
episodes, `D` still clips around a third of the time, because under a fading policy
the loss term eventually wins and `D` sits at 0.

**This needs a supervisor decision, and it matters most for D5.** A ten-year
Monte Carlo runs far longer than 120 episodes, so under these equations `R` and
`D` would spend most of a simulated decade pinned at a bound, and any longitudinal
contrast built on them would be a contrast of clipping rates. The options are to
add a saturating term to both equations (changing the brief), to interpret them
only over short horizons, or to accept the bound as substantive.

The clip counts are written into every run log and validation report so the
problem stays visible rather than being absorbed by a choice of rates.

### F2. A choice-scoring Minitaur engine yields no explanation

§7.3 requires the engine's output to be parsed into **answer, confidence,
explanation and requested support**. Minitaur is a next-choice predictor used with
`max_new_tokens=1`; scoring a discrete choice set gives three of the four cleanly
and cannot give the fourth.

Free-text generation would recover explanations at a large cost in speed and in
parsing reliability. **This is a supervisor question, recorded rather than decided
in code.**

### F3. `F` saturates if its intercept is too large

Not a defect, but a trap worth recording. `traditional` and `ai_scaffolding`
differ *only* through `Adaptation`, which enters eq. (20). With the initial
coefficients, `F` sat near its ceiling for every condition, the two conditions
became indistinguishable, and the cause was arithmetic rather than substantive.
`f0`–`f2` were retuned to put `F` in mid-range, and
`tests/test_effort.py::test_adaptation_still_moves_effectiveness_at_the_defaults`
guards against it recurring.

---

## Where to look next

| File | Holds |
|---|---|
| [`docs/learner_equations.md`](learner_equations.md) | generated: every equation, its formula and the direction of each term |
| [`docs/data_dictionary_phase3.md`](data_dictionary_phase3.md) | every output column, with type, units and provenance |
| [`config/learners.yaml`](../config/learners.yaml) | every parameter, with low / medium / high arms |
| [`src/learners/CLAUDE.md`](../src/learners/CLAUDE.md) | the package's design rules |

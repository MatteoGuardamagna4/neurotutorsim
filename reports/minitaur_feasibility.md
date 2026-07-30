# Minitaur feasibility probe (§7.3, §10.2) — NOT YET RUN

**Status: no measurements exist.** This file is a placeholder so that the pointer
in `src.learners.engine.MinitaurEngine`'s `NotImplementedError` resolves to
something. It is overwritten in full by

```bash
python scripts/benchmark_minitaur.py --episodes 20
```

which must run on a Colab GPU. **Every number in the generated version comes from
that run.** Nothing here is estimated, projected, or carried over from another
model — a feasibility report with invented numbers would be worse than no report,
because a GO / NO-GO decision would then rest on them.

## What the probe will answer

One question: *is Minitaur usable as the §10.2 validation engine on a small
sample?* It is a sanity check, not infrastructure. The transparent logistic
baseline (`src/learners/responses.py`) is the main engine regardless of the
outcome — brief §9.3 asks for ~1.2e10 simulated episodes, and no LLM produces
that many.

## Method, preregistered here before any run

1. **Preflight before any download.** Read the Hub metadata for
   `marcelbinz/Llama-3.1-Minitaur-8B` (the merged checkpoint) and report whether
   it is gated. The merged model is preferred over
   `marcelbinz/Llama-3.1-Minitaur-8B-adapter`, which additionally needs the gated
   Meta base `meta-llama/Llama-3.1-8B` — a second per-account approval. **If the
   merged model turns out to be gated, the probe stops and reports.** It does not
   fall back to the adapter.
2. **Pin the revision** to a commit SHA and record it (§6.1). `transformers`
   accepts a `revision` argument, so unlike the TRIBE path this needs no
   workaround.
3. **Load in 4-bit** — bitsandbytes nf4, double quantisation, fp16 compute. The
   checkpoint is ~8.03B parameters, ~16 GB in BF16; a T4 has 16 GB and, being
   Turing, no native BF16. Report peak VRAM from
   `torch.cuda.max_memory_allocated()`.
4. **Score, do not generate.** Minitaur is a next-choice predictor trained on
   Psych-101, used with `max_new_tokens=1`; it will not produce a break-even
   calculation. For each option in a three-way choice set — correct answer, the
   unit's documented misconception, request a hint — sum the log-likelihood of its
   tokens under teacher forcing, then softmax over the set. `P(correct)` from that
   is directly comparable to the baseline's `p_correct`.
5. **Measure over at least 20 episodes**: model load time, peak VRAM, mean and SD
   seconds per scored episode, mean prompt token count.
6. **Extrapolate explicitly** to the §10.2 target of ~50 learners × 100 episodes
   = 5,000 scorings, showing the arithmetic, and state a GO / NO-GO
   recommendation. The Colab compute-unit rate is a caller-supplied figure, not a
   measured one.

## Already known without running it

- **A choice-scoring engine yields no explanation.** §7.3 asks for the engine's
  output to be parsed into answer, confidence, explanation and requested support.
  Scoring gives three of the four cleanly and cannot give the fourth. Free-text
  generation would recover explanations at a large cost in speed and parsing
  reliability. This is a supervisor question, recorded in `CLAUDE.md`, not a
  decision taken in code.
- **`MinitaurEngine` in Track B stays unimplemented** until this probe has run.
  Nothing under `src/learners/` may import torch, transformers, unsloth,
  bitsandbytes or accelerate (`tests/test_learners_cpu_purity.py` enforces it), so
  a working LLM engine cannot live there as written; if the probe says GO, where
  it *should* live is the first design question to settle.

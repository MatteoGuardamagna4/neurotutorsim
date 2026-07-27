# NeuroTutorSim

A purely computational ESADE (Master in Business Analytics) study comparing three
instructional regimes, traditional instruction, AI scaffolding and AI substitution, by
combining Meta's TRIBE v2 fMRI encoding model with synthetic-learner simulations and
10-year projections. No participants, no real experiment.

## Two tracks

| | Track A | Track B |
| --- | --- | --- |
| Does | TRIBE v2 inference: `Z(u,c,p)` cortical-response lookup | Synthetic learners, Monte Carlo, analysis |
| Needs | GPU | CPU only |
| Runs on | Google Colab | your laptop |
| Runs | once, ever | millions of times |

The two tracks are **parallel, not serial** — they meet only inside the plasticity
equation `N ← (1-δ)N + η · E(i,u,t) · Z(u,c,p)`. TRIBE never feeds an LLM.

## Setup

```bash
# Track B (local) — CPU stack, no CUDA torch
uv sync --extra dev --extra validation

# Track A (Colab) :
uv sync --frozen --active --extra tribe --extra llm
```

`uv.lock` is committed and is the core of the project's reproducibility guarantee (§4.3):
`uv sync` on a laptop yields the CPU-only stack; the `tribe`/`llm` extras add torch, nilearn,
nibabel, and the quantized behavioral engine only on Colab.

## Infrastructure

Track A runs on **Google Colab**, which supplies the free GPU. Colab is a *shell, not a
workspace*: a notebook clones this repo, installs the environment, imports functions from
`src/`, and calls them. Project logic never lives in a cell, anything written inline dies
with the session and breaks the "every figure generated from a script" rule (§4.3).

Outputs are persisted to **Google Drive**, which Colab mounts as a local path. Colab itself
has no persistent disk and sessions die at 12h, on disconnect, or on idle — so results are
written to Drive *inside* the processing loop, never accumulated in memory and saved at the
end. Inference is idempotent: a stimulus whose parquet already exists on Drive is skipped, so
a dropped session resumes rather than restarts.

```text
Colab (GPU, disposable)                         Google Drive (persistent, 15 TB)
──────────────────────                          ────────────────────────────────
clone repo ──► import src/ ──► TRIBE inference ──► parquet written per-stimulus, in-loop
                                     ▲                          │
                              HF_TOKEN from                     └─ skip if already present
                              Colab Secrets (🔑)                   (idempotent resume)
```

Secrets (the `HF_TOKEN` gating access to Llama-3.2-3B) live in **Colab Secrets (🔑)**, never
in a file or a cell.

## Phase II: TRIBE inference and neural analysis

Phase II spans brief §6.1–§6.7 plus the §8.2 handoff. It ends at `Z(u,c,p)` — the static
lookup table that is the only thing Track A hands to Track B.

Run in this order. Steps 1–3 are Track A (Colab GPU); step 4 is Track B (laptop).

```bash
# 0. one-off: build the parcellation file (needs the Schaefer fsaverage5 .annot files)
python scripts/build_atlas.py --config config/tribe.yaml

# 1. pin the checkpoint revision (once, after Meta approves Llama-3.2-3B access)
python scripts/run_tribe_verification.py --config config/tribe.yaml --resolve-revision
#    paste the SHA into config/tribe.yaml, then:
python scripts/run_tribe_verification.py --config config/tribe.yaml --write-lock

# 2. decision gate 17: verify the checkpoint, reproduce Meta's official example
python scripts/run_tribe_verification.py --config config/tribe.yaml
#    -> outputs/verification/gate17.json + docs/tribe_environment.md

# 3. inference over the corpus (idempotent; skips anything already cached)
python scripts/run_tribe_inference.py --config config/tribe.yaml \
    --cache-root /content/drive/MyDrive/NeuroTutorSim/tribe_cache

# 4. analysis: parcel data -> metrics -> contrasts -> RSA -> reports/phase2_report.md
python scripts/run_neural_analysis.py --config config/tribe.yaml
```

On Colab, steps 2 and 3 are driven by `notebooks/01_tribe_verification.ipynb` and
`notebooks/02_tribe_inference.ipynb`. Both are thin: bootstrap cells (GPU check, Drive mount,
`HF_HOME` **before** any HuggingFace import, `uv pip install --system`, auth from Colab
Secrets) and then a single call into a script.

Two things are enforced mechanically rather than by convention:

- **Step 3 refuses to run until step 2 has passed** for the pinned revision. `gate_17_passed()`
  checks `outputs/verification/gate17.json` and compares its revision to the config's.
- **Step 4 is expected to fail today.** The corpus is one unit; §6.6 and §6.7 need the ten
  units of decision gate 16, so it stops with a `ValueError` naming that gate. Use
  `--descriptive-only` for the part that is defined at n ≥ 1 (metrics and paired deltas).

Supporting scripts: `scripts/build_metrics_reference.py` regenerates
`docs/metrics_reference.md` from the metric docstrings (decision gate 19, `--check` verifies
it is current); `scripts/build_vertex_retention_set.py` regenerates the list of stimuli whose
vertex-level output is kept.

Further reading: [`docs/caching_policy.md`](docs/caching_policy.md),
[`docs/parcellation.md`](docs/parcellation.md),
[`docs/data_dictionary_phase2.md`](docs/data_dictionary_phase2.md),
[`docs/metrics_reference.md`](docs/metrics_reference.md).

## Layout

```text
config/
data/
├── DATA_DICTIONARY.md
├── raw/
│   └── bda.example_concept_maps.json
├── interim/
└── processed/
notebooks/
└── 00_tribe_demo.ipynb
outputs/
├── figures/
└── tables/
reports/
src/
├── generation/                      # teaching-corpus generation
├── validation/                      # correctness, coverage, leakage, matching validators
│   ├── validators.py                # actual answers validation
│   └── test_validators.py           # checking correctness of validators
├── tribe/                           # TRIBE inference + vertex→parcel/network aggregation
├── learners/                        # synthetic learners, knowledge tracing
├── plasticity/                      # plasticity models A/B/C/D
├── analysis/                        # scenario contrasts, specification curve
└── visualization/                   # figures
stimuli/
├── traditional/
├── ai_scaffolding/
└── ai_substitution/
tests/
└── test_imports.py                  # package-structure gate
```

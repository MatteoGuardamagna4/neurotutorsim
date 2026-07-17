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

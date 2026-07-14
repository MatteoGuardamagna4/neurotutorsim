# NeuroTutorSim

A purely computational ESADE (Master in Business Analytics) study comparing three
instructional regimes — traditional instruction, AI scaffolding, AI substitution — by
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

# Track A (Colab) — see 00_bootstrap_colab.ipynb, which runs:
uv sync --frozen --active --extra tribe --extra llm
```

`uv.lock` is committed and is the core of the project's reproducibility guarantee (§4.3):
`uv sync` on a laptop yields the CPU-only stack; the `tribe`/`llm` extras add torch, nilearn,
nibabel, and the quantized behavioral engine only on Colab.

## Layout

```text
src/
├── generation/      # teaching-corpus generation
├── validation/      # correctness, coverage, leakage, matching validators
├── tribe/           # TRIBE inference + vertex→parcel/network aggregation
├── learners/        # synthetic learners, knowledge tracing
├── plasticity/      # plasticity models A/B/C/D
├── analysis/        # scenario contrasts, specification curve
└── visualization/   # figures
tests/               # package-structure gate + unit tests
config/              # run configuration
```

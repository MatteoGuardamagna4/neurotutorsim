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

### Required Colab notebook snippets

Every Track A notebook must open with these cells (in order). They are boilerplate — the
scientific logic stays in `src/`.

**1. Mount Drive** (the persistence layer):

```python
from google.colab import drive
drive.mount("/content/drive")
```

**2. Clone the repo and install the GPU stack.** `tribev2` is not on PyPI and is pinned to a
commit hash — never `main` (§6.1):

```python
!git clone https://github.com/MatteoGuardamagna4/neurotutorsim.git
%cd neurotutorsim
!pip install -q uv
!uv sync --frozen --active --extra tribe --extra llm
```

**3. Load secrets** from Colab Secrets (🔑) — sets the token TRIBE needs to reach the gated
Llama text encoder:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**4. Import from `src/`** — call project functions, don't reimplement them in the cell:

```python
import sys
sys.path.insert(0, "/content/neurotutorsim")

from src.tribe.inference import run_inference      # Track A entry point
from src.tribe.aggregate import vertex_to_parcel   # eq. 6-7
```

**5. Define saving paths on Drive and write idempotently, inside the loop:**

```python
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/neurotutorsim")
OUT_PARCEL = DRIVE_ROOT / "outputs" / "tribe" / "parcel"   # ~200 KB/stimulus
OUT_PARCEL.mkdir(parents=True, exist_ok=True)

for stimulus in corpus:
    out_path = OUT_PARCEL / f"{stimulus.id}.parquet"
    if out_path.exists():
        continue                     # §4.3: never silently regenerate an existing item
    Z = run_inference(stimulus)      # heavy GPU work
    vertex_to_parcel(Z).to_parquet(out_path)   # write to Drive immediately
```

## Layout

```text
config/              # run configuration
data/
notebooks/
outputs/
reports/
src/
├── generation/      # teaching-corpus generation
├── validation/      # correctness, coverage, leakage, matching validators
├── tribe/           # TRIBE inference + vertex→parcel/network aggregation
├── learners/        # synthetic learners, knowledge tracing
├── plasticity/      # plasticity models A/B/C/D
├── analysis/        # scenario contrasts, specification curve
└── visualization/   # figures
stimuli/
tests/               # package-structure gate + unit tests
```

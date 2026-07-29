# Phase II data dictionary

Every column of every Phase II output table: type, units, provenance.

Phase II spans brief §6.1–§6.7 plus the §8.2 handoff. It ends at `Z(u,c,p)`;
nothing here concerns learners, effort, or plasticity.

Conventions used throughout:

* `stimulus_id` — declared in stimulus front matter, e.g. `be_001_traditional_primary`. Never parsed by slicing the string; `unit_id`, `condition` and `variant` are read from front matter.
* `condition` ∈ `traditional`, `ai_scaffolding`, `ai_substitution`.
* `parcel_id` — 1…400 (Schaefer-400). **`-1` means whole cortex**, not a parcel.
* "predicted BOLD units" — TRIBE's arbitrary output scale. Not a physiological measurement.

---

## 1. Cache manifest — `{cache_root}/manifest.parquet`

Append-only, one row per cache entry. Written by `src/tribe/cache.py`.

| column | type | units | provenance |
|---|---|---|---|
| `vertex_key` | str (64 hex) | — | `sha256(text_hash, reading_rate_wpm, checkpoint_revision, precision)` |
| `parcel_key` | str (64 hex) | — | `sha256(vertex_key, atlas, parcel_weighting)` |
| `stimulus_id` | str | — | stimulus front matter |
| `level` | str | — | `vertex` or `parcel` |
| `path` | str | — | path relative to `cache_root` |
| `size_bytes` | int | bytes | filesystem, at write time |
| `timestamp_utc` | str | ISO 8601 | write time |
| `text_hash` | str (64 hex) | — | `sha256` of the newline-normalised stimulus body |
| `reading_rate_wpm` | float | words/minute | config; 220 main, 180/260 robustness |
| `checkpoint` | str | — | config, e.g. `facebook/tribev2` |
| `revision` | str (40 hex) | — | pinned checkpoint commit SHA |
| `precision` | str | — | `fp32` \| `fp16` |
| `atlas` | str | — | config, e.g. `schaefer400` |
| `parcel_weighting` | str | — | `area` \| `equal` |
| `n_timesteps` | int | samples | prediction shape |
| `n_vertices` | int | vertices | 20 484 (fsaverage5), asserted |
| `n_parcels` | int | parcels | 400 |
| `runtime_s` | float | seconds | wall clock for that stimulus |

---

## 2. Vertex cache entry — `{cache_root}/vertex/**/*.npz`

| array | type | shape | provenance |
|---|---|---|---|
| `predictions` | float32 | `(n_timesteps, 20484)` | `TribeModel.predict` |
| `metadata_json` | str | scalar | canonical JSON of the §4.4 metadata dict |

Complete time-resolved output only. A static per-vertex summary is rejected
(§6.3). Retained only for ids in `config/vertex_retention_set.txt`.

---

## 3. Parcel table — `{cache_root}/parcel/**/*.parquet`

Equation (6). Written by `src/tribe/aggregate.vertices_to_parcels`.

| column | type | units | provenance |
|---|---|---|---|
| `stimulus_id` | str | — | front matter |
| `time_index` | int | samples (TR = 1 s) | 0-based sample index |
| `parcel_id` | int | — | atlas |
| `network` | str | — | Yeo network from the Schaefer label |
| `mean_bold` | float | predicted BOLD | eq. (6), unweighted mean over the parcel's vertices |
| `sd_bold` | float | predicted BOLD | SD across vertices in the parcel, `ddof=0` |

`.attrs["provenance"]` carries the atlas name, version, file path, file
`sha256`, vertex/parcel counts, equation number and weighting.

---

## 4. Network table (in memory / optional output)

Equation (7). Written by `src/tribe/aggregate.parcels_to_networks`.

| column | type | units | provenance |
|---|---|---|---|
| `stimulus_id` | str | — | front matter |
| `time_index` | int | samples | — |
| `network` | str | — | atlas |
| `mean_bold` | float | predicted BOLD | eq. (7), weighted by `weighting` |
| `weighting` | str | — | `area` (main) or `equal` (robustness) — recorded per row, not only in config |

---

## 5. Metrics table — `outputs/tables/phase2_metrics.csv`

§6.5. Written by `scripts/run_neural_analysis.py` from
`src/analysis/neural_metrics.compute_all_metrics`. One row per
`(stimulus_id, parcel_id, metric)`.

| column | type | units | provenance |
|---|---|---|---|
| `stimulus_id` | str | — | front matter |
| `parcel_id` | int | — | atlas; **`-1` = whole cortex** |
| `network` | str | — | atlas; `__all__` on whole-cortex rows |
| `metric` | str | — | one of the ten §6.5 metrics |
| `value` | float | varies by metric | see [`metrics_reference.md`](metrics_reference.md) |
| `smoothing_kernel` | str | — | config `metrics.smoothing_kernel` |
| `smoothing_width_s` | float | seconds | config |
| `tr_seconds` | float | seconds | config; 1.0 for TRIBE v2 |
| `instructional_window_s` | str | seconds | config; `full` when unwindowed |
| `baseline_percentile` | float | percentile | config |
| `sustained_threshold_sd` | float | SD | config |
| `spatial_basis` | str | — | non-negative summary the spatial metrics use (`abs_auc`) |
| `unit_id` | str | — | front matter (joined by the script) |
| `condition` | str | — | front matter |
| `variant` | str | — | front matter |

Per-metric units and direction: [`metrics_reference.md`](metrics_reference.md),
generated from the docstrings (decision gate 19).

---

## 6. Paired deltas — `outputs/tables/phase2_deltas.csv`

§6.6 equations (10)–(12). Valid at n ≥ 1.

| column | type | units | provenance |
|---|---|---|---|
| `unit_id` | str | — | front matter |
| `regeneration` | int | — | regeneration index; `0` when the corpus has none |
| `parcel_id` | int | — | atlas; `-1` = whole cortex |
| `metric` | str | — | as in the metrics table |
| `contrast` | str | — | `S-T`, `U-T`, `S-U` |
| `delta` | float | metric's own units | difference of two matched stimuli of the same unit |

Sign convention: `S-T` = `ai_scaffolding` − `traditional`; `U-T` =
`ai_substitution` − `traditional`; `S-U` = `ai_scaffolding` − `ai_substitution`.

---

## 7. Cluster bootstrap — `outputs/tables/phase2_bootstrap.csv`

§6.6. Requires n_units ≥ 10 (decision gate 16).

| column | type | units | provenance |
|---|---|---|---|
| `contrast` | str | — | as above |
| `metric` | str | — | as above |
| `parcel_id` | int | — | atlas |
| `estimate` | float | metric's units | mean delta over units |
| `boot_se` | float | metric's units | SD of the bootstrap distribution, `ddof=1` |
| `ci_lower`, `ci_upper` | float | metric's units | percentile interval at `alpha` |
| `n_boot` | int | draws | config `analysis.n_bootstrap` |
| `n_units` | int | units | corpus size at run time |
| `seed` | int | — | `master_seed` |
| `alpha` | float | — | interval width; 0.05 default |

Resampling is over **units, then regenerations within unit**. Parcels are not
resampled and **no parcel- or vertex-level p-values are produced** (§6.6
prohibits them).

---

## 8. Z table — `outputs/tables/Z_ucp.parquet` (+ `Z_ucp_unwinsorized.parquet`)

§8.2 equation (28). The sole Track A → Track B interface.

| column | type | units | provenance |
|---|---|---|---|
| `unit_id` | str | — | front matter |
| `condition` | str | — | front matter |
| `parcel_id` | int | — | atlas |
| `auc` | float | predicted BOLD × s | metrics table, metric `auc` |
| `z` | float | SD units | `(auc − mu_p) / sigma_p`, moments pooled **across the full corpus** |
| `winsorized` | bool | — | whether this row's `z` was clipped |

`mu_p` and `sigma_p` are computed per parcel over all units **and all
conditions**. Standardizing within condition would remove the between-condition
difference the study measures.

Main table is winsorized at the 1st/99th percentile of the pooled `z`;
`Z_ucp_unwinsorized.parquet` is the same table without clipping and always
carries `winsorized = False`.

---

## 9. Gate 17 artifact — `outputs/verification/gate17.json`

§6.1. `src/tribe/inference.py` refuses to run without a matching one.

| key | type | provenance |
|---|---|---|
| `gate`, `passed` | int, bool | `17`, `true` — written only on a full pass |
| `checkpoint`, `checkpoint_revision`, `precision` | str | config at verification time |
| `checksum` | object | per-file `sha256` of the downloaded snapshot vs `config/checkpoints.lock` |
| `official_example` | object | timesteps, vertices, runtime, summary statistics |
| `environment` | object | GPU, VRAM, CUDA, Python, platform, package versions |
| `written_utc` | str | ISO 8601 |

A gate artifact recorded against a *different* revision is not a pass — that is
exactly what the gate exists to catch.

---

## 10. Timing discrepancy report — `reports/timing_discrepancy.md`

Written only when TRIBE's own event timings disagree with our 220 wpm onsets.
Our onsets are already authoritative by the time this is written; neither
source is adjusted to match the other.

| column | units | provenance |
|---|---|---|
| `stimulus_id` | — | front matter |
| `r (wpm)` | words/minute | config |
| `n words` | words | our tokenizer |
| `max abs offset` | seconds | max over words of \|TRIBE onset − our onset\| |
| `mean abs offset` | seconds | mean of the same |
| `TRIBE duration` | seconds | last TRIBE word onset + duration |
| `our duration` | seconds | `60 × words / r` |

---

## Negative-control provenance

Any artifact from `src/analysis/negative_controls.py` carries
`DataFrame.attrs["negative_control"]` and `attrs["is_negative_control"] = True`,
with `control`, `seed`, `source`, `detail` and `created_utc`. A control is
designed to be indistinguishable from substantive output, so the stamp is the
only thing separating them — it is never optional.

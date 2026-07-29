# Phase II structure

What every file added in Phase II does, how it is called, and how they connect.

Phase II covers brief §6.1–§6.7 plus the §8.2 handoff. It starts at stimulus
text and ends at `Z(u,c,p)`. It contains no learner, no effort model and no
plasticity — those are Phase III/IV.

---

## 1. The pipeline

```
config/tribe.yaml ──────────────────────────────────────────────┐
                                                                │ (every stage)
stimuli/<condition>/*.txt                                       │
   │                                                            │
   │ stimulus_io.build_stimulus_index                           │
   ▼                                                            │
{stimulus_id: unit_id, condition, variant, body}                │
   │                                                            │
   │ events.build_events        (220 wpm onsets, Track B)       │
   ▼                                                            │
canonical event table ──► events.measure_timing_discrepancy     │
   (measured against TRIBE's own frame, never written into it)  │
   │                                                            │
   │           ┌─── GATE 17 ──────────────────────────┐         │
   │           │ verification.gate_17_passed(config)  │         │
   │           └──────────────┬───────────────────────┘         │
   ▼                          ▼                                 │
inference.predict_stimulus ──► (T, 20484) float32               │  TRACK A
   │                                                            │  (Colab GPU)
   │ aggregate.vertices_to_parcels        (eq. 6)               │
   ▼                                                            │
(T, 400) parcel table                                           │
   │                                                            │
   │ cache.write_parcel  (always) / cache.write_vertex (retention set only)
   ▼                                                            │
{cache_root}/  parcel/*.parquet   vertex/*.npz   manifest.parquet
   │                                                            │
═══╪════════════════════════════════════════════════════════════╪══════════
   │                                                            │  TRACK B
   │ cache.resolve(need_vertex=False)                           │  (laptop CPU)
   ▼                                                            │
neural_metrics.compute_all_metrics   (eq. 8, 9 + 8 more)        │
   │                                                            │
   ├──► contrasts.paired_deltas          (eq. 10–12)  n ≥ 1     │
   │       └─► fit_unit_fe_model (eq. 13, n ≥ 2)                │
   │       └─► cluster_bootstrap         (n ≥ 10, gate 16)      │
   ├──► rsa.unit_pattern_vectors ─► build_rdm (eq. 14, n ≥ 3)   │
   │       └─► compare_rdms / permutation_test (n ≥ 10)         │
   │       └─► classify_geometry → GeometryVerdict              │
   └──► standardize.standardize_auc      (eq. 28, full corpus)  │
              │                                                 │
              ▼                                                 │
        outputs/tables/Z_ucp.parquet   ◄── the ONLY Track A → B handoff
```

`negative_controls` sits beside this, generating shuffled / permuted / donor
variants of any of these inputs, each stamped so it cannot be confused with
substantive output.

---

## 2. Run order

```bash
# 0. one-off: build the parcellation                        (needs nilearn/nibabel)
python scripts/build_atlas.py --config config/tribe.yaml

# 1. one-off: pin the checkpoint revision                   (needs HF access)
python scripts/run_tribe_verification.py --config config/tribe.yaml --resolve-revision
#    paste the SHA into config/tribe.yaml, then
python scripts/run_tribe_verification.py --config config/tribe.yaml --write-lock

# 2. GATE 17: checksum + reproduce Meta's example           (Colab GPU)
python scripts/run_tribe_verification.py --config config/tribe.yaml

# 3. inference over the corpus, idempotent                  (Colab GPU)
python scripts/run_tribe_inference.py --config config/tribe.yaml \
    --cache-root /content/drive/MyDrive/NeuroTutorSim/tribe_cache

# 4. analysis                                               (laptop CPU)
python scripts/run_neural_analysis.py --config config/tribe.yaml
```

Steps 2 and 3 are what the two notebooks drive. Step 3 refuses to start until
step 2 has passed **for the same pinned revision**. Step 4 currently stops with
a `ValueError` naming decision gate 16 — the corpus is one unit.

---

## 3. Config files

| File | Purpose | How it is used |
|---|---|---|
| `config/tribe.yaml` | The single source of truth for both tracks: checkpoint, revision, precision, reading rate, atlas, weighting, cache root, seeds, metric parameters, analysis parameters. | `--config config/tribe.yaml` on every script. Edited by hand. Unknown keys raise. |
| `config/checkpoints.lock` | `sha256` of every file in the pinned checkpoint snapshot, keyed `<repo>@<revision>`. | Read by `verify_checkpoint`, written only by the explicit `--write-lock` flag. Ships empty. |
| `config/vertex_retention_set.txt` | The stimulus ids whose full `(T, V)` vertex array is kept permanently. Everything else is aggregated to parcels and the vertices are **discarded irreversibly**. | Read by `TribeConfig.read_vertex_retention_set()`; regenerated by `scripts/build_vertex_retention_set.py`. |

---

## 4. `src/tribe/` — Track A + shared

| File | Purpose | Key entry points |
|---|---|---|
| `config.py` | Frozen, self-validating config. `TribeConfig`, `MetricsConfig`, `AnalysisConfig`. Validates a 40-hex revision, allowed precision/atlas/weighting/reading rate; refuses the all-zero placeholder revision for Track A. | `load_config(path, cache_root=None)`, `cfg.identity()`, `cfg.require_resolved_revision()`, `cfg.read_vertex_retention_set()` |
| `events.py` | The canonical word-event table at 220 wpm (`onset_j = 60·cum_words/r`), plus the isolated TRIBE-schema reader. Onsets are delegated to `features.word_onsets`, never reimplemented. The table governs Track B and is **not** written into TRIBE's events frame — the tokenisations cannot be aligned and TRIBE encodes a TTS waveform, so the gap is measured instead. | `build_events(text, wpm)` → DataFrame; `measure_timing_discrepancy(tribe_df, events_df, stimulus_id=)` → `TimingDiscrepancy`; `write_timing_discrepancy_report(...)` |
| `cache.py` | Content-addressed hybrid cache. Two nested `sha256` keys; three-branch `resolve`; append-only manifest. **A parcel entry is not a vertex hit.** | `TribeCache(config)`, `.resolve(stimulus_id, text, need_vertex=, load_data=True)`, `.write_parcel/.write_vertex`, `.manifest()`, `.verify_manifest()`; `keys_for(text, config)`, `text_hash(text)` |
| `aggregate.py` | Atlas loading + validation, eq. (6) vertex→parcel, eq. (7) parcel→network. Pure CPU/numpy — no nilearn at import. | `load_atlas(path)` → `Atlas`; `vertices_to_parcels(B_hat, atlas, weighting, stimulus_id=)`; `parcels_to_networks(parcel_df, weighting, atlas)`; `aggregate_prediction(...)`; `file_sha256(path)` |
| `verification.py` | **Decision gate 17.** Checksums the pinned checkpoint, reproduces Meta's published example (with TRIBE's *own* timings), records the environment, writes the gate artifact. | `resolve_remote_revision`, `verify_checkpoint`, `write_lock`, `pinned_snapshot_dir`, `load_model`, `run_official_example`, `record_environment`, `write_gate_17_artifact`, **`gate_17_passed(config)`** |
| `inference.py` | The idempotent GPU loop. Imports torch — never import from Track B. | `run_inference(config, stimuli, limit=None)`; `predict_stimulus(model, id, text, config)`; `set_determinism(config)`; `assert_bitwise_reproducible(...)`; `load_stimuli_csv(path)`, `iter_repo_stimuli(root)` |

Modified: **`src/generation/stimulus_io.py`** gains `build_stimulus_index(root)`
— maps `stimulus_id → {unit_id, condition, variant, body, path}` read from front
matter, never by slicing the id string.

---

## 5. `src/analysis/` — Track B (CPU only, no torch anywhere)

| File | Purpose | Key entry points |
|---|---|---|
| `gates.py` | The sample-size thresholds in one place, so modules and scripts cannot drift on what "enough units" means. | `MIN_UNITS_GATE_16=10`, `MIN_UNITS_FIXED_EFFECTS=2`, `MIN_UNITS_RDM=3`; `require_gate_16(n, what=)`, `require_fixed_effects_units`, `require_rdm_units` |
| `neural_metrics.py` | The ten §6.5 metrics, each pure and independently testable, each carrying a gate-19 docstring block. | `mean_response`, `peak_response`, `auc`, `time_to_peak`, `sustained_engagement`, `spatial_dispersion`, `spatial_entropy`, `network_integration`, `representational_differentiation`, `within_concept_consistency`; driver `compute_all_metrics(parcel_df, config.metrics)`; helpers `smooth`, `window_slice`; `WHOLE_CORTEX = -1`, `SPATIAL_BASIS = "abs_auc"` |
| `contrasts.py` | §6.6 eq. (10)–(13) and the cluster bootstrap over **units and regenerations**. No vertex-level p-values anywhere. | `paired_deltas(metrics_df)`; `fit_unit_fe_model(level_df, covariates=None, metric=)`; `cluster_bootstrap(deltas_df, n_boot, seed, alpha=0.05)`; `CONTRASTS`, `TRADITIONAL/AI_SCAFFOLDING/AI_SUBSTITUTION` |
| `rsa.py` | §6.7 eq. (14). RDMs, upper-triangle-only comparison, within-unit label permutation, and the preserve/sharpen/homogenize answer as a dataclass. | `unit_pattern_vectors(metrics_df, condition, metric=)`, `pattern_unit_order`, `build_rdm`, `upper_triangle`, `compare_rdms` → `SpearmanResult`, `permutation_test` → `PermResult`, `classify_geometry` → `GeometryVerdict` |
| `standardize.py` | §8.2 eq. (28). Standardizes AUC per parcel against **corpus-wide** moments and writes the winsorized + unwinsorized tables. | `standardize_auc(metrics_df, config.analysis, winsorize=True)`; `write_z_table(metrics_df, config.analysis, out_dir)` |
| `negative_controls.py` | §10.3 generators only. Every return value is `(data, ControlProvenance)` and stamped into `DataFrame.attrs`. | `temporally_shuffled(text, seed)`, `irrelevant_matched(text, donor_corpus, ...)`, `permute_predictions(parcel_df, seed)`, `permute_condition_labels(df, seed)`; `CONTROLS`, `CONTROL_FLAG` |

---

## 6. `scripts/`

Every script takes `--config`, uses argparse, hard-codes no paths.

| Script | Track | What it does | Writes |
|---|---|---|---|
| `run_tribe_verification.py` | A | Gate 17. Flags: `--resolve-revision` (print the SHA), `--write-lock` (record checksums), `--skip-example` (checksum only), `--cache-root`. | `outputs/verification/gate17.json`, `docs/tribe_environment.md` |
| `run_tribe_inference.py` | A | The idempotent loop. Flags: `--stimuli` (CSV index; defaults to the repo's `stimuli/`), `--cache-root`, `--limit`, `--check-determinism`, `--verify-manifest`. | parcel/vertex cache entries, `manifest.parquet`, `reports/timing_discrepancy.md` |
| `run_neural_analysis.py` | B | Metrics → deltas → bootstrap → FE → RSA → report. Flags: `--descriptive-only` (skip every gate-16 step), `--metric`, `--cache-root`. | `outputs/tables/phase2_metrics.csv`, `phase2_deltas.csv`, `phase2_bootstrap.csv`, `reports/phase2_report.md` |
| `build_metrics_reference.py` | B | **Gate 19 as code.** Parses the metric docstrings and fails if any lacks Formula / Units / Increase means / Direction. `--check` verifies the doc is current without writing. | `docs/metrics_reference.md` |
| `build_atlas.py` | A | Builds the Schaefer-400 fsaverage5 `.npz` from the two `.annot` files. Asserts 10 242 vertices/hemisphere; refuses the volumetric MNI atlas. Flags: `--lh-annot`, `--rh-annot`, `--out`, `--force`. | `data/raw/atlas/schaefer400_fsaverage5.npz` |
| `build_vertex_retention_set.py` | B | Regenerates the retention list (policy: every `primary` variant). Never silently drops an id already in the file. | `config/vertex_retention_set.txt` |

---

## 7. `notebooks/`

Both are **thin**: bootstrap cells plus one call into a script. No project
logic — a cell dies with the session.

| Notebook | Purpose |
|---|---|
| `01_tribe_verification.ipynb` | GPU check → Drive mount → `HF_HOME` **before** any HuggingFace import → clone + `uv pip install --system` → auth from Colab Secrets → `run_tribe_verification.py`. Includes the one-off revision-pinning cells. |
| `02_tribe_inference.ipynb` | Same bootstrap → gate-17 check → `run_tribe_inference.py` → optional determinism check → manifest verification. The "commit before the session closes" warning is a callout at the top. |

---

## 8. `tests/` — all CPU, no GPU, no network

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: `toy_atlas` (6 vertices, 3 parcels, 2 networks — hand-checkable), `tribe_config`, `metrics_config`, `analysis_config`, `synthetic_metrics`, `make_parcel_frame` |
| `test_events.py` | Onsets at 180/220/260 wpm vs hand-computed values, monotonicity, sentence boundaries, adapter purity, misalignment raising |
| `test_cache.py` | All three `resolve` branches, the `parcel_only_vertex_required` miss, key sensitivity to `r`/revision/precision/atlas/weighting, manifest round-trip, orphan and missing-file detection |
| `test_aggregate.py` | Eq. (6) against a hand-computed mean, eq. (7) under both weightings, atlas-coverage failure, single-assignment guarantee |
| `test_metrics.py` | One test per metric against an analytic input (AUC of a triangle wave, entropy of a uniform profile), entropy raising on all-zero/negative, concept and variant guards |
| `test_contrasts.py` | Exact antisymmetry, triangle identity, FE raising at n=1, bootstrap raising below 10 with the exact gate-16 message, seed reproducibility |
| `test_rsa.py` | RDM symmetry and zero diagonal, raising below 3 units, permutation raising below 10, within-unit pairing preserved, upper-triangle-only comparison |
| `test_determinism.py` | Cache keys identical across subprocesses with different `PYTHONHASHSEED`; a grep-level guard against builtin `hash()` |
| `test_tribe_config.py` | Unknown keys, malformed revisions, disallowed precision/rate/weighting, path resolution, retention-set parsing |
| `test_standardize.py` | Partial-corpus refusal, corpus-wide (not within-condition) pooling, zero-variance parcel, winsorization flags |
| `test_negative_controls.py` | Word-count preservation, caliper enforcement, within-unit permutation, provenance stamping |

Run: `uv run pytest tests/ -v` → **218 passed**.

---

## 9. `docs/`

| File | Content | Generated? |
|---|---|---|
| `metrics_reference.md` | Every metric's formula, units, meaning of an increase, and whether its direction is meaningful or ambiguous. | **Yes** — `build_metrics_reference.py`. Do not hand-edit. |
| `parcellation.md` | Atlas version and space, `.npz` format, validation rules, eq. (6)/(7), network naming, weighting. | No |
| `data_dictionary_phase2.md` | Every column of every Phase II table: type, units, provenance. | No |
| `caching_policy.md` | Why the cache is hybrid, the size arithmetic, key derivation, the one-way-averaging constraint. | No |
| `phase2_structure.md` | This file. | No |
| `tribe_environment.md` | GPU, VRAM, CUDA, package versions of the verification run. | **Yes** — `record_environment()`. Does not exist until gate 17 runs on real hardware. |

Also updated: `README.md` (Phase II run order), `CLAUDE.md` (module map, settled
timing decision, guard/gate table, open questions), `.gitignore` (cache roots).

---

## 10. Where things stop on purpose

| Situation | What happens |
|---|---|
| Revision is the all-zero placeholder | Track B loads fine; Track A raises with the command to resolve it |
| `precision: fp16` | `load_model` raises — the cache key would claim fp16 while the model loaded fp32 |
| Installed `tribev2` takes no `revision` (the published build) | `load_model` pins it anyway: `snapshot_download(revision=<sha>)`, SHA checked in the returned path, directory passed as `checkpoint_dir` |
| Snapshot path lacks the pinned SHA, or holds 0 / >1 `.ckpt` files | `load_model` raises — the weights cannot be shown to be the pinned ones |
| Installed `tribev2` takes neither `revision` nor `checkpoint_dir` | `load_model` raises — no way left to pin, the build changed |
| Gate 17 artifact missing or from another revision | `run_inference` refuses to touch a project stimulus |
| Prediction has ≠ 20 484 vertices | `predict_stimulus` raises — every parcel mapping downstream would be wrong |
| Vertex needed, only parcels cached | `resolve` returns a **miss**, reason `parcel_only_vertex_required` — never fabricated |
| n_units < 2 | `fit_unit_fe_model` raises |
| n_units < 3 | `build_rdm` raises |
| n_units < 10 | `cluster_bootstrap`, `permutation_test` and `run_neural_analysis.py` raise, naming decision gate 16 |
| n_units < `corpus_n_units` | `standardize_auc` raises — corpus-wide moments on a partial corpus are silently wrong |
| Metric docstring missing a gate-19 field | `build_metrics_reference.py` fails |
| Atlas does not cover the mesh | `assert_covers_mesh` raises |
| Entropy of an all-zero / negative vector | `spatial_entropy` raises rather than returning 0 |

**`run_neural_analysis.py` failing today is the system working.** One unit, gate
16 open. Use `--descriptive-only` for the part that is defined at n ≥ 1.

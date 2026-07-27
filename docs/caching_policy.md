# Caching policy (§4.3, §6.3)

Why the Phase II cache is hybrid rather than uniform, how its keys are derived,
and what the one-way nature of parcel averaging forces on the design.

Implemented in [`src/tribe/cache.py`](../src/tribe/cache.py); tested in
[`tests/test_cache.py`](../tests/test_cache.py).

## The constraint everything follows from

**Parcel averaging cannot be undone.** Equation (6) reduces a
`(T, ~20 484)` vertex array to `(T, 400)` parcel means. Nothing downstream can
recover the vertices from that mean. Recomputing them costs another GPU run
against a gated checkpoint, on free-tier Colab, with a 12-hour session ceiling.

So the decision "keep vertices or not" is irreversible at the moment it is
made, and the cache is built to make that decision explicit, recorded, and
impossible to take accidentally.

## Size arithmetic

Per stimulus, at fsaverage5 with ~1 prediction per second:

| level | shape | float32 size | compressed |
|---|---|---|---|
| vertex | ~120 x 20 484 | ~9.4 MB | ~4 MB `.npz` |
| parcel | ~120 x 400 | ~190 KB | ~200 KB parquet |

At the planned full grid (~3 240 stimuli: units x 3 conditions x variants and
regenerations):

| policy | storage | verdict |
|---|---|---|
| vertex for everything | ~13 GB | does not fit free Drive (15 GB total) |
| parcel for everything | ~1.2 GB | fits, but discards all vertex detail |
| **hybrid (adopted)** | ~1.2 GB + ~3.5 GB = **~4.7 GB** | fits, keeps a robustness subset |

The hybrid policy keeps parcel output for every stimulus and vertex output for
the ~360 `primary` stimuli listed in
[`config/vertex_retention_set.txt`](../config/vertex_retention_set.txt).

> ⚠️ This deviates from brief §6.3, which asks for vertex-level output
> throughout. It is recorded as an open question for the supervisor in
> `CLAUDE.md`, not treated as settled.

## Key derivation

Two nested, content-addressed keys, both `sha256` over canonical JSON:

```
vertex_key = sha256(text_hash, reading_rate_wpm, checkpoint_revision, precision)
parcel_key = sha256(vertex_key, atlas, parcel_weighting)
```

* `text_hash` is `sha256` of the stimulus body with newlines normalised, so a
  CRLF checkout and an LF checkout agree.
* `reading_rate_wpm` is in the key because our 220 wpm onsets (§6.2) determine
  the model's input timing. Rerunning at 180 or 260 wpm is a *different
  prediction*, not the same one.
* Parcel settings nest **under** the vertex key. Changing the atlas or the
  weighting re-derives parcels from a retained vertex array; it does not
  invalidate the GPU work.

`hashlib` is used throughout. Python's builtin `hash()` is salted per process,
so a key derived from it would silently miss every entry a previous Colab
session wrote. [`tests/test_determinism.py`](../tests/test_determinism.py)
enforces this across two subprocesses with different `PYTHONHASHSEED` values.

## Layout

```
{cache_root}/
  manifest.parquet                 append-only, one row per entry
  vertex/<first-2-hex>/<vertex_key>.npz        float32 (T, V), compressed
  parcel/<first-2-hex>/<parcel_key>.parquet    long format, snappy
```

The two-hex fan-out keeps any single directory small enough for Drive to list
comfortably.

Only complete time-resolved output is stored. §6.3 forbids substituting a
static per-vertex summary for the full `(T, V)` matrix, and `write_vertex`
rejects a 1-D array rather than reshaping it.

## The three-branch resolve

`resolve(stimulus_id, text, need_vertex)` returns one of three outcomes:

| state on disk | `need_vertex=False` | `need_vertex=True` |
|---|---|---|
| vertex entry exists | `hit_vertex` (parcels derivable) | `hit_vertex` |
| parcel entry only | `hit_parcel` | **`miss`**, reason `parcel_only_vertex_required` |
| neither | `miss`, reason `no_entry` | `miss`, reason `no_entry` |

The bold cell is the point of the whole design. A parcel entry is *not* a
vertex hit. The cache never synthesises vertex data from parcel means, and the
distinct miss reason lets the caller tell "never computed" apart from
"computed, then deliberately discarded".

## Idempotence

§4.3: never silently regenerate an existing item. Every stimulus is resolved
before any GPU work; a hit is skipped and logged. A Colab session that dies at
hour 11 resumes where it stopped.

Writes happen **inside** the loop, per stimulus, parcel first — parcel data is
what the whole corpus needs and what survives a session that dies mid-write. A
manifest assembled in memory and flushed at the end is a manifest that is
usually lost.

## Manifest

One append-only parquet at `{cache_root}/manifest.parquet`, one row per entry,
carrying both keys, the stimulus id, level, relative path, size, timestamp and
every metadata field from §4.4 (text hash, reading rate, checkpoint, revision,
precision, timesteps, vertices, parcels, runtime).

`verify_manifest()` cross-checks rows against files and reports two categories:

* **missing** — recorded in the manifest, absent on disk;
* **orphans** — on disk, absent from the manifest.

It deletes nothing. Every entry cost a GPU run against a gated model, so
removal is always a human decision.

## What is committed

Nothing under `cache_root`. The cache lives on Drive (Colab) or under
`data/interim/` (laptop); both are gitignored, as are `*.npz` and `*.parquet`.
Reproducibility comes from the pinned revision, the config, and the key
derivation — not from committing outputs.

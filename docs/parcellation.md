# Parcellation (§6.4, Appendix A)

The atlas, the vertex→parcel mapping, the network assignment, and the weighting
choice at each aggregation step.

Implemented in [`src/tribe/aggregate.py`](../src/tribe/aggregate.py); the atlas
file is built by [`scripts/build_atlas.py`](../scripts/build_atlas.py).

## Atlas

| | |
|---|---|
| Parcellation | Schaefer 2018, 400 parcels, 7 Yeo networks |
| Space | `fsaverage5` surface, 10 242 vertices per hemisphere, **20 484 total** |
| Source | `lh/rh.Schaefer2018_400Parcels_7Networks_order.annot` (CBIG release, `FreeSurfer5.3/fsaverage5/label/`) |
| Stored as | `data/raw/atlas/schaefer400_fsaverage5.npz` |
| Committed | no — `data/raw/` is gitignored; the file's `sha256` is recorded in every table it touches |

TRIBE v2 predicts onto fsaverage5, so the parcellation must be the **surface**
annotation in that same space. The volumetric MNI Schaefer atlas is a different
asset; `build_atlas.py` refuses to substitute it, and asserts 10 242 vertices
per hemisphere before writing anything.

### File format

`.npz`, read by `aggregate.load_atlas`:

| array | shape | meaning |
|---|---|---|
| `labels` | `(20484,)` int64 | parcel id per vertex; `0` = background (medial wall) |
| `parcel_id` | `(400,)` int64 | 1…400, LH first then RH |
| `parcel_name` | `(400,)` str | e.g. `7Networks_LH_Vis_1` |
| `network` | `(400,)` str | Yeo network parsed from the label name |
| `area_mm2` | `(400,)` float | parcel surface area |
| `name`, `version` | scalar str | `schaefer400`; version embeds both annot checksums |
| `background_label` | scalar int | `0` |

### Validation on load

`load_atlas` raises, rather than coping, if any of these fail:

* `labels` is not 1-D;
* a vertex label has no row in the parcel table;
* a declared parcel has no vertices;
* a parcel id is duplicated;
* any parcel area is ≤ 0.

`assert_covers_mesh(atlas, n_vertices)` then checks the mapping against the
actual prediction array. **`len(labels) == n_vertices` is simultaneously the
coverage check and the no-double-assignment guarantee**: one label slot per
vertex, no more, no less — a vertex cannot belong to two parcels by
construction.

## Vertex → parcel, equation (6)

```
Bbar[t, p] = (1 / |V_p|) * sum_{v in V_p} Bhat[t, v]
```

A **plain unweighted mean** over the parcel's vertex set. No area weighting at
this level: fsaverage5 vertices are near-uniformly distributed by construction,
and the brief defines equation (6) as an unweighted mean.

Output is long format, one row per `(stimulus_id, time_index, parcel_id)`:

| column | meaning |
|---|---|
| `mean_bold` | equation (6) |
| `sd_bold` | SD across vertices within the parcel at that timepoint, `ddof=0` |

`sd_bold` is the information equation (6) discards, retained as a scalar so the
aggregation loss is at least visible in the table rather than invisible.

### The `weighting` argument on `vertices_to_parcels`

`vertices_to_parcels(B_hat, atlas, weighting, ...)` accepts a weighting
argument that **does not change the computation** — equation (6) is defined as
unweighted. It is carried into the output provenance so a parcel table records
which scheme its downstream network aggregation used, instead of that choice
living only in a config file that may not travel with the data.

## Parcel → network, equation (7)

```
Bbar[t, n] = sum_{p in n} w_p * Bbar[t, p] / sum_{p in n} w_p
```

| weighting | `w_p` | role |
|---|---|---|
| `area` | parcel surface area in mm² | **main specification** |
| `equal` | 1 | robustness |

Parcel areas come from the atlas file, computed by summing per-vertex areas
(the standard barycentric third-of-each-incident-triangle split) over each
parcel's vertices on the fsaverage5 pial surface.

The chosen weighting is written into a `weighting` **column** of the network
table, not just into config, so any table can answer how it was built without
its config. The two weightings give measurably different network means — this
is asserted in
[`tests/test_aggregate.py`](../tests/test_aggregate.py), so the choice can
never silently become cosmetic.

## Networks

The seven Yeo networks, parsed from the third underscore-delimited field of the
Schaefer label name (`7Networks_LH_Vis_1` → `Vis`):

`Vis`, `SomMot`, `DorsAttn`, `SalVentAttn`, `Limbic`, `Cont`, `Default`.

A label that does not parse into that shape stops the atlas build rather than
being assigned a fallback network.

## Provenance recorded downstream

Every table produced by `aggregate.py` carries, in `DataFrame.attrs["provenance"]`:

`atlas_name`, `atlas_version`, `atlas_file`, `atlas_file_sha256`, `n_vertices`,
`n_parcels`, `n_background_vertices`, the equation number, and the weighting.

Since the atlas file itself is not committed, that `sha256` is what ties a
result back to the exact parcellation that produced it.

## Whole-cortex rows

Metrics that describe the whole cortical sheet rather than one parcel (spatial
dispersion, spatial entropy, network integration) are stored with
`parcel_id = -1` and `network = "__all__"`. Schaefer parcel ids start at 1, so
the sentinel cannot collide with a real parcel.

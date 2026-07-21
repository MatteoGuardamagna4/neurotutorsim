# Data dictionary — corpus tables

## Phase 1 note

Content now comes from MBA curricula (managerial accounting, starting with break-even
analysis via contribution margin), superseding the earlier two-domain design. Development
is agile: this phase builds **one** fully validated unit
(`be_001`) end to end before any corpus scaling is attempted. Do not treat the row counts
below as targets yet.

`data/interim/units/*.json` is the **authored source of truth** — one JSON file per unit,
holding the canonical problem, worked solution, common misconception, practice problem,
near transfer, and far transfer (see `be_001.json` for the schema). `units.csv` below is a
**derived validation index**, regenerated from the JSON files by
`scripts/build_units_csv.py`. Never hand-edit `units.csv`.

## units.csv — one row per knowledge unit

| field                       | meaning                                                              |
|------------------------------|----------------------------------------------------------------------|
| unit_id                     | Stable ID, constant across all three conditions (§3.1).              |
| domain                      | e.g. `managerial_accounting`.                                        |
| concept                     | `concept_id` for the unit's anchor concept.                          |
| difficulty                  | Difficulty score 1–5 (§5.1.5).                                       |
| prerequisites                | Semicolon-separated `concept_id`s.                                   |
| validator_reference          | Validator name for the canonical (reference) problem.                |
| validator_reference_params   | JSON of params passed to the reference validator.                    |
| reference_answer             | Canonical answer; must equal validator output (§5.5, 100%).          |
| validator_transfer           | Validator name for the far-transfer problem.                         |
| validator_transfer_params    | JSON of params passed to the transfer validator.                     |
| transfer_answer              | Answer to the far-transfer problem, validated the same way.          |

## stimuli.csv — one row per condition-specific stimulus (§4.2)

| field         | meaning                                                            |
|---------------|-------------------------------------------------------------------|
| stimulus_id   | Unique per (unit_id, condition, variant).                         |
| unit_id       | FK to units.csv.                                                   |
| condition     | `traditional` \| `ai_scaffolding` \| `ai_substitution`.           |
| text          | Path to the stimulus text file under stimuli/<condition>/.        |
| modality      | `text` (main) or `audio` if TTS is generated (§5.2.10).           |
| duration      | Presentation seconds from the 220 wpm mapping (§6.2).             |
| word_count    | Matching feature (§5.3).                                          |
| readability   | Flesch–Kincaid grade (§5.4).                                      |
| equation_count| Matching feature.                                                 |
| example_count | Matching feature.                                                 |
| variant       | `primary` \| `robustness_1` \| `robustness_2` (§5.2.9).           |

Matching target: all |SMD| < 0.10; duration caliper within 10% (§5.3, §5.5). At n=1 unit
(this slice), SMD and Mahalanobis distance are mathematically undefined — they require a
pooled covariance/variance across a sample of stimuli — so `src/generation/matching.py`
raises `ValueError` instead of returning a placeholder. Only raw per-feature deltas and the
duration caliper check are reported until at least 10 units exist.

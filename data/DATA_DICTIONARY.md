# Data dictionary — corpus tables (§4.2)

## units.csv — one row per knowledge unit (60 per domain, 120 total)

| field             | meaning                                                        |
|-------------------|----------------------------------------------------------------|
| unit_id           | Stable ID, constant across all three conditions (§3.1).        |
| domain            | `bayesian_decision_analysis` or `causal_inference`.            |
| concept           | `concept_id` from the domain concept map.                      |
| difficulty        | Difficulty score 1–5 (§5.1.5).                                 |
| prerequisites     | Semicolon-separated `concept_id`s.                             |
| validator         | Name of the deterministic validator (see src/validation).      |
| validator_params  | JSON of params passed to the validator.                        |
| reference_answer  | Canonical answer; must equal validator output (§5.5, 100%).    |
| transfer_answer   | Answer to the far-transfer problem, validated the same way.    |

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

Matching target: all |SMD| < 0.10; duration caliper within 10% (§5.3, §5.5).

"""Phase 1 vertical-slice runner.

Runs the whole be_001 chain -- validator checks, feature extraction, matching
diagnostics, and content validation -- and writes:

  outputs/tables/slice_features.csv   feature vector per stimulus
  outputs/tables/slice_report.md      human-readable acceptance report

No TRIBE, no Colab, no inference: this is Track B, laptop-only, Phase 1 only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from src.generation import features as F
from src.generation import matching as M
from src.generation.stimulus_io import parse_front_matter
from src.validation import checks as C
from src.validation import validators as V

ROOT = Path(__file__).resolve().parent.parent
UNIT_PATH = ROOT / "data" / "interim" / "units" / "be_001.json"
STIMULI = {
    "traditional": ROOT / "stimuli" / "traditional" / "be_001_primary.txt",
    "ai_scaffolding": ROOT / "stimuli" / "ai_scaffolding" / "be_001_primary.txt",
    "ai_substitution": ROOT / "stimuli" / "ai_substitution" / "be_001_primary.txt",
}
ANCHOR_CONDITION = "traditional"
FEATURES_OUT = ROOT / "outputs" / "tables" / "slice_features.csv"
REPORT_OUT = ROOT / "outputs" / "tables" / "slice_report.md"

FEATURE_FIELDS = [
    "words",
    "characters",
    "sentences",
    "fk_grade",
    "equation_count",
    "example_count",
    "lexical_diversity",
    "duration_seconds",
]


def run_validator_checks(unit: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for problem_name in ("canonical_problem", "practice_problem", "near_transfer", "far_transfer"):
        p = unit[problem_name]
        results[problem_name] = V.get(p["validator"]).check(params=p["params"], answer=p["answer"])

    canonical = unit["canonical_problem"]
    wrong_answer = unit["common_misconception"]["wrong_answer"]
    results["misconception_correctly_fails"] = V.get(canonical["validator"]).check(
        params=canonical["params"], answer=wrong_answer
    )
    return results


def load_stimuli() -> Dict[str, Dict[str, Any]]:
    loaded = {}
    for condition, path in STIMULI.items():
        meta, body = parse_front_matter(path)
        loaded[condition] = {"meta": meta, "body": body}
    return loaded


def extract_all_features(stimuli: Dict[str, Dict[str, Any]]) -> Dict[str, F.FeatureVector]:
    return {
        condition: F.extract_features(data["body"], example_count=data["meta"]["example_count"])
        for condition, data in stimuli.items()
    }


def write_feature_csv(feature_vectors: Dict[str, F.FeatureVector], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", *FEATURE_FIELDS])
        writer.writeheader()
        for condition, fv in feature_vectors.items():
            writer.writerow({"condition": condition, **fv.as_dict()})


def build_report(
    unit: Dict[str, Any],
    validator_results: Dict[str, Any],
    stimuli: Dict[str, Dict[str, Any]],
    feature_vectors: Dict[str, F.FeatureVector],
) -> str:
    lines = ["# Phase 1 slice acceptance report -- be_001", ""]

    # 1. Validator results
    lines += ["## 1. Validator results", ""]
    for name, result in validator_results.items():
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"- **{name}** ({result.validator}): {status} -- {result.detail}")
    misconception_ok = not validator_results["misconception_correctly_fails"].passed
    lines.append("")
    lines.append(
        f"Misconception discriminates correctly (wrong_answer fails validation): "
        f"{'YES' if misconception_ok else 'NO -- validator is not discriminating'}"
    )
    lines.append("")

    # 2. Feature table
    lines += ["## 2. Feature table across conditions", ""]
    header = "| condition | " + " | ".join(FEATURE_FIELDS) + " |"
    sep = "|---|" + "|".join(["---"] * len(FEATURE_FIELDS)) + "|"
    lines += [header, sep]
    for condition, fv in feature_vectors.items():
        d = fv.as_dict()
        row = "| " + condition + " | " + " | ".join(f"{d[k]:.3g}" if isinstance(d[k], float) else str(d[k]) for k in FEATURE_FIELDS) + " |"
        lines.append(row)
    lines.append("")

    # 3. Per-feature deltas vs the traditional anchor
    lines += [f"## 3. Per-feature deltas vs the {ANCHOR_CONDITION} anchor", ""]
    anchor_dict = feature_vectors[ANCHOR_CONDITION].as_dict()
    for condition, fv in feature_vectors.items():
        if condition == ANCHOR_CONDITION:
            continue
        deltas = M.raw_feature_deltas(anchor_dict, fv.as_dict())
        lines.append(f"- **{condition}**: " + ", ".join(f"{k}={v:+.3g}" for k, v in deltas.items()))
    lines.append("")

    # 4. Duration caliper check (§5.5)
    lines += ["## 4. Duration caliper check (within 10% of the traditional anchor)", ""]
    anchor_duration = anchor_dict["duration_seconds"]
    for condition, fv in feature_vectors.items():
        if condition == ANCHOR_CONDITION:
            continue
        caliper = M.duration_caliper_check(anchor_duration, fv.as_dict()["duration_seconds"])
        status = "PASS" if caliper["passed"] else "FAIL"
        lines.append(f"- **{condition}**: {status} (relative difference {caliper['relative_difference']:.1%})")
    lines.append("")

    # 5. SMD / Mahalanobis distance (expected undefined at n=1)
    lines += ["## 5. Standardized mean difference / Mahalanobis distance", ""]
    try:
        M.standardized_mean_difference([anchor_dict["words"]], [feature_vectors["ai_scaffolding"].as_dict()["words"]])
    except ValueError as e:
        lines.append(f"Undefined at n=1, as expected: {e}")
    lines.append("")

    # 6. Content validation checks (§5.4)
    lines += ["## 6. Content validation checks", ""]
    for condition in ("ai_scaffolding", "ai_substitution"):
        body = stimuli[condition]["body"]
        canonical_answer = unit["canonical_problem"]["answer"]
        leakage = C.check_leakage(body, final_answer=canonical_answer)
        labels = C.classify_tutor_turns(body)
        counts = {label: labels.count(label) for label in set(labels)}
        lines.append(f"- **{condition}**: leaked_before_first_attempt={leakage.leaked}, turn_labels={counts}")
    readability = C.check_readability({c: stimuli[c]["body"] for c in stimuli})
    lines.append(
        f"- **readability spread**: {readability['spread']:.2f} grade levels "
        f"({'FLAGGED, exceeds 1.0' if readability['flagged'] else 'within 1.0 grade level'})"
    )
    lines.append("")

    # 7. Limitations
    lines += [
        "## 7. Limitations",
        "",
        "- This is a single-unit (n=1) slice. SMD and Mahalanobis distance require a "
        "pooled covariance/variance across a sample of stimuli (§5.3 eq. 2-3) and are "
        "undefined here by design -- not approximated, not silently skipped.",
        "- The `ai_scaffolding` and `ai_substitution` stimuli are hand-authored, scripted "
        "tutor/learner transcripts, not learner-generated interactions. The scripted "
        "learner error reproduces the documented misconception "
        f"({unit['common_misconception']['wrong_method']} = {unit['common_misconception']['wrong_answer']}). "
        "Real learner-driven trajectories are out of scope until the Phase III learner "
        "engine exists.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    unit = json.loads(UNIT_PATH.read_text(encoding="utf-8"))
    validator_results = run_validator_checks(unit)
    stimuli = load_stimuli()
    feature_vectors = extract_all_features(stimuli)

    write_feature_csv(feature_vectors, FEATURES_OUT)
    report = build_report(unit, validator_results, stimuli, feature_vectors)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report, encoding="utf-8")

    print(f"wrote {FEATURES_OUT}")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()

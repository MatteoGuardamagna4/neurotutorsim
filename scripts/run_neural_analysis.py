"""Track B: parcel predictions -> metrics -> contrasts -> RSA -> Phase II report.

    python scripts/run_neural_analysis.py --config config/tribe.yaml
    python scripts/run_neural_analysis.py --config config/tribe.yaml --descriptive-only

CPU only. Imports no GPU library.

**This script is expected to fail today.** The corpus is one unit; every
inferential quantity in §6.6 and §6.7 needs the 10 units of decision gate 16,
so the run stops with a ValueError naming that gate before it computes
anything. A clean run on a one-unit corpus would mean a guard is missing.

`--descriptive-only` runs the part that *is* defined at n >= 1 -- per-stimulus
metrics and paired deltas -- and writes them clearly labelled as descriptive.
It does not fit a model, bootstrap, or test anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import contrasts, rsa  # noqa: E402
from src.analysis.gates import MIN_UNITS_GATE_16, require_gate_16  # noqa: E402
from src.analysis.neural_metrics import compute_all_metrics  # noqa: E402
from src.generation.stimulus_io import build_stimulus_index  # noqa: E402
from src.tribe.cache import TribeCache  # noqa: E402
from src.tribe.config import TribeConfig, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to config/tribe.yaml")
    parser.add_argument("--cache-root", default=None, help="override cache_root")
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="metrics and paired deltas only; skips every gate-16 inferential step",
    )
    parser.add_argument(
        "--metric", default="auc", help="metric used for RSA pattern vectors (default: auc)"
    )
    return parser.parse_args()


def load_parcel_tables(config: TribeConfig, index: Dict[str, dict]) -> pd.DataFrame:
    """Pull every cached parcel table for the corpus, annotated with unit/condition.

    A stimulus with no cached prediction is reported, not skipped silently: a
    partial corpus that looks complete is how a corpus-wide standardization
    (§8.2) goes quietly wrong.
    """
    cache = TribeCache(config)
    frames: List[pd.DataFrame] = []
    absent: List[str] = []

    for stimulus_id, entry in sorted(index.items()):
        resolved = cache.resolve(stimulus_id, entry["body"], need_vertex=False, load_data=True)
        if not resolved.is_hit:
            absent.append(f"{stimulus_id} ({resolved.reason})")
            continue
        frame = resolved.data.copy()
        frame["unit_id"] = entry["unit_id"]
        frame["condition"] = entry["condition"]
        frame["variant"] = entry["variant"]
        frames.append(frame)

    if absent:
        raise SystemExit(
            f"no cached parcel prediction for {len(absent)} of {len(index)} stimuli:\n  "
            + "\n  ".join(absent)
            + f"\n\nRun Track A first (scripts/run_tribe_inference.py) against cache root "
            f"{config.cache_root}."
        )
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, cache_root=args.cache_root)
    index = build_stimulus_index(config.project_root)

    units = sorted({entry["unit_id"] for entry in index.values()})
    print(f"corpus: {len(index)} stimuli across {len(units)} unit(s): {units}")

    if not args.descriptive_only:
        # Everything this script produces beyond the descriptive tables is
        # inferential, and every inferential quantity in §6.6/§6.7 needs the
        # units of decision gate 16. Fail here rather than part-way through.
        require_gate_16(len(units), what="Phase II neural analysis (contrasts, bootstrap, RSA)")

    parcel_df = load_parcel_tables(config, index)
    print(f"loaded {len(parcel_df)} parcel-timepoint rows")

    metrics_df = compute_all_metrics(parcel_df, config.metrics)
    keys = parcel_df.drop_duplicates("stimulus_id").set_index("stimulus_id")
    for column in ("unit_id", "condition", "variant"):
        metrics_df[column] = metrics_df["stimulus_id"].map(keys[column])
    print(f"computed {len(metrics_df)} metric rows")

    tables_dir = config.project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(tables_dir / "phase2_metrics.csv", index=False)

    deltas = contrasts.paired_deltas(metrics_df)
    deltas.to_csv(tables_dir / "phase2_deltas.csv", index=False)
    print(f"wrote {tables_dir / 'phase2_metrics.csv'} and {tables_dir / 'phase2_deltas.csv'}")

    if args.descriptive_only:
        print(
            f"\n--descriptive-only: stopped before every inferential step. "
            f"{len(units)} unit(s) < {MIN_UNITS_GATE_16} required by decision gate 16."
        )
        return 0

    bootstrap = contrasts.cluster_bootstrap(
        deltas, n_boot=config.analysis.n_bootstrap, seed=config.master_seed
    )
    bootstrap.to_csv(tables_dir / "phase2_bootstrap.csv", index=False)

    models = {}
    for metric in sorted(metrics_df["metric"].unique()):
        models[metric] = contrasts.fit_unit_fe_model(metrics_df, metric=metric)

    verdicts = []
    reference = rsa.unit_pattern_vectors(metrics_df, contrasts.TRADITIONAL, metric=args.metric)
    patterns = {contrasts.TRADITIONAL: reference}
    for condition in (contrasts.AI_SCAFFOLDING, contrasts.AI_SUBSTITUTION):
        patterns[condition] = rsa.unit_pattern_vectors(
            metrics_df, condition, metric=args.metric
        )
        verdicts.append(
            rsa.classify_geometry(
                patterns[condition],
                reference,
                condition=condition,
                reference_condition=contrasts.TRADITIONAL,
            )
        )
    permutation = rsa.permutation_test(
        patterns, n_perm=config.analysis.n_permutations, seed=config.master_seed
    )

    report_path = config.project_root / "reports" / "phase2_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(config, units, metrics_df, deltas, bootstrap, models, verdicts, permutation),
        encoding="utf-8",
    )
    print(f"wrote {report_path}")
    return 0


def _render_report(config, units, metrics_df, deltas, bootstrap, models, verdicts, permutation) -> str:
    lines = [
        "# Phase II report: predicted cortical response by instructional condition",
        "",
        "Generated by `scripts/run_neural_analysis.py`. Do not edit by hand.",
        "",
        f"- units: {len(units)}",
        f"- stimuli: {metrics_df['stimulus_id'].nunique()}",
        f"- metrics: {sorted(metrics_df['metric'].unique())}",
        f"- atlas: `{config.atlas}`, parcel weighting `{config.parcel_weighting}`",
        f"- reading rate: {config.reading_rate_wpm} wpm",
        "",
        "## Paired deltas (§6.6 eq. 10-12)",
        "",
        f"{len(deltas)} within-unit differences across "
        f"{sorted(deltas['contrast'].unique())}.",
        "",
        "## Cluster bootstrap (§6.6)",
        "",
        f"Resampled over units and regenerations, {config.analysis.n_bootstrap} draws, "
        f"seed {config.master_seed}. Parcels are not resampled and no parcel- or "
        f"vertex-level p-values are reported.",
        "",
        "| contrast | metric | parcels | mean estimate |",
        "|---|---|---|---|",
    ]
    summary = bootstrap.groupby(["contrast", "metric"], as_index=False).agg(
        parcels=("parcel_id", "nunique"), mean_estimate=("estimate", "mean")
    )
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['parcels']} | "
            f"{row['mean_estimate']:+.4g} |"
        )

    lines += ["", "## Unit fixed-effects models (§6.6 eq. 13)", ""]
    for metric, model in models.items():
        lines.append(f"- **{metric}**: n={model.n_observations}, formula `{model.formula}`")

    lines += ["", "## Representational geometry (§6.7)", ""]
    for verdict in verdicts:
        rho = verdict.rdm_similarity.rho if verdict.rdm_similarity else float("nan")
        lines.append(
            f"- **{verdict.condition}** vs {verdict.reference_condition}: "
            f"`{verdict.verdict}` (distance ratio {verdict.ratio:.3f}, "
            f"RDM Spearman rho {rho:.3f})"
        )
    lines += [
        "",
        f"Permutation test (labels shuffled within matched units, "
        f"{permutation.n_perm} permutations, seed {permutation.seed}): "
        f"observed {permutation.observed:.4f}, p = {permutation.p_value:.4f}.",
        "",
        "## Language",
        "",
        "Every quantity above is a *predicted cortical response* from an encoding model "
        "applied to synthetic stimuli. It is a scenario contrast, not a causal effect, and "
        "no participant was measured.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

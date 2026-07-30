"""Generate a purely synthetic units table for Phase III development.

    python scripts/make_placeholder_units.py --n-units 120 --seed 42

The learner engine is corpus-independent by design (§2 of the Phase III task
spec): it simulates over whatever units exist. But the real corpus is one unit
and decision gate 16 blocks its expansion, so `load_curriculum` refuses to run.
This script produces the development fixture that guard is written to allow --
and only that.

Every row is stamped `provenance = "synthetic_placeholder"`, which is the *only*
provenance the gate-16 guard waives. The unit ids are `SYNTH_<domain>_<nnn>` so
they cannot be mistaken for a real id at a glance, and the answers are tokens
rather than numbers so a leaked value cannot look like a validated result.

**No output derived from this table is a project result.** Any report generated
from it carries a placeholder banner (see
`src.learners.simulate.render_validation_report`).

Track B. CPU, no torch, no network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learners.curriculum import SYNTHETIC_PLACEHOLDER  # noqa: E402

DEFAULT_DOMAINS = (
    "managerial_accounting",
    "corporate_finance",
    "operations",
    "marketing_analytics",
    "microeconomics",
)

#: §5.1 item 2: one third introductory, one third intermediate, one third
#: advanced. The difficulty integers each band draws from are a choice about the
#: 1-5 scale, not a claim about real units.
DIFFICULTY_BANDS = (
    ("introductory", (1, 2)),
    ("intermediate", (3,)),
    ("advanced", (4, 5)),
)

BANNER = """
================================================================================
  PLACEHOLDER CORPUS -- THESE UNITS ARE FICTITIOUS
================================================================================
  Every unit written by this script is invented. The domains, concepts, answers
  and misconceptions are placeholders with no instructional content behind them.

  They exist for ONE purpose: to exercise the Phase III learner engine while
  decision gate 16 (10 real units passing matching + correctness tests) is still
  open. `load_curriculum` waives the gate-16 guard for rows stamped
  provenance = "synthetic_placeholder", and for nothing else.

  NO NUMBER DERIVED FROM THIS TABLE IS A PROJECT RESULT.
  It must never appear in the manuscript, in a figure, or in a reported metric.
================================================================================
"""


def build_placeholder_units(
    n_units: int, seed: int, domains: Sequence[str] = DEFAULT_DOMAINS
) -> pd.DataFrame:
    """Build the synthetic units table.

    Parameters
    ----------
    n_units
        Number of units. The default of 120 matches the eventual corpus size so
        that a development run has the same shape as the real one.
    seed
        Master seed. Drawn from a local `default_rng`; the global `np.random`
        functions are never called anywhere in this repo.
    domains
        Domain names to cycle through.

    Returns
    -------
    pandas.DataFrame
        One row per unit, every row stamped `synthetic_placeholder`.
    """
    if n_units < 1:
        raise ValueError(f"--n-units must be >= 1; got {n_units}")
    if not domains:
        raise ValueError("--domains must name at least one domain")
    rng = np.random.default_rng(seed)

    # Equal thirds, with any remainder going to the earlier bands, so the split
    # is exact rather than approximately right.
    base, remainder = divmod(n_units, len(DIFFICULTY_BANDS))
    counts = [base + (1 if index < remainder else 0) for index in range(len(DIFFICULTY_BANDS))]

    bands: List[str] = []
    difficulties: List[int] = []
    for (band_name, levels), count in zip(DIFFICULTY_BANDS, counts):
        bands.extend([band_name] * count)
        difficulties.extend(rng.choice(np.asarray(levels), size=count).tolist())

    order = rng.permutation(n_units)
    bands = [bands[index] for index in order]
    difficulties = [int(difficulties[index]) for index in order]

    rows = []
    per_domain: dict[str, int] = {}
    for index, (band, difficulty) in enumerate(zip(bands, difficulties)):
        domain = domains[index % len(domains)]
        ordinal = per_domain.get(domain, 0) + 1
        per_domain[domain] = ordinal
        unit_id = f"SYNTH_{domain}_{ordinal:03d}"
        rows.append(
            {
                "unit_id": unit_id,
                "domain": domain,
                "concept": f"synthetic_concept_{band}_{ordinal:03d}",
                "difficulty": difficulty,
                "band": band,
                "prerequisites": "",
                "reference_answer": f"SYNTH_ANSWER_{index:03d}",
                "misconception_answer": f"SYNTH_MISCONCEPTION_{index:03d}",
                "misconception_description": (
                    f"placeholder error mode {index:03d}; no instructional content behind it"
                ),
                "provenance": SYNTHETIC_PLACEHOLDER,
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-units", type=int, default=120, help="number of units (default 120)")
    parser.add_argument("--seed", type=int, default=42, help="master seed (default 42)")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help="domain names to cycle through",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path (default data/synthetic/units_placeholder.csv)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    out_path = (
        Path(args.out)
        if args.out
        else project_root / "data" / "synthetic" / "units_placeholder.csv"
    )

    units = build_placeholder_units(args.n_units, args.seed, args.domains)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(out_path, index=False, lineterminator="\n")

    print(BANNER)
    print(f"wrote {out_path}")
    print(f"  units: {len(units)}")
    print(f"  seed: {args.seed}")
    print(f"  domains: {sorted(units['domain'].unique())}")
    print("  difficulty band split:")
    for band, count in units["band"].value_counts().sort_index().items():
        print(f"    {band}: {count} ({count / len(units):.1%})")
    print("  difficulty values:")
    for difficulty, count in units["difficulty"].value_counts().sort_index().items():
        print(f"    {difficulty}: {count}")
    print(f"  provenance: {sorted(units['provenance'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

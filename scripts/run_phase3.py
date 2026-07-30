"""Track B entry point: synthetic learners -> responses, state, checkpoints (§7).

    python scripts/run_phase3.py --units data/synthetic/units_placeholder.csv \
        --n-learners 500 --episodes 40

CPU only. Imports no GPU library, and by rule cannot: nothing under
`src/learners/` may import torch, transformers, unsloth, bitsandbytes or
accelerate.

**This script is expected to fail on the real corpus today.** `data/processed/
units.csv` holds one unit and decision gate 16 requires ten, so
`load_curriculum` raises a `ValueError` naming that gate before anything is
simulated. A clean run there would mean the guard went missing. There is no
override flag, deliberately: the failure mode gate 16 exists to prevent is a
plausible-looking simulation over a corpus nobody has validated.

Outputs
-------
=========================================  =====================================
`data/processed/responses.csv`             one row per scored response
`data/processed/learner_state.parquet`     one row per learner x episode x
                                           condition
`outputs/tables/phase3_checkpoints.csv`    one row per learner x checkpoint x
                                           condition
`outputs/tables/phase3_validation_report.md`  generated, never hand-edited
`outputs/logs/phase3_run_<timestamp>.json`    config hash, seed, git commit,
                                           package versions, row counts, clip
                                           counts, wall time
=========================================  =====================================
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learners.config import load_learner_config  # noqa: E402
from src.learners.curriculum import load_curriculum  # noqa: E402
from src.learners.simulate import checkpoint_episodes, run_simulation, sha256_file  # noqa: E402

DEFAULT_CONFIG = "config/learners.yaml"
DEFAULT_UNITS = "data/processed/units.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config/learners.yaml")
    parser.add_argument(
        "--units",
        default=DEFAULT_UNITS,
        help="explicit path to the units table; there is no corpus discovery",
    )
    parser.add_argument("--n-learners", type=int, default=None, help="override population size")
    parser.add_argument("--episodes", type=int, default=None, help="override episode count")
    parser.add_argument(
        "--conditions", nargs="+", default=None, help="override the instructional conditions"
    )
    parser.add_argument("--out", default=None, help="outputs/ root (default <project>/outputs)")
    parser.add_argument(
        "--data-dir", default=None, help="data/processed root (default <project>/data/processed)"
    )
    parser.add_argument(
        "--setting",
        default=None,
        choices=("low", "medium", "high"),
        help="§7.2 sensitivity arm; overrides parameter_setting in the config file",
    )
    parser.add_argument(
        "--engine", default=None, choices=("logistic", "minitaur"), help="override engine.name"
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="fix the run-log filename, so two runs can be compared byte for byte",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="print RuntimeWarnings as they are raised as well as recording them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_learner_config(
        args.config,
        setting=args.setting,
        n_learners=args.n_learners,
        episodes=args.episodes,
        conditions=args.conditions,
        engine=args.engine,
    )

    print(f"config: {args.config} (setting={cfg.parameter_setting}, engine={cfg.engine.name})")
    print(f"config hash: {cfg.config_hash()}")
    print(f"units: {args.units}")

    # Raises ValueError naming decision gate 16 on a real corpus of fewer than
    # ten units. Deliberately not caught: the traceback is the evidence that the
    # gate is enforced in code and not merely documented.
    units = load_curriculum(args.units, cfg.curriculum)
    print(f"loaded {len(units)} unit(s); provenance {sorted(units['provenance'].unique())}")

    print(
        f"simulating {cfg.population.n_learners} learners x {cfg.run.episodes} episodes x "
        f"{len(cfg.run.conditions)} condition(s) {list(cfg.run.conditions)}"
    )
    print(f"checkpoints inside this run: {list(checkpoint_episodes(cfg))}")

    with warnings.catch_warnings(record=not args.show_warnings) as caught:
        if not args.show_warnings:
            warnings.simplefilter("always")
        result = run_simulation(
            cfg,
            units,
            out_dir=Path(args.out) if args.out else None,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            timestamp=args.timestamp,
        )
    if caught:
        for entry in caught:
            print(f"warning: {entry.message}")

    print()
    print(f"wall time: {result.wall_time_s:.2f} s")
    print(
        f"rows: responses={result.response_rows}, learner_state={result.state_rows}, "
        f"checkpoints={result.checkpoint_rows}"
    )
    print("clip counts (§7.6 -- heavy clipping means the parameters are wrong):")
    if result.clip_counts:
        for line in _clip_lines(result):
            print(f"  {line}")
    else:
        print("  none")

    if result.separation:
        print("§10.6 condition separation after support removal:")
        for verdict in result.separation:
            state = "separated" if verdict.separated else "NOT SEPARATED"
            print(
                f"  episode {verdict.checkpoint_episode}: {verdict.difference:+.4f} "
                f"(threshold {verdict.threshold:.4f}) -> {state}"
            )
    else:
        print(
            "§10.6 condition separation not assessed: the run does not contain both "
            "ai_scaffolding and ai_substitution."
        )

    print()
    print(f"wrote {result.paths.responses_csv}")
    print(f"wrote {result.paths.learner_state_parquet}")
    print(f"  sha256 {sha256_file(result.paths.learner_state_parquet)}")
    print(f"wrote {result.paths.checkpoints_csv}")
    print(f"wrote {result.paths.validation_report}")
    print(f"wrote {result.paths.run_log}")

    if result.placeholder_corpus:
        print()
        print(
            "*** PLACEHOLDER CORPUS: every unit is stamped synthetic_placeholder. No number "
            "from this run is a project result. ***"
        )
    return 0


def _clip_lines(result) -> list[str]:
    return [
        f"{name}: {values['clipped']:.0f} / {values['total']:.0f} clipped ({values['share']:.4%})"
        for name, values in result.clip_counts.items()
    ]


if __name__ == "__main__":
    raise SystemExit(main())

"""Track A: TRIBE inference over the stimulus corpus.

    python scripts/run_tribe_inference.py --config config/tribe.yaml
    python scripts/run_tribe_inference.py --config config/tribe.yaml \
        --stimuli data/processed/stimuli.csv --cache-root /content/drive/MyDrive/NeuroTutorSim

Resolves every stimulus against the content-addressed cache before any GPU
work, writes parcel output for all of them and vertex output only for the
retention set, and appends a manifest row per item as it completes. Refuses to
start until decision gate 17 has been cleared for the pinned revision.

Safe to re-run: a completed stimulus is skipped, never regenerated (§4.3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tribe import inference  # noqa: E402
from src.tribe.cache import TribeCache  # noqa: E402
from src.tribe.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to config/tribe.yaml")
    parser.add_argument(
        "--stimuli",
        default=None,
        help="stimulus index CSV (stimulus_id + text|path). Defaults to the repo's stimuli/.",
    )
    parser.add_argument("--cache-root", default=None, help="override cache_root")
    parser.add_argument("--limit", type=int, default=None, help="process at most N stimuli")
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="run the first stimulus twice and require bitwise-equal output (§10.1)",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="cross-check the manifest against files on disk, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, cache_root=args.cache_root)

    if args.verify_manifest:
        report = TribeCache(config).verify_manifest()
        print(f"cache root: {config.cache_root}")
        print(f"missing (in manifest, absent on disk): {len(report['missing'])}")
        for path in report["missing"]:
            print(f"  MISSING {path}")
        print(f"orphans (on disk, absent from manifest): {len(report['orphans'])}")
        for path in report["orphans"]:
            print(f"  ORPHAN  {path}")
        return 1 if (report["missing"] or report["orphans"]) else 0

    if args.stimuli:
        stimuli = inference.load_stimuli_csv(args.stimuli)
    else:
        stimuli = list(inference.iter_repo_stimuli(config.project_root))
        print(f"no --stimuli index given; using {len(stimuli)} stimulus file(s) from the repo")

    if args.check_determinism:
        if not stimuli:
            raise SystemExit("no stimuli to check")
        inference.set_determinism(config)
        model = inference.load_model(config, cache_folder=config.cache_root / "hf")
        first = stimuli[0]
        print(f"determinism check on {first['stimulus_id']} (two full runs)")
        inference.assert_bitwise_reproducible(
            model, first["stimulus_id"], first["text"], config
        )
        print("bitwise equal across runs")
        return 0

    outcomes = inference.run_inference(config, stimuli, limit=args.limit)

    predicted = sum(1 for o in outcomes if o.action == "predicted")
    skipped = len(outcomes) - predicted
    retained = sum(1 for o in outcomes if o.vertex_retained)
    print(
        f"\n{len(outcomes)} stimulus/stimuli: {predicted} predicted, {skipped} skipped "
        f"(cache hits), {retained} with vertex output retained"
    )
    print(f"manifest: {TribeCache(config).manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

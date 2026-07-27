"""Decision gate 17: verify the pinned checkpoint and reproduce Meta's example.

Track A (Colab GPU). Writes `outputs/verification/gate17.json`, which
`src/tribe/inference.py` requires before it will touch a project stimulus.

    python scripts/run_tribe_verification.py --config config/tribe.yaml
    python scripts/run_tribe_verification.py --config config/tribe.yaml --resolve-revision
    python scripts/run_tribe_verification.py --config config/tribe.yaml --write-lock

`--resolve-revision` and `--write-lock` are separate, explicit steps: pinning a
revision and recording its checksums are decisions, not side effects of a run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tribe import verification  # noqa: E402
from src.tribe.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to config/tribe.yaml")
    parser.add_argument(
        "--cache-root", default=None, help="override cache_root (Colab: the mounted Drive path)"
    )
    parser.add_argument(
        "--resolve-revision",
        action="store_true",
        help="print the current commit SHA of the checkpoint repo and exit",
    )
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="record checksums for the pinned revision into checkpoints.lock and exit",
    )
    parser.add_argument(
        "--skip-example",
        action="store_true",
        help="checksum only; does NOT write a gate 17 artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, cache_root=args.cache_root)

    if args.resolve_revision:
        sha = verification.resolve_remote_revision(config)
        print(f"{config.checkpoint} currently resolves to:\n  {sha}")
        print(
            "\nPaste this into `checkpoint_revision` in your config, then run --write-lock "
            "to record its checksums. Commit both."
        )
        return 0

    if args.write_lock:
        hashes = verification.write_lock(config)
        print(f"recorded {len(hashes)} file checksum(s) in {config.checkpoints_lock}")
        return 0

    started = time.perf_counter()

    print(f"[1/4] checksumming {config.checkpoint}@{config.checkpoint_revision}")
    checksum = verification.verify_checkpoint(config)
    if not checksum.matched:
        print("CHECKSUM MISMATCH -- gate 17 fails.", file=sys.stderr)
        for field_name in ("mismatches", "missing_from_lock", "unexpected_in_lock"):
            values = getattr(checksum, field_name)
            if values:
                print(f"  {field_name}: {values}", file=sys.stderr)
        return 1
    print(f"      {len(checksum.files)} file(s) match the lock")

    if args.skip_example:
        print("[--skip-example] stopping before the official example; no gate artifact written")
        return 0

    print("[2/4] reproducing the official TRIBE example")
    example = verification.run_official_example(config)
    print(
        f"      predictions: {example.n_timesteps} timesteps x {example.n_vertices} vertices "
        f"in {example.runtime_s:.1f}s"
    )
    print(f"      figure: {example.figure_path}")

    print("[3/4] recording the environment")
    environment = verification.record_environment(config, runtime_s=time.perf_counter() - started)

    print("[4/4] writing the gate 17 artifact")
    artifact = verification.write_gate_17_artifact(config, checksum, example, environment)
    print(f"wrote {artifact}")
    print(f"gate_17_passed = {verification.gate_17_passed(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

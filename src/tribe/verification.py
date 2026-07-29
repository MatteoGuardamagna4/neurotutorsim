"""Checkpoint verification and reproduction of the official example (§6.1, items 11-15).

This module is the mechanical form of **decision gate 17**: no TRIBE inference
runs on project stimuli until the pinned checkpoint has been checksummed and
Meta's own published example has been reproduced on this machine. The gate is
not a note in a document -- `src/tribe/inference.py` calls `gate_17_passed()`
and refuses to start without it.

Track A. Heavy imports (torch, tribev2, huggingface_hub) happen *inside*
functions so that Track B can import this module on a laptop to read the gate
artifact.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tribe.config import TribeConfig

GATE_17_ARTIFACT = Path("outputs") / "verification" / "gate17.json"

#: The text used by Meta's published TRIBE v2 demo notebook for the text
#: pathway (the opening of Hamlet's soliloquy). Reproducing gate 17 means
#: running *their* example exactly as they ship it -- including their own
#: TTS/whisperx timings. Our 220 wpm override (§6.2) applies to project
#: stimuli only; applying it here would mean we were no longer reproducing
#: the official example.
OFFICIAL_EXAMPLE_TEXT = """
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
"""

#: fsaverage5 has 10242 vertices per hemisphere.
EXPECTED_N_VERTICES = 20484


class VerificationError(RuntimeError):
    """Raised when checkpoint verification fails or cannot be completed."""


@dataclass(frozen=True)
class ChecksumReport:
    checkpoint: str
    revision: str
    files: Dict[str, str]
    matched: bool
    mismatches: List[str] = field(default_factory=list)
    missing_from_lock: List[str] = field(default_factory=list)
    unexpected_in_lock: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "files": self.files,
            "matched": self.matched,
            "mismatches": self.mismatches,
            "missing_from_lock": self.missing_from_lock,
            "unexpected_in_lock": self.unexpected_in_lock,
        }


@dataclass(frozen=True)
class ExampleReport:
    n_timesteps: int
    n_vertices: int
    runtime_s: float
    figure_path: Optional[str]
    summary: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_timesteps": self.n_timesteps,
            "n_vertices": self.n_vertices,
            "runtime_s": self.runtime_s,
            "figure_path": self.figure_path,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Revision resolution and checksums
# ---------------------------------------------------------------------------


def resolve_remote_revision(config: TribeConfig) -> str:
    """Ask the Hub for the current commit SHA of `config.checkpoint`.

    Printed, never written automatically: pinning is a deliberate act recorded
    in a tracked config file, not something a script decides mid-run (§6.1).
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(config.checkpoint)
    sha = getattr(info, "sha", None)
    if not sha:
        raise VerificationError(
            f"the Hub returned no commit SHA for {config.checkpoint!r}; cannot pin a revision"
        )
    return str(sha)


def _lock_key(config: TribeConfig) -> str:
    return f"{config.checkpoint}@{config.require_resolved_revision()}"


def _read_lock(config: TribeConfig) -> Dict[str, Any]:
    path = config.checkpoints_lock
    if not path.exists():
        raise VerificationError(f"checkpoint lock file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "entries" not in data or not isinstance(data["entries"], dict):
        raise VerificationError(f"{path}: lock file must contain an 'entries' mapping")
    return data


def download_checkpoint(config: TribeConfig) -> Path:
    """Download the pinned revision and return the local snapshot directory."""
    from huggingface_hub import snapshot_download

    revision = config.require_resolved_revision()
    return Path(
        snapshot_download(repo_id=config.checkpoint, revision=revision, repo_type="model")
    )


def hash_snapshot(snapshot_dir: str | Path) -> Dict[str, str]:
    """sha256 of every regular file in the snapshot, keyed by relative path.

    The `.cache` skip targets huggingface_hub's own download bookkeeping
    *inside* the snapshot, so it is tested on the path relative to the snapshot
    root. Testing the absolute path would make the result depend on where the
    hub cache happens to live: the default `~/.cache/huggingface` would exclude
    every file, while a Drive-backed `HF_HOME` would include them all -- the
    same revision hashing to two different locks.
    """
    from src.tribe.aggregate import file_sha256

    root = Path(snapshot_dir)
    hashes: Dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if item.is_file() and ".cache" not in relative.parts:
            hashes[relative.as_posix()] = file_sha256(item)
    if not hashes:
        raise VerificationError(f"snapshot directory {root} contains no files to checksum")
    return hashes


def verify_checkpoint(config: TribeConfig) -> ChecksumReport:
    """Download the pinned checkpoint and compare it to `config/checkpoints.lock`.

    Any mismatch, and any file absent from the lock, fails the report. A lock
    entry that does not exist yet is an error, not an invitation to create one:
    use `write_lock()` explicitly (see `--write-lock`).
    """
    lock = _read_lock(config)
    key = _lock_key(config)
    if key not in lock["entries"]:
        raise VerificationError(
            f"no entry for {key!r} in {config.checkpoints_lock}. Record one explicitly with "
            f"`python scripts/run_tribe_verification.py --config <cfg> --write-lock` after "
            f"confirming the download is the checkpoint you intend to use. Auto-creating the "
            f"entry here would make the lock verify itself."
        )

    expected = dict(lock["entries"][key])
    observed = hash_snapshot(download_checkpoint(config))

    mismatches = sorted(f for f in expected if f in observed and expected[f] != observed[f])
    missing_from_lock = sorted(set(observed) - set(expected))
    unexpected_in_lock = sorted(set(expected) - set(observed))

    return ChecksumReport(
        checkpoint=config.checkpoint,
        revision=config.checkpoint_revision,
        files=observed,
        matched=not (mismatches or missing_from_lock or unexpected_in_lock),
        mismatches=mismatches,
        missing_from_lock=missing_from_lock,
        unexpected_in_lock=unexpected_in_lock,
    )


def write_lock(config: TribeConfig) -> Dict[str, str]:
    """Record checksums for the pinned revision. Explicit opt-in only.

    Refuses to overwrite an existing entry -- if the checksums for a revision
    changed, either the revision is not what it was or the download is corrupt,
    and both deserve a human.
    """
    lock = _read_lock(config)
    key = _lock_key(config)
    if key in lock["entries"]:
        raise VerificationError(
            f"{config.checkpoints_lock} already has an entry for {key!r}. Refusing to "
            f"overwrite: a changed checksum for a pinned revision means either the pin or "
            f"the download is wrong. Investigate before editing the lock by hand."
        )
    hashes = hash_snapshot(download_checkpoint(config))
    lock["entries"][key] = hashes
    config.checkpoints_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return hashes


# ---------------------------------------------------------------------------
# Official example (§6.1 item 14)
# ---------------------------------------------------------------------------


def pinned_snapshot_dir(config: TribeConfig) -> Path:
    """Local directory holding *the pinned revision* of the checkpoint.

    `snapshot_download(revision=<sha>)` is what does the pinning; the returned
    path is a `.../snapshots/<sha>/` directory, and the SHA in that path is
    checked here so a silently redirected download cannot pass for a pin.
    """
    revision = config.require_resolved_revision()
    snapshot = download_checkpoint(config)
    if revision not in snapshot.parts:
        raise VerificationError(
            f"snapshot_download({config.checkpoint!r}, revision={revision!r}) returned "
            f"{snapshot}, whose path does not contain the pinned revision. Refusing to load: "
            f"the directory cannot be shown to be the pinned commit."
        )
    return snapshot


def _pinned_checkpoint_name(snapshot: Path, default: str) -> str:
    """The `.ckpt` file to load out of a pinned snapshot directory.

    Prefers the build's own default name; falls back to a single unambiguous
    `.ckpt`. Zero or several candidates is a stop, not a guess -- loading the
    wrong weights would be invisible downstream.
    """
    if (snapshot / default).is_file():
        return default
    candidates = sorted(p.name for p in snapshot.glob("*.ckpt") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    raise VerificationError(
        f"the pinned snapshot {snapshot} contains no file named {default!r} and "
        f"{len(candidates)} alternative .ckpt file(s) ({candidates}). Cannot decide which "
        f"weights the pinned revision means. Stop and report this."
    )


def load_model(config: TribeConfig, cache_folder: str | Path | None = None):
    """`TribeModel.from_pretrained` at the pinned revision, in eval mode.

    The published tribev2 build takes `checkpoint_dir` and resolves the repo
    itself, with no way to ask for a commit -- calling it with the bare repo id
    would load whatever `main` points at today, silently breaking §6.1 and
    every cache key derived from the revision. So the pin is applied on our
    side: `snapshot_download(revision=<sha>)` (the same download
    `verify_checkpoint` checksums, so it is already on disk by the time gate 17
    reaches here) and the resulting local directory is handed over. A build
    that *does* expose `revision` is used through that argument instead.
    """
    import inspect

    from tribev2.demo_utils import TribeModel

    if config.precision != "fp32":
        raise VerificationError(
            f"precision {config.precision!r} is declared in the config and therefore in every "
            f"cache key, but this codebase has no verified way to force the installed tribev2 "
            f"build (and the Llama-3.2-3B encoder it calls) into that dtype end to end. "
            f"Loading in fp32 while the key says fp16 would make the cache lie about what it "
            f"holds. Stop and report: either establish and implement the supported dtype "
            f"switch, or keep precision at 'fp32'."
        )

    revision = config.require_resolved_revision()
    signature = inspect.signature(TribeModel.from_pretrained)

    kwargs: Dict[str, Any] = {}
    if cache_folder is not None and "cache_folder" in signature.parameters:
        kwargs["cache_folder"] = str(cache_folder)

    if "revision" in signature.parameters:
        target: str | Path = config.checkpoint
        kwargs["revision"] = revision
        pinned_by = "revision argument"
    else:
        if "checkpoint_dir" not in signature.parameters:
            raise VerificationError(
                "the installed tribev2 build's TribeModel.from_pretrained accepts neither a "
                f"`revision` nor a `checkpoint_dir` argument (signature: {signature}). Phase II "
                "pins the checkpoint by commit SHA (§6.1) and has no way to pin this build. "
                "Stop and report this: the pinned tribev2 commit changed."
            )
        target = pinned_snapshot_dir(config)
        pinned_by = "local snapshot of the pinned revision"
        name_param = signature.parameters.get("checkpoint_name")
        if name_param is not None and name_param.default is not inspect.Parameter.empty:
            kwargs["checkpoint_name"] = _pinned_checkpoint_name(
                Path(target), str(name_param.default)
            )

    model = TribeModel.from_pretrained(target, **kwargs)
    print(f"[tribe] loaded {config.checkpoint} at revision {revision} ({pinned_by})")

    inner = getattr(model, "model", model)
    if hasattr(inner, "eval"):
        inner.eval()
    return model


#: Instructions attached to every headless-rendering failure. Kept in one place
#: because the failure is silent otherwise: VTK calls `abort()` on a missing X
#: server, so the process dies with no Python traceback to search for.
_XVFB_HINT = (
    "The official example's figure (§6.1 item 14) is rendered by VTK, which needs an X "
    "display. Colab has none, and a `!python script.py` subprocess cannot use pyvista's "
    "notebook backend. Install a virtual framebuffer once per session:\n"
    "    !apt-get install -qq xvfb libgl1-mesa-glx\n"
    "then rerun. `ensure_offscreen_display()` starts Xvfb itself once the package exists."
)


def ensure_offscreen_display() -> None:
    """Give this process a display VTK can render into, or stop.

    Called before the model loads rather than before the plot: a missing X
    server makes VTK abort the interpreter outright -- no exception, no
    traceback, no gate artifact -- and finding that out after a checkpoint load
    and a forward pass wastes minutes of GPU time per attempt.
    """
    import os

    if os.environ.get("DISPLAY"):
        return

    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - depends on the Track A stack
        raise VerificationError(
            f"no DISPLAY is set and pyvista is not installed, so no virtual framebuffer can "
            f"be started.\n{_XVFB_HINT}"
        ) from exc

    try:
        pyvista.OFF_SCREEN = True
        pyvista.start_xvfb()
    except Exception as exc:
        raise VerificationError(f"could not start a virtual framebuffer.\n{_XVFB_HINT}") from exc

    if not os.environ.get("DISPLAY"):
        raise VerificationError(
            f"pyvista.start_xvfb() returned without setting DISPLAY, so VTK would still abort "
            f"the process.\n{_XVFB_HINT}"
        )
    print(f"[tribe] started a virtual framebuffer (DISPLAY={os.environ['DISPLAY']})")


def run_official_example(
    config: TribeConfig, work_dir: str | Path | None = None
) -> ExampleReport:
    """Run TRIBE's published text example and reproduce one of its summaries.

    Uses TRIBE's own `get_events_dataframe` timings, unmodified. This is the
    one place in the project where we do *not* substitute our 220 wpm onsets:
    the point of gate 17 is to reproduce Meta's example, not to run a variant
    of it.
    """
    work = Path(work_dir) if work_dir else (config.project_root / "outputs" / "verification")
    work.mkdir(parents=True, exist_ok=True)

    # Before the model, not after: the figure needs a display, and discovering
    # that at the end costs a checkpoint load plus a GPU forward pass. VTK also
    # aborts the interpreter rather than raising, so the run would end with no
    # traceback and nothing written.
    ensure_offscreen_display()

    model = load_model(config, cache_folder=work / "tribe_cache")

    text_path = work / "official_example.txt"
    text_path.write_text(OFFICIAL_EXAMPLE_TEXT, encoding="utf-8")

    started = time.perf_counter()
    events = model.get_events_dataframe(text_path=text_path)
    predictions, segments = model.predict(events=events)
    runtime_s = time.perf_counter() - started

    import numpy as np

    array = np.asarray(predictions)
    if array.ndim != 2:
        raise VerificationError(
            f"official example returned predictions of shape {array.shape}; expected 2-D "
            f"(n_timesteps, n_vertices)"
        )
    n_timesteps, n_vertices = array.shape
    if n_vertices != EXPECTED_N_VERTICES:
        raise VerificationError(
            f"official example predicted {n_vertices} vertices; fsaverage5 has "
            f"{EXPECTED_N_VERTICES}. Stop: the output mesh is not what Phase II assumes and "
            f"every parcel mapping downstream would be wrong."
        )

    figure_path = _plot_example(model, array, segments, work)

    return ExampleReport(
        n_timesteps=int(n_timesteps),
        n_vertices=int(n_vertices),
        runtime_s=float(runtime_s),
        figure_path=str(figure_path) if figure_path else None,
        summary={
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=0)),
            "min": float(array.min()),
            "max": float(array.max()),
        },
    )


def _plot_example(model, array, segments, work: Path) -> Optional[Path]:
    """Reproduce the demo's timestep visualisation (§6.1 item 14)."""
    from tribev2.plotting import PlotBrain

    n = min(15, array.shape[0])
    plotter = PlotBrain(mesh="fsaverage5")
    figure = plotter.plot_timesteps(
        array[:n],
        segments=segments[:n],
        cmap="fire",
        norm_percentile=99,
        vmin=0.6,
        alpha_cmap=(0, 0.2),
        show_stimuli=True,
    )
    out = work / "official_example_timesteps.png"
    figure.savefig(out, dpi=120, bbox_inches="tight")
    return out


# ---------------------------------------------------------------------------
# Environment record (§6.1 item 15)
# ---------------------------------------------------------------------------


def record_environment(config: TribeConfig, runtime_s: Optional[float] = None) -> Dict[str, Any]:
    """Capture the exact conditions of this verification run and write
    `docs/tribe_environment.md`.
    """
    from importlib import metadata as importlib_metadata

    packages = {}
    for name in (
        "torch", "torchaudio", "torchvision", "transformers", "huggingface-hub",
        "tribev2", "numpy", "pandas", "pyarrow", "scipy", "statsmodels", "nilearn", "nibabel",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not installed"

    gpu: Dict[str, Any] = {"available": False}
    try:
        import torch

        packages.setdefault("torch", torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {
                "available": True,
                "name": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "cuda": torch.version.cuda,
                "capability": f"{props.major}.{props.minor}",
            }
    except ImportError:
        gpu = {"available": False, "note": "torch not installed (Track B machine)"}

    record = {
        "checkpoint": config.checkpoint,
        "checkpoint_revision": config.checkpoint_revision,
        "precision": config.precision,
        "batch_size": config.batch_size,
        "reading_rate_wpm": config.reading_rate_wpm,
        "atlas": config.atlas,
        "master_seed": config.master_seed,
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "wall_clock_runtime_s": runtime_s,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = config.project_root / "docs" / "tribe_environment.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_environment(record), encoding="utf-8")
    return record


def _render_environment(record: Dict[str, Any]) -> str:
    lines = [
        "# TRIBE environment record",
        "",
        "Generated by `src.tribe.verification.record_environment` (§6.1 item 15).",
        "Do not edit by hand -- rerun `scripts/run_tribe_verification.py`.",
        "",
        f"Recorded (UTC): `{record['recorded_utc']}`",
        "",
        "## Checkpoint",
        "",
        f"- repo: `{record['checkpoint']}`",
        f"- revision: `{record['checkpoint_revision']}`",
        f"- precision: `{record['precision']}`",
        f"- batch size: `{record['batch_size']}`",
        f"- reading rate: `{record['reading_rate_wpm']}` wpm",
        f"- atlas: `{record['atlas']}`",
        f"- master seed: `{record['master_seed']}`",
        "",
        "## Hardware",
        "",
    ]
    gpu = record["gpu"]
    if gpu.get("available"):
        lines += [
            f"- GPU: `{gpu['name']}`",
            f"- VRAM: `{gpu['vram_gb']} GB`",
            f"- CUDA: `{gpu['cuda']}` (capability {gpu['capability']})",
        ]
    else:
        lines.append(f"- GPU: none detected ({gpu.get('note', 'no CUDA device')})")
    lines += [
        f"- Python: `{record['python']}`",
        f"- Platform: `{record['platform']}`",
        f"- Wall-clock runtime: `{record['wall_clock_runtime_s']}` s",
        "",
        "## Packages",
        "",
        "| package | version |",
        "|---|---|",
    ]
    for name, version in sorted(record["packages"].items()):
        lines.append(f"| `{name}` | `{version}` |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def gate_17_artifact_path(config: TribeConfig) -> Path:
    return config.project_root / GATE_17_ARTIFACT


def write_gate_17_artifact(
    config: TribeConfig,
    checksum: ChecksumReport,
    example: ExampleReport,
    environment: Dict[str, Any],
) -> Path:
    """Write `outputs/verification/gate17.json`. Only called on a full pass."""
    if not checksum.matched:
        raise VerificationError(
            "refusing to write a gate 17 artifact: checkpoint checksums did not match "
            f"({len(checksum.mismatches)} mismatch(es), "
            f"{len(checksum.missing_from_lock)} unlocked file(s))"
        )
    payload = {
        "gate": 17,
        "passed": True,
        "checkpoint": config.checkpoint,
        "checkpoint_revision": config.checkpoint_revision,
        "precision": config.precision,
        "checksum": checksum.as_dict(),
        "official_example": example.as_dict(),
        "environment": environment,
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = gate_17_artifact_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def gate_17_passed(config: TribeConfig) -> bool:
    """True only if a verification artifact exists for *this* pinned revision.

    A gate artifact from a different revision is not a pass -- that is exactly
    the situation the gate exists to catch.
    """
    path = gate_17_artifact_path(config)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(
        payload.get("passed")
        and payload.get("checkpoint") == config.checkpoint
        and payload.get("checkpoint_revision") == config.checkpoint_revision
    )

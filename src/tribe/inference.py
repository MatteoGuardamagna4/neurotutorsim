"""TRIBE v2 inference over project stimuli (§6.2-§6.3). Track A, GPU only.

Three properties this module is built around:

* **Gated.** `run_inference` calls `verification.gate_17_passed()` first and
  refuses to touch project stimuli until the checkpoint has been verified and
  Meta's official example reproduced. Decision gate 17 is enforced in code, not
  in a checklist.
* **Idempotent.** Every stimulus is resolved against the content-addressed
  cache before any GPU work. A hit is skipped and logged. A disconnected Colab
  session resumes; it never silently regenerates an existing prediction (§4.3).
* **Written as it goes.** Predictions, parcel tables and manifest rows are
  flushed per stimulus, inside the loop. Colab sessions die at 12h, on
  disconnect, or on idle, and anything held in memory dies with them.

Timing: our 220 wpm onsets (§6.2) overwrite TRIBE's TTS/whisperx-derived
timings for every project stimulus, and the disagreement is recorded to
`reports/timing_discrepancy.md`.

This module imports torch. Do not import it from Track B code.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.tribe import events as events_mod
from src.tribe import verification
from src.tribe.aggregate import Atlas, aggregate_prediction, load_atlas
from src.tribe.cache import TribeCache, text_hash
from src.tribe.config import TribeConfig
from src.tribe.verification import EXPECTED_N_VERTICES, load_model  # re-exported for scripts

__all__ = [
    "InferenceOutcome",
    "load_model",
    "predict_stimulus",
    "run_inference",
    "set_determinism",
    "assert_bitwise_reproducible",
]


class InferenceError(RuntimeError):
    """Raised when inference cannot proceed or produced an unusable result."""


@dataclass(frozen=True)
class InferenceOutcome:
    """What happened to one stimulus in the loop."""

    stimulus_id: str
    action: str  # "predicted" | "skipped_vertex" | "skipped_parcel" | "aggregated_from_vertex"
    reason: Optional[str]
    vertex_key: str
    parcel_key: str
    vertex_retained: bool
    runtime_s: float


def set_determinism(config: TribeConfig) -> None:
    """Pin every seed and switch off nondeterministic kernels (§10.1).

    Two runs on identical input under identical settings must produce
    bitwise-equal output. If a kernel here has no deterministic implementation,
    torch raises -- which is the correct outcome. Do not relax this to a
    tolerance; report it instead.
    """
    import random

    import torch

    random.seed(config.master_seed)
    np.random.seed(config.master_seed)
    torch.manual_seed(config.master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.master_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def predict_stimulus(
    model, stimulus_id: str, text: str, config: TribeConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Predict the cortical response to one stimulus.

    Returns a `(T, V)` float32 array on the fsaverage5 mesh plus a metadata
    dict. `V` is asserted to be exactly `EXPECTED_N_VERTICES` -- a different
    mesh means every parcel mapping downstream is wrong, so it stops here.
    """
    import torch

    canonical = events_mod.build_events(text, config.reading_rate_wpm)

    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / f"{stimulus_id}.txt"
        text_path.write_text(text, encoding="utf-8")

        started = time.perf_counter()
        # TRIBE renders the text to speech and re-transcribes it to build this
        # frame. We keep its audio and its row structure; we replace its
        # timings with ours (§6.2, settled).
        tribe_events = model.get_events_dataframe(text_path=text_path)
        corrected, discrepancy = events_mod.apply_our_timings(
            tribe_events, canonical, stimulus_id=stimulus_id
        )

        with torch.inference_mode():
            predictions, _segments = model.predict(events=corrected)
        runtime_s = time.perf_counter() - started

    array = np.asarray(predictions)
    if array.ndim != 2:
        raise InferenceError(
            f"{stimulus_id}: predictions have shape {array.shape}; expected 2-D "
            f"(n_timesteps, n_vertices)"
        )
    n_timesteps, n_vertices = array.shape
    if n_vertices != EXPECTED_N_VERTICES:
        raise InferenceError(
            f"{stimulus_id}: predictions have {n_vertices} vertices but fsaverage5 has "
            f"{EXPECTED_N_VERTICES}. Stop and report -- the output mesh is not the one "
            f"Phase II's parcellation assumes."
        )
    array = array.astype(np.float32, copy=False)

    metadata: Dict[str, Any] = {
        "stimulus_id": stimulus_id,
        "text_hash": text_hash(text),
        "reading_rate_wpm": float(config.reading_rate_wpm),
        "checkpoint": config.checkpoint,
        "revision": config.checkpoint_revision,
        "precision": config.precision,
        "n_timesteps": int(n_timesteps),
        "n_vertices": int(n_vertices),
        "runtime_s": float(runtime_s),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "atlas": config.atlas,
        "parcel_weighting": config.parcel_weighting,
    }
    return array, {"metadata": metadata, "timing_discrepancy": discrepancy}


def assert_bitwise_reproducible(model, stimulus_id: str, text: str, config: TribeConfig) -> None:
    """Run the same stimulus twice and require bitwise-equal output (§10.1).

    Deliberately not a tolerance check. If this fails on the target hardware,
    stop and report it -- a nondeterministic encoder makes the cache keys lie
    about what they identify.
    """
    set_determinism(config)
    first, _ = predict_stimulus(model, stimulus_id, text, config)
    set_determinism(config)
    second, _ = predict_stimulus(model, stimulus_id, text, config)
    if not np.array_equal(first, second):
        n_diff = int((first != second).sum())
        max_diff = float(np.abs(first.astype(np.float64) - second.astype(np.float64)).max())
        raise InferenceError(
            f"{stimulus_id}: two runs under identical settings are not bitwise equal "
            f"({n_diff} differing values, max |delta| = {max_diff:g}). §10.1 requires bitwise "
            f"equality. Stop and report; do not relax this to a tolerance."
        )


def run_inference(
    config: TribeConfig,
    stimuli: Sequence[Dict[str, str]],
    *,
    model=None,
    atlas: Optional[Atlas] = None,
    limit: Optional[int] = None,
) -> List[InferenceOutcome]:
    """Idempotent inference loop over `stimuli`.

    Each element is a mapping with at least `stimulus_id` and `text`.

    Retention policy (§4.5): parcel output is written for every stimulus;
    vertex output only for ids in `config.vertex_retention_set`. For everything
    else the `(T, V)` array is aggregated and then dropped -- deliberately, and
    irreversibly.
    """
    if not verification.gate_17_passed(config):
        raise InferenceError(
            f"decision gate 17 has not been cleared for {config.checkpoint}@"
            f"{config.checkpoint_revision}: no matching artifact at "
            f"{verification.gate_17_artifact_path(config)}. Run "
            f"`python scripts/run_tribe_verification.py --config <cfg>` first. Phase II does "
            f"not run inference on project stimuli against an unverified checkpoint (§6.1)."
        )

    cache = TribeCache(config)
    retention = set(config.read_vertex_retention_set())
    if atlas is None:
        atlas = load_atlas(config.atlas_file)

    outcomes: List[InferenceOutcome] = []
    discrepancies: List[events_mod.TimingDiscrepancy] = []
    lazy_model = model

    for item in list(stimuli)[:limit]:
        stimulus_id = item["stimulus_id"]
        text = item["text"]
        need_vertex = stimulus_id in retention

        resolved = cache.resolve(stimulus_id, text, need_vertex=need_vertex, load_data=False)
        if resolved.is_hit:
            print(f"[skip] {stimulus_id}: cache {resolved.status} ({resolved.vertex_key[:12]})")
            outcomes.append(
                InferenceOutcome(
                    stimulus_id=stimulus_id,
                    action=f"skipped_{resolved.status.removeprefix('hit_')}",
                    reason=None,
                    vertex_key=resolved.vertex_key,
                    parcel_key=resolved.parcel_key,
                    vertex_retained=need_vertex,
                    runtime_s=0.0,
                )
            )
            continue

        if lazy_model is None:
            set_determinism(config)
            # Not the HuggingFace cache -- HF_HOME points at the ephemeral disk,
            # because 13 GB of re-downloadable weights would crowd the corpus off
            # a 15 GB Drive. This is TRIBE's own working cache (TTS audio and the
            # whisperx word table), which is worth keeping: it costs ~2 minutes
            # per stimulus to rebuild and a few hundred KB to store.
            lazy_model = load_model(config, cache_folder=config.cache_root / "tribe_workdir")

        array, extra = predict_stimulus(lazy_model, stimulus_id, text, config)
        metadata = dict(extra["metadata"])
        discrepancies.append(extra["timing_discrepancy"])

        parcels = aggregate_prediction(
            array,
            atlas,
            stimulus_id=stimulus_id,
            weighting=config.parcel_weighting,
            metadata=metadata,
        )
        metadata["n_parcels"] = int(parcels["parcel_id"].nunique())
        metadata["vertex_key"] = resolved.vertex_key
        metadata["parcel_key"] = resolved.parcel_key

        # Write inside the loop, parcel first: parcel data is what the whole
        # corpus needs and what survives a session that dies mid-write.
        cache.write_parcel(resolved.parcel_key, parcels, metadata)
        if need_vertex:
            cache.write_vertex(resolved.vertex_key, array, metadata)
        else:
            del array  # aggregated and dropped, per the retention policy

        print(
            f"[done] {stimulus_id}: {metadata['n_timesteps']} timesteps, "
            f"vertex_retained={need_vertex}, {metadata['runtime_s']:.1f}s"
        )
        outcomes.append(
            InferenceOutcome(
                stimulus_id=stimulus_id,
                action="predicted",
                reason=resolved.reason,
                vertex_key=resolved.vertex_key,
                parcel_key=resolved.parcel_key,
                vertex_retained=need_vertex,
                runtime_s=float(metadata["runtime_s"]),
            )
        )

    if discrepancies:
        report = events_mod.write_timing_discrepancy_report(
            discrepancies, config.project_root / "reports" / "timing_discrepancy.md"
        )
        print(f"[timing] wrote {report}")

    return outcomes


def load_stimuli_csv(path: str | Path) -> List[Dict[str, str]]:
    """Read the stimulus index used by the inference loop.

    Requires `stimulus_id` and either `text` or `path` (a file whose front
    matter is stripped before use). Missing columns raise: a stimulus index
    that silently yields nothing would look like a fully cached corpus.
    """
    from src.generation.stimulus_io import parse_front_matter

    frame = pd.read_csv(path)
    if "stimulus_id" not in frame.columns:
        raise InferenceError(f"{path}: stimulus index needs a 'stimulus_id' column")
    if "text" not in frame.columns and "path" not in frame.columns:
        raise InferenceError(f"{path}: stimulus index needs either a 'text' or a 'path' column")

    root = Path(path).resolve().parent
    items: List[Dict[str, str]] = []
    for row in frame.to_dict("records"):
        if "text" in frame.columns and isinstance(row.get("text"), str):
            body = row["text"]
        else:
            stim_path = Path(str(row["path"]))
            if not stim_path.is_absolute():
                stim_path = (root / stim_path).resolve()
            _meta, body = parse_front_matter(stim_path)
        items.append({"stimulus_id": str(row["stimulus_id"]), "text": body})
    return items


def iter_repo_stimuli(project_root: str | Path) -> Iterable[Dict[str, str]]:
    """Fallback stimulus source: every `stimuli/<condition>/*.txt` in the repo.

    Used when no explicit `stimuli.csv` index exists yet -- which is the case
    while the corpus is a single unit and decision gate 16 is open.
    """
    from src.generation.stimulus_io import parse_front_matter

    for path in sorted(Path(project_root).glob("stimuli/*/*.txt")):
        meta, body = parse_front_matter(path)
        if "stimulus_id" not in meta:
            raise InferenceError(f"{path}: front matter has no stimulus_id")
        yield {"stimulus_id": str(meta["stimulus_id"]), "text": body}

"""Minitaur feasibility probe (§7.3, §10.2). GPU, Colab, run once.

    python scripts/benchmark_minitaur.py --episodes 20

Answers one question: **is Minitaur usable as the §10.2 validation engine on a
small sample?** It is a sanity check, not infrastructure. Centaur-scale
simulation is arithmetically impossible at §9.3 volumes -- ~1.2e10 episodes, one
LLM call each -- so the transparent logistic baseline stays the main engine
whatever this probe concludes.

Scoring, not generating
----------------------
Minitaur is a next-choice predictor trained on Psych-101, where the participant's
choice sits inside `<<` `>>` and inference uses `max_new_tokens=1`. It will not
produce a break-even calculation. So the probe computes the summed
log-likelihood of each choice's tokens under teacher forcing and softmaxes over
the choice set. `P(A)` from that is directly comparable to the baseline's
`p_correct`.

Why 4-bit
---------
The merged checkpoint is ~8.03B parameters in BF16, i.e. ~16 GB of weights. A T4
has 16 GB **and no native BF16** -- it is Turing, not Ampere. BF16 weights would
not fit alongside activations even if the dtype were supported. So the load is
bitsandbytes nf4 with double quantisation and fp16 compute, and peak VRAM is
reported via `torch.cuda.max_memory_allocated()`.

Isolation
---------
This script imports exactly one module from `src/learners/`: `minitaur_prompt`,
which is pure standard library. Nothing under `src/learners/` may import torch --
the learner engine runs millions of times on a laptop while this runs once on a
GPU box, and coupling them would make the cheap path depend on the expensive one.
torch, transformers and bitsandbytes are imported **inside** the functions that
need them, after preflight, so a preflight failure costs no download.

The Phase III task spec asks this script to mirror
`scripts/benchmark_tribe_timing.py`. **That script does not exist in this
repository** (the measured TRIBE timings in `src/tribe/CLAUDE.md` came from a
Colab session, not from a committed benchmark). This script therefore mirrors
`scripts/run_tribe_verification.py` instead: preflight before any download, an
environment record, measurement, then a generated report.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learners.minitaur_prompt import (  # noqa: E402
    EpisodeHistory,
    EpisodeTurn,
    choice_set_for_unit,
    scored_continuations,
    summarise_prompt,
    to_psych101_prompt,
)

#: The merged checkpoint. Preferred over `...-8B-adapter`, which additionally
#: needs the gated Meta base `meta-llama/Llama-3.1-8B` -- a second per-account
#: approval this probe deliberately does not depend on.
MERGED_MODEL = "marcelbinz/Llama-3.1-Minitaur-8B"
ADAPTER_MODEL = "marcelbinz/Llama-3.1-Minitaur-8B-adapter"
ADAPTER_BASE = "meta-llama/Llama-3.1-8B"

#: §10.2 validation target: ~50 learners x 100 episodes.
TARGET_LEARNERS = 50
TARGET_EPISODES = 100

#: Colab compute-unit burn rate, per GPU-hour. **A user-supplied rate, not a
#: fact**: Colab's pricing and its GPU classes both change, and hard-coding a
#: number here would let a stale figure drive a GO / NO-GO decision. Override
#: with --compute-units-per-hour.
DEFAULT_COMPUTE_UNITS_PER_HOUR = 1.96

REPORT_PATH = Path("reports") / "minitaur_feasibility.md"


class PreflightError(RuntimeError):
    """Raised when the probe must stop before downloading anything."""


@dataclass
class Preflight:
    """What the Hub says about the checkpoint, before a single weight is fetched."""

    model_id: str
    accessible: bool
    gated: object
    revision: Optional[str]
    detail: str
    siblings: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "accessible": self.accessible,
            "gated": self.gated,
            "revision": self.revision,
            "detail": self.detail,
            "files": self.siblings,
        }


@dataclass
class Measurement:
    """Timings and memory from the measured run."""

    n_episodes: int
    load_seconds: float
    peak_vram_bytes: int
    seconds_per_episode: List[float] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)
    p_correct: List[float] = field(default_factory=list)

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.seconds_per_episode)

    @property
    def sd_seconds(self) -> float:
        if len(self.seconds_per_episode) < 2:
            return 0.0
        return statistics.stdev(self.seconds_per_episode)

    @property
    def mean_prompt_tokens(self) -> float:
        return statistics.fmean(self.prompt_tokens)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_episodes": self.n_episodes,
            "load_seconds": self.load_seconds,
            "peak_vram_gb": self.peak_vram_bytes / 1024**3,
            "mean_seconds_per_episode": self.mean_seconds,
            "sd_seconds_per_episode": self.sd_seconds,
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "mean_p_correct": statistics.fmean(self.p_correct) if self.p_correct else None,
        }


def preflight(model_id: str, *, revision: Optional[str] = None) -> Preflight:
    """Ask the Hub about `model_id` without downloading weights.

    Raises
    ------
    PreflightError
        If the repository is gated for this account, or missing. Reported rather
        than worked around: falling back to the adapter would silently introduce
        a dependency on a *second* gated repository (`meta-llama/Llama-3.1-8B`),
        and a probe whose measurements came from a different checkpoint than the
        one it names is worse than no probe.
    """
    from huggingface_hub import model_info
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        info = model_info(model_id, revision=revision)
    except GatedRepoError as error:
        raise PreflightError(
            f"{model_id} is GATED for this account: {error}. Request access on the model page "
            f"and wait for approval. Do NOT fall back to {ADAPTER_MODEL}: the adapter needs the "
            f"gated Meta base {ADAPTER_BASE} as well, which is a second per-account approval "
            f"this probe is designed not to depend on. Stopping and reporting."
        ) from error
    except RepositoryNotFoundError as error:
        raise PreflightError(
            f"{model_id} was not found on the Hub (or the token cannot see it): {error}"
        ) from error

    resolved = revision or getattr(info, "sha", None)
    if not resolved:
        raise PreflightError(
            f"could not resolve a commit SHA for {model_id}. §6.1 requires a pinned revision; "
            f"this probe will not measure against `main`, because the numbers would describe "
            f"whatever the branch pointed at on the day."
        )
    return Preflight(
        model_id=model_id,
        accessible=True,
        gated=getattr(info, "gated", False),
        revision=resolved,
        detail="reachable; metadata read without downloading weights",
        siblings=len(getattr(info, "siblings", []) or []),
    )


def load_model_4bit(model_id: str, revision: str) -> Tuple[object, object, float, int]:
    """Load the merged checkpoint in nf4, fp16 compute. Returns (model, tokenizer, seconds, peak).

    The revision is passed explicitly. `transformers` accepts one, so unlike the
    TRIBE path (whose `from_pretrained` has no `revision` argument) the §6.1 pin
    needs no workaround here.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise PreflightError(
            "no CUDA device visible. This probe measures GPU wall-clock and peak VRAM; both are "
            "meaningless on CPU, so it stops rather than reporting numbers that cannot inform a "
            "GO / NO-GO decision."
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization,
        device_map="auto",
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    return model, tokenizer, load_seconds, int(torch.cuda.max_memory_allocated())


def score_choices(
    model: object, tokenizer: object, prompt: str, continuations: Sequence[str]
) -> Tuple[List[float], List[float]]:
    """Summed log-likelihood of each continuation under teacher forcing, then softmax.

    Returns
    -------
    (list of float, list of float)
        Log-likelihoods and the softmax probabilities over the choice set.

    Only the continuation's own tokens are scored: the prompt's log-likelihood is
    identical across choices, so including it would add a constant to every score
    and leave the softmax unchanged while making the numbers unreadable.
    """
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[1]
    log_likelihoods: List[float] = []

    with torch.no_grad():
        for continuation in continuations:
            full = tokenizer(prompt + continuation, return_tensors="pt").input_ids
            full = full.to(model.device)
            logits = model(full).logits.float()
            # Next-token prediction: position t predicts token t+1.
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = full[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scored = token_log_probs[:, n_prompt - 1 :]
            log_likelihoods.append(float(scored.sum().item()))

    tensor = torch.tensor(log_likelihoods)
    probabilities = torch.softmax(tensor, dim=0).tolist()
    return log_likelihoods, probabilities


def build_probe_episode(index: int, n_history: int) -> Tuple[EpisodeHistory, EpisodeTurn]:
    """A synthetic episode with `n_history` completed trials plus one open trial.

    Deliberately synthetic and deliberately fictitious. The probe measures
    wall-clock and VRAM, not instructional content, and using real corpus text
    here would make the timing depend on which units happened to be authored.
    """
    turns = []
    for step in range(n_history):
        correct = step % 2 == 0
        turns.append(
            EpisodeTurn(
                problem_text=(
                    f"A firm has fixed costs of {100 + step * 10} thousand euros, a unit price "
                    f"of {80 + step}, and variable costs of {50 + step} per unit. How many "
                    f"units must it sell to break even?"
                ),
                chosen="A" if correct else "B",
                correct=correct,
                hint_given=not correct,
                answer_revealed=False,
                confidence=0.5 + 0.1 * (step % 4),
            )
        )
    current = EpisodeTurn(
        problem_text=(
            f"A subscription business has fixed costs of {90 + index} thousand euros and "
            f"variable costs of {12 + index % 5} per subscriber. What monthly price lets it "
            f"break even at 3,000 subscribers?"
        ),
        hint_given=False,
    )
    return EpisodeHistory(tuple(turns)), current


def measure(
    model: object,
    tokenizer: object,
    *,
    n_episodes: int,
    n_history: int,
    load_seconds: float,
    peak_after_load: int,
) -> Measurement:
    """Score `n_episodes` episodes and record per-episode wall-clock."""
    import torch

    choice_set = choice_set_for_unit(
        reference_answer="42",
        misconception_answer="30",
        misconception_description="dividing fixed costs by price instead of contribution margin",
    )
    continuations = scored_continuations(choice_set)
    result = Measurement(
        n_episodes=n_episodes, load_seconds=load_seconds, peak_vram_bytes=peak_after_load
    )

    for index in range(n_episodes):
        history, current = build_probe_episode(index, n_history)
        prompt = to_psych101_prompt(history, choice_set, current=current)
        started = time.perf_counter()
        _, probabilities = score_choices(model, tokenizer, prompt, continuations)
        result.seconds_per_episode.append(time.perf_counter() - started)
        result.prompt_tokens.append(int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1]))
        result.p_correct.append(probabilities[0])

    result.peak_vram_bytes = max(result.peak_vram_bytes, int(torch.cuda.max_memory_allocated()))
    return result


@dataclass
class Projection:
    """The §10.2 extrapolation, with the arithmetic kept alongside the numbers."""

    n_scorings: int
    seconds: float
    hours: float
    compute_units: float
    compute_units_per_hour: float
    session_limit_hours: float
    fits_one_session: bool
    recommendation: str
    rationale: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_scorings": self.n_scorings,
            "projected_seconds": self.seconds,
            "projected_hours": self.hours,
            "compute_units": self.compute_units,
            "compute_units_per_hour": self.compute_units_per_hour,
            "session_limit_hours": self.session_limit_hours,
            "fits_one_session": self.fits_one_session,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


def extrapolate(
    measurement: Measurement,
    *,
    learners: int = TARGET_LEARNERS,
    episodes: int = TARGET_EPISODES,
    compute_units_per_hour: float = DEFAULT_COMPUTE_UNITS_PER_HOUR,
    session_limit_hours: float = 12.0,
) -> Projection:
    """Project the §10.2 validation cost from the measured per-episode rate."""
    n_scorings = learners * episodes
    seconds = measurement.mean_seconds * n_scorings
    hours = seconds / 3600.0
    fits = hours <= session_limit_hours
    if fits and hours <= session_limit_hours / 4:
        recommendation = "GO"
        rationale = (
            f"{hours:.2f} h is well inside a single {session_limit_hours:.0f} h session, with "
            f"room for the reruns a first attempt always needs."
        )
    elif fits:
        recommendation = "GO WITH CAUTION"
        rationale = (
            f"{hours:.2f} h fits a {session_limit_hours:.0f} h session but leaves little margin. "
            f"Checkpoint partial results to Drive so a disconnect does not cost the whole run, "
            f"and consider reducing the sample below {learners} x {episodes}."
        )
    else:
        recommendation = "NO-GO at this sample size"
        rationale = (
            f"{hours:.2f} h exceeds a single {session_limit_hours:.0f} h Colab session, so the "
            f"validation cannot complete in one sitting. Either cut the sample -- "
            f"{int(session_limit_hours * 3600 / measurement.mean_seconds)} scorings fit -- or "
            f"drop the LLM comparison. The transparent baseline remains the main engine either "
            f"way; §10.2 is a sanity check, not infrastructure."
        )
    return Projection(
        n_scorings=n_scorings,
        seconds=seconds,
        hours=hours,
        compute_units=hours * compute_units_per_hour,
        compute_units_per_hour=compute_units_per_hour,
        session_limit_hours=session_limit_hours,
        fits_one_session=fits,
        recommendation=recommendation,
        rationale=rationale,
    )


def environment_record() -> Dict[str, object]:
    """Package versions and GPU identity, as `record_environment` does for Track A."""
    from importlib import metadata as importlib_metadata

    packages: Dict[str, str] = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface-hub"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not installed"

    gpu: Dict[str, object] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "available": True,
                "name": properties.name,
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "cuda": torch.version.cuda,
                "capability": f"{properties.major}.{properties.minor}",
                "native_bf16": properties.major >= 8,
            }
    except ImportError:  # pragma: no cover - Track B machine
        gpu = {"available": False, "note": "torch not installed"}

    return {
        "packages": packages,
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def render_report(
    flight: Preflight,
    measurement: Measurement,
    projection: Projection,
    environment: Dict[str, object],
    *,
    n_history: int,
) -> str:
    """Generate `reports/minitaur_feasibility.md`."""
    gpu = environment.get("gpu", {})
    lines = [
        "# Minitaur feasibility probe (§7.3, §10.2)",
        "",
        "Generated by `scripts/benchmark_minitaur.py`. **Do not edit by hand.**",
        "",
        "This probe answers one question: is Minitaur usable as the §10.2 validation engine "
        "on a small sample? It is a sanity check, not infrastructure. The transparent "
        "logistic baseline is the main engine regardless of the answer, because ~1.2e10 "
        "episodes at §9.3 volumes cannot be produced by any LLM.",
        "",
        "## Recommendation",
        "",
        f"### {projection.recommendation}",
        "",
        projection.rationale,
        "",
        "## Checkpoint identity (§6.1)",
        "",
        f"- model: `{flight.model_id}` (merged; the adapter would additionally require the "
        f"gated `{ADAPTER_BASE}`)",
        f"- revision: `{flight.revision}`",
        f"- gated: `{flight.gated}`",
        f"- files in repo: {flight.siblings}",
        "",
        "## Environment",
        "",
        f"- GPU: `{gpu.get('name', 'unknown')}`, {gpu.get('vram_gb', '?')} GB, "
        f"capability {gpu.get('capability', '?')}, native BF16: "
        f"`{gpu.get('native_bf16', '?')}`",
        f"- CUDA: `{gpu.get('cuda', '?')}`",
        f"- Python: `{environment.get('python')}`",
        f"- platform: `{environment.get('platform')}`",
        "",
        "| package | version |",
        "|---|---|",
    ]
    for name, version in sorted(dict(environment.get("packages", {})).items()):
        lines.append(f"| `{name}` | {version} |")

    lines += [
        "",
        "## Measured",
        "",
        "Loaded in 4-bit (bitsandbytes nf4, double quantisation, fp16 compute). The merged "
        "checkpoint is ~8.03B parameters, ~16 GB in BF16; a 16 GB Turing card has neither the "
        "headroom nor native BF16, so 4-bit is not an optimisation but the only way it fits.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| episodes scored | {measurement.n_episodes} |",
        f"| trials of history per prompt | {n_history} |",
        f"| model load time | {measurement.load_seconds:.1f} s |",
        f"| peak VRAM | {measurement.peak_vram_bytes / 1024**3:.2f} GB |",
        f"| seconds per scored episode (mean) | {measurement.mean_seconds:.3f} |",
        f"| seconds per scored episode (SD) | {measurement.sd_seconds:.3f} |",
        f"| prompt tokens (mean) | {measurement.mean_prompt_tokens:.0f} |",
        f"| mean P(correct choice) | "
        f"{statistics.fmean(measurement.p_correct):.4f} |" if measurement.p_correct else "| mean P(correct choice) | n/a |",
        "",
        "`P(correct choice)` is the softmax over the three-option choice set, so it is "
        "directly comparable to the baseline engine's `p_correct`. It is reported here as "
        "evidence the scoring path works, **not** as a validation result: these episodes are "
        "synthetic probe fixtures, not corpus units.",
        "",
        "## Extrapolation to the §10.2 target",
        "",
        f"The validation target is ~{TARGET_LEARNERS} learners x {TARGET_EPISODES} episodes "
        f"= {projection.n_scorings:,} scorings.",
        "",
        "```text",
        f"  {measurement.mean_seconds:.3f} s/scoring x {projection.n_scorings:,} scorings",
        f"= {projection.seconds:,.0f} s",
        f"= {projection.hours:.2f} GPU-hours",
        f"x {projection.compute_units_per_hour} compute units/hour",
        f"= {projection.compute_units:.1f} Colab compute units",
        "```",
        "",
        f"- fits one {projection.session_limit_hours:.0f} h session: "
        f"`{projection.fits_one_session}`",
        f"- model load ({measurement.load_seconds:.0f} s) is paid once and is excluded from "
        f"the per-scoring rate above.",
        "",
        "⚠️ The compute-unit rate is **supplied by the caller**, not measured. Colab's pricing "
        "and GPU classes both change; treat the compute-unit figure as an order of magnitude "
        "and re-check the current rate before spending on it.",
        "",
        "## What this probe does not establish",
        "",
        "- Nothing about agreement between Minitaur and the baseline. The probe scores "
        "synthetic fixtures to measure throughput; the §10.2 comparison is a separate task.",
        "- Nothing about explanations. A choice-scoring engine yields an answer, a confidence "
        "and a support request, but **not** an explanation, which §7.3 also asks for. That "
        "trade-off is a supervisor question, recorded in `CLAUDE.md`.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MERGED_MODEL, help="HuggingFace repo id")
    parser.add_argument(
        "--revision",
        default=None,
        help="commit SHA to pin; resolved from the Hub and recorded if omitted",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="episodes to score (>= 20 per §6.2 item 5)"
    )
    parser.add_argument(
        "--history", type=int, default=5, help="completed trials per prompt (default 5)"
    )
    parser.add_argument("--learners", type=int, default=TARGET_LEARNERS)
    parser.add_argument("--target-episodes", type=int, default=TARGET_EPISODES)
    parser.add_argument(
        "--compute-units-per-hour",
        type=float,
        default=DEFAULT_COMPUTE_UNITS_PER_HOUR,
        help="Colab burn rate; a caller-supplied figure, not a measured one",
    )
    parser.add_argument("--session-limit-hours", type=float, default=12.0)
    parser.add_argument("--out", default=None, help=f"report path (default {REPORT_PATH})")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check Hub access and stop, downloading nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out) if args.out else project_root / REPORT_PATH

    if args.episodes < 20:
        raise SystemExit(
            f"--episodes must be >= 20: with fewer, the SD of the per-episode rate is too "
            f"noisy to extrapolate {args.learners * args.target_episodes} scorings from. "
            f"Got {args.episodes}."
        )

    print("=== preflight (no weights downloaded) ===")
    flight = preflight(args.model, revision=args.revision)
    print(json.dumps(flight.as_dict(), indent=2))
    if args.preflight_only:
        print("--preflight-only: stopping before the download.")
        return 0

    print("\n=== loading in 4-bit ===")
    model, tokenizer, load_seconds, peak_after_load = load_model_4bit(
        args.model, str(flight.revision)
    )
    print(f"loaded in {load_seconds:.1f} s; peak VRAM {peak_after_load / 1024**3:.2f} GB")

    print(f"\n=== scoring {args.episodes} episodes ===")
    measurement = measure(
        model,
        tokenizer,
        n_episodes=args.episodes,
        n_history=args.history,
        load_seconds=load_seconds,
        peak_after_load=peak_after_load,
    )
    print(json.dumps(measurement.as_dict(), indent=2))

    projection = extrapolate(
        measurement,
        learners=args.learners,
        episodes=args.target_episodes,
        compute_units_per_hour=args.compute_units_per_hour,
        session_limit_hours=args.session_limit_hours,
    )
    print("\n=== extrapolation ===")
    print(json.dumps(projection.as_dict(), indent=2))

    environment = environment_record()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(flight, measurement, projection, environment, n_history=args.history),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    print(f"recommendation: {projection.recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _example_prompt() -> str:
    """A prompt built on CPU, for the unit tests and for eyeballing the format."""
    choice_set = choice_set_for_unit(
        reference_answer="42",
        misconception_answer="30",
        misconception_description="dividing fixed costs by price",
    )
    history, current = build_probe_episode(0, 3)
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert summarise_prompt(prompt)["ends_with_open_marker"]
    return prompt

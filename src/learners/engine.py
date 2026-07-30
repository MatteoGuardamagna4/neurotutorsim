"""The response-engine seam (§7.3, §10.2).

§10.2 asks whether an LLM's simulated trajectories resemble the transparent
baseline's. That question is only answerable cheaply if swapping the engine is
**one config line** and nothing else -- otherwise the comparison drags in a
second code path and stops being a comparison of engines.

So the episode loop never computes a response itself. It builds an
`EpisodeBatch`, hands it to whatever satisfies `ResponseEngine`, and reads back
a `ResponseOutcome`. Both are frozen dataclasses of arrays over learners.

Two implementations exist:

`LogisticEngine`
    The main engine. Equations (17)-(18), vectorised. Built here.

`MinitaurEngine`
    Declared, **not implemented**. Its `respond` raises. Track B must not grow
    speculative LLM inference code: nothing under `src/learners/` may import
    torch, transformers, unsloth, bitsandbytes or accelerate, because this
    package runs millions of times on a laptop while the Minitaur probe runs
    once on a Colab GPU. Coupling them would make the cheap path depend on the
    expensive one. `tests/test_learners_cpu_purity.py` enforces it.

Track B. numpy only; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from src.learners import responses
from src.learners.config import STATE_LOWER, STATE_UPPER, LearnerConfig
from src.learners.seeds import SeedStreams

#: Stream episode responses are drawn from.
RESPONSE_STREAM = "responses"

#: Stream §7.7 checkpoint probes are drawn from. A probe engine is built on this
#: stream so that changing the checkpoint schedule cannot shift the response
#: draws of the episodes around it.
CHECKPOINT_STREAM = "checkpoints"

#: The four points in an episode at which a response is scored. `first` and
#: `transfer` are unaided (`h = 0`); `supported` carries the tutor's support
#: level; `probe` is a §7.7 checkpoint item, outside the episode loop.
STAGE_FIRST = "first"
STAGE_SUPPORTED = "supported"
STAGE_TRANSFER = "transfer"
STAGE_PROBE = "probe"
ALLOWED_STAGES: Tuple[str, ...] = (STAGE_FIRST, STAGE_SUPPORTED, STAGE_TRANSFER, STAGE_PROBE)

#: Message the unimplemented engine raises with. Named so a test can assert the
#: pointer stays valid rather than matching a prose fragment.
MINITAUR_REPORT = "reports/minitaur_feasibility.md"


@dataclass(frozen=True)
class EpisodeBatch:
    """One scored response, for every learner at once.

    Every array is 1-D of length `n_learners` and aligned on position. `b_u` is
    per-learner rather than per-unit because the transfer probes shift it by a
    configured offset, and a checkpoint may probe learners on different items.

    Nothing latent about the *engine* is carried here beyond what eq. (17)-(18)
    need. That matters for the Minitaur path: a prompt built from this batch
    must expose observable history only, never `K`, `C`, `alpha` or `theta`
    (see `minitaur_prompt`).
    """

    unit_id: str
    episode: int
    condition: str
    stage: str
    b_u: np.ndarray
    theta: np.ndarray
    memory: np.ndarray
    reasoning: np.ndarray
    dependence: np.ndarray
    support_level: np.ndarray
    confidence_bias: np.ndarray

    _ARRAYS = (
        "b_u",
        "theta",
        "memory",
        "reasoning",
        "dependence",
        "support_level",
        "confidence_bias",
    )

    def __post_init__(self) -> None:
        if self.stage not in ALLOWED_STAGES:
            raise ValueError(
                f"stage must be one of {list(ALLOWED_STAGES)}; got {self.stage!r}"
            )
        sizes = {name: np.asarray(getattr(self, name)).shape for name in self._ARRAYS}
        distinct = set(sizes.values())
        if len(distinct) != 1:
            raise ValueError(
                f"every EpisodeBatch array must have the same shape (one entry per learner); "
                f"got {sizes}"
            )
        if len(next(iter(distinct))) != 1:
            raise ValueError(
                f"EpisodeBatch arrays must be 1-D over learners; got shape {next(iter(distinct))}"
            )
        support = np.asarray(self.support_level, dtype=np.float64)
        if support.min() < STATE_LOWER or support.max() > STATE_UPPER:
            raise ValueError(
                f"support_level h must lie in [{STATE_LOWER}, {STATE_UPPER}]; got "
                f"[{support.min()}, {support.max()}]"
            )
        if self.stage in (STAGE_FIRST, STAGE_TRANSFER) and support.max() > STATE_LOWER:
            raise ValueError(
                f"stage {self.stage!r} is unaided by definition (§3.1 steps 1 and 4) but carries "
                f"support up to h={support.max()}. The episode order of operations is not "
                f"rearranged to give support before the first attempt."
            )

    @property
    def n_learners(self) -> int:
        return int(np.asarray(self.theta).size)


@dataclass(frozen=True)
class ResponseOutcome:
    """What an engine returns for one `EpisodeBatch`.

    `p_correct` is the engine's *own* probability. It is kept for diagnostics --
    it is what makes the §10.2 comparison quantitative rather than a comparison
    of accuracy rates -- and for the choice-scoring Minitaur engine it is the
    softmax over the choice set.
    """

    answer_correct: np.ndarray
    confidence: np.ndarray
    support_requested: np.ndarray
    p_correct: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            name: np.asarray(getattr(self, name)).shape
            for name in ("answer_correct", "confidence", "support_requested", "p_correct")
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"every ResponseOutcome array must have the same shape; got {shapes}")
        for name in ("confidence", "p_correct"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"ResponseOutcome.{name} contains non-finite values")
            if values.min() < STATE_LOWER or values.max() > STATE_UPPER:
                raise ValueError(
                    f"ResponseOutcome.{name} must lie in [{STATE_LOWER}, {STATE_UPPER}]; got "
                    f"[{values.min()}, {values.max()}]"
                )

    @property
    def n_learners(self) -> int:
        return int(np.asarray(self.p_correct).size)


@runtime_checkable
class ResponseEngine(Protocol):
    """The seam. One method, one config line to swap."""

    name: str

    def respond(self, batch: EpisodeBatch) -> ResponseOutcome:
        """Score one batch of responses, one entry per learner."""
        ...


class LogisticEngine:
    """The transparent baseline (§7.3, eq. 17-18), vectorised over learners.

    Draws come from the `responses` seed stream in a fixed order --
    correctness, then confidence, then support request -- so the stream position
    after an episode depends only on the number of learners and stages, never on
    which learners happened to answer correctly. That is what makes two runs
    with the same master seed bitwise identical (§10.1).
    """

    name = "logistic"

    def __init__(
        self, cfg: LearnerConfig, streams: SeedStreams, *, stream: str = RESPONSE_STREAM
    ) -> None:
        self._cfg = cfg
        self._stream = stream
        self._rng = streams.generator(stream)

    def respond(self, batch: EpisodeBatch) -> ResponseOutcome:
        """Equations (17)-(18) plus the confidence and support-request models."""
        response_cfg = self._cfg.response
        p_correct = responses.probability_correct(
            batch.theta,
            batch.b_u,
            batch.reasoning,
            batch.memory,
            batch.support_level,
            response_cfg,
        )
        answer_correct = responses.draw_correct(p_correct, self._rng)
        confidence = responses.draw_confidence(
            p_correct, batch.confidence_bias, response_cfg, self._rng
        )
        request_probability = responses.support_request_probability(
            batch.theta, batch.b_u, batch.dependence, response_cfg
        )
        support_requested = responses.draw_support_request(request_probability, self._rng)
        return ResponseOutcome(
            answer_correct=answer_correct,
            confidence=confidence,
            support_requested=support_requested,
            p_correct=p_correct,
        )


class MinitaurEngine:
    """Declared seam for the §10.2 validation engine. **Not implemented.**

    Minitaur is a next-choice predictor, not a dialogue partner: it was trained
    on Psych-101, where human choices sit inside `<<` `>>` tokens, and it is used
    with `max_new_tokens=1`. Used as an engine it would *score* a small discrete
    choice set, not generate a solution -- see `minitaur_prompt`.

    Whether it is usable at all is an empirical question about VRAM and
    wall-clock on free tooling, and `scripts/benchmark_minitaur.py` is the probe
    that answers it. Until that probe has run there is nothing to implement
    against, so this raises rather than guessing.
    """

    name = "minitaur"

    def __init__(
        self, cfg: LearnerConfig, streams: SeedStreams, *, stream: str = RESPONSE_STREAM
    ) -> None:
        self._cfg = cfg
        self._streams = streams
        self._stream = stream

    def respond(self, batch: EpisodeBatch) -> ResponseOutcome:
        raise NotImplementedError(
            f"MinitaurEngine is a declared seam, not a working engine. Track B is CPU-pure by "
            f"rule: nothing under src/learners/ may import torch, transformers, unsloth, "
            f"bitsandbytes or accelerate, so LLM inference cannot live here. Run "
            f"scripts/benchmark_minitaur.py on a Colab GPU first; it writes {MINITAUR_REPORT} "
            f"with the measured VRAM, seconds per scored episode and the GO / NO-GO "
            f"recommendation that decides whether this engine is worth implementing at all. "
            f"The transparent LogisticEngine remains the main engine regardless of the outcome "
            f"(engine.name: logistic)."
        )


def build_engine(
    cfg: LearnerConfig, streams: SeedStreams, *, stream: str = RESPONSE_STREAM
) -> ResponseEngine:
    """Construct the engine named by `engine.name`.

    This function is the entire cost of swapping engines. `EngineConfig` has
    already rejected any name that is not in `ALLOWED_ENGINES`, so the fallback
    below fires only if a name is added to the config layer and not here.

    `stream` selects which named RNG substream the engine draws from --
    `RESPONSE_STREAM` for the episode loop, `CHECKPOINT_STREAM` for §7.7 probes.
    """
    if cfg.engine.name == LogisticEngine.name:
        return LogisticEngine(cfg, streams, stream=stream)
    if cfg.engine.name == MinitaurEngine.name:
        return MinitaurEngine(cfg, streams, stream=stream)
    raise ValueError(  # pragma: no cover - unreachable while config validates first
        f"no engine implementation registered for engine.name={cfg.engine.name!r}"
    )

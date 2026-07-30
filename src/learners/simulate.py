"""Simulation orchestrator (§3.9): episodes x conditions -> streamed output.

One population, several counterfactual conditions
-------------------------------------------------
The same learners are run through every condition. That is what makes the
condition contrast a within-learner comparison rather than a comparison of two
populations that happened to be drawn differently. So the population is drawn
**once**, from the `population` stream with no condition label, and each
condition then gets independent `responses`, `tutor` and `checkpoints` substreams
via `SeedStreams.derive(condition)`. Consequences worth knowing:

* running `[traditional]` alone gives the traditional arm exactly the draws it
  would have received as part of `[traditional, ai_scaffolding,
  ai_substitution]`, so a partial run is comparable to a full one;
* every learner appears once per condition with the same initial state, so
  `learner_id` joins arms.

Streaming
---------
At 5,000 learners x 120 episodes x 3 conditions the response table is ~5.4M
rows, so nothing is held whole in memory: episode records are buffered for
`run.chunk_episodes` episodes and flushed to a parquet row group and a CSV
append. Checkpoints are small (learners x checkpoints x conditions) and are kept
in memory.

Determinism
-----------
Row order is fixed (condition, then episode, then learner), no wall-clock value
enters an output table, and every draw comes from a named substream. Two runs
with the same master seed therefore produce a bitwise-identical
`learner_state.parquet`; `tests/test_learner_validation.py` checks the sha256.

Track B. numpy, pandas, pyarrow; never torch.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.learners import checkpoints as checkpoints_module
from src.learners.config import LearnerConfig
from src.learners.curriculum import UnitView, is_placeholder_corpus, units_to_views
from src.learners.effort import resolve_effectiveness_columns
from src.learners.engine import CHECKPOINT_STREAM, build_engine
from src.learners.episode import (
    RESPONSE_COLUMNS,
    STATE_COLUMNS,
    EpisodeRecords,
    run_episode,
)
from src.learners.population import POPULATION_STREAM, init_population, init_state
from src.learners.seeds import SeedStreams
from src.learners.support import TUTOR_STREAM
from src.learners.updates import ClipCounter

#: `condition` is an **addition** to both §4.2 schemas. Three conditions share a
#: `(learner_id, time)` key, so without it `learner_state` has no primary key.
#: Recorded in `docs/data_dictionary_phase3.md`.
STATE_OUTPUT_COLUMNS: Tuple[str, ...] = ("condition", *STATE_COLUMNS)


@dataclass(frozen=True)
class OutputPaths:
    """Where a run writes. Every path is derived, never guessed at write time."""

    responses_csv: Path
    learner_state_parquet: Path
    checkpoints_csv: Path
    validation_report: Path
    run_log: Path

    @classmethod
    def build(
        cls,
        project_root: Path,
        *,
        out_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> "OutputPaths":
        outputs = Path(out_dir) if out_dir is not None else project_root / "outputs"
        processed = Path(data_dir) if data_dir is not None else project_root / "data" / "processed"
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(
            responses_csv=processed / "responses.csv",
            learner_state_parquet=processed / "learner_state.parquet",
            checkpoints_csv=outputs / "tables" / "phase3_checkpoints.csv",
            validation_report=outputs / "tables" / "phase3_validation_report.md",
            run_log=outputs / "logs" / f"phase3_run_{stamp}.json",
        )

    def mkdirs(self) -> None:
        for path in (
            self.responses_csv,
            self.learner_state_parquet,
            self.checkpoints_csv,
            self.validation_report,
            self.run_log,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationResult:
    """Everything a run produced, minus the streamed tables themselves."""

    config_hash: str
    master_seed: int
    paths: OutputPaths
    n_learners: int
    n_episodes: int
    conditions: Tuple[str, ...]
    n_units: int
    placeholder_corpus: bool
    response_rows: int
    state_rows: int
    checkpoint_rows: int
    clip_counts: Dict[str, Dict[str, float]]
    warnings: List[str]
    wall_time_s: float
    checkpoints: pd.DataFrame
    diagnostics: pd.DataFrame
    separation: Tuple[checkpoints_module.SeparationVerdict, ...] = ()
    environment: Dict[str, object] = field(default_factory=dict)

    def as_log(self) -> Dict[str, object]:
        """The `outputs/logs/phase3_run_<timestamp>.json` payload."""
        return {
            "config_hash": self.config_hash,
            "master_seed": self.master_seed,
            "git_commit": self.environment.get("git_commit"),
            "packages": self.environment.get("packages"),
            "python": self.environment.get("python"),
            "platform": self.environment.get("platform"),
            "n_learners": self.n_learners,
            "n_episodes": self.n_episodes,
            "conditions": list(self.conditions),
            "n_units": self.n_units,
            "placeholder_corpus": self.placeholder_corpus,
            "row_counts": {
                "responses": self.response_rows,
                "learner_state": self.state_rows,
                "checkpoints": self.checkpoint_rows,
            },
            "clip_counts": self.clip_counts,
            "warnings": self.warnings,
            "wall_time_s": self.wall_time_s,
            "separation": [
                {
                    "checkpoint_episode": verdict.checkpoint_episode,
                    "metric": verdict.metric,
                    "difference": verdict.difference,
                    "threshold": verdict.threshold,
                    "separated": verdict.separated,
                }
                for verdict in self.separation
            ],
            "outputs": {
                "responses_csv": str(self.paths.responses_csv),
                "learner_state_parquet": str(self.paths.learner_state_parquet),
                "checkpoints_csv": str(self.paths.checkpoints_csv),
                "validation_report": str(self.paths.validation_report),
            },
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }


class _ChunkedWriter:
    """Buffers episode records and flushes them to parquet + CSV.

    The parquet schema is taken from the first chunk and reused, so a later chunk
    that somehow produced a different dtype fails loudly instead of writing a
    file whose row groups disagree.
    """

    def __init__(self, paths: OutputPaths) -> None:
        self._paths = paths
        self._responses: List[Dict[str, np.ndarray]] = []
        self._states: List[Dict[str, np.ndarray]] = []
        self._writer = None
        self._schema = None
        self._csv_header_written = False
        self.response_rows = 0
        self.state_rows = 0

    def add(self, records: EpisodeRecords, condition: str) -> None:
        self._responses.append(records.responses)
        state = {"condition": np.full(records.state["learner_id"].size, condition, dtype=object)}
        state.update(records.state)
        self._states.append(state)

    @staticmethod
    def _concat(chunks: Sequence[Dict[str, np.ndarray]], columns: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                name: np.concatenate([chunk[name] for chunk in chunks])
                for name in columns
            }
        )

    def flush(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._responses:
            frame = self._concat(self._responses, RESPONSE_COLUMNS)
            frame.to_csv(
                self._paths.responses_csv,
                mode="a" if self._csv_header_written else "w",
                header=not self._csv_header_written,
                index=False,
                lineterminator="\n",
            )
            self._csv_header_written = True
            self.response_rows += len(frame)
            self._responses = []

        if self._states:
            frame = self._concat(self._states, STATE_OUTPUT_COLUMNS)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._writer is None:
                self._schema = table.schema
                self._writer = pq.ParquetWriter(self._paths.learner_state_parquet, self._schema)
            elif not table.schema.equals(self._schema):
                raise ValueError(
                    f"learner_state chunk schema drifted mid-run:\nexpected {self._schema}\n"
                    f"got {table.schema}"
                )
            self._writer.write_table(table)
            self.state_rows += len(frame)
            self._states = []

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _git_commit(project_root: Path) -> Optional[str]:
    """Current commit, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return result.stdout.strip() or None


def _environment(project_root: Path) -> Dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "pyyaml"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            packages[name] = "not installed"
    return {
        "git_commit": _git_commit(project_root),
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def sha256_file(path: Path) -> str:
    """sha256 of a file, for the §10.1 determinism check.

    `hashlib`, never Python's `hash()`: the built-in is salted per process and
    would differ between the two runs the check compares.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_episodes(cfg: LearnerConfig) -> Tuple[int, ...]:
    """Configured checkpoint indices that fall inside this run's episode count.

    A checkpoint at episode 0 is dropped: no unit has been taught yet, so there
    are no trained items to probe.
    """
    return tuple(
        index for index in cfg.checkpoints.episodes if 0 < index < cfg.run.episodes
    )


def run_simulation(
    cfg: LearnerConfig,
    units: pd.DataFrame,
    *,
    out_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> SimulationResult:
    """Run every condition for `cfg.run.episodes` episodes and write the outputs.

    Parameters
    ----------
    cfg
        Full learner configuration.
    units
        Units frame from `curriculum.load_curriculum`. §3.9's signature is
        `run_simulation(cfg)`; the corpus arrives as an argument because §2
        requires it to be passed by explicit path and never discovered.
    out_dir, data_dir
        Override the `outputs/` and `data/processed/` roots.
    timestamp
        Fixes the run-log filename, so a determinism check can compare two runs
        without their log names differing.

    Returns
    -------
    SimulationResult
    """
    started = time.perf_counter()
    units, warnings_raised = resolve_effectiveness_columns(units, cfg.effectiveness)
    unit_views = units_to_views(units)
    if not unit_views:
        raise ValueError("the units table is empty; there is nothing to simulate")

    paths = OutputPaths.build(
        cfg.project_root, out_dir=out_dir, data_dir=data_dir, timestamp=timestamp
    )
    paths.mkdirs()

    base = SeedStreams(cfg.seeds.master, cfg.seeds.streams)
    population = init_population(cfg.population.n_learners, cfg, base)

    counter = ClipCounter()
    writer = _ChunkedWriter(paths)
    checkpoint_frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, float]] = []
    scheduled = checkpoint_episodes(cfg)

    try:
        for condition in cfg.run.conditions:
            streams = base.derive(condition)
            state = init_state(population)
            engine = build_engine(cfg, streams)
            probe_engine = build_engine(cfg, streams, stream=CHECKPOINT_STREAM)
            tutor_rng = streams.generator(TUTOR_STREAM)
            checkpoint_rng = streams.generator(CHECKPOINT_STREAM)
            seen: List[UnitView] = []

            for episode in range(cfg.run.episodes):
                unit = unit_views[episode % len(unit_views)]
                state, records = run_episode(
                    state,
                    unit,
                    condition,
                    engine,
                    cfg,
                    tutor_rng,
                    episode=episode,
                    counter=counter,
                )
                seen.append(unit)
                writer.add(records, condition)
                diagnostics.append({"condition": condition, **records.diagnostics})

                if episode in scheduled:
                    checkpoint_frames.append(
                        checkpoints_module.run_checkpoint(
                            state,
                            seen,
                            condition,
                            probe_engine,
                            cfg,
                            checkpoint_rng,
                            episode=episode,
                            counter=counter,
                        )
                    )

                if (episode + 1) % cfg.run.chunk_episodes == 0:
                    writer.flush()
            writer.flush()
    finally:
        writer.close()

    checkpoint_table = (
        pd.concat(checkpoint_frames, ignore_index=True)
        if checkpoint_frames
        else pd.DataFrame(columns=list(checkpoints_module.CHECKPOINT_COLUMNS))
    )
    checkpoint_table.to_csv(paths.checkpoints_csv, index=False, lineterminator="\n")
    diagnostics_table = pd.DataFrame(diagnostics)

    result = SimulationResult(
        config_hash=cfg.config_hash(),
        master_seed=cfg.seeds.master,
        paths=paths,
        n_learners=cfg.population.n_learners,
        n_episodes=cfg.run.episodes,
        conditions=tuple(cfg.run.conditions),
        n_units=len(unit_views),
        placeholder_corpus=is_placeholder_corpus(units),
        response_rows=writer.response_rows,
        state_rows=writer.state_rows,
        checkpoint_rows=len(checkpoint_table),
        clip_counts=counter.as_dict(),
        warnings=list(warnings_raised),
        wall_time_s=time.perf_counter() - started,
        checkpoints=checkpoint_table,
        diagnostics=diagnostics_table,
        separation=checkpoints_module.condition_separation(checkpoint_table, cfg),
        environment=_environment(cfg.project_root),
    )

    paths.validation_report.write_text(render_validation_report(result, cfg), encoding="utf-8")
    paths.run_log.write_text(
        json.dumps(result.as_log(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def render_validation_report(result: SimulationResult, cfg: LearnerConfig) -> str:
    """Generate `outputs/tables/phase3_validation_report.md`. Never hand-edited.

    The §10.6 separation verdict is written whether or not the conditions
    separated. A run where scaffolding and substitution are indistinguishable
    after support removal is a falsification signal about the model, and the
    report is where it is recorded rather than passed over.
    """
    lines = [
        "# Phase III validation report",
        "",
        "Generated by `src.learners.simulate.render_validation_report` "
        "(via `scripts/run_phase3.py`). **Do not edit by hand.**",
        "",
        "Every learner below is a *simulated* learner from a declared generative model. "
        "No participant was measured, nothing here is a causal effect, and no quantity in "
        "this report is a neural or cortical measurement.",
        "",
        "## Run",
        "",
        f"- config hash: `{result.config_hash}`",
        f"- parameter setting: `{cfg.parameter_setting}`",
        f"- master seed: `{result.master_seed}`",
        f"- engine: `{cfg.engine.name}`",
        f"- persistence policy: `{cfg.support.persistence_policy}`",
        f"- learners: {result.n_learners}",
        f"- episodes: {result.n_episodes}",
        f"- conditions: {list(result.conditions)}",
        f"- units: {result.n_units}",
        f"- wall time: {result.wall_time_s:.2f} s",
        "",
    ]

    if result.placeholder_corpus:
        lines += [
            "> ## ⚠️ PLACEHOLDER CORPUS",
            ">",
            "> Every unit in this run is stamped `synthetic_placeholder`. The units are "
            "fictitious, generated by `scripts/make_placeholder_units.py` for development. "
            "**No number in this report is a project result** and none of it may be reported, "
            "cited or carried into the manuscript.",
            "",
        ]

    if result.warnings:
        lines += ["## Warnings raised during the run", ""]
        lines += [f"- {message}" for message in result.warnings]
        lines += [""]

    lines += ["## Clip counts (§7.6)", ""]
    if result.clip_counts:
        lines += [
            "Heavy clipping means the parameters are wrong, not that the learners are unusual.",
            "",
            "| quantity | clipped | total | share |",
            "|---|---|---|---|",
        ]
        for name, values in result.clip_counts.items():
            lines.append(
                f"| `{name}` | {values['clipped']:.0f} | {values['total']:.0f} | "
                f"{values['share']:.4%} |"
            )
    else:
        lines.append("No clipped values recorded.")
    lines += [""]

    if not result.diagnostics.empty:
        summary = result.diagnostics.groupby("condition", as_index=False).agg(
            mean_effort=("mean_effort", "mean"),
            mean_effectiveness=("mean_effectiveness", "mean"),
            mean_support=("mean_support", "mean"),
            unaided_accuracy=("mean_unaided_correct", "mean"),
            transfer_accuracy=("mean_transfer_correct", "mean"),
            answer_provided=("mean_answer_provided", "mean"),
            offloading=("mean_offloading", "mean"),
        )
        lines += [
            "## Episode means by condition",
            "",
            "`mean_effort` is eq. (19) `E`, the quantity Phase IV multiplies by the predicted "
            "cortical response. Phase III ends here; no plasticity is computed.",
            "",
            "| condition | E | F | h | unaided acc | transfer acc | answer given | offloading |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in summary.to_dict("records"):
            lines.append(
                f"| {row['condition']} | {row['mean_effort']:.4f} | "
                f"{row['mean_effectiveness']:.4f} | {row['mean_support']:.4f} | "
                f"{row['unaided_accuracy']:.4f} | {row['transfer_accuracy']:.4f} | "
                f"{row['answer_provided']:.4f} | {row['offloading']:.4f} |"
            )
        lines += [""]

    lines += ["## Checkpoints (§7.7), all support removed", ""]
    if result.checkpoints.empty:
        lines += [
            "No checkpoint fell inside this run's episode range "
            f"(configured: {list(cfg.checkpoints.episodes)}, episodes: {result.n_episodes}).",
            "",
        ]
    else:
        grouped = result.checkpoints.groupby(
            ["condition", "checkpoint_episode"], as_index=False
        ).mean(numeric_only=True)
        lines += [
            "| condition | ep | unaided | supported | gap (eq. 26) | near | far | retention "
            "| Brier | ECE | dependence |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in grouped.to_dict("records"):
            lines.append(
                f"| {row['condition']} | {int(row['checkpoint_episode'])} | "
                f"{row['unaided_accuracy_trained']:.4f} | "
                f"{row['supported_accuracy_trained']:.4f} | {row['support_gap']:+.4f} | "
                f"{row['near_transfer_accuracy']:.4f} | {row['far_transfer_accuracy']:.4f} | "
                f"{row['retention_accuracy']:.4f} | {row['brier_score']:.4f} | "
                f"{row['expected_calibration_error']:.4f} | {row['dependence']:.4f} |"
            )
        lines += [""]

    lines += ["## §10.6 condition separation after support removal", ""]
    if not result.separation:
        lines += [
            "Not assessed: the run does not contain both `ai_scaffolding` and "
            "`ai_substitution`, and two conditions cannot be separated when one is absent.",
            "",
        ]
    else:
        unseparated = [verdict for verdict in result.separation if not verdict.separated]
        lines += [
            "| ep | metric | condition A | condition B | difference | threshold | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [verdict.as_line() for verdict in result.separation]
        lines += [""]
        if unseparated:
            lines += [
                f"⚠️ **§10.6 falsification signal at {len(unseparated)} of "
                f"{len(result.separation)} checkpoint(s).** Scaffolding and substitution are "
                "not distinguishable there once support is withdrawn. This is recorded rather "
                "than passed over: either the effort mechanism is too weak at these parameter "
                "values, or the model does not imply the difference the study is looking for.",
                "",
            ]
        else:
            lines += [
                "Scaffolding and substitution are distinguishable at every checkpoint once "
                "support is withdrawn.",
                "",
            ]

    lines += [
        "## Language",
        "",
        "Every quantity above is a *model-implied* trajectory of a *simulated* learner. "
        "Differences between conditions are scenario contrasts, not causal effects, and the "
        "parameter values that produce them are assumptions documented in "
        "`docs/phase3_assumptions.md`.",
        "",
    ]
    return "\n".join(lines)

"""Neural response metrics over parcel-level TRIBE predictions (§6.5).

**Decision gate 19 is enforced here.** No cortical metric may be interpreted
until its direction and meaning are documented, so every metric function below
carries a structured docstring block:

    Formula:       the arithmetic, explicitly
    Units:         what the number is measured in
    Increase means: what a larger value corresponds to
    Direction:     `meaningful` if larger is interpretable in one direction,
                   `ambiguous` if it is not

`scripts/build_metrics_reference.py` parses those four fields to generate
`docs/metrics_reference.md`, and fails if any registered metric omits one. The
documentation cannot silently drift from the code, and a metric cannot enter
the pipeline undocumented.

Every function here is pure and independently testable. Track B: numpy, pandas
and scipy only -- no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.tribe.config import MetricsConfig

# numpy renamed trapz -> trapezoid in 2.0; pyproject allows >=1.26, so bind once
# here rather than sprinkling version checks through the metric functions.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: `parcel_id` used for rows that describe the whole cortical sheet rather than
#: one parcel (spatial dispersion, spatial entropy, network integration). A
#: real Schaefer parcel id is always >= 1, so -1 cannot collide.
WHOLE_CORTEX = -1

#: Non-negative spatial summary that the whole-cortex metrics are computed
#: over. Recorded in the output rather than left implicit: entropy and
#: dispersion are only defined on a magnitude, and choosing that magnitude is a
#: modelling decision, not a detail.
SPATIAL_BASIS = "abs_auc"

TIMESERIES_METRICS = (
    "mean_response",
    "peak_response",
    "auc",
    "time_to_peak",
    "sustained_engagement",
)
SPATIAL_METRICS = ("spatial_dispersion", "spatial_entropy", "network_integration")
CROSS_STIMULUS_METRICS = ("representational_differentiation", "within_concept_consistency")


# ---------------------------------------------------------------------------
# Helpers (not metrics -- no docstring block, not exported to the reference)
# ---------------------------------------------------------------------------


def _as_series(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D time series; got shape {array.shape}")
    if array.size == 0:
        raise ValueError("time series is empty")
    if not np.isfinite(array).all():
        raise ValueError("time series contains non-finite values")
    return array


def window_slice(
    n_timesteps: int, tr_seconds: float, window: Optional[Tuple[float, float]]
) -> slice:
    """Index range of the instructional window, in samples.

    A window that selects nothing raises: an empty window silently turns every
    metric into a statistic of no data.
    """
    if window is None:
        return slice(0, n_timesteps)
    start = int(np.floor(window[0] / tr_seconds))
    stop = int(np.ceil(window[1] / tr_seconds))
    start, stop = max(start, 0), min(stop, n_timesteps)
    if stop <= start:
        raise ValueError(
            f"instructional window {window} selects no samples from a {n_timesteps}-sample "
            f"series at TR={tr_seconds}s"
        )
    return slice(start, stop)


def smooth(values: Sequence[float], kernel: str, width_s: float, tr_seconds: float) -> np.ndarray:
    """Temporal smoothing used before peak detection.

    `boxcar` is a centred moving average of `round(width_s / TR)` samples, with
    edge-replicated padding so the first and last samples are not artificially
    attenuated toward zero. `gaussian` uses sigma = width_s / TR.
    """
    series = _as_series(values)
    if kernel == "boxcar":
        n = max(int(round(width_s / tr_seconds)), 1)
        if n % 2 == 0:
            n += 1  # centred window needs an odd length
        if n <= 1:
            return series.copy()
        pad = n // 2
        padded = np.pad(series, pad, mode="edge")
        return np.convolve(padded, np.ones(n) / n, mode="valid")
    if kernel == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        return np.asarray(
            gaussian_filter1d(series, sigma=width_s / tr_seconds, mode="nearest"), dtype=np.float64
        )
    raise ValueError(f"unknown smoothing kernel {kernel!r}; expected 'boxcar' or 'gaussian'")


# ---------------------------------------------------------------------------
# Time-series metrics (one value per stimulus x parcel)
# ---------------------------------------------------------------------------


def mean_response(series: Sequence[float]) -> float:
    """Average predicted response over the instructional window.

    Formula: (1/T) * sum_t x_t, over the samples inside the instructional window.
    Units: predicted BOLD units (arbitrary, TRIBE's output scale).
    Increase means: a higher average predicted response across the window.
    Direction: ambiguous -- a larger predicted response is not by itself better
        or worse learning; it is only interpretable relative to a matched
        contrast condition.
    """
    return float(_as_series(series).mean())


def peak_response(
    series: Sequence[float], kernel: str, width_s: float, tr_seconds: float
) -> float:
    """Maximum of the temporally smoothed response.

    Formula: max_t (x * k)_t, where k is the configured smoothing kernel.
    Units: predicted BOLD units.
    Increase means: a stronger maximal response once high-frequency fluctuation
        has been smoothed away.
    Direction: ambiguous -- peak amplitude conflates the size of a response with
        how concentrated it is in time.
    """
    return float(smooth(series, kernel, width_s, tr_seconds).max())


def auc(series: Sequence[float], tr_seconds: float) -> float:
    """Area under the response curve, equation (8).

    Formula: trapezoidal integral sum_t (x_t + x_{t+1}) / 2 * TR.
    Units: predicted BOLD units x seconds.
    Increase means: more total predicted response integrated over the window --
        a larger response, a longer one, or both.
    Direction: meaningful -- AUC is the quantity §8.2 standardizes into Z(u,c,p),
        and larger unambiguously means more integrated predicted response. It
        does not, on its own, mean more learning.
    """
    values = _as_series(series)
    if tr_seconds <= 0:
        raise ValueError(f"tr_seconds must be > 0; got {tr_seconds}")
    return float(_trapezoid(values, dx=tr_seconds))


def time_to_peak(
    series: Sequence[float], kernel: str, width_s: float, tr_seconds: float
) -> float:
    """Latency of the smoothed maximum, from the start of the window.

    Formula: TR * argmax_t (x * k)_t.
    Units: seconds.
    Increase means: the strongest predicted response arrives later in the
        stimulus.
    Direction: meaningful -- later is later. Whether late peaks are desirable is
        a separate question this metric does not answer.
    """
    smoothed = smooth(series, kernel, width_s, tr_seconds)
    return float(int(np.argmax(smoothed)) * tr_seconds)


def sustained_engagement(
    series: Sequence[float],
    tr_seconds: float,
    baseline_percentile: float,
    threshold_sd: float,
) -> float:
    """Time spent above a stimulus-specific baseline threshold.

    Formula: TR * #{t : x_t > b + k*sd(x)}, where b is the `baseline_percentile`
        percentile of this stimulus's own series and k is `threshold_sd`.
    Units: seconds.
    Increase means: the predicted response stays elevated above its own baseline
        for longer.
    Direction: meaningful -- a longer time above threshold is unambiguously more
        sustained. The baseline is per stimulus, so this is not comparable in
        absolute terms across stimuli of different overall amplitude.
    """
    values = _as_series(series)
    if not 0.0 <= baseline_percentile <= 100.0:
        raise ValueError(f"baseline_percentile must be in [0, 100]; got {baseline_percentile}")
    baseline = float(np.percentile(values, baseline_percentile))
    threshold = baseline + threshold_sd * float(values.std(ddof=0))
    return float((values > threshold).sum() * tr_seconds)


# ---------------------------------------------------------------------------
# Spatial metrics (one value per stimulus, over the parcel vector)
# ---------------------------------------------------------------------------


def spatial_dispersion(parcel_values: Sequence[float]) -> float:
    """Spread of the response magnitude across parcels.

    Formula: population standard deviation over parcels, sd_p(x_p) with ddof=0.
    Units: same units as the parcel summary it is computed on (BOLD x seconds
        when the summary is |AUC|).
    Increase means: response magnitude is distributed more unevenly across
        cortex -- some parcels far above others.
    Direction: ambiguous -- high dispersion is consistent with a focal,
        well-targeted response and with a noisy one, and the metric cannot
        distinguish them.
    """
    values = np.asarray(parcel_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"spatial_dispersion needs a 1-D vector of at least 2 parcels; got shape "
            f"{values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("parcel vector contains non-finite values")
    return float(values.std(ddof=0))


def spatial_entropy(parcel_values: Sequence[float]) -> float:
    """Shannon entropy of the normalized spatial response profile, equation (9).

    Formula: p_p = x_p / sum_p x_p ; H = -sum_p p_p * log(p_p), natural log,
        with the convention 0*log 0 = 0.
    Units: nats (0 = all response in one parcel; log(P) = perfectly uniform).
    Increase means: predicted response is spread more evenly across parcels.
    Direction: meaningful -- higher entropy is unambiguously a flatter spatial
        distribution. Whether flatter is better is not a claim this metric makes.

    Requires a non-negative vector that sums to a positive number, because a
    normalized profile is only defined there. An all-zero, all-negative or
    mixed-sign vector raises: there is no defensible entropy to return, and
    returning 0 would be indistinguishable from a genuinely concentrated
    response.
    """
    values = np.asarray(parcel_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"spatial_entropy needs a 1-D vector of at least 2 parcels; got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("parcel vector contains non-finite values")
    if (values < 0).any():
        raise ValueError(
            "spatial_entropy requires a non-negative spatial summary (equation 9 normalizes "
            "the vector into a probability distribution). Supply a magnitude such as |AUC|; "
            "this function will not shift or rectify the input for you."
        )
    total = values.sum()
    if total <= 0:
        raise ValueError(
            "spatial_entropy is undefined for an all-zero parcel vector: there is no "
            "distribution to normalize. Returning 0 would falsely report a maximally "
            "concentrated response."
        )
    p = values / total
    nonzero = p[p > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def network_integration(
    parcel_timeseries: np.ndarray, networks: Sequence[str]
) -> float:
    """Ratio of between-network to within-network parcel coupling.

    Formula: mean r(p, q) for parcel pairs in different networks, divided by
        mean r(p, q) for pairs in the same network, where r is the Pearson
        correlation of the two parcels' predicted time series.
    Units: dimensionless ratio (1.0 = between-network coupling as strong as
        within-network; < 1 = networks are segregated).
    Increase means: parcels in different networks behave more alike -- a more
        integrated, less segregated predicted response.
    Direction: meaningful -- higher is more integrated. Note this is the inverse
        sense of the segregation indices used elsewhere in the fMRI literature,
        so the sign convention must be stated whenever it is reported.

    Raises if mean within-network coupling is not positive: the ratio flips sign
    or explodes there, and a ratio through zero is not interpretable.
    """
    matrix = np.asarray(parcel_timeseries, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"network_integration needs a 2-D (n_parcels, n_timesteps) array; got "
            f"{matrix.shape}"
        )
    labels = np.asarray(list(networks))
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"got {labels.shape[0]} network labels for {matrix.shape[0]} parcels"
        )
    if len(set(labels.tolist())) < 2:
        raise ValueError(
            f"network_integration requires >= 2 distinct networks; got "
            f"{sorted(set(labels.tolist()))}. With one network there are no between-network "
            f"pairs to average."
        )
    if matrix.shape[1] < 2:
        raise ValueError("network_integration needs at least 2 timepoints to correlate")

    sd = matrix.std(axis=1, ddof=0)
    if (sd == 0).any():
        raise ValueError(
            f"{int((sd == 0).sum())} parcel(s) have a constant time series; their correlation "
            f"with anything is undefined"
        )
    corr = np.corrcoef(matrix)
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    within = corr[upper & same]
    between = corr[upper & ~same]
    if within.size == 0 or between.size == 0:
        raise ValueError("need at least one within-network and one between-network parcel pair")

    mean_within = float(within.mean())
    if mean_within <= 0:
        raise ValueError(
            f"mean within-network coupling is {mean_within:.4f} (not positive); the "
            f"between/within ratio is not interpretable through zero"
        )
    return float(between.mean() / mean_within)


# ---------------------------------------------------------------------------
# Cross-stimulus metrics (need more than one concept / variant)
# ---------------------------------------------------------------------------


def representational_differentiation(patterns_by_concept: Mapping[str, np.ndarray]) -> float:
    """How distinct different concepts' cortical patterns are from each other.

    Formula: mean over concept pairs of 1 - r(z_a, z_b), where z is a concept's
        parcel-pattern vector and r is the Pearson correlation across parcels.
    Units: correlation distance (0 = identical patterns, 1 = uncorrelated,
        2 = perfectly anticorrelated).
    Increase means: concepts are represented more distinctly from one another.
    Direction: meaningful -- higher is more differentiated.

    Requires >= 2 concepts. One concept has no pairs, and the mean of an empty
    set is not zero differentiation, it is no measurement at all.
    """
    concepts = sorted(patterns_by_concept)
    if len(concepts) < 2:
        raise ValueError(
            f"representational_differentiation requires >= 2 concepts; got {len(concepts)} "
            f"({concepts}). With a single concept there are no between-concept pairs and the "
            f"metric is undefined -- not zero."
        )
    matrix = _stack_patterns({c: patterns_by_concept[c] for c in concepts})
    corr = np.corrcoef(matrix)
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float((1.0 - corr[upper]).mean())


def within_concept_consistency(patterns_by_variant: Mapping[str, np.ndarray]) -> float:
    """How reproducible one concept's cortical pattern is across its variants.

    Formula: mean over variant pairs of r(z_i, z_j), Pearson correlation across
        parcels between two variants of the same concept.
    Units: correlation coefficient in [-1, 1].
    Increase means: regenerations and variants of the same concept produce more
        similar predicted patterns -- a more stable representation.
    Direction: meaningful -- higher is more consistent.

    Requires >= 2 variants of the concept, for the same reason as
    `representational_differentiation`.
    """
    variants = sorted(patterns_by_variant)
    if len(variants) < 2:
        raise ValueError(
            f"within_concept_consistency requires >= 2 variants per concept; got "
            f"{len(variants)} ({variants}). A single variant has nothing to be consistent "
            f"with -- the metric is undefined, not 1.0."
        )
    matrix = _stack_patterns({v: patterns_by_variant[v] for v in variants})
    corr = np.corrcoef(matrix)
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[upper].mean())


def _stack_patterns(patterns: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = []
    length = None
    for name, vector in patterns.items():
        array = np.asarray(vector, dtype=np.float64).ravel()
        if length is None:
            length = array.size
        elif array.size != length:
            raise ValueError(
                f"pattern {name!r} has {array.size} parcels but a previous pattern has "
                f"{length}; patterns must share one parcellation"
            )
        if array.size < 2:
            raise ValueError(f"pattern {name!r} has fewer than 2 parcels")
        if array.std(ddof=0) == 0:
            raise ValueError(
                f"pattern {name!r} is constant across parcels; its correlation with any "
                f"other pattern is undefined"
            )
        rows.append(array)
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

METRIC_OUTPUT_COLUMNS = (
    "stimulus_id",
    "parcel_id",
    "network",
    "metric",
    "value",
    "smoothing_kernel",
    "smoothing_width_s",
    "tr_seconds",
    "instructional_window_s",
    "baseline_percentile",
    "sustained_threshold_sd",
    "spatial_basis",
)


@dataclass(frozen=True)
class _ParcelSeries:
    parcel_ids: np.ndarray
    networks: np.ndarray
    matrix: np.ndarray  # (n_parcels, n_timesteps)


def _pivot(parcel_df: pd.DataFrame, stimulus_id: str) -> _ParcelSeries:
    frame = parcel_df[parcel_df["stimulus_id"] == stimulus_id]
    wide = frame.pivot(index="parcel_id", columns="time_index", values="mean_bold").sort_index()
    if wide.isna().to_numpy().any():
        raise ValueError(
            f"{stimulus_id}: parcel table is ragged -- some parcel/timepoint combinations are "
            f"missing. Metrics over a partially observed series would be silently wrong."
        )
    networks = (
        frame.drop_duplicates("parcel_id").set_index("parcel_id")["network"].reindex(wide.index)
    )
    return _ParcelSeries(
        parcel_ids=wide.index.to_numpy(),
        networks=networks.to_numpy(),
        matrix=wide.to_numpy(dtype=np.float64),
    )


def compute_all_metrics(parcel_df: pd.DataFrame, config: MetricsConfig) -> pd.DataFrame:
    """Every per-stimulus metric, keyed by `stimulus_id` and `parcel_id`.

    Rows describing the whole cortical sheet rather than one parcel carry
    `parcel_id == WHOLE_CORTEX` (-1) and `network == "__all__"`.

    The cross-stimulus metrics (`representational_differentiation`,
    `within_concept_consistency`) are *not* computed here: they are defined over
    sets of concepts and variants, not over one stimulus, and they have their
    own sample-size requirements. Call them directly.

    Every free parameter used is written into the output, so a metrics table can
    be interpreted without its config file (decision gate 19).
    """
    required = {"stimulus_id", "time_index", "parcel_id", "network", "mean_bold"}
    missing = sorted(required - set(parcel_df.columns))
    if missing:
        raise ValueError(f"parcel table is missing column(s) {missing}")
    if parcel_df.empty:
        raise ValueError("parcel table is empty; there is nothing to compute metrics over")

    params = {
        "smoothing_kernel": config.smoothing_kernel,
        "smoothing_width_s": config.smoothing_width_s,
        "tr_seconds": config.tr_seconds,
        "instructional_window_s": (
            "full" if config.instructional_window_s is None else str(list(config.instructional_window_s))
        ),
        "baseline_percentile": config.baseline_percentile,
        "sustained_threshold_sd": config.sustained_threshold_sd,
        "spatial_basis": SPATIAL_BASIS,
    }

    rows: list[dict[str, Any]] = []
    for stimulus_id in sorted(parcel_df["stimulus_id"].unique()):
        series = _pivot(parcel_df, stimulus_id)
        window = window_slice(series.matrix.shape[1], config.tr_seconds, config.instructional_window_s)
        windowed = series.matrix[:, window]

        per_parcel_auc = np.empty(windowed.shape[0], dtype=np.float64)
        for i, parcel_id in enumerate(series.parcel_ids):
            x = windowed[i]
            values = {
                "mean_response": mean_response(x),
                "peak_response": peak_response(
                    x, config.smoothing_kernel, config.smoothing_width_s, config.tr_seconds
                ),
                "auc": auc(x, config.tr_seconds),
                "time_to_peak": time_to_peak(
                    x, config.smoothing_kernel, config.smoothing_width_s, config.tr_seconds
                ),
                "sustained_engagement": sustained_engagement(
                    x, config.tr_seconds, config.baseline_percentile, config.sustained_threshold_sd
                ),
            }
            per_parcel_auc[i] = values["auc"]
            for metric, value in values.items():
                rows.append(
                    {
                        "stimulus_id": stimulus_id,
                        "parcel_id": int(parcel_id),
                        "network": str(series.networks[i]),
                        "metric": metric,
                        "value": float(value),
                        **params,
                    }
                )

        # Whole-cortex metrics. `abs(auc)` is the non-negative spatial summary
        # entropy needs; the choice is recorded in `spatial_basis`.
        magnitude = np.abs(per_parcel_auc)
        for metric, value in (
            ("spatial_dispersion", spatial_dispersion(magnitude)),
            ("spatial_entropy", spatial_entropy(magnitude)),
            ("network_integration", network_integration(windowed, series.networks)),
        ):
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "parcel_id": WHOLE_CORTEX,
                    "network": "__all__",
                    "metric": metric,
                    "value": float(value),
                    **params,
                }
            )

    return pd.DataFrame(rows, columns=list(METRIC_OUTPUT_COLUMNS))

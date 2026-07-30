"""Generate `docs/learner_equations.md` from the Phase III docstrings.

    python scripts/build_learner_equations.py
    python scripts/build_learner_equations.py --check

The same discipline `scripts/build_metrics_reference.py` applies to the §6.5
cortical metrics (decision gate 19), applied to the behavioural quantities of §7:
no equation is documented by hand, and none may enter the pipeline without
stating its formula and the direction of each term. If the reference disagrees
with the code, the code wins and this script is what makes that true.

Every registered function must carry both fields in its docstring::

    Equation:  the arithmetic, explicitly (or `Definition:` for a proxy)
    Direction: what a larger value of each term does

A missing field fails the build.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learners import checkpoints, curriculum, effort, population, responses  # noqa: E402
from src.learners import support, updates  # noqa: E402

_FIELD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<field>Equation|Definition|Direction):\s*(?P<value>.*)$"
)

REQUIRED_DIRECTION = "Direction"
FORMULA_FIELDS = ("Equation", "Definition")

#: (equation label, function). The label is the brief's equation number where one
#: exists, and a descriptive name where the quantity is a mapping or a proxy the
#: brief names but does not number.
SECTIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, object], ...]], ...] = (
    (
        "Scales the equations are written on",
        "Neither map is given by the brief. Both are monotone, configurable and "
        "labelled ASSUMPTION in `docs/phase3_assumptions.md`.",
        (
            ("difficulty -> b_u", curriculum.difficulty_to_b_u),
            ("K -> theta", population.theta_from_k),
        ),
    ),
    (
        "Response model (§7.3)",
        "Equation (18) is equation (17) plus `omega * h`, so both are one "
        "implementation rather than two that could drift apart.",
        (
            ("17-18", responses.response_logit),
            ("17-18 (probability)", responses.probability_correct),
            ("confidence", responses.draw_confidence),
            ("support request", responses.support_request_probability),
            ("latency proxy", responses.latency_proxy),
        ),
    ),
    (
        "Effort and instructional effectiveness (§7.4)",
        "`E` is the quantity Phase IV multiplies by the predicted cortical "
        "response `Z(u,c,p)`. Phase III ends at `E`.",
        (
            ("19", effort.instructional_effort),
            ("20", effort.instructional_effectiveness),
        ),
    ),
    (
        "Operational proxies (§7.4)",
        "The brief specifies the equations but not their inputs. Every proxy "
        "below is an observable of the tutor interaction, never a free "
        "parameter, so none of them can be tuned to produce a result.",
        (
            ("Retrieval", effort.retrieval_proxy),
            ("Explanation", effort.explanation_proxy),
            ("Offloading", effort.offloading_proxy),
            ("CorrectAfterError", effort.correct_after_error_proxy),
            ("IndependentSuccess", effort.independent_success_proxy),
            ("SupportFaded", support.support_faded),
            ("Mismatch", support.difficulty_mismatch),
            ("support persistence", support.apply_persistence),
        ),
    ),
    (
        "State updates (§7.6)",
        "Equation (24) is **not** implemented literally: §7.6 mandates "
        "`1 - running Brier` instead. Every other equation is as written.",
        (
            ("21", updates.update_knowledge),
            ("22", updates.update_memory),
            ("22 (decay rate)", updates.memory_decay_rate),
            ("23", updates.update_reasoning),
            ("24", updates.calibration_from_brier),
            ("25", updates.update_dependence),
            ("retention projection", updates.decay_only),
        ),
    ),
    (
        "Checkpoint outcomes (§7.7)",
        "Scored with all support removed, except the eq. (26) supported probe.",
        (
            ("26", checkpoints.support_gap),
            ("27", responses.brier_score),
            ("expected calibration error", checkpoints.expected_calibration_error),
        ),
    ),
)


class DocstringError(ValueError):
    """Raised when a registered equation's docstring is incomplete."""


def parse_docstring(name: str, doc: str | None) -> Tuple[str, Dict[str, str]]:
    """Split a docstring into its summary and its structured fields.

    A field block runs until a blank line or the next field label. Blank-line
    termination rather than the deeper-indent rule of
    `scripts/build_metrics_reference.py`: these docstrings wrap their prose at the
    field's own indent, and requiring a deeper indent silently truncated every
    multi-line Direction at its first wrap -- which produced a reference full of
    half-sentences that still passed the build.
    """
    if not doc:
        raise DocstringError(f"{name}: has no docstring")

    lines = inspect.cleandoc(doc).splitlines()
    summary = lines[0].strip() if lines else ""

    fields: Dict[str, str] = {}
    current: str | None = None
    for line in lines:
        match = _FIELD_RE.match(line)
        if match:
            current = match.group("field")
            fields[current] = match.group("value").strip()
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped:
            fields[current] = f"{fields[current]} {stripped}".strip()
        else:
            current = None

    formula = next((fields[key] for key in FORMULA_FIELDS if fields.get(key)), None)
    if not formula:
        raise DocstringError(
            f"{name}: docstring states no {' or '.join(FORMULA_FIELDS)}. Every Phase III "
            f"equation and proxy must write its arithmetic out explicitly."
        )
    if not fields.get(REQUIRED_DIRECTION):
        raise DocstringError(
            f"{name}: docstring states no Direction. A behavioural quantity whose direction is "
            f"undocumented cannot be interpreted -- the same rule decision gate 19 applies to "
            f"cortical metrics."
        )
    fields["_formula"] = formula
    return summary, fields


def render() -> str:
    out: List[str] = [
        "# Learner equations reference (§7)",
        "",
        "Generated by `scripts/build_learner_equations.py` from the docstrings in",
        "`src/learners/`. **Do not edit by hand** -- rerun the script.",
        "",
        "Every quantity below describes a **simulated** learner produced by a declared",
        "generative model. None of it is a measurement of a person, none of it is a causal",
        "effect, and none of it is a neural quantity: Phase III computes no cortical response",
        "and reads no TRIBE artefact.",
        "",
        "The parameter values behind these equations are assumptions, not findings (§7.2).",
        "They live in `config/learners.yaml` with low / medium / high arms, and the ones with",
        "no anchor in the brief or in data are listed in `docs/phase3_assumptions.md`.",
        "",
        "## The state vector (eq. 15)",
        "",
        "| symbol | name | meaning |",
        "|---|---|---|",
        "| `K` | knowledge | expressed competence; the state eq. (21) grows |",
        "| `M` | memory strength | the consolidated trace of eq. (22); decays more slowly "
        "than `K` |",
        "| `R` | independent reasoning | eq. (23); the state offloading erodes |",
        "| `C` | calibration | implemented as `1 - running Brier` (§7.6) |",
        "| `D` | dependence | eq. (25); the propensity to lean on support |",
        "",
        "Every component lies in [0, 1]. Per-learner parameters (eq. 16) are",
        "`alpha_i ~ LogNormal(mu_alpha, sigma_alpha)` and `delta_i ~ Beta(a_delta, b_delta)`.",
        "",
    ]

    count = 0
    for heading, preamble, entries in SECTIONS:
        out += [f"## {heading}", "", preamble, ""]
        for label, function in entries:
            qualified = f"{function.__module__.split('.')[-1]}.{function.__name__}"
            summary, fields = parse_docstring(qualified, function.__doc__)
            count += 1
            out += [
                f"### {label} — `{qualified}`",
                "",
                f"`{function.__name__}{inspect.signature(function)}`",
                "",
                summary,
                "",
                f"- **Equation:** {fields['_formula']}",
                f"- **Direction:** {fields[REQUIRED_DIRECTION]}",
                "",
            ]

    out += [
        "## Summary",
        "",
        f"- documented equations and proxies: **{count}**",
        "- equations implemented exactly as the brief writes them: 21, 22, 23, 25, 26, 27",
        "- the one mandated deviation: **eq. 24**, implemented as `1 - running Brier` "
        "because §7.6 requires it",
        "",
    ]
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "learner_equations.md"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the docstrings and report drift without writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    content = render()
    out_path = Path(args.out)

    if args.check:
        if not out_path.exists():
            print(f"{out_path} does not exist", file=sys.stderr)
            return 1
        if out_path.read_text(encoding="utf-8") != content:
            print(f"{out_path} is out of date; rerun without --check", file=sys.stderr)
            return 1
        print(f"{out_path} is up to date")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

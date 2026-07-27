"""Parsing for stimulus text files (§5.2).

Each stimulus file begins with a YAML-like front matter block delimited by
`---` lines, e.g.::

    ---
    stimulus_id: be_001_traditional_primary
    unit_id: be_001
    condition: traditional
    variant: primary
    example_count: 1
    learner_error_scripted: false
    ---
    <body text>

Front matter is stripped before any downstream use (feature extraction,
TRIBE) so those consumers only ever see clean body text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

_DELIMITER = "---"


def parse_front_matter(path: Path) -> Tuple[Dict[str, Any], str]:
    """Split a stimulus file into (front-matter dict, body text).

    Raises ValueError if the file does not open with a `---` delimited
    front-matter block.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")

    if not lines or lines[0].strip() != _DELIMITER:
        raise ValueError(f"{path}: missing opening '---' front-matter delimiter")

    try:
        closing_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == _DELIMITER)
    except StopIteration:
        raise ValueError(f"{path}: missing closing '---' front-matter delimiter") from None

    meta_lines = lines[1:closing_idx]
    body = "\n".join(lines[closing_idx + 1:]).strip("\n")

    meta: Dict[str, Any] = {}
    for line in meta_lines:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}: malformed front-matter line {line!r} (expected 'key: value')")
        key, _, value = line.partition(":")
        meta[key.strip()] = _coerce(value.strip())

    if not meta:
        raise ValueError(f"{path}: front matter is empty")

    return meta, body


#: Front-matter keys every stimulus must declare for Phase II to index it.
INDEX_KEYS = ("stimulus_id", "unit_id", "condition", "variant")


def build_stimulus_index(root: Path) -> Dict[str, Dict[str, Any]]:
    """Index every `stimuli/<condition>/*.txt` under `root` by stimulus_id.

    Each entry carries the front matter plus the stripped `body` and the source
    `path`. Phase II needs this because a cached prediction is keyed only by
    stimulus_id, while contrasts (§6.6) and RSA (§6.7) are organised by unit and
    condition -- and those are declared in the front matter, never inferred by
    slicing the id string.

    Raises on a missing key or a duplicate stimulus_id: both would silently
    misattribute a prediction to the wrong unit or condition.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(root).glob("stimuli/*/*.txt")):
        meta, body = parse_front_matter(path)
        missing = [key for key in INDEX_KEYS if key not in meta]
        if missing:
            raise ValueError(f"{path}: front matter is missing {missing}")
        stimulus_id = str(meta["stimulus_id"])
        if stimulus_id in index:
            raise ValueError(
                f"duplicate stimulus_id {stimulus_id!r} in {path} and "
                f"{index[stimulus_id]['path']}"
            )
        index[stimulus_id] = {**meta, "body": body, "path": str(path)}
    if not index:
        raise ValueError(f"no stimuli found under {Path(root) / 'stimuli'}")
    return index


def _coerce(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value

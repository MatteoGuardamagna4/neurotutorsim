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


def _coerce(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value

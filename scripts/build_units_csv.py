"""Build data/processed/units.csv from the authored unit JSON files.

units.csv is a validation index derived from data/interim/units/*.json (the
authored source of truth). Regenerate it any time a unit JSON changes; never
hand-edit the CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

UNITS_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "units"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "units.csv"

HEADER = [
    "unit_id",
    "domain",
    "concept",
    "difficulty",
    "prerequisites",
    "validator_reference",
    "validator_reference_params",
    "reference_answer",
    "validator_transfer",
    "validator_transfer_params",
    "transfer_answer",
]


def build_units_csv(units_dir: Path = UNITS_DIR, out_path: Path = OUT_PATH) -> None:
    rows = []
    for unit_path in sorted(units_dir.glob("*.json")):
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "domain": unit["domain"],
                "concept": unit["concept"],
                "difficulty": unit["difficulty"],
                "prerequisites": ";".join(unit["prerequisites"]),
                "validator_reference": unit["canonical_problem"]["validator"],
                "validator_reference_params": json.dumps(unit["canonical_problem"]["params"]),
                "reference_answer": unit["canonical_problem"]["answer"],
                "validator_transfer": unit["far_transfer"]["validator"],
                "validator_transfer_params": json.dumps(unit["far_transfer"]["params"]),
                "transfer_answer": unit["far_transfer"]["answer"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build_units_csv()
    print(f"wrote {OUT_PATH}")

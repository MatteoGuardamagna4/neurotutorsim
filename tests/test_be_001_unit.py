"""Verification gate C: be_001.json's declared problems must validate correctly,
and the scripted misconception's wrong answer must fail validation."""

import json
from pathlib import Path

from src.validation import validators as V

UNIT_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "units" / "be_001.json"


def _load_unit():
    return json.loads(UNIT_PATH.read_text(encoding="utf-8"))


def test_canonical_problem_validates():
    unit = _load_unit()
    p = unit["canonical_problem"]
    result = V.get(p["validator"]).check(params=p["params"], answer=p["answer"])
    assert result.passed, result.detail


def test_practice_problem_validates():
    unit = _load_unit()
    p = unit["practice_problem"]
    result = V.get(p["validator"]).check(params=p["params"], answer=p["answer"])
    assert result.passed, result.detail


def test_near_transfer_validates():
    unit = _load_unit()
    p = unit["near_transfer"]
    result = V.get(p["validator"]).check(params=p["params"], answer=p["answer"])
    assert result.passed, result.detail


def test_far_transfer_validates():
    unit = _load_unit()
    p = unit["far_transfer"]
    result = V.get(p["validator"]).check(params=p["params"], answer=p["answer"])
    assert result.passed, result.detail


def test_misconception_wrong_answer_fails_validation():
    unit = _load_unit()
    canonical = unit["canonical_problem"]
    wrong_answer = unit["common_misconception"]["wrong_answer"]
    result = V.get(canonical["validator"]).check(params=canonical["params"], answer=wrong_answer)
    assert not result.passed, "misconception's wrong_answer must not pass validation"

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantitativeactuarial as quact
from tests.cases import CASES


RESULTS_PATH = Path(__file__).parent / "fixtures" / "results.json"


@pytest.fixture(scope="module")
def saved_results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def public_function_names() -> set[str]:
    return {name for name in quact.__all__ if inspect.isfunction(getattr(quact, name, None))}


def normalize_result(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "columns": list(value.columns),
            "records": normalize_result(value.to_dict("records")),
        }

    if isinstance(value, pd.Series):
        return normalize_result(value.to_dict())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return normalize_result(value.tolist())

    if isinstance(value, np.generic):
        return normalize_result(value.item())

    if isinstance(value, tuple):
        return [normalize_result(item) for item in value]

    if isinstance(value, list):
        return [normalize_result(item) for item in value]

    if isinstance(value, dict):
        return {str(key): normalize_result(item) for key, item in value.items()}

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    return value


def assert_matches_saved_result(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert_matches_saved_result(actual[key], value)
        return

    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_matches_saved_result(actual_item, expected_item)
        return

    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
        return

    assert actual == expected


def test_every_public_function_has_a_saved_result_case() -> None:
    assert set(CASES) == public_function_names()


@pytest.mark.parametrize("function_name", sorted(CASES))
def test_function_matches_saved_result(function_name: str, saved_results: dict[str, Any]) -> None:
    actual = normalize_result(CASES[function_name]())
    assert_matches_saved_result(actual, saved_results[function_name])

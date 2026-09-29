"""Benchmark tests: run with `pytest -q`. Offline (no API key needed).

Covers: case loading, ground-truth validation, benchmark execution,
metric calculation, and result serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.runner import (
    DETECT_THRESHOLD,
    DIMENSIONS,
    build_results,
    compute_metrics,
    load_cases,
    run_benchmark,
    save_results,
    validate_cases,
    write_report,
)

REQUIRED_ERROR_TYPES = {
    "none",                          # A. correct answer
    "incorrect_numerical_value",     # B. incorrect numerical value
    "incorrect_analytical_method",   # C. incorrect analytical method
    "incomplete_answer",             # D. incomplete answer
    "hallucinated_column",           # E. hallucinated/nonexistent column
    "unsupported_claim",             # F. unsupported claim
    "unnecessary_wording",           # G. correct answer with unnecessary wording
    "multiple_errors",               # H. multiple simultaneous errors
}


# --- benchmark case loading -------------------------------------------------

def test_case_loading():
    cases = load_cases("benchmark/cases.json")
    assert len(cases) >= 8
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["description"]
        assert case["question"]
        assert case["injected_error"]
        assert case["mutation"]  # every case documents exactly what changed
        assert isinstance(case["trace"], dict)
        assert case["trace"]["question"] == case["question"]


def test_all_required_error_types_present():
    cases = load_cases("benchmark/cases.json")
    types = {c["injected_error"] for c in cases}
    assert REQUIRED_ERROR_TYPES <= types


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_cases("benchmark/does_not_exist.json")


# --- ground-truth validation ------------------------------------------------

def test_ground_truth_covers_all_dimensions():
    cases = load_cases("benchmark/cases.json")
    for case in cases:
        assert set(case["ground_truth"]) == set(DIMENSIONS)
        assert set(case["expected_evaluator"]) == set(DIMENSIONS)
        for label in case["ground_truth"].values():
            assert label in ("correct", "incorrect")
        for lo, hi in case["expected_evaluator"].values():
            assert 0.0 <= lo <= hi <= 1.0
        assert isinstance(case["expected_pass"], bool)


def test_expected_pass_rule_consistent_with_labels():
    """expected_pass is true only when every dimension is labeled correct."""
    cases = load_cases("benchmark/cases.json")
    for case in cases:
        all_correct = all(v == "correct" for v in case["ground_truth"].values())
        assert case["expected_pass"] == all_correct, case["case_id"]


def test_control_cases_have_no_injected_error():
    cases = load_cases("benchmark/cases.json")
    controls = [c for c in cases if c["expected_pass"]]
    assert controls, "benchmark must contain at least one clean control case"
    for case in controls:
        assert case["injected_error"] in ("none", "unnecessary_wording")


def test_invalid_case_rejected():
    cases = load_cases("benchmark/cases.json")
    broken = json.loads(json.dumps(cases))
    broken[0]["ground_truth"].pop("completeness")
    with pytest.raises(ValueError, match="ground_truth"):
        validate_cases(broken)


def test_duplicate_case_id_rejected():
    cases = load_cases("benchmark/cases.json")
    broken = json.loads(json.dumps(cases))
    broken[1]["case_id"] = broken[0]["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(broken)


def test_bad_expected_range_rejected():
    cases = load_cases("benchmark/cases.json")
    broken = json.loads(json.dumps(cases))
    broken[0]["expected_evaluator"]["completeness"] = [1.0, 0.5]  # lo > hi
    with pytest.raises(ValueError, match="expected range"):
        validate_cases(broken)


# --- benchmark execution ----------------------------------------------------

def test_benchmark_execution(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cases = load_cases("benchmark/cases.json")
    results = run_benchmark(cases)
    assert len(results) == len(cases)
    for r in results:
        assert set(r["evaluator_scores"]) == set(DIMENSIONS)
        for dim in DIMENSIONS:
            assert 0.0 <= r["evaluator_scores"][dim] <= 1.0
        for dr in r["dimension_results"]:
            assert dr["source"] == "deterministic"  # offline in tests
        assert r["llm_judge_used"] is False  # offline in tests
        assert len(r["dimension_results"]) == len(DIMENSIONS)


def test_benchmark_deterministic_and_reproducible(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cases = load_cases("benchmark/cases.json")
    first = run_benchmark(cases)
    second = run_benchmark(cases)
    for a, b in zip(first, second):
        assert a["evaluator_scores"] == b["evaluator_scores"]
        assert a["evaluator_pass"] == b["evaluator_pass"]


# --- metric calculation -----------------------------------------------------

def _fake_result(case_id, labels, scores, expected_pass, evaluator_pass):
    """Build one run_case-shaped result for synthetic metric tests."""
    dims = []
    for dim in DIMENSIONS:
        detected = scores[dim] < DETECT_THRESHOLD
        dims.append(
            {
                "dimension": dim,
                "label": labels[dim],
                "score": scores[dim],
                "expected_range": [0.0, 1.0],
                "within_expected_range": True,
                "detected": detected,
                "detection_correct": detected == (labels[dim] == "incorrect"),
                "source": "deterministic",
            }
        )
    return {
        "case_id": case_id,
        "evaluator_scores": scores,
        "evaluator_pass": evaluator_pass,
        "expected_pass": expected_pass,
        "verdict_correct": evaluator_pass == expected_pass,
        "dimension_results": dims,
    }


def test_metric_calculation():
    good = {d: "correct" for d in DIMENSIONS}
    bad = {d: "correct" for d in DIMENSIONS}
    bad["numerical_correctness"] = "incorrect"

    high = {d: 1.0 for d in DIMENSIONS}
    low_num = dict(high, numerical_correctness=0.0)

    results = [
        # TP: error injected and detected
        _fake_result("tp_case", bad, low_num, False, False),
        # FN: error injected but missed
        _fake_result("fn_case", bad, high, False, True),
        # FP: no error but flagged
        _fake_result("fp_case", good, low_num, True, False),
        # TN: no error and not flagged
        _fake_result("tn_case", good, high, True, True),
    ]
    metrics = compute_metrics(results)

    num = metrics["per_dimension"]["numerical_correctness"]
    assert num == {**num, "tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert num["accuracy"] == 0.5
    assert num["precision"] == 0.5
    assert num["recall"] == 0.5

    # Every other dimension is a clean TN in all four cases.
    for dim in DIMENSIONS[1:]:
        m = metrics["per_dimension"][dim]
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 0, 4, 0)
        assert m["accuracy"] == 1.0

    conf = metrics["dimension_confusion"]
    assert (conf["tp"], conf["fp"], conf["tn"], conf["fn"]) == (1, 1, 17, 1)

    pl = metrics["pass_level"]
    assert pl["correct_verdicts"] == 2
    assert pl["total"] == 4
    assert set(pl["missed_failures"]) == {"fn_case"}
    assert set(pl["false_failures"]) == {"fp_case"}

    assert metrics["score_sources"]["deterministic"] == 20


def test_metric_zero_denominator_is_none_not_crash():
    results = [_fake_result("only_tp", {**{d: "correct" for d in DIMENSIONS}, "completeness": "incorrect"},
                            dict({d: 1.0 for d in DIMENSIONS}, completeness=0.4), False, False)]
    metrics = compute_metrics(results)
    assert metrics["per_dimension"]["numerical_correctness"]["precision"] is None
    assert metrics["per_dimension"]["completeness"]["recall"] == 1.0


# --- result serialization ---------------------------------------------------

def test_result_serialization(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cases = load_cases("benchmark/cases.json")
    results = build_results(cases, run_benchmark(cases))

    out = save_results(results, tmp_path / "results.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))

    for key in ("benchmark_version", "generated_at", "detect_threshold",
                "evaluator_mode", "summary", "cases"):
        assert key in loaded
    summary = loaded["summary"]
    for key in ("n_cases", "per_dimension", "dimension_confusion", "pass_level",
                "score_sources", "expected_range_agreement"):
        assert key in summary
    assert summary["n_cases"] == len(cases)
    for dim in DIMENSIONS:
        assert dim in summary["per_dimension"]
    assert loaded["evaluator_mode"] in ("deterministic", "deterministic+llm")

    # Every dimension comparison round-trips with its label and detection verdict.
    for case in loaded["cases"]:
        assert len(case["dimension_results"]) == len(DIMENSIONS)
        for dr in case["dimension_results"]:
            assert dr["label"] in ("correct", "incorrect")
            assert isinstance(dr["detected"], bool)
            assert isinstance(dr["detection_correct"], bool)
            assert dr["source"] in ("deterministic", "deterministic+llm")


def test_report_written(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cases = load_cases("benchmark/cases.json")
    results = build_results(cases, run_benchmark(cases))
    report = write_report(results, tmp_path / "REPORT.md")
    text = report.read_text(encoding="utf-8")
    for heading in (
        "# Benchmark Report",
        "## 1. Research question",
        "## 6. Per-dimension performance",
        "## 7. False positives",
        "## 8. False negatives",
        "## 11. Limitations",
        "## 13. What should be tested next",
    ):
        assert heading in text
    assert "not** evidence that the evaluator is generally reliable" in text

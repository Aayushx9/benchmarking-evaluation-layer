"""Evaluator tests: run with `pytest -q`. Offline (no API key needed)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

from src.agent import DataAnalysisAgent
from src.evaluator import evaluate_trace, evaluate_trace_file

CSV = "data/sample_eval_results.csv"
AGENT = DataAnalysisAgent()
DIMS = (
    "numerical_correctness",
    "data_grounding",
    "method_correctness",
    "completeness",
    "hallucination_free",
)


def _fresh_trace(question: str, tmp_path, **kw):
    out = AGENT.ask(CSV, question, trace_dir=str(tmp_path / "traces"))
    trace = json.loads(Path(out["trace_file"]).read_text(encoding="utf-8"))
    trace.update(kw)
    return trace


def test_good_trace_passes(tmp_path):
    trace = _fresh_trace("Which model has the highest average accuracy?", tmp_path)
    ev = evaluate_trace(trace, trace_id="good")
    assert ev["pass"] is True
    assert ev["overall_score"] >= 0.8
    assert ev["llm_judge_used"] is False  # offline in tests
    for dim in DIMS:
        assert dim in ev["scores"]
        assert 0.0 <= ev["scores"][dim] <= 1.0
    assert ev["issues"] == []


def test_tampered_number_fails_numerical(tmp_path):
    trace = _fresh_trace("Which model has the highest average accuracy?", tmp_path)
    bad = copy.deepcopy(trace)
    bad["final_answer"] = bad["final_answer"].replace("0.8367", "0.9999")
    bad["answer"] = bad["final_answer"]
    ev = evaluate_trace(bad, trace_id="tampered")
    assert ev["scores"]["numerical_correctness"] < 1.0
    assert ev["scores"]["hallucination_free"] < 1.0
    assert any("0.9999" in i["message"] for i in ev["issues"])


def test_invented_column_flagged(tmp_path):
    trace = _fresh_trace("What is the average accuracy by model?", tmp_path)
    bad = copy.deepcopy(trace)
    bad["generated_code"] = "result = df.groupby('model')['f1_score'].mean().to_dict()"
    bad["code_used"] = bad["generated_code"]
    bad["final_answer"] = "Average f1_score by model: gpt-4o-mini: 0.9"
    bad["answer"] = bad["final_answer"]
    ev = evaluate_trace(bad, trace_id="invented")
    assert ev["scores"]["data_grounding"] < 1.0
    assert any("f1_score" in i["message"] for i in ev["issues"])


def test_wrong_method_penalized(tmp_path):
    trace = _fresh_trace("Which task appears to be the most difficult?", tmp_path)
    bad = copy.deepcopy(trace)
    bad["analysis_plan"] = {**bad["analysis_plan"], "analysis_type": "count_by_group"}
    bad["plan"] = bad["analysis_plan"]
    ev = evaluate_trace(bad, trace_id="wrong-method")
    assert ev["scores"]["method_correctness"] < 1.0


def test_failed_execution_fails(tmp_path):
    trace = _fresh_trace("Which model has the highest average accuracy?", tmp_path)
    bad = copy.deepcopy(trace)
    bad["execution_result"] = None
    bad["result"] = None
    bad["errors"] = "ValueError: boom"
    bad["error"] = "ValueError: boom"
    ev = evaluate_trace(bad, trace_id="failed")
    assert ev["pass"] is False
    assert ev["scores"]["numerical_correctness"] == 0.0


def test_legacy_trace_still_evaluates(tmp_path):
    legacy = {
        "question": "What is the average accuracy by model?",
        "csv_path": CSV,
        "inspection": {
            "columns": ["model", "accuracy"],
            "dtypes": {"model": "str", "accuracy": "float64"},
            "n_rows": 6,
            "n_cols": 2,
            "null_counts": {},
            "numeric_summary": {"accuracy": {"mean": 0.8, "min": 0.7, "max": 0.9, "count": 6}},
            "categorical_values": {"model": ["a", "b"]},
        },
        "plan": {"analysis_type": "mean_by_group", "group_col": "model", "metric_col": "accuracy"},
        "code_used": "result = df.groupby('model')['accuracy'].mean().to_dict()",
        "result": {"a": 0.8, "b": 0.85},
        "answer": "Average accuracy by model: a: 0.8; b: 0.85",
        "evidence": {"a": 0.8, "b": 0.85},
        "error": None,
    }
    ev = evaluate_trace(legacy, trace_id="legacy")
    assert ev["pass"] is True
    assert ev["overall_score"] >= 0.7


def test_evaluate_file_writes_json(tmp_path):
    trace = _fresh_trace("How many rows per task?", tmp_path)
    trace_path = tmp_path / "trace_x.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    ev = evaluate_trace_file(str(trace_path), out_dir=str(tmp_path / "evaluations"))
    eval_path = Path(ev["_eval_file"])
    assert eval_path.exists()
    saved = json.loads(eval_path.read_text(encoding="utf-8"))
    assert saved["trace_id"] == "trace_x"
    for key in ("scores", "overall_score", "pass", "issues", "explanation"):
        assert key in saved
    for dim in DIMS:
        assert dim in saved["scores"]


def test_evaluate_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        evaluate_trace_file(str(tmp_path / "nope.json"), out_dir=str(tmp_path))


def test_cli_evaluates_trace(tmp_path):
    trace = _fresh_trace("Which model has the lowest average latency?", tmp_path)
    trace_path = tmp_path / "trace_cli.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    out_dir = tmp_path / "evals"
    proc = subprocess.run(
        [sys.executable, "evaluate.py", str(trace_path), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OVERALL" in proc.stdout
    assert (out_dir / "eval_trace_cli.json").exists()

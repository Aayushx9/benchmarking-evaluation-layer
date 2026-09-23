"""Agent tests: run with `pytest -q`. Offline by default (no API key needed)."""

import json
import os
from pathlib import Path

import pytest

from src.agent import DataAnalysisAgent
from src.executor import execute, validate_code
from src.inspector import load_csv

CSV = "data/sample_eval_results.csv"
AGENT = DataAnalysisAgent()


def test_average_accuracy_by_model(tmp_path):
    out = AGENT.ask(
        CSV, "What is the average accuracy by model?", trace_dir=str(tmp_path)
    )
    assert out["analysis_type"] == "mean_by_group"
    assert set(out["evidence"]) == {"gpt-4o-mini", "claude-3-haiku", "llama-3-8b"}
    # gpt-4o-mini has the highest mean accuracy in the fixture
    assert out["evidence"]["gpt-4o-mini"] > out["evidence"]["llama-3-8b"]
    assert "gpt-4o-mini" in out["answer"]


def test_which_model_highest_accuracy(tmp_path):
    out = AGENT.ask(
        CSV, "Which model has the highest accuracy?", trace_dir=str(tmp_path)
    )
    assert out["analysis_type"] == "top_1"
    assert "gpt-4o-mini" in out["answer"]


def test_count_by_task(tmp_path):
    out = AGENT.ask(CSV, "How many rows per task?", trace_dir=str(tmp_path))
    assert out["analysis_type"] == "count_by_group"
    assert out["evidence"] == {"classification": 6, "qa": 6, "summarization": 6}


def test_trace_file_is_valid_json(tmp_path):
    out = AGENT.ask(
        CSV,
        "What is the correlation between accuracy and latency_ms?",
        trace_dir=str(tmp_path),
    )
    assert out["analysis_type"] == "correlation"
    trace_path = Path(out["trace_file"])
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for key in (
        "question",
        "csv_path",
        "inspection",
        "plan",
        "code_used",
        "result",
        "answer",
        "evidence",
    ):
        assert key in trace


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AGENT.ask("data/does_not_exist.csv", "What is the average?", trace_dir=str(tmp_path))


# --- Broader question handling (dynamic reasoner, offline) ---

def test_lowest_average_latency(tmp_path):
    out = AGENT.ask(
        CSV, "Which model has the lowest average latency?", trace_dir=str(tmp_path)
    )
    assert out["analysis_type"] == "top_1"
    assert "llama-3-8b" in out["answer"]  # fastest = lowest latency_ms


def test_cost_accuracy_relationship(tmp_path):
    out = AGENT.ask(
        CSV, "What is the relationship between cost and accuracy?", trace_dir=str(tmp_path)
    )
    assert out["analysis_type"] == "correlation"
    assert isinstance(out["evidence"], dict)
    assert "correlation" in out["evidence"]
    assert isinstance(out["evidence"]["correlation"], float)


def test_most_difficult_task(tmp_path):
    out = AGENT.ask(
        CSV, "Which task appears to be the most difficult?", trace_dir=str(tmp_path)
    )
    assert out["analysis_type"] == "difficulty"
    assert "summarization" in out["answer"]  # lowest mean accuracy in fixture


def test_compare_models_multi_metric(tmp_path):
    out = AGENT.ask(
        CSV,
        "Compare the models across accuracy, latency, and cost.",
        trace_dir=str(tmp_path),
    )
    assert out["analysis_type"] == "compare_multi"
    for model in ("gpt-4o-mini", "claude-3-haiku", "llama-3-8b"):
        assert model in out["evidence"]
        for metric in ("accuracy", "latency_ms", "cost_per_1k"):
            assert metric in out["evidence"][model]


def test_outlier_question_returns_bounds(tmp_path):
    out = AGENT.ask(
        CSV,
        "Are there any unusual or extreme values in the dataset?",
        trace_dir=str(tmp_path),
    )
    assert out["analysis_type"] == "outliers"
    # Clean fixture: no outliers, but bounds must still be reported numerically.
    assert "No unusual" in out["answer"] or "Unusual values" in out["answer"]
    assert out["evidence"]  # bounds or flagged values


def test_synonym_resolution_speed(tmp_path):
    out = AGENT.ask(CSV, "Which model is the fastest?", trace_dir=str(tmp_path))
    assert out["analysis_type"] == "top_1"
    assert "llama-3-8b" in out["answer"]


def test_trace_has_rich_keys(tmp_path):
    out = AGENT.ask(
        CSV, "Which task appears to be the most difficult?", trace_dir=str(tmp_path)
    )
    trace = json.loads(Path(out["trace_file"]).read_text(encoding="utf-8"))
    for key in (
        "question",
        "dataset_schema",
        "reasoning",
        "analysis_plan",
        "generated_code",
        "execution_result",
        "final_answer",
        "numerical_evidence",
        "errors",
        # legacy aliases preserved
        "inspection",
        "plan",
        "code_used",
        "result",
        "answer",
        "evidence",
    ):
        assert key in trace
    assert trace["reasoning"]  # non-empty reasoning string
    assert trace["generated_code"]


def test_answer_only_from_execution(tmp_path):
    out = AGENT.ask(
        CSV, "Which model has the highest average accuracy?", trace_dir=str(tmp_path)
    )
    # Evidence values must match a fresh independent computation.
    df = load_csv(CSV)
    expected = df.groupby("model")["accuracy"].mean().round(4).to_dict()
    assert out["evidence"] == {str(k): v for k, v in expected.items()}


def test_unsafe_code_blocked(tmp_path):
    df = load_csv(CSV)
    for bad in (
        "import os\nresult = 1",
        "result = open('data/sample_eval_results.csv').read()",
        "result = __import__('os').listdir('.')",
    ):
        with pytest.raises(ValueError):
            execute(df, bad)
    with pytest.raises(ValueError):
        validate_code("x = 1")  # must set `result`


def test_no_llm_key_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = AGENT.ask(CSV, "Which model has the highest average accuracy?", trace_dir=str(tmp_path))
    assert out["llm_used"] is False
    assert "gpt-4o-mini" in out["answer"]

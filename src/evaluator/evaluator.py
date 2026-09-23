"""Evaluator orchestrator: trace dict -> scored evaluation dict + JSON file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import checks
from .llm_judge import judge_with_llm

PASS_THRESHOLD = 0.7

__all__ = ["PASS_THRESHOLD", "normalize_trace", "evaluate_trace", "evaluate_trace_file"]


def normalize_trace(trace: dict) -> dict:
    """Accept new trace keys with legacy fallbacks (old MVP traces still work)."""
    schema = trace.get("dataset_schema") or trace.get("inspection") or {}
    return {
        "question": trace.get("question", ""),
        "schema": schema,
        "plan": trace.get("analysis_plan") or trace.get("plan") or {},
        "code": trace.get("generated_code") or trace.get("code_used") or "",
        "result": trace.get("execution_result", trace.get("result")),
        "answer": trace.get("final_answer") or trace.get("answer") or "",
        "evidence": trace.get("numerical_evidence", trace.get("evidence")),
        "errors": trace.get("errors", trace.get("error")),
    }


def _merge_llm(dim: str, det: tuple, llm: dict | None) -> tuple[float, str, list[str], str]:
    d_score, d_expl, d_issues = det
    if llm is None or dim not in llm:
        return d_score, d_expl, [f"{dim}: {m}" for m in d_issues], "deterministic"
    l = llm[dim]
    score = round((d_score + l["score"]) / 2, 2)
    expl = f"det: {d_expl} | llm: {l['explanation']}"
    issues = [f"{dim}: {m}" for m in d_issues] + [f"{dim} (llm): {m}" for m in l["issues"]]
    return score, expl, issues, "deterministic+llm"


def evaluate_trace(trace: dict, trace_id: str = "unknown") -> dict:
    """Score a trace dict. Pure function (no I/O); offline unless LLM key set."""
    norm = normalize_trace(trace)
    det = {
        "numerical_correctness": checks.check_numerical(norm),
        "data_grounding": checks.check_grounding(norm),
        "method_correctness": checks.check_method(norm),
        "completeness": checks.check_completeness(norm),
        "hallucination_free": checks.check_hallucination(norm),
    }
    llm = judge_with_llm(norm, det)
    llm_used = llm is not None

    scores: dict[str, float] = {}
    explanations: dict[str, str] = {}
    issues: list[dict[str, str]] = []
    detail: dict[str, dict] = {}
    for dim in checks.DIMENSIONS:
        if dim in ("method_correctness", "completeness", "hallucination_free"):
            score, expl, dim_issues, source = _merge_llm(dim, det[dim], llm)
        else:
            score, expl, dim_issues = det[dim][0], det[dim][1], [f"{dim}: {m}" for m in det[dim][2]]
            source = "deterministic"
        scores[dim] = score
        explanations[dim] = expl
        detail[dim] = {"score": score, "explanation": expl, "issues": dim_issues, "source": source}
        issues.extend({"dimension": dim, "message": m} for m in dim_issues)

    overall = round(sum(scores.values()) / len(scores), 2)
    passed = (
        overall >= PASS_THRESHOLD
        and scores["numerical_correctness"] >= 0.5
        and scores["hallucination_free"] >= 0.5
    )
    summary = (
        f"overall {overall} ({'PASS' if passed else 'FAIL'}); "
        + ", ".join(f"{d}={s}" for d, s in scores.items())
    )
    return {
        "trace_id": trace_id,
        "question": norm["question"],
        "scores": scores,
        "overall_score": overall,
        "pass": passed,
        "issues": issues,
        "explanation": summary,
        "explanations": explanations,
        "detail": detail,
        "llm_judge_used": llm_used,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_trace_file(trace_path: str, out_dir: str = "evaluations") -> dict:
    """Evaluate a trace JSON file, save the evaluation JSON, return it + paths."""
    trace_path_p = Path(trace_path)
    if not trace_path_p.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    trace = json.loads(trace_path_p.read_text(encoding="utf-8"))
    trace_id = trace_path_p.stem
    evaluation = evaluate_trace(trace, trace_id=trace_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_path = out / f"eval_{trace_id}.json"
    eval_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    evaluation["_eval_file"] = str(eval_path)
    evaluation["_trace_file"] = str(trace_path_p)
    return evaluation

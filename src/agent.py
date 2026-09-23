"""Data Analysis Agent: CSV + question -> answer + evidence + trace.

Flow: inspect -> reason (LLM if configured, else local dynamic reasoner)
  -> dynamic codegen -> safe execute -> answer strictly from the result.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .codegen import generate_code
from .executor import execute, validate_code
from .inspector import inspect, load_csv
from .llm_backend import plan_with_llm
from .reasoner import reason
from .trace import save_trace


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _build_answer(question: str, plan_: dict, result) -> tuple[str, object]:
    """Answer strictly from the executed result. Evidence mirrors the numbers."""
    t = plan_.get("analysis_type", "describe")

    if t == "mean_by_group":
        lines = [f"{k}: {v}" for k, v in result.items()]
        answer = f"Average {plan_['metric_col']} by {plan_['group_col']}: " + "; ".join(lines)
        return answer, _to_serializable(result)

    if t == "overall_mean":
        val = round(float(result), 4)
        return f"Overall mean of {plan_['metric_col']}: {val}", {
            "metric": plan_["metric_col"],
            "mean": val,
        }

    if t == "count_by_group":
        total = sum(result.values())
        answer = (
            f"Row counts by {plan_['group_col']}: "
            + "; ".join(f"{k}: {v}" for k, v in result.items())
            + f" (total {total})"
        )
        return answer, _to_serializable(result)

    if t == "top_1":
        ordered = list(result.items())
        winner, value = ordered[0]
        label = "lowest" if plan_.get("direction", "top") == "bottom" else "highest"
        answer = (
            f"{winner} has the {label} average {plan_.get('metric_col')} "
            f"by {plan_.get('group_col')} ({value}). Full ranking: "
            + "; ".join(f"{k}: {v}" for k, v in result.items())
        )
        return answer, _to_serializable(result)

    if t == "correlation":
        if isinstance(result, dict):  # relationship intent: corr + means
            corr = result.get("correlation")
            answer = (
                f"Pearson correlation between {plan_.get('col_x')} and "
                f"{plan_.get('col_y')}: {corr}. Context: {result}"
            )
            return answer, _to_serializable(result)
        val = round(float(result), 4)
        return (
            f"Pearson correlation between {plan_['col_x']} and {plan_['col_y']}: {val}",
            {"col_x": plan_["col_x"], "col_y": plan_["col_y"], "correlation": val},
        )

    if t == "compare_multi":
        g = plan_.get("group_col", "model")
        parts = []
        for group, metrics in result.items():
            parts.append(f"{group}: " + ", ".join(f"{m}={v}" for m, v in metrics.items()))
        return f"Comparison by {g}: " + "; ".join(parts), _to_serializable(result)

    if t == "difficulty":
        ordered = list(result.items())
        hardest, val = ordered[0]
        return (
            f"{hardest} appears most difficult (lowest average "
            f"{plan_.get('metric_col', 'accuracy')}: {val}). Full ranking: "
            + "; ".join(f"{k}: {v}" for k, v in result.items()),
            _to_serializable(result),
        )

    if t == "outliers":
        flagged = {c: info for c, info in result.items() if info.get("count", 0) > 0}
        if not flagged:
            bounds = {c: {"lower": info["lower"], "upper": info["upper"]} for c, info in result.items()}
            return (
                "No unusual or extreme values found (all values inside IQR bounds).",
                {"outliers": {}, "bounds": bounds},
            )
        lines = [f"{c}: {info['values']}" for c, info in flagged.items()]
        return "Unusual values found: " + "; ".join(lines), _to_serializable(result)

    # describe / generic fallback: report the executed result, never invent numbers.
    if isinstance(result, dict):
        keys = list(result.keys())[:8]
        return (
            f"Summary statistics for the dataset (question: {question}). Keys: {keys}.",
            _to_serializable(result),
        )
    return f"Result for '{question}': {result}", _to_serializable(result)


class DataAnalysisAgent:
    """Orchestrator: inspect -> LLM-or-local reason -> codegen -> safe execute."""

    def ask(self, csv_path: str, question: str, trace_dir: str = "traces") -> dict:
        started = datetime.now(timezone.utc).isoformat()
        df = load_csv(csv_path)  # raises FileNotFoundError if missing
        inspection = inspect(df)

        # 1. Reasoning: LLM first (if key configured + valid), else local reasoner.
        llm_used = False
        llm_out = plan_with_llm(question, inspection)
        if llm_out is not None:
            try:
                validate_code(llm_out["code"])
                plan_ = {
                    "analysis_type": llm_out.get("analysis_type", "describe"),
                    "intent": "llm",
                    "reasoning": llm_out.get("reasoning", ""),
                    "relevant_columns": llm_out.get("relevant_columns", []),
                    "steps": ["llm_generated"],
                }
                # Carry structural keys when the LLM echoes them.
                for k in ("group_col", "metric_col", "metric_cols", "col_x", "col_y", "direction"):
                    if k in llm_out:
                        plan_[k] = llm_out[k]
                code = llm_out["code"]
                llm_used = True
                reasoning = llm_out.get("reasoning", "")
            except Exception:
                llm_out = None  # fall through to local reasoner

        if not llm_used:
            plan_ = reason(question, inspection)
            reasoning = plan_.get("reasoning", "")
            code = generate_code(plan_)

        # 2. Execute safely; keep the trace complete even on failure.
        try:
            result = execute(df, code)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            answer, evidence = _build_answer(question, plan_, result)
        else:
            answer, evidence = f"Analysis failed: {error}", {}

        trace = {
            # Required by the evaluation-layer contract (new names)...
            "question": question,
            "dataset_schema": inspection,
            "reasoning": reasoning,
            "analysis_plan": plan_,
            "generated_code": code,
            "execution_result": _to_serializable(result),
            "final_answer": answer,
            "numerical_evidence": _to_serializable(evidence),
            "errors": error,
            # ...plus legacy aliases so existing tests/traces keep working.
            "csv_path": csv_path,
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "inspection": inspection,
            "plan": plan_,
            "code_used": code,
            "result": _to_serializable(result),
            "answer": answer,
            "evidence": _to_serializable(evidence),
            "error": error,
            "llm_used": llm_used,
        }
        trace_file = save_trace(trace, trace_dir)

        return {
            "answer": answer,
            "evidence": _to_serializable(evidence),
            "analysis_type": plan_.get("analysis_type", "describe"),
            "reasoning": reasoning,
            "plan": plan_,
            "code_used": code,
            "generated_code": code,
            "execution_result": _to_serializable(result),
            "inspection": inspection,
            "trace_file": trace_file,
            "llm_used": llm_used,
            "error": error,
        }

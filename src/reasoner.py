"""Local dynamic reasoner: schema-aware analysis planning without an LLM.

Used when no LLM key is configured (offline default) and as the deterministic
core the LLM path falls back to. Unlike the old keyword planner, it:
  - resolves synonyms to real columns ("speed" -> latency_ms, "price" -> cost_per_1k),
  - picks an intent from the question semantics, not just trigger words,
  - outputs composable steps so codegen builds code dynamically.

Returns a plan dict that always includes the legacy `analysis_type` key so
existing tests/traces keep working, plus richer fields for the new traces.
"""

from __future__ import annotations

# Synonym -> real column. Kept small and dataset-aware; extend as CSVs change.
_SYNONYMS = {
    "accuracy": "accuracy",
    "accurate": "accuracy",
    "correct": "accuracy",
    "score": "accuracy",
    "performance": "accuracy",
    "f1": "accuracy",
    "latency": "latency_ms",
    "latency_ms": "latency_ms",
    "speed": "latency_ms",
    "slow": "latency_ms",
    "fast": "latency_ms",
    "response time": "latency_ms",
    "ms": "latency_ms",
    "cost": "cost_per_1k",
    "cost_per_1k": "cost_per_1k",
    "price": "cost_per_1k",
    "expensive": "cost_per_1k",
    "cheap": "cost_per_1k",
    "model": "model",
    "task": "task",
}

_NUMERIC_HINTS = ("accuracy", "latency_ms", "cost_per_1k")


def _resolve_columns(question: str, inspection: dict) -> list[str]:
    q = question.lower()
    cols = inspection["columns"]
    found: list[str] = []
    for syn, real in _SYNONYMS.items():
        if syn in q and real in cols and real not in found:
            found.append(real)
    # Also match exact column names.
    for c in cols:
        if c.lower() in q and c not in found:
            found.append(c)
    return found


def _numeric_cols(inspection: dict) -> list[str]:
    return list(inspection["numeric_summary"].keys())


def _categorical_cols(inspection: dict) -> list[str]:
    return list(inspection["categorical_values"].keys())


def reason(question: str, inspection: dict) -> dict:
    """Build a reasoning plan from question + schema."""
    q = question.lower()
    columns = inspection["columns"]
    numeric = _numeric_cols(inspection)
    categorical = _categorical_cols(inspection)
    relevant = _resolve_columns(question, inspection)

    def first_numeric(default: str | None = None) -> str | None:
        for c in relevant:
            if c in numeric:
                return c
        for c in _NUMERIC_HINTS:
            if c in numeric:
                return c
        return default or (numeric[0] if numeric else None)

    def group_default(prefer: str = "model") -> str:
        for c in relevant:
            if c in categorical:
                # "compare the models" -> model; "task ... difficult" -> task
                if prefer == "task" and c == "task":
                    return c
                if prefer == "model" and c == "model":
                    return c
        if prefer in categorical:
            return prefer
        return categorical[0] if categorical else columns[0]

    # 1. Outliers / extreme values
    if any(w in q for w in ("unusual", "extreme", "anomal", "outlier", "strange", "odd")):
        metrics = [c for c in relevant if c in numeric] or numeric
        return {
            "reasoning": (
                f"Relevant columns: {metrics}. The question asks for unusual/extreme "
                "values, so I will compute IQR bounds per numeric column and list "
                "rows outside the bounds."
            ),
            "analysis_type": "outliers",
            "intent": "outliers",
            "relevant_columns": metrics,
            "metric_cols": metrics,
            "steps": [f"iqr_outliers({', '.join(metrics)})"],
        }

    # 2. Relationship / correlation between two metrics
    if any(w in q for w in ("relationsh", "correl", "tradeoff", "trade-off", "associated", "linked")):
        pair = [c for c in relevant if c in numeric]
        if len(pair) < 2:
            pair = (pair + [c for c in numeric if c not in pair])[:2]
        return {
            "reasoning": (
                f"Relevant columns: {pair}. The question asks about the relationship, "
                "so I will compute the Pearson correlation plus per-model means "
                "for context."
            ),
            "analysis_type": "correlation",
            "intent": "relationship",
            "relevant_columns": pair,
            "col_x": pair[0],
            "col_y": pair[1],
            "steps": [f"correlation({pair[0]}, {pair[1]})", "group_means"],
        }

    # 3. Most difficult task (lowest mean accuracy per task)
    if any(w in q for w in ("difficult", "hardest", "hard ", "challenging", "weakest", "worst task")):
        return {
            "reasoning": (
                "Relevant columns: task, accuracy. 'Most difficult' means the task "
                "with the lowest average accuracy, so I will rank tasks by mean accuracy."
            ),
            "analysis_type": "difficulty",
            "intent": "difficulty",
            "relevant_columns": ["task", "accuracy"],
            "group_col": "task",
            "metric_col": "accuracy",
            "steps": ["group_mean(task, accuracy)", "rank_ascending"],
        }

    # 4. Multi-metric comparison ("compare models across accuracy, latency, cost")
    if "compar" in q or ("across" in q and len([c for c in relevant if c in numeric]) >= 2) or (
        "vs" in q or "versus" in q
    ):
        metrics = [c for c in relevant if c in numeric] or numeric
        g = group_default("model")
        return {
            "reasoning": (
                f"Relevant columns: {[g] + metrics}. The question asks for a comparison, "
                f"so I will compute mean of each metric grouped by {g}."
            ),
            "analysis_type": "compare_multi",
            "intent": "compare",
            "relevant_columns": [g] + metrics,
            "group_col": g,
            "metric_cols": metrics,
            "steps": [f"group_means({g}, {', '.join(metrics)})"],
        }

    # 5. Ranking: highest/lowest/best/worst/fastest/slowest + model/task
    if any(
        w in q
        for w in ("highest", "lowest", "best", "worst", "top", "fastest", "slowest", "cheapest", "most expensive", "which")
    ):
        metric = first_numeric()
        # "fastest"/"slowest"/"lowest latency" all resolve to latency direction
        wants_low = any(
            w in q
            for w in ("lowest", "worst", "slowest", "fastest", "cheapest", "minimum", "least")
        )
        # "fastest model" means lowest latency even though the word says "fastest"
        if "fastest" in q and metric == "latency_ms":
            wants_low = True
        g = group_default("task" if "task" in q and "model" not in q else "model")
        direction = "bottom" if wants_low else "top"
        return {
            "reasoning": (
                f"Relevant columns: {g}, {metric}. The question asks for a ranking "
                f"({'lowest' if wants_low else 'highest'} average {metric} per {g}), "
                "so I will compute group means and sort."
            ),
            "analysis_type": "top_1",
            "intent": "rank",
            "relevant_columns": [g, metric],
            "group_col": g,
            "metric_col": metric,
            "direction": direction,
            "steps": [f"group_mean({g}, {metric})", f"rank_{'ascending' if wants_low else 'descending'}"],
        }

    # 6. Counts
    if "how many" in q or "count" in q or "rows per" in q or "rows by" in q:
        g = group_default("task" if "task" in q else "model")
        return {
            "reasoning": (
                f"Relevant column: {g}. The question asks for counts, "
                f"so I will count rows grouped by {g}."
            ),
            "analysis_type": "count_by_group",
            "intent": "count",
            "relevant_columns": [g],
            "group_col": g,
            "steps": [f"count_by({g})"],
        }

    # 7. Average by group
    if "averag" in q or "mean" in q or " by " in q or " per " in q:
        metric = first_numeric()
        g = group_default()
        if metric is not None:
            return {
                "reasoning": (
                    f"Relevant columns: {g}, {metric}. The question asks for an "
                    f"average of {metric} grouped by {g}."
                ),
                "analysis_type": "mean_by_group",
                "intent": "aggregate",
                "relevant_columns": [g, metric],
                "group_col": g,
                "metric_col": metric,
                "steps": [f"group_mean({g}, {metric})"],
            }

    # 8. Fallback: describe named metric or whole frame
    metric = first_numeric()
    if metric is not None:
        return {
            "reasoning": (
                f"Relevant column: {metric}. No specific grouping detected, "
                f"so I will summarize {metric} overall."
            ),
            "analysis_type": "overall_mean",
            "intent": "aggregate",
            "relevant_columns": [metric],
            "metric_col": metric,
            "steps": [f"overall_mean({metric})"],
        }
    return {
        "reasoning": "No clear metric detected; summarizing the dataset.",
        "analysis_type": "describe",
        "intent": "describe",
        "relevant_columns": columns,
        "steps": ["describe_all"],
    }

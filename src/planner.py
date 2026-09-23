"""Rule-based planner: map a natural-language question to one analysis type.

MVP only — keyword matching, no LLM. Keeps behavior deterministic and
testable offline. The evaluation layer (later) will judge these decisions.
"""

from __future__ import annotations


def _find_column(question: str, columns: list[str]) -> str | None:
    q = question.lower()
    for col in columns:
        if col.lower() in q:
            return col
    return None


def _find_group_column(question: str, columns: list[str], exclude: str | None = None) -> str | None:
    q = question.lower()
    # Prefer explicit "by <col>" / "per <col>" mention.
    for col in columns:
        if exclude and col == exclude:
            continue
        if f"by {col.lower()}" in q or f"per {col.lower()}" in q or f"each {col.lower()}" in q:
            return col
    # Fall back to common group columns present in the dataset.
    for candidate in ("model", "task"):
        if candidate in columns and candidate != exclude and candidate in q:
            return candidate
    return None


def plan(question: str, inspection: dict) -> dict:
    """Return a plan dict with analysis_type and resolved columns."""
    q = question.lower()
    columns: list[str] = inspection["columns"]
    numeric_cols = list(inspection["numeric_summary"].keys())

    value_col = _find_column(q, numeric_cols) or (numeric_cols[0] if numeric_cols else None)
    group_col = _find_column(q, columns)

    # Correlation: "correlation between A and B"
    if "correl" in q:
        cols_found = [c for c in numeric_cols if c.lower() in q]
        if len(cols_found) >= 2:
            return {
                "analysis_type": "correlation",
                "col_x": cols_found[0],
                "col_y": cols_found[1],
            }
        if len(numeric_cols) >= 2:
            return {
                "analysis_type": "correlation",
                "col_x": numeric_cols[0],
                "col_y": numeric_cols[1],
            }

    # Count: "how many ..." / "count ..."
    if "how many" in q or "count" in q or "rows per" in q or "rows by" in q:
        g = _find_group_column(q, columns) or group_col
        if g is None:
            # default to first categorical column
            cat_cols = list(inspection["categorical_values"].keys())
            g = cat_cols[0] if cat_cols else columns[0]
        return {"analysis_type": "count_by_group", "group_col": g}

    # Top/best: "which ... highest/lowest/best ..."
    if any(w in q for w in ("which", "highest", "lowest", "best", "worst", "top")):
        g = _find_group_column(q, columns) or group_col
        if g is None:
            cat_cols = list(inspection["categorical_values"].keys())
            g = cat_cols[0] if cat_cols else columns[0]
        metric = value_col or (numeric_cols[0] if numeric_cols else columns[0])
        direction = "bottom" if any(w in q for w in ("lowest", "worst", "slowest")) else "top"
        return {
            "analysis_type": "top_1",
            "group_col": g,
            "metric_col": metric,
            "direction": direction,
        }

    # Average by group: "average X by Y"
    if "averag" in q or "mean" in q or " by " in q or " per " in q:
        g = _find_group_column(q, columns) or group_col
        if g is not None and value_col is not None:
            return {
                "analysis_type": "mean_by_group",
                "group_col": g,
                "metric_col": value_col,
            }
        if value_col is not None:
            return {"analysis_type": "overall_mean", "metric_col": value_col}

    # Overall mean fallback when a numeric column is named.
    if value_col is not None and any(w in q for w in ("overall", "total average", "mean")):
        return {"analysis_type": "overall_mean", "metric_col": value_col}

    # Default: describe the named metric, or the full frame.
    if value_col is not None:
        return {"analysis_type": "overall_mean", "metric_col": value_col}
    return {"analysis_type": "describe"}

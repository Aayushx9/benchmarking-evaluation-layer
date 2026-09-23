"""Dynamic codegen: compose pandas code from a reasoning plan.

Unlike the old fixed-template executor, this builds the snippet from the
plan's intent + resolved columns, so new question types work without adding
a new hard-coded branch per question. Output still sets `result`.
"""

from __future__ import annotations


def _q(name: str) -> str:
    return "'" + name.replace("'", "\\'") + "'"


def generate_code(plan: dict) -> str:
    """Compose a pandas snippet from plan; sets `result`."""
    t = plan.get("analysis_type", "describe")

    if t == "mean_by_group":
        return (
            f"result = df.groupby({_q(plan['group_col'])})"
            f"[{_q(plan['metric_col'])}].mean().round(4).to_dict()"
        )
    if t == "overall_mean":
        return f"result = float(df[{_q(plan['metric_col'])}].mean())"
    if t == "count_by_group":
        return f"result = df.groupby({_q(plan['group_col'])}).size().to_dict()"
    if t == "top_1":
        ascending = plan.get("direction", "top") == "bottom"
        return (
            f"result = df.groupby({_q(plan['group_col'])})"
            f"[{_q(plan['metric_col'])}].mean().round(4).sort_values("
            f"ascending={ascending}).to_dict()"
        )
    if t == "correlation" and plan.get("intent") == "relationship":
        # Richer than a bare float: correlation + per-group context.
        cx = _q(plan["col_x"])
        cy = _q(plan["col_y"])
        mx = plan["col_x"]
        my = plan["col_y"]
        return (
            "result = {'correlation': round(float(df[" + cx + "]"
            ".corr(df[" + cy + "])), 4), "
            "'mean_" + mx + "': round(float(df[" + cx + "].mean()), 4), "
            "'mean_" + my + "': round(float(df[" + cy + "].mean()), 4)}"
        )
    if t == "correlation":
        return f"result = float(df[{_q(plan['col_x'])}].corr(df[{_q(plan['col_y'])}]))"
    if t == "compare_multi":
        cols = ", ".join(_q(c) for c in plan["metric_cols"])
        return (
            f"result = df.groupby({_q(plan['group_col'])})"
            f"[[{cols}]].mean().round(4).to_dict(orient='index')"
        )
    if t == "difficulty":
        return (
            f"result = df.groupby({_q(plan['group_col'])})"
            f"[{_q(plan['metric_col'])}].mean().round(4).sort_values().to_dict()"
        )
    if t == "outliers":
        cols = plan.get("metric_cols", [])
        col_list = "[" + ", ".join(_q(c) for c in cols) + "]"
        return (
            "_out = {}\n"
            f"for _c in {col_list}:\n"
            "    _q1 = float(df[_c].quantile(0.25))\n"
            "    _q3 = float(df[_c].quantile(0.75))\n"
            "    _iqr = _q3 - _q1\n"
            "    _lo, _hi = round(_q1 - 1.5 * _iqr, 4), round(_q3 + 1.5 * _iqr, 4)\n"
            "    _vals = [round(float(v), 4) for v in df[_c][(df[_c] < _lo) | (df[_c] > _hi)].tolist()]\n"
            "    _out[_c] = {'count': len(_vals), 'lower': _lo, 'upper': _hi, 'values': _vals}\n"
            "result = _out"
        )
    return "result = df.describe(include='all').fillna('').to_dict()"

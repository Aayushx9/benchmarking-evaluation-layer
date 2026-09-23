"""Deterministic evaluation checks (offline, no network).

Each check returns (score 0..1, explanation, issues list).
Scores are rule-based so results are reproducible for research.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.-])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_QUOTED_RE = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_\- .]*)['"]""")
_TOL = 5e-4

DIMENSIONS = (
    "numerical_correctness",
    "data_grounding",
    "method_correctness",
    "completeness",
    "hallucination_free",
)


def flatten_numbers(obj) -> list[float]:
    """Collect all int/float values (excluding bools) from nested structures.

    Dict keys are skipped: entity names like 'gpt-4o-mini' contain digits
    that are not data values.
    """
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_numbers(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_numbers(v))
        return out
    if isinstance(obj, str):
        return [float(m) for m in _NUMBER_RE.findall(obj)]
    return out


def numbers_match(a: float, b: float) -> bool:
    return abs(a - b) <= _TOL


def _contains_number(pool: list[float], value: float) -> bool:
    return any(numbers_match(p, value) for p in pool)


def known_columns(schema: dict) -> list[str]:
    cols = schema.get("columns") or []
    return [str(c) for c in cols]


def known_values(schema: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col, vals in (schema.get("categorical_values") or {}).items():
        out[str(col)] = [str(v) for v in vals]
    return out


def code_columns(code: str) -> list[str]:
    """Column names referenced via brackets, groupby(), or df.attr access.

    Handles chained indexing (df.groupby('m')['col']) by scanning every
    bracket pair, not just ones directly after `df`.
    """
    found: list[str] = []

    def add(token: str) -> None:
        if token not in found:
            found.append(token)

    for m in re.finditer(r"\[([^\[\]]*)\]", code or ""):
        for q in _QUOTED_RE.finditer(m.group(1)):
            add(q.group(1))
    for m in re.finditer(r"""groupby\(\s*['"]([^'"]+)['"]""", code or ""):
        add(m.group(1))
    for m in re.finditer(r"""df\.([A-Za-z_][A-Za-z0-9_]*)""", code or ""):
        if m.group(1) not in (
            "groupby", "mean", "corr", "describe", "quantile", "size",
            "sort_values", "to_dict", "fillna", "astype", "head", "round",
        ):
            add(m.group(1))
    return found


def check_numerical(norm: dict) -> tuple[float, str, list[str]]:
    """Are reported numbers consistent with the execution result?"""
    result, evidence, answer = norm["result"], norm["evidence"], norm["answer"]
    issues: list[str] = []
    if result is None:
        if flatten_numbers(answer) or flatten_numbers(evidence):
            return 0.0, "Execution failed but the answer/evidence reports numbers.", [
                "numbers reported despite execution failure"
            ]
        return 0.0, "Execution failed; no valid result to verify.", ["execution produced no result"]
    pool = flatten_numbers(result)
    allowed_extra: list[float] = []
    schema = norm["schema"]
    if isinstance(schema.get("n_rows"), int):
        allowed_extra.append(float(schema["n_rows"]))
    if isinstance(schema.get("n_cols"), int):
        allowed_extra.append(float(schema["n_cols"]))
    if isinstance(result, dict):
        allowed_extra.append(float(len(result)))
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in result.values()):
            allowed_extra.append(float(sum(result.values())))
    pool_all = pool + allowed_extra

    answer_nums = flatten_numbers(answer) if isinstance(answer, str) else flatten_numbers(answer)
    evidence_nums = flatten_numbers(evidence)
    bad_answer = [n for n in answer_nums if not _contains_number(pool_all, n)]
    bad_evidence = [n for n in evidence_nums if not _contains_number(pool, n)]
    for n in bad_answer:
        issues.append(f"answer number {n} not found in execution result")
    for n in bad_evidence:
        issues.append(f"evidence number {n} not found in execution result")
    penalty = 0.5 * (len(bad_answer) + len(bad_evidence))
    score = max(0.0, round(1.0 - penalty, 2))
    if score == 1.0:
        expl = f"All {len(answer_nums)} answer number(s) and {len(evidence_nums)} evidence number(s) match the execution result."
    else:
        expl = f"{len(bad_answer)} answer number(s) and {len(bad_evidence)} evidence number(s) mismatch the execution result."
    return score, expl, issues


def check_grounding(norm: dict) -> tuple[float, str, list[str]]:
    """Are claims supported by the dataset schema and execution result?"""
    schema, code, answer = norm["schema"], norm["code"], norm["answer"]
    issues: list[str] = []
    cols = known_columns(schema)
    colset = {c.lower(): c for c in cols}

    for ref in code_columns(code):
        if ref not in cols and ref.lower() not in colset:
            issues.append(f"code references unknown column '{ref}' (schema: {cols})")

    if isinstance(answer, str):
        for m in _QUOTED_RE.finditer(answer):
            token = m.group(1)
            if token not in cols and token.lower() not in colset:
                vals = {v.lower() for vs in known_values(schema).values() for v in vs}
                if token.lower() not in vals and not _NUMBER_RE.fullmatch(token):
                    issues.append(f"answer mentions unknown column/value '{token}'")
        # Bare column-like tokens (e.g. f1_score): identifiers containing an
        # underscore are almost always column references in this domain.
        for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*)\b", answer):
            token = m.group(1)
            if token not in cols and token.lower() not in colset:
                issues.append(f"answer mentions unknown column-like token '{token}'")

    score = max(0.0, round(1.0 - 0.5 * len(issues), 2))
    expl = "All code/answer references match the schema." if score == 1.0 else f"{len(issues)} ungrounded reference(s) found."
    return score, expl, issues


def expected_types(question: str) -> set[str] | None:
    """Analysis types acceptable for a question; None = no constraint."""
    q = (question or "").lower()
    if any(w in q for w in ("difficult", "hardest", "challenging", "weakest")):
        return {"difficulty"}
    if any(w in q for w in ("unusual", "extreme", "anomal", "outlier", "strange")):
        return {"outliers"}
    if any(w in q for w in ("relationsh", "correl", "tradeoff", "trade-off")):
        return {"correlation"}
    if "compar" in q or " versus " in q or " vs " in q:
        return {"compare_multi"}
    if "how many" in q or "count" in q:
        return {"count_by_group"}
    if any(w in q for w in ("highest", "lowest", "best", "worst", "fastest", "slowest", "cheapest", "which", "top")):
        return {"top_1", "mean_by_group"}
    if "averag" in q or "mean" in q:
        return {"mean_by_group", "overall_mean", "top_1"}
    return None


def check_method(norm: dict) -> tuple[float, str, list[str]]:
    """Was the analysis method appropriate for the question? (heuristic)"""
    question, plan, result = norm["question"], norm["plan"], norm["result"]
    actual = (plan or {}).get("analysis_type", "describe")
    if norm.get("errors") or result is None:
        return 0.3, "Execution failed, so the method produced no usable result.", ["execution error"]
    expected = expected_types(question)
    if expected is None:
        return 1.0, f"No method constraint for this question; '{actual}' accepted.", []
    if actual in expected:
        return 1.0, f"Method '{actual}' fits the question (expected one of {sorted(expected)}).", []
    return 0.4, f"Method '{actual}' looks wrong for this question (expected one of {sorted(expected)}).", [
        f"expected one of {sorted(expected)}, got '{actual}'"
    ]


def check_completeness(norm: dict) -> tuple[float, str, list[str]]:
    """Did the answer address the full question? (proxy checks)"""
    q = (norm["question"] or "").lower()
    answer = norm["answer"] if isinstance(norm["answer"], str) else str(norm["answer"])
    result = norm["result"]
    ans_low = answer.lower()
    issues: list[str] = []
    if not answer.strip():
        return 0.0, "Empty answer.", ["empty answer"]
    if result is None:
        return 0.2, "No execution result, answer cannot be complete.", ["no result"]

    vals = {v.lower() for vs in known_values(norm["schema"]).values() for v in vs}
    mentioned = {v for v in vals if v in ans_low}

    if "compar" in q:
        groups = list(result.keys()) if isinstance(result, dict) else []
        named = sum(1 for g in groups if str(g).lower() in ans_low)
        if isinstance(result, dict) and result and isinstance(next(iter(result.values())), dict):
            metrics_ok = True  # multi-metric table present; answer text summarizes
        else:
            metrics_ok = named >= 2
        if named >= 2 and metrics_ok:
            return 1.0, "Answer compares multiple groups.", []
        issues.append(f"comparison names only {named} of {len(groups)} groups")
        return 0.5, "Answer only partially compares the groups.", issues
    if any(w in q for w in ("relationsh", "correl")):
        if re.search(r"-?\d*\.\d+", answer):
            return 1.0, "Answer reports the relationship strength numerically.", []
        issues.append("no correlation value in answer")
        return 0.4, "Answer discusses no numeric relationship.", issues
    if any(w in q for w in ("which", "highest", "lowest", "best", "worst", "fastest", "slowest", "difficult", "hardest")):
        first_key = str(next(iter(result.keys()))) if isinstance(result, dict) and result else ""
        if first_key and first_key.lower() in ans_low:
            extra = "" if len(mentioned) > 1 or len(result) <= 2 else " (only the winner named)"
            return 1.0 if not extra else 0.8, f"Answer names the winner '{first_key}'{extra}.", []
        issues.append("winner/answer entity not named in answer")
        return 0.3, "Answer does not name the winning entity.", issues
    if "how many" in q or "count" in q:
        if re.search(r"\d+", answer):
            return 1.0, "Answer reports counts.", []
        return 0.4, "Answer reports no counts.", ["no counts in answer"]
    if mentioned or re.search(r"\d", answer):
        return 0.9, "Answer references data entities/numbers.", []
    issues.append("answer references no data entities or numbers")
    return 0.4, "Answer is generic with no data references.", issues


def check_hallucination(norm: dict) -> tuple[float, str, list[str]]:
    """Did the agent invent columns, values, or facts? (deterministic part)"""
    issues: list[str] = []
    g_score, _, g_issues = check_grounding(norm)
    issues.extend(f"grounding: {m}" for m in g_issues)
    n_score, _, n_issues = check_numerical(norm)
    for m in n_issues:
        issues.append(f"number: {m}")
    unsupported_penalty = (0.0 if g_score == 1.0 else 0.5) + (0.0 if n_score == 1.0 else 0.5)
    score = max(0.0, round(1.0 - unsupported_penalty, 2))
    expl = "No invented columns, values, or numbers detected." if score == 1.0 else f"{len(issues)} unsupported claim(s) detected."
    return score, expl, issues

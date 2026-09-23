"""Optional LLM judge: used only where deterministic checks are insufficient.

Covers method_correctness, completeness, and hallucination_free.
Numerical correctness and data grounding are always purely deterministic.

Same env vars as the agent: OPENAI_API_KEY enables it, with optional
OPENAI_BASE_URL / EVAL_MODEL (fallback OPENAI_MODEL, default gpt-4o-mini).
Returns None when unconfigured or on any failure -> deterministic-only mode.
"""

from __future__ import annotations

import json
import os
import urllib.request

_JUDGED = ("method_correctness", "completeness", "hallucination_free")

_PROMPT = """You judge a data-analysis agent's output. Be strict but fair. Output JSON only.

Question: {question}

Schema columns: {columns}
Categorical values: {categorical}

Analysis plan: {plan}
Generated code: {code}
Execution result: {result}
Final answer: {answer}
Numerical evidence: {evidence}

Deterministic findings (already established, do not re-judge numbers):
{det_findings}

Score these 3 dimensions 0..1 and list concrete issues:
- "method_correctness": was the analysis method appropriate for the question?
- "completeness": did the answer address the full question?
- "hallucination_free": 1.0 = no invented columns/values/trends/explanations; penalize causal claims or facts not supported by the data.

Return JSON: {{"method_correctness": {{"score": 0..1, "explanation": "...", "issues": [...]}},
 "completeness": {{"score": 0..1, "explanation": "...", "issues": [...]}},
 "hallucination_free": {{"score": 0..1, "explanation": "...", "issues": [...]}}}}
"""


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, round(float(x), 2)))
    except (TypeError, ValueError):
        return 0.5


def judge_with_llm(norm: dict, det: dict) -> dict | None:
    """Return {dim: {score, explanation, issues}} for the 3 judged dims, or None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("EVAL_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    schema = norm.get("schema", {}) or {}

    findings = "; ".join(
        f"{d}: {det[d][0]} ({det[d][1]})" for d in ("numerical_correctness", "data_grounding")
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        question=norm.get("question", ""),
                        columns=schema.get("columns"),
                        categorical=schema.get("categorical_values"),
                        plan=norm.get("plan"),
                        code=norm.get("code", ""),
                        result=json.dumps(norm.get("result"), default=str)[:3000],
                        answer=norm.get("answer", ""),
                        evidence=json.dumps(norm.get("evidence"), default=str)[:2000],
                        det_findings=findings,
                    ),
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        parsed = json.loads(payload["choices"][0]["message"]["content"])
        out: dict = {}
        for dim in _JUDGED:
            entry = parsed.get(dim, {}) if isinstance(parsed.get(dim), dict) else {}
            out[dim] = {
                "score": _clamp(entry.get("score", 0.5)),
                "explanation": str(entry.get("explanation", "")),
                "issues": [str(i) for i in entry.get("issues", []) if str(i)],
            }
        return out
    except Exception:
        return None

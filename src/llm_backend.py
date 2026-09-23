"""LLM backend: optional OpenAI-compatible reasoning + codegen.

If no API key is configured, returns None and the agent falls back to the
local dynamic reasoner (offline, deterministic). This keeps the project
runnable for students with no keys while being genuinely LLM-powered when
a key is present.

Env vars:
  OPENAI_API_KEY   - enables the LLM path (any OpenAI-compatible endpoint)
  OPENAI_BASE_URL  - default https://api.openai.com/v1
  OPENAI_MODEL     - default gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import urllib.request


_PROMPT_TEMPLATE = """You are a data analyst. Given a CSV schema and a question, output JSON only.

Schema:
{schema}

Question: {question}

Available in execution: `pd` (pandas), `df` (the loaded DataFrame).
Your code must be 1-8 lines of pandas that sets a variable `result`.
No imports, no file/network access, no plotting.

Return JSON with keys:
- "reasoning": 1-3 sentences: which columns are relevant and what analysis is needed
- "relevant_columns": list of column names from the schema
- "analysis_type": one of mean_by_group, overall_mean, count_by_group, top_1, correlation, compare_multi, difficulty, outliers, describe
- "code": python code string setting `result`
"""


def _schema_text(inspection: dict) -> str:
    lines = [
        f"columns: {inspection.get('columns')}",
        f"dtypes: {inspection.get('dtypes')}",
        f"numeric_summary: {inspection.get('numeric_summary')}",
        f"categorical_values: {inspection.get('categorical_values')}",
    ]
    return "\n".join(lines)


def plan_with_llm(question: str, inspection: dict) -> dict | None:
    """Try the LLM; return {reasoning, relevant_columns, analysis_type, code} or None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": _PROMPT_TEMPLATE.format(
                        schema=_schema_text(inspection), question=question
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
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed.get("code"), str) or not parsed["code"].strip():
            return None
        return {
            "reasoning": str(parsed.get("reasoning", "")),
            "relevant_columns": list(parsed.get("relevant_columns", [])),
            "analysis_type": str(parsed.get("analysis_type", "describe")),
            "code": parsed["code"],
        }
    except Exception:
        return None

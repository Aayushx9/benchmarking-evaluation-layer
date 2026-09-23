# Benchmarking the Evaluation Layer — Data Analysis Agent + Evaluator

CSV + question -> answer + evidence + JSON trace, then trace -> scored evaluation.
No auth, DB, or frontend.

The first working miniature of the system - The data-analysis agent accepts a CSV and natural-language questions, performs pandas-based analysis, produces an answer with numerical evidence and records a structured trace. I've then added an evaluation layer that evaluates those traces across numerical correctness, data grounding, method correctness, completeness and hallucination. The next step is to deliberately introduce controlled errors into agent outputs and test whether the evaluator consistently detects them.

## Setup

Full-path Python is used because `python` resolves to a Store stub on this machine:

```powershell
& "C:\Users\Aayush\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

## Run

```powershell
& "C:\Users\Aayush\AppData\Local\Programs\Python\Python312\python.exe" run_agent.py data/sample_eval_results.csv "Which task appears to be the most difficult?"
```

## Test

```powershell
& "C:\Users\Aayush\AppData\Local\Programs\Python\Python312\python.exe" -m pytest -q
```

## How it works

1. `src/inspector.py` — loads CSV with pandas, returns schema/summary.
2. `src/llm_backend.py` — tries an OpenAI-compatible LLM (`OPENAI_API_KEY`,
   optional `OPENAI_BASE_URL`/`OPENAI_MODEL`) for reasoning + code. Returns
   `None` when unconfigured/failing so the run stays offline.
3. `src/reasoner.py` — offline fallback: resolves synonyms to real columns
   (`speed`->`latency_ms`, `price`->`cost_per_1k`), picks an intent
   (`rank | compare | relationship | difficulty | outliers | count | aggregate`),
   emits reasoning + composable steps.
4. `src/codegen.py` — dynamically composes the pandas snippet from the plan
   (sets `result`). Not fixed per-question templates.
5. `src/executor.py` — AST-validates the code (no imports/file/network calls,
   must set `result`) then executes it in a restricted namespace.
6. `src/agent.py` — answers strictly from the execution result, saves the trace.
7. `src/planner.py` — legacy keyword planner, kept for reference only.

Traces land in `traces/trace_*.json`. Evaluations land in
`evaluations/eval_<trace_id>.json`.

## Evaluate a trace

```powershell
& "C:\Users\Aayush\AppData\Local\Programs\Python\Python312\python.exe" evaluate.py traces/trace_20260923T043041407326.json
# multiple files: evaluate.py <trace1> <trace2> [--out-dir evaluations]
```

## How the evaluator works

- `src/evaluator/checks.py` — deterministic checks (offline): number-vs-result
  matching with tolerance, schema/column grounding, method-vs-question heuristics,
  completeness proxies, invented-column/value detection.
- `src/evaluator/llm_judge.py` — optional LLM overlay for method, completeness,
  and hallucination only (same `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`EVAL_MODEL`
  vars). Skipped when unconfigured; per-dimension `source` records what ran.
- `src/evaluator/evaluator.py` — `evaluate_trace()` (pure) + `evaluate_trace_file()`
  (reads trace, writes `evaluations/` JSON). Overall = mean of 5 dims;
  pass = overall >= 0.7 with numerical and hallucination both >= 0.5.

## Sample questions

See `examples/sample_questions.md`.

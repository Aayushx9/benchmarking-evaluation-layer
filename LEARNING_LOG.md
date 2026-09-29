# LEARNING_LOG.md

Research log for **"Benchmarking the Evaluation Layer."**

**Provenance rules used in this document:**

- Sections marked **[RECONSTRUCTED]** are inferred *only* from existing project
  artifacts (README.md, AGENTS.md, source docstrings, tests, and trace files).
  Where a detail cannot be verified from those artifacts, the line
  "Historical detail not directly recoverable from the current project artifacts."
  is written explicitly. Reconstructed statements may be incomplete; they are not
  invented events.
- Sections marked **[NEW EXPERIMENT]** describe work actually performed during
  Phase 3 (2026-09-28) and are supported by committed files
  (`benchmark/`, `tests/test_benchmark.py`, `benchmark/results.json`).

---

## 1. Project motivation

**[RECONSTRUCTED]** The project README is titled "Benchmarking the Evaluation
Layer — Data Analysis Agent + Evaluator" and describes a pipeline in which
"CSV + question -> answer + evidence + JSON trace, then trace -> scored
evaluation." The structure of the repository — an agent, a separate evaluator
package, a trace contract explicitly described in AGENTS.md as "the contract for
the future evaluation layer," and a legacy planner whose docstring says "The
evaluation layer (later) will judge these decisions" — shows the project was
organized from early on around building a judge for AI-agent outputs rather than
only an agent.

The deeper personal or course motivation for the project (e.g., which class or
capstone it belongs to) is
Historical detail not directly recoverable from the current project artifacts.

## 2. Research question

**[RECONSTRUCTED]** Phase 1–2 artifacts do not record an explicit written
research question; the evaluator instead specifies operational goals in code and
tests (five dimensions, pass threshold 0.7).

**[NEW EXPERIMENT]** The current, explicit research question for Phase 3 is:

> "How might we tell whether the systems that judge AI agents are themselves
> any good?"

## 3. Initial Data Analysis Agent design

**[RECONSTRUCTED]** The earliest design is visible in `src/planner.py`, which
describes itself as: "Rule-based planner: map a natural-language question to one
analysis type. MVP only — keyword matching, no LLM. Keeps behavior deterministic
and testable offline." It matches trigger words ("correl", "how many",
"highest/lowest/best/worst", "averag"/"mean") to analysis types
(`correlation`, `count_by_group`, `top_1`, `mean_by_group`, `overall_mean`,
`describe`) and does simple column lookup against the inspected schema.

## 4. MVP implementation

**[RECONSTRUCTED]** Evidence of the MVP's interface survives in two places:

- Legacy trace keys preserved in `src/agent.py` (`csv_path`, `inspection`,
  `plan`, `code_used`, `result`, `answer`, `evidence`, `error`) and in
  `src/evaluator/evaluator.py::normalize_trace`, which falls back to them so
  "old MVP traces still work."
- `tests/test_evaluator.py::test_legacy_trace_still_evaluates` feeds a trace
  containing exactly those legacy keys and asserts it still evaluates.

The exact date range and step-by-step order of the MVP's construction is
Historical detail not directly recoverable from the current project artifacts.

## 5. Agent architecture

**[RECONSTRUCTED]** The architecture is documented in AGENTS.md and README.md
and implemented in `src/`:

`src/inspector.py` (loads CSV, returns schema/summary) -> `src/llm_backend.py`
(optional LLM reasoning + code; returns `None` with no `OPENAI_API_KEY`) ->
`src/reasoner.py` (offline synonym/intent fallback) -> `src/codegen.py`
(dynamic pandas builder that sets `result`) -> `src/executor.py` (AST blocklist
+ restricted builtins) -> `src/agent.py` (answer only from the execution
result) -> `src/trace.py` (JSON written to `traces/`). Entry points:
`run_agent.py` and `DataAnalysisAgent.ask(csv, question)`.

Supported intents (AGENTS.md): `rank | aggregate | count |
relationship(correlation) | compare_multi | difficulty | outliers | describe`.
The legacy `analysis_type` key is kept for old tests. `src/planner.py` is
explicitly marked "legacy, do not extend."

## 6. Initial limitations

**[RECONSTRUCTED]** The current code documents the earlier limitations it was
built to fix:

- `src/planner.py`: keyword matching only, no LLM, no synonym handling.
- `src/codegen.py`: contrasts itself with "the old fixed-template executor,"
  implying the first version used hard-coded templates per question type.
- `src/reasoner.py`: notes it resolves synonyms and picks intents "from the
  question semantics, not just trigger words," implying the first version could
  not answer e.g. "Which model is the fastest?" against a `latency_ms` column.

Which of these limitations were observed first, and in what order, is
Historical detail not directly recoverable from the current project artifacts.

## 7. Evolution of the reasoning/planning system

**[RECONSTRUCTED]** `src/reasoner.py` documents the replacement of the keyword
planner: it "resolves synonyms to real columns (`speed`->`latency_ms`,
`price`->`cost_per_1k`), picks an intent from the question semantics, not just
trigger words," and "outputs composable steps so codegen builds code
dynamically." It prioritizes question patterns in a fixed order (outliers ->
relationship -> difficulty -> compare -> rank -> count -> aggregate -> describe)
and always emits the legacy `analysis_type` key "so existing tests/traces keep
working."

## 8. Offline fallback

**[RECONSTRUCTED]** `src/llm_backend.py` documents the design: it "tries an
OpenAI-compatible LLM (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`/
`OPENAI_MODEL`)" and "Returns `None` when unconfigured/failing so the run stays
offline." `src/agent.py` calls `plan_with_llm` first and falls back to
`reason(...)` + `generate_code(...)` when the LLM path returns `None` or its
code fails `validate_code`.
`tests/test_agent.py::test_no_llm_key_offline` asserts `llm_used is False`
without a key. The motivation statement "keeps the project runnable for
students with no keys" appears in the `llm_backend.py` docstring.

## 9. Dynamic code generation

**[RECONSTRUCTED]** `src/codegen.py::generate_code(plan)` composes a pandas
snippet from the plan's `analysis_type` + resolved columns ("Not fixed
per-question templates," per README). Every branch sets `result`, including a
multi-line IQR loop for the `outliers` type.

## 10. Safe execution

**[RECONSTRUCTED]** `src/executor.py::validate_code` parses the code with `ast`
and rejects imports, blocked calls (`open`, `exec`, `eval`, `compile`,
`__import__`, `input`), blocked attributes (`os`, `sys`, `subprocess`,
`socket`, `requests`, `urllib`, `pathlib`, `shutil`, ...), and code that does
not set `result`. `execute` runs the code in a namespace whose `__builtins__`
is a restricted dict and which only exposes `pd` and `df`.
`tests/test_agent.py::test_unsafe_code_blocked` verifies these rejections.

## 11. Structured trace design

**[RECONSTRUCTED]** AGENTS.md fixes the trace contract: new keys
`question, dataset_schema, reasoning, analysis_plan, generated_code,
execution_result, final_answer, numerical_evidence, errors` plus legacy aliases
`csv_path, inspection, plan, code_used, result, answer, evidence, error`, and
states these traces "are the contract for the future evaluation layer."
`src/agent.py` writes both sets; `src/trace.py` serializes with
`json.dumps(trace, default=str)` into `traces/trace_<timestamp>.json`.

## 12. Agent testing

**[RECONSTRUCTED]** `tests/test_agent.py` contains 15 offline tests covering:
mean-by-group, top-1 ranking, count-by-group, correlation JSON validity, missing
CSV, lowest-latency ranking, cost/accuracy relationship, difficulty, multi-metric
comparison, outlier bounds, synonym resolution ("fastest" -> `latency_ms`),
rich trace keys, answer-only-from-execution (recomputed independently with
pandas), unsafe-code blocking, and offline behavior. These pass locally
(see section 23).

## 13. Evaluation Layer design

**[RECONSTRUCTED]** AGENTS.md: the evaluator (`src/evaluator/`, entry
`evaluate.py`, tests `tests/test_evaluator.py`) "is separate from the agent."
`evaluate_trace()` is a pure function; `evaluate_trace_file()` reads a trace
file and writes `evaluations/eval_<trace_id>.json`. Overall = mean of 5 dims;
pass = overall >= 0.7 **and** numerical >= 0.5 **and** hallucination >= 0.5
(`PASS_THRESHOLD = 0.7` in `src/evaluator/evaluator.py`).

## 14. Five evaluation dimensions

**[RECONSTRUCTED]** Defined in `src/evaluator/checks.py::DIMENSIONS`:

1. `numerical_correctness` — do reported numbers match the execution result?
2. `data_grounding` — do code/answer references exist in the schema?
3. `method_correctness` — does the analysis type fit the question?
4. `completeness` — does the answer address the full question?
5. `hallucination_free` — were columns/values/facts invented?

## 15. Deterministic checks

**[RECONSTRUCTED]** `src/evaluator/checks.py` is documented as "Deterministic
evaluation checks (offline, no network). Each check returns (score 0..1,
explanation, issues list). Scores are rule-based so results are reproducible for
research." Key mechanics: `_NUMBER_RE` extracts numbers with tolerance
`_TOL = 5e-4`; `check_numerical` penalizes 0.5 per answer/evidence number not
found in the result (with allowances for `n_rows`, `n_cols`, dict length, dict
sum); `check_grounding` scans code column references plus quoted and
underscore-bearing tokens in the answer; `check_method` uses
`expected_types(question)` keyword heuristics; `check_completeness` uses
question-type-specific proxies; `check_hallucination` composes grounding +
numerical results.

## 16. Optional LLM judging

**[RECONSTRUCTED]** `src/evaluator/llm_judge.py` "used only where deterministic
checks are insufficient. Covers method_correctness, completeness, and
hallucination_free. Numerical correctness and data grounding are always purely
deterministic." Enabled by `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`,
`EVAL_MODEL`/`OPENAI_MODEL`); returns `None` when unconfigured or on any
failure. When active, `evaluator.py::_merge_llm` averages deterministic and LLM
scores and records per-dimension `source` as `deterministic` or
`deterministic+llm`.

## 17. Evaluator testing

**[RECONSTRUCTED]** `tests/test_evaluator.py` contains 9 offline tests: good
trace passes (>= 0.8, `llm_judge_used is False`), tampered number fails
numerical, invented column flagged, wrong method penalized, failed execution
fails, legacy trace still evaluates, file evaluation writes JSON, missing file
raises, and the CLI evaluates a trace end-to-end.

## 18. Negative testing / tampered outputs

**[RECONSTRUCTED]** Negative testing already existed before Phase 3: three
tests mutate a freshly generated correct trace —
`test_tampered_number_fails_numerical` (0.8367 -> 0.9999),
`test_invented_column_flagged` (`f1_score` injected into code and answer), and
`test_wrong_method_penalized` (`analysis_type` changed to `count_by_group`) —
plus `test_failed_execution_fails` (result set to `None`). These verify the
evaluator reacts to bad inputs, but they do not measure how reliably it detects
errors across a suite. That gap is what Phase 3 addresses.

## 19. Current benchmark phase

**[NEW EXPERIMENT]** Phase 3 (2026-09-28) adds `benchmark/`:

- `benchmark/cases.json` — 8 controlled cases (A–H) built from two existing
  traces, each with case_id, description, question, source trace reference,
  exact mutation description, ground-truth labels for all five dimensions,
  pre-registered expected score ranges, and expected pass/fail.
- `benchmark/runner.py` — loads and validates cases, calls the **existing**
  `src.evaluator.evaluate_trace` (no second evaluator), compares scores against
  ground truth, computes confusion matrices and pass/fail agreement, writes
  `benchmark/results.json` and `benchmark/REPORT.md`.
- `tests/test_benchmark.py` — 15 tests for loading, validation, execution,
  determinism, metric math, serialization, and report generation.

The Data Analysis Agent and Evaluation Layer were **not** modified.

## 20. Benchmark design decisions

**[NEW EXPERIMENT]** Decisions and why:

1. **Mutation of known-correct traces, not random generation.** Each bad case
   copies an existing passing trace and applies exactly one documented edit, so
   the injected error is obvious and reproducible. (Spec requirement: no random
   mutations.)
2. **Binary detection rule.** A dimension is "flagged" when the evaluator
   scores it < 0.9. Threshold chosen so that legitimate partial credit (e.g.
   completeness 0.8 for naming only the winner) still counts as detected, while
   clean scores (1.0) do not.
3. **Ground truth is rule-based:** the injected error determines which
   dimension is `incorrect`; `expected_pass` = all dimensions correct. This is
   honest but weak — it is not independent human judgment (documented as a
   limitation in REPORT.md §11).
4. **Pre-registered expected score ranges** were written into `cases.json`
   *before* the first benchmark run, derived from reading the check code. The
   run matched all 40 ranges — that is a code-reading sanity check, not
   independent validation.
5. **Case F deliberately encodes a known gap** (non-numeric unsupported claim)
   that deterministic checks are expected to miss, so the benchmark can show a
   real false negative rather than a suspiciously perfect score.
6. **No API key required.** The runner works in deterministic mode and records
   the `source` of every dimension score so deterministic and LLM-judged
   results can never be conflated.

## 21. Problems encountered

**[NEW EXPERIMENT]**

1. **Repeated response timeouts while first writing the benchmark files.**
   The initial large-file write attempts were interrupted before saving
   anything. Resumed in smaller steps, file by file, without redesign.
2. **Wrong expected value in a new test.** `test_metric_calculation` expected
   `tn = 16` for the synthetic 4-case x 5-dimension matrix; the correct count
   is 17 (20 comparisons - TP 1 - FP 1 - FN 1). Caught by the first test run.
3. **Leftover dead expression in `write_report`** (an `if False else ""`
   note-lookup remnant) — removed before the first run.

Problems encountered during Phases 1–2 are
Historical detail not directly recoverable from the current project artifacts.

## 22. Fixes and solutions

**[NEW EXPERIMENT]**

1. Timeouts -> switched to incremental file creation; each file saved before
   starting the next (no redesign, no discarded work).
2. Test arithmetic -> corrected the expected tuple to `(1, 1, 17, 1)`; the
   per-dimension assertions already encoded the same math correctly.
3. Dead code -> deleted the expression rather than leaving misleading logic.

## 23. Test results

**[NEW EXPERIMENT]** All commands use the full-path Python from AGENTS.md.

- `python -m pytest -q` -> **39 passed** (15 agent + 9 evaluator + 15
  benchmark; benchmark tests run in < 1 s because everything is offline).
- `python benchmark/runner.py` -> completed in deterministic mode:
  - 8 cases, 40 dimension comparisons
  - Dimension accuracy **0.975** (TP=10, FP=0, TN=29, FN=1)
  - Per dimension: numerical 1.0, grounding 1.0, method 1.0, completeness 1.0,
    hallucination_free 0.875
  - Pre-registered range agreement: 40/40
  - Case-level pass/fail agreement: **4/8**; missed failures: C, D, E, F
  - Score sources: 40 deterministic, 0 deterministic+llm
- Artifacts: `benchmark/results.json`, `benchmark/REPORT.md`.

## 24. What was learned

**[NEW EXPERIMENT]**

1. **Dimension scoring and case-level verdicts disagree.** The evaluator
   detected 10/11 injected dimension errors with 0 false positives, yet passed
   4 of 6 faulty cases: averaging five dimensions dilutes a single-dimension
   failure below the 0.7 gate (e.g. grounding 0.0 still yields overall 0.70).
   `pass` alone under-reports problems; per-dimension scores are the sharper
   signal.
2. **The deterministic hallucination check has a structural blind spot.**
   Case F (unsupported causal claim, all numbers correct) scored 1.0 on
   hallucination_free — the check only verifies columns and numbers, exactly as
   its docstring implies. This is the gap the optional LLM judge exists to fill.
3. **Pre-registering expectations before running exposed honest disagreement.**
   Encoding the expected miss in `cases.json` first prevented post-hoc
   rationalization of the false negative.
4. **Rule-based ground truth is cheap but circular.** It rewards checks that
   mirror the injection procedure; independent human labels are needed before
   trusting these numbers beyond the constructed set.

## 25. Current limitations

**[NEW EXPERIMENT]**

- 8 cases built from only 2 base traces on one 18-row fixture CSV.
- Ground truth authored by the same team as the evaluator (shared-assumption
  risk); not human-annotated.
- Only one unsupported-claim case; no within-tolerance numerical errors, no
  near-miss column names, no reordered/correct-but-unusual answers.
- LLM-judged dimensions were exercised in 0 comparisons (no key in this
  environment), so deterministic+llm merging is untested by the benchmark.
- The 0.9 detection threshold and expected ranges are design choices of this
  benchmark, not validated operating points.
- Pass/fail agreement of 4/8 reflects this rule-based `expected_pass`
  definition; a different aggregation policy could change it — that is a finding
  about the evaluator's design, not a general accuracy claim.

## 26. Next research questions

1. Does the optional LLM judge catch case F and its variants, and does it
   introduce false positives on cases A/G?
2. How well does the evaluator's `pass` rule correlate with human judgments —
   should the gate weight or require per-dimension floors?
3. What happens on adversarial traces: numbers wrong by less than the 5e-4
   tolerance, columns like `latency` vs `latency_ms`, rankings off by one?
4. Do independent human labels on natural (non-mutated) traces agree with the
   rule-based ground truth used here?
5. How stable are scores across evaluator code changes (regression benchmark
   in CI)?

---

## Maintenance policy (going forward)

After each major implementation or experiment, append an entry to this log
recording:

- what changed
- why
- problems encountered
- fixes
- tests
- results
- what was learned

Sections 1–18 above are reconstructed from project artifacts; sections 19–26
record Phase 3 experiments actually executed on 2026-09-28.

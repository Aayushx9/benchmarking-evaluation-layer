"""Benchmark runner for the Evaluation Layer itself.

Research question: "How might we tell whether the systems that judge AI agents
are themselves any good?"

This runner does NOT reimplement judging. It loads controlled benchmark cases
from benchmark/cases.json, calls the existing Evaluation Layer
(src.evaluator.evaluate_trace), and compares the evaluator's scores against
pre-registered ground-truth labels.

Runs fully offline. The LLM judge is used only when OPENAI_API_KEY is set; the
source of every dimension score (deterministic vs deterministic+llm) is
recorded in the results either way.

Usage (from the project root):
  ...python.exe benchmark/runner.py
  ...python.exe benchmark/runner.py --cases benchmark/cases.json \
      --results benchmark/results.json --report benchmark/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python benchmark/runner.py` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import evaluate_trace  # noqa: E402

DIMENSIONS = (
    "numerical_correctness",
    "data_grounding",
    "method_correctness",
    "completeness",
    "hallucination_free",
)
DETECT_THRESHOLD = 0.9
LABELS = ("correct", "incorrect")
REQUIRED_CASE_FIELDS = (
    "case_id",
    "description",
    "question",
    "injected_error",
    "mutation",
    "expected_pass",
    "ground_truth",
    "expected_evaluator",
    "trace",
)
DEFAULT_CASES = Path("benchmark/cases.json")
DEFAULT_RESULTS = Path("benchmark/results.json")
DEFAULT_REPORT = Path("benchmark/REPORT.md")


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    """Load benchmark cases from JSON and validate them. Raises ValueError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"benchmark cases not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    validate_cases(cases)
    return cases


def validate_cases(cases) -> None:
    """Raise ValueError unless every case has complete, well-formed labels."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark must contain at least one case")
    seen: set[str] = set()
    for case in cases:
        cid = case.get("case_id", "<missing case_id>")
        if not isinstance(case, dict):
            raise ValueError(f"case {cid!r} is not an object")
        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                raise ValueError(f"{cid}: missing required field {field!r}")
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id {case['case_id']!r}")
        seen.add(case["case_id"])

        gt = case["ground_truth"]
        if set(gt) != set(DIMENSIONS):
            raise ValueError(
                f"{cid}: ground_truth must cover exactly {list(DIMENSIONS)}"
            )
        for dim, label in gt.items():
            if label not in LABELS:
                raise ValueError(f"{cid}: {dim} label must be one of {LABELS}, got {label!r}")

        ranges = case["expected_evaluator"]
        if set(ranges) != set(DIMENSIONS):
            raise ValueError(
                f"{cid}: expected_evaluator must cover exactly {list(DIMENSIONS)}"
            )
        for dim, rng in ranges.items():
            if (
                not isinstance(rng, list)
                or len(rng) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng)
                or not (0.0 <= rng[0] <= rng[1] <= 1.0)
            ):
                raise ValueError(f"{cid}: {dim} expected range must be [lo, hi] within 0..1 with lo <= hi")

        if not isinstance(case["expected_pass"], bool):
            raise ValueError(f"{cid}: expected_pass must be a boolean")
        trace = case["trace"]
        if not isinstance(trace, dict) or "question" not in trace:
            raise ValueError(f"{cid}: trace must be a dict containing a 'question' key")


# ---------------------------------------------------------------------------
# Running the EXISTING evaluator on each case
# ---------------------------------------------------------------------------

def run_case(case: dict) -> dict:
    """Evaluate one benchmark case with src.evaluator.evaluate_trace and compare."""
    evaluation = evaluate_trace(case["trace"], trace_id=case["case_id"])

    dimension_results = []
    for dim in DIMENSIONS:
        score = evaluation["scores"][dim]
        label = case["ground_truth"][dim]
        rng = case["expected_evaluator"][dim]
        detected = score < DETECT_THRESHOLD
        labeled_incorrect = label == "incorrect"
        dimension_results.append(
            {
                "dimension": dim,
                "label": label,
                "score": score,
                "expected_range": rng,
                "within_expected_range": rng[0] <= score <= rng[1],
                "detected": detected,
                "detection_correct": detected == labeled_incorrect,
                "source": evaluation["detail"][dim]["source"],
            }
        )

    evaluator_pass = bool(evaluation["pass"])
    expected_pass = bool(case["expected_pass"])
    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "question": case["question"],
        "injected_error": case["injected_error"],
        "mutation": case["mutation"],
        "evaluator_scores": evaluation["scores"],
        "evaluator_overall_score": evaluation["overall_score"],
        "evaluator_pass": evaluator_pass,
        "expected_pass": expected_pass,
        "verdict_correct": evaluator_pass == expected_pass,
        "llm_judge_used": evaluation["llm_judge_used"],
        "issues": evaluation["issues"],
        "explanations": evaluation["explanations"],
        "dimension_results": dimension_results,
    }


def run_benchmark(cases: list[dict]) -> list[dict]:
    """Run the existing evaluator on every benchmark case, in order."""
    return [run_case(case) for case in cases]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(case_results: list[dict]) -> dict:
    """Per-dimension binary confusion matrices (positive = 'incorrect' ground truth)
    plus case-level pass/fail agreement, plus score-source accounting."""
    counts = {d: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for d in DIMENSIONS}
    sources = {"deterministic": 0, "deterministic+llm": 0}
    range_agreement = 0

    for result in case_results:
        for dr in result["dimension_results"]:
            bucket = counts[dr["dimension"]]
            if dr["label"] == "incorrect":
                bucket["tp" if dr["detected"] else "fn"] += 1
            else:
                bucket["fp" if dr["detected"] else "tn"] += 1
            key = "deterministic+llm" if "+" in dr["source"] else "deterministic"
            sources[key] += 1
            if dr["within_expected_range"]:
                range_agreement += 1

    per_dimension = {}
    for dim, g in counts.items():
        total = g["tp"] + g["fp"] + g["tn"] + g["fn"]
        per_dimension[dim] = {
            **g,
            "total": total,
            "accuracy": _ratio(g["tp"] + g["tn"], total),
            "precision": _ratio(g["tp"], g["tp"] + g["fp"]),
            "recall": _ratio(g["tp"], g["tp"] + g["fn"]),
            "detected_errors": g["tp"],
            "missed_errors": g["fn"],
            "false_positives": g["fp"],
        }

    totals = {k: sum(counts[d][k] for d in DIMENSIONS) for k in ("tp", "fp", "tn", "fn")}
    n_dim = sum(totals.values())

    missed_failures = [r["case_id"] for r in case_results if not r["expected_pass"] and r["evaluator_pass"]]
    false_failures = [r["case_id"] for r in case_results if r["expected_pass"] and not r["evaluator_pass"]]
    correct_verdicts = [r["case_id"] for r in case_results if r["verdict_correct"]]
    fully_correct = [
        r["case_id"]
        for r in case_results
        if r["verdict_correct"] and all(d["detection_correct"] for d in r["dimension_results"])
    ]

    return {
        "n_cases": len(case_results),
        "n_dimensions": len(DIMENSIONS),
        "n_dimension_comparisons": n_dim,
        "per_dimension": per_dimension,
        "dimension_confusion": {
            **totals,
            "accuracy": _ratio(totals["tp"] + totals["tn"], n_dim),
        },
        "expected_range_agreement": {
            "matches": range_agreement,
            "total": n_dim,
            "rate": _ratio(range_agreement, n_dim),
        },
        "pass_level": {
            "correct_verdicts": len(correct_verdicts),
            "total": len(case_results),
            "accuracy": _ratio(len(correct_verdicts), len(case_results)),
            "missed_failures": missed_failures,
            "false_failures": false_failures,
            "fully_correct_cases": fully_correct,
        },
        "score_sources": sources,
    }


def build_results(cases: list[dict], case_results: list[dict]) -> dict:
    any_llm = any(r["llm_judge_used"] for r in case_results)
    return {
        "benchmark_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detect_threshold": DETECT_THRESHOLD,
        "evaluator_mode": "deterministic+llm" if any_llm else "deterministic",
        "llm_judge_available": bool(os.environ.get("OPENAI_API_KEY")),
        "summary": compute_metrics(case_results),
        "cases": case_results,
    }


def save_results(results: dict, path: str | Path = DEFAULT_RESULTS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _yes_no(flag: bool) -> str:
    return "yes" if flag else "NO"


def write_report(results: dict, path: str | Path = DEFAULT_REPORT) -> Path:
    s = results["summary"]
    pd_ = s["per_dimension"]
    conf = s["dimension_confusion"]
    pl = s["pass_level"]
    src = s["score_sources"]

    lines: list[str] = []
    add = lines.append

    add("# Benchmark Report: Evaluating the Evaluation Layer")
    add("")
    add(f"Generated: {results['generated_at']}  ")
    add(f"Evaluator mode for this run: **{results['evaluator_mode']}** "
        f"(LLM judge {'available' if results['llm_judge_available'] else 'not available'}; "
        f"score sources = {src['deterministic']} deterministic, "
        f"{src['deterministic+llm']} deterministic+llm)  ")
    add(f"Detection threshold: a dimension is *flagged* when the evaluator scores it < {results['detect_threshold']}.")
    add("")
    add("> **Honesty note:** these metrics measure the evaluator only on this small,")
    add("> constructed, controlled benchmark. They do **not** prove the evaluator is")
    add("> generally reliable on unseen agent outputs.")
    add("")

    add("## 1. Research question")
    add("")
    add("> How might we tell whether the systems that judge AI agents are themselves any good?")
    add("")

    add("## 2. Benchmark design")
    add("")
    add("- Each case starts from a known-correct existing agent trace (or a minimal variation of it).")
    add("- Exactly one error type is injected per case, except case H which combines three.")
    add("- Every mutation is documented in the `mutation` field of `benchmark/cases.json` (no random mutations).")
    add("- Ground-truth labels cover all five evaluator dimensions, each `correct` or `incorrect`.")
    add("- `expected_pass` is true only when every dimension in the case is labeled `correct`.")
    add("- The runner calls the **existing** `src.evaluator.evaluate_trace`; no second evaluator exists.")
    add("")

    add("## 3. Benchmark cases")
    add("")
    add("| Case | Injected error | Question | Expected pass |")
    add("|---|---|---|---|")
    for r in results["cases"]:
        add(f"| {r['case_id']} | {r['injected_error']} | {r['question']} | {r['expected_pass']} |")
    add("")

    add("## 4. Ground-truth labels")
    add("")
    add("| Case | " + " | ".join(d.replace("_", " ") for d in DIMENSIONS) + " | pass |")
    add("|---" * (len(DIMENSIONS) + 2) + "|")
    for r in results["cases"]:
        labels = " | ".join(
            next(d["label"] for d in r["dimension_results"] if d["dimension"] == dim)
            for dim in DIMENSIONS
        )
        add(f"| {r['case_id']} | {labels} | {r['expected_pass']} |")
    add("")

    add("## 5. Evaluator results")
    add("")
    add("| Case | " + " | ".join(d.replace("_", " ") for d in DIMENSIONS) + " | overall | pass | verdict ok |")
    add("|---" * (len(DIMENSIONS) + 4) + "|")
    for r in results["cases"]:
        scores = " | ".join(str(r["evaluator_scores"][d]) for d in DIMENSIONS)
        add(
            f"| {r['case_id']} | {scores} | {r['evaluator_overall_score']} "
            f"| {r['evaluator_pass']} | {_yes_no(r['verdict_correct'])} |"
        )
    add("")

    add("## 6. Per-dimension performance")
    add("")
    add(f"Aggregated over {s['n_dimension_comparisons']} dimension comparisons "
        f"({s['n_cases']} cases x {s['n_dimensions']} dimensions). "
        "Positive class = ground truth `incorrect` (an error was injected in that dimension).")
    add("")
    add("| Dimension | TP | FP | TN | FN | accuracy | precision | recall |")
    add("|---|---|---|---|---|---|---|---|")
    for dim in DIMENSIONS:
        m = pd_[dim]
        add(
            f"| {dim} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} "
            f"| {m['accuracy']} | {m['precision']} | {m['recall']} |"
        )
    add(f"| **ALL** | {conf['tp']} | {conf['fp']} | {conf['tn']} | {conf['fn']} | **{conf['accuracy']}** | - | - |")
    add("")
    agree = s["expected_range_agreement"]
    add(f"Score landed inside the pre-registered expected range in {agree['matches']}/{agree['total']} "
        f"dimension comparisons ({agree['rate']}).")
    add("")
    add(f"Case-level pass/fail agreement: {pl['correct_verdicts']}/{pl['total']} ({pl['accuracy']}).")
    add("")

    add("## 7. False positives")
    add("")
    fps = [
        (r["case_id"], d["dimension"], d["score"])
        for r in results["cases"]
        for d in r["dimension_results"]
        if d["label"] == "correct" and d["detected"]
    ]
    if fps:
        for cid, dim, score in fps:
            add(f"- **{cid}** / {dim}: scored {score} (< {results['detect_threshold']}) despite being correct.")
    else:
        add("- None. The evaluator did not flag any dimension that ground truth labeled `correct`.")
    add("")

    add("## 8. False negatives (missed errors)")
    add("")
    fns = [
        (r["case_id"], d["dimension"], d["score"])
        for r in results["cases"]
        for d in r["dimension_results"]
        if d["label"] == "incorrect" and not d["detected"]
    ]
    if fns:
        for cid, dim, score in fns:
            add(f"- **{cid}** / {dim}: scored {score} (>= {results['detect_threshold']}) even though the error was injected.")
    else:
        add("- None. Every injected error was flagged.")
    add("")
    if pl["missed_failures"]:
        add(f"Case-level misses (evaluator PASSED a case that should have failed): {', '.join(pl['missed_failures'])}")
    if pl["false_failures"]:
        add(f"Case-level false failures (evaluator FAILED a case that should have passed): {', '.join(pl['false_failures'])}")
    add("")

    add("## 9. Cases the evaluator handled correctly")
    add("")
    for cid in pl["fully_correct_cases"]:
        r = next(x for x in results["cases"] if x["case_id"] == cid)
        add(f"- **{cid}**: {r['injected_error']} — all dimension detections and the pass/fail verdict matched ground truth.")
    add("")

    add("## 10. Cases where the evaluator failed")
    add("")
    failures = []
    for r in results["cases"]:
        bad_dims = [d["dimension"] for d in r["dimension_results"] if not d["detection_correct"]]
        if bad_dims or not r["verdict_correct"]:
            failures.append((r["case_id"], bad_dims, r["verdict_correct"]))
    if failures:
        for cid, bad_dims, verdict_ok in failures:
            parts = []
            if bad_dims:
                parts.append(f"dimension detection wrong on: {', '.join(bad_dims)}")
            if not verdict_ok:
                parts.append("pass/fail verdict disagreed with ground truth")
            add(f"- **{cid}**: " + "; ".join(parts) + ".")
    else:
        add("- None on this benchmark.")
    add("")

    add("## 11. Limitations")
    add("")
    add("- Small, constructed benchmark built from one 18-row fixture CSV and a handful of existing traces.")
    add("- Cases were created by the same team that wrote the evaluator, so shared assumptions can hide evaluator blind spots.")
    add("- Ground truth is rule-based (injected error => dimension incorrect), not independent human annotation.")
    add("- Only one non-numeric unsupported-claim case (F) exists; hallucination coverage beyond column/number errors is thin.")
    add(f"- The run above used mode `{results['evaluator_mode']}`; LLM-judged dimensions may behave differently when a key is present.")
    add("- Correct cases are all variations of two base traces, so the true-positive side is narrow.")
    add("")

    add("## 12. What the results mean")
    add("")
    add(f"- On this benchmark the evaluator flagged {conf['tp']} of {conf['tp'] + conf['fn']} injected "
        f"dimension errors (recall {round((conf['tp'] / (conf['tp'] + conf['fn'])) * 100, 1) if (conf['tp'] + conf['fn']) else 0}%) "
        f"with {conf['fp']} false positives.")
    add(f"- Case-level pass/fail verdicts agreed with ground truth on {pl['correct_verdicts']}/{pl['total']} cases.")
    add("- This is evidence that the evaluator handled *these constructed cases* correctly (or identifies exactly where it did not).")
    add("- It is **not** evidence that the evaluator is generally reliable; the benchmark is too small and too closely tied to the evaluator's own design.")
    add("- Case F is the expected structural gap: deterministic checks cannot see non-numeric unsupported claims; that is what the optional LLM judge is for.")
    add("- Averaging five dimensions means a single-dimension error can still leave overall >= 0.7, so `pass` alone under-reports problems — the per-dimension scores are the sharper signal.")
    add("")

    add("## 13. What should be tested next")
    add("")
    add("- More unsupported-claim cases (causal, comparative, and speculative statements) with and without the LLM judge.")
    add("- Adversarial traces: subtly wrong numbers within tolerance, near-miss columns (`latency` vs `latency_ms`), off-by-one rankings.")
    add("- Correct-but-unusual outputs (different phrasings, reordered answers) to hunt for false positives.")
    add("- Independent human labels on natural agent traces (not just injected mutations).")
    add("- Correlation between evaluator scores and human ratings across a larger trace set.")
    add("- Deterministic-vs-LLM agreement study: same cases under both modes, per-dimension delta.")
    add("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    s = results["summary"]
    conf = s["dimension_confusion"]
    pl = s["pass_level"]
    src = s["score_sources"]
    print("=" * 60)
    print("BENCHMARK SUMMARY: Evaluation Layer vs controlled cases")
    print("=" * 60)
    print(f"Mode: {results['evaluator_mode']} "
          f"(sources: {src['deterministic']} deterministic, {src['deterministic+llm']} deterministic+llm)")
    print(f"Cases: {s['n_cases']} | dimension comparisons: {s['n_dimension_comparisons']}")
    print(f"Dimension accuracy: {conf['accuracy']} "
          f"(TP={conf['tp']} FP={conf['fp']} TN={conf['tn']} FN={conf['fn']})")
    print("Per-dimension:")
    for dim in DIMENSIONS:
        m = s["per_dimension"][dim]
        print(f"  {dim:26s} acc={m['accuracy']}  TP={m['tp']} FP={m['fp']} "
              f"TN={m['tn']} FN={m['fn']}")
    print(f"Pass/fail agreement: {pl['correct_verdicts']}/{pl['total']} ({pl['accuracy']})")
    if pl["missed_failures"]:
        print(f"  Missed failures: {', '.join(pl['missed_failures'])}")
    if pl["false_failures"]:
        print(f"  False failures: {', '.join(pl['false_failures'])}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Evaluation Layer against controlled cases")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    case_results = run_benchmark(cases)
    results = build_results(cases, case_results)
    results_path = save_results(results, args.results)
    report_path = write_report(results, args.report)
    print_summary(results)
    print(f"RESULTS: {results_path}")
    print(f"REPORT:  {report_path}")


if __name__ == "__main__":
    main()

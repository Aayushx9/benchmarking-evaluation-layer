"""CLI: evaluate one or more agent trace files.

Usage:
  ...python.exe evaluate.py traces/trace_abc.json [--out-dir evaluations]
  ...python.exe evaluate.py traces/trace_a.json traces/trace_b.json
"""

from __future__ import annotations

import argparse
import json

from src.evaluator import evaluate_trace_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Data Analysis Agent traces")
    parser.add_argument("traces", nargs="+", help="Path(s) to trace JSON file(s)")
    parser.add_argument("--out-dir", default="evaluations")
    args = parser.parse_args()

    for trace_path in args.traces:
        ev = evaluate_trace_file(trace_path, out_dir=args.out_dir)
        print(f"TRACE: {ev['_trace_file']}")
        print(f"EVAL:  {ev['_eval_file']}")
        print(f"OVERALL: {ev['overall_score']} ({'PASS' if ev['pass'] else 'FAIL'})")
        print("SCORES:")
        print(json.dumps(ev["scores"], indent=2))
        if ev["issues"]:
            print("ISSUES:")
            for item in ev["issues"]:
                print(f"  - [{item['dimension']}] {item['message']}")
        else:
            print("ISSUES: none")
        print(f"llm_judge_used={ev['llm_judge_used']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

"""CLI: python run_agent.py <csv> "<question>" [--trace-dir traces]."""

from __future__ import annotations

import argparse
import json

from src.agent import DataAnalysisAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Analysis Agent (LLM + offline fallback)")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("question", help="Natural-language question about the CSV")
    parser.add_argument("--trace-dir", default="traces")
    args = parser.parse_args()

    out = DataAnalysisAgent().ask(args.csv, args.question, trace_dir=args.trace_dir)
    print("ANSWER:")
    print(out["answer"])
    print("\nEVIDENCE:")
    print(json.dumps(out["evidence"], indent=2))
    print(f"\nREASONING: {out.get('reasoning', '')}")
    print(f"ANALYSIS TYPE: {out['analysis_type']} (llm_used={out.get('llm_used')})")
    print(f"CODE: {out['code_used']}")
    print(f"TRACE: {out['trace_file']}")
    if out.get("error"):
        print(f"ERROR: {out['error']}")


if __name__ == "__main__":
    main()

"""Evaluation layer public API (kept separate from the Data Analysis Agent)."""

from .evaluator import PASS_THRESHOLD, evaluate_trace, evaluate_trace_file

__all__ = ["PASS_THRESHOLD", "evaluate_trace", "evaluate_trace_file"]

"""Safe generation + execution of pandas code.

`generate_code` is re-exported from codegen (dynamic composer). `execute`
validates the snippet with an AST blocklist before running it in a
restricted namespace. No imports, no file/network access.
"""

from __future__ import annotations

import ast

import pandas as pd

from .codegen import generate_code  # noqa: F401 - kept here for backwards compat

__all__ = ["generate_code", "validate_code", "execute"]

_BLOCKED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input"}
_BLOCKED_ATTRS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "mkdir",
    "write",
    "remove",
    "unlink",
    "rmdir",
    "popen",
    "system",
    "call",
}

_SAFE_BUILTINS = {
    "round": round,
    "float": float,
    "int": int,
    "len": len,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "range": range,
    "dict": dict,
    "list": list,
    "str": str,
    "abs": abs,
}


def validate_code(code: str) -> None:
    """Raise ValueError if code uses blocked constructs. Must set `result`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"invalid python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                raise ValueError(f"blocked call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTRS:
                raise ValueError(f"blocked attribute call: {func.attr}")
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRS:
            raise ValueError(f"blocked attribute: {node.attr}")
    if "result" not in code:
        raise ValueError("code must set a `result` variable")


def execute(df: pd.DataFrame, code: str):
    """Validate then execute code with `df` and `pd` in scope; return `result`."""
    validate_code(code)
    namespace: dict = {"pd": pd, "df": df}
    exec(code, {"__builtins__": dict(_SAFE_BUILTINS)}, namespace)  # noqa: S102
    if "result" not in namespace:
        raise ValueError("code did not set `result`")
    return namespace["result"]

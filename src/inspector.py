"""CSV loading and inspection using pandas."""

from __future__ import annotations

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file. Raises FileNotFoundError if missing."""
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> dict:
    """Return a JSON-serializable summary of the dataframe."""
    numeric_summary = {}
    for col in df.select_dtypes(include="number").columns:
        numeric_summary[col] = {
            "mean": float(df[col].mean()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "count": int(df[col].count()),
        }

    categorical_values: dict[str, list] = {}
    for col in df.select_dtypes(include=["object", "string", "str"]).columns:
        categorical_values[col] = sorted(df[col].dropna().unique().tolist())[:50]

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(n) for c, n in df.isna().sum().items()},
        "numeric_summary": numeric_summary,
        "categorical_values": categorical_values,
    }

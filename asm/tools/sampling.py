"""Sample values and lightweight column statistics."""
from typing import Any, Callable, Dict
import re

import pandas as pd

from asm.tools.profiling import _infer_value_pattern


def build_lightweight_sample_profile(series: pd.Series, sample_size: int) -> Dict[str, Any]:
    non_null = series.dropna()
    non_null_count = int(non_null.shape[0])
    total_count = int(series.shape[0])
    sample_non_null = non_null.head(sample_size).astype(str).tolist()
    top_values = []
    if non_null_count:
        vc = non_null.astype(str).value_counts().head(5)
        top_values = [
            {"value": value, "count": int(count)}
            for value, count in vc.items()
        ]

    lengths = non_null.astype(str).str.len() if non_null_count else pd.Series(dtype="float64")
    if len(lengths) > 0:
        length_stats = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "avg": round(float(lengths.mean()), 2),
        }
    else:
        length_stats = {"min": 0, "max": 0, "avg": 0.0}

    sample_strings = sample_non_null[: min(10, len(sample_non_null))]
    joined = " ".join(sample_strings).lower()
    distinctive_tokens = sorted({
        tok for tok in re.split(r"[^a-z0-9]+", joined)
        if len(tok) >= 3 and not tok.isdigit()
    })[:8]

    pattern = _infer_value_pattern(series, sample_strings)
    upper_ratio = 0.0
    if sample_strings:
        upper_ratio = sum(1 for v in sample_strings if v.isupper()) / len(sample_strings)

    ontology_markers = ("bao", "go_", "hp:", "ncit", "uberon", "snomed", "mondo")
    has_ontology_marker = any(token in joined for token in ontology_markers)
    is_code_like = pattern == "code_like" or has_ontology_marker
    is_ontology_like = has_ontology_marker

    if is_ontology_like:
        summary = "Ontology/code-system identifiers dominate the populated values"
    elif is_code_like:
        summary = "Short controlled identifiers or compact codes dominate the populated values"
    elif pattern in {"short_enum", "binary_flag"}:
        summary = "Dense low-cardinality categorical field"
    elif pattern in {"numeric", "positive_numeric"}:
        summary = "Mostly numeric field"
    else:
        summary = "Mostly free-text or descriptive values"

    return {
        "fill_rate": f"{(non_null_count / total_count):.0%}" if total_count else "0%",
        "value_pattern": pattern,
        "top_values": top_values,
        "sample_non_null_values": sample_non_null,
        "length_stats": length_stats,
        "is_code_like": is_code_like,
        "is_ontology_like": is_ontology_like,
        "uppercase_ratio": round(float(upper_ratio), 2),
        "distinctive_tokens": distinctive_tokens,
        "profile_summary": summary,
    }


def get_sample_data(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    table_type: str,
    column: str,
    sample_size: int,
    get_schema_profile: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    if source_df is None or target_df is None:
        return {"error": "No task loaded"}

    if table_type == "source":
        df = source_df
    elif table_type == "target":
        df = target_df
    else:
        return {"error": f"Invalid table_type: {table_type}, use 'source' or 'target'"}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in {table_type} table"}

    sample = df[column].head(sample_size).tolist()
    col_data = df[column]
    stats = {
        "dtype": str(col_data.dtype),
        "non_null_count": int(col_data.notna().sum()),
        "null_count": int(col_data.isna().sum()),
        "unique_count": int(col_data.nunique()),
    }
    profile = build_lightweight_sample_profile(col_data, sample_size)
    schema_profile = get_schema_profile(table_type, column)
    return {
        "table_type": table_type,
        "column": column,
        "sample_data": sample,
        "statistics": stats,
        "profile": profile,
        "schema_card": schema_profile.get("schema_card"),
    }

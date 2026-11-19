"""Column profiles, schema roles, and structural summaries."""
from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd

from asm.tools.text import (
    _dedupe_list,
    _generate_abbreviation_signatures,
    _name_candidate_variants,
    _split_identifier,
)


_ATTRIBUTE_FAMILY_KEYWORDS = {
    "identifier": {"id", "identifier", "submitter"},
    "code": {"code", "icd", "ajcc", "figo", "tnm"},
    "site": {"site", "location", "organ", "tissue", "origin", "anatomic", "anatomical", "laterality"},
    "country": {"country", "residence", "birth", "citizenship"},
    "age": {"age", "birth", "dob", "year", "days"},
    "sex_gender": {"sex", "gender"},
    "race_ethnicity": {"race", "ethnicity"},
    "diagnosis": {"diagnosis", "disease", "histology", "morphology", "tumor", "cancer"},
    "stage_grade": {"stage", "grade", "class", "classification", "score"},
    "size_measurement": {"size", "weight", "height", "length", "width", "diameter", "depth", "volume", "percent", "count", "quantity", "measurement"},
    "time_date": {"date", "time", "days", "month", "year", "onset", "follow", "progression", "recurrence", "collection"},
    "status": {"status", "present", "indicator", "flag", "result", "response", "outcome"},
    "sample_specimen": {"sample", "specimen", "aliquot", "portion", "analyte", "slide", "biospecimen"},
    "treatment": {"treatment", "therapy", "dose", "regimen", "drug", "radiation"},
}

_TIME_ANCHOR_KEYWORDS = {
    "birth": {"birth", "dob"},
    "diagnosis": {"diagnosis", "diag"},
    "index": {"index"},
    "enrollment": {"enrollment", "enroll"},
    "follow_up": {"follow", "followup", "fup"},
    "death": {"death", "deceased"},
    "treatment": {"treatment", "therapy"},
    "collection": {"collection", "procurement", "sample"},
    "progression": {"progression", "recurrence", "relapse"},
}

_EVENT_SPECIFIC_TOKENS = {
    "biopsy", "resection", "surgery", "procedure", "enrollment", "follow", "index",
    "collection", "procurement", "test", "diagnosis", "treatment", "progression",
}

_ORIGIN_TOKENS = {"origin", "organ", "tissue", "primary"}

_LOCATION_TOKENS = {"site", "location", "laterality", "country", "residence", "birth"}

_SCHEMA_ROLE_KEYWORDS = {
    "identifier": {"id", "identifier", "key", "uid", "uuid"},
    "code": {"code", "cd", "icd", "ajcc", "figo", "tnm"},
    "name_label": {"name", "label", "title", "desc", "description", "nm"},
    "type_class": {"type", "class", "category", "kind", "subtype", "group"},
    "flag_status": {"flag", "status", "marker", "indicator", "present", "response", "outcome"},
    "percent_rate": {"percent", "pct", "perc", "rate", "ratio", "share"},
}

_ROLE_SUFFIXES = {
    "identifier": ["identifier", "id", "key", "uid", "uuid", "iid", "sid", "d"],
    "code": ["code", "cd"],
    "name_label": ["description", "desc", "label", "name", "nm", "ttl"],
    "type_class": ["category", "class", "group", "type", "kind", "cat", "tp"],
    "flag_status": ["indicator", "status", "marker", "flag", "ind", "flg", "mrkr"],
    "percent_rate": ["percent", "ratio", "share", "rate", "pct", "perc", "prcnt"],
}

_ROLE_PAIR_SETS = {
    frozenset({"identifier", "name_label"}),
    frozenset({"identifier", "type_class"}),
    frozenset({"code", "name_label"}),
    frozenset({"code", "type_class"}),
}

_ROLE_STOPWORDS = {
    "source", "target", "table", "column", "field", "value", "record", "entity",
    "identifier", "id", "key", "code", "name", "label", "desc", "description",
    "type", "class", "category", "kind", "flag", "status", "marker", "indicator",
    "percent", "pct", "perc", "rate", "ratio", "share",
}

ROLE_STOPWORDS = _ROLE_STOPWORDS


def _infer_attribute_family(tokens: List[str]) -> str:
    for family, keywords in _ATTRIBUTE_FAMILY_KEYWORDS.items():
        if any(token in keywords for token in tokens):
            return family
    return "unknown"


def _infer_time_anchor(tokens: List[str]) -> str:
    for anchor, keywords in _TIME_ANCHOR_KEYWORDS.items():
        if any(token in keywords for token in tokens):
            return anchor
    return "unknown"


def _infer_granularity(tokens: List[str]) -> str:
    token_set = set(tokens)
    if token_set & _EVENT_SPECIFIC_TOKENS:
        return "event_specific"
    if token_set & _ORIGIN_TOKENS:
        return "organ_or_origin"
    if token_set & _LOCATION_TOKENS:
        return "location"
    return "generic"


def _looks_like_date(value: str) -> bool:
    return bool(re.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$", value)) or bool(
        re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", value)
    )


def _detect_unit_hints(tokens: List[str], samples: List[str]) -> List[str]:
    hints = []
    token_set = set(tokens)
    keyword_units = {
        "kg": {"weight", "mass", "body"},
        "lb": {"pound", "lbs"},
        "cm": {"cm", "length", "height", "width", "diameter", "depth", "size"},
        "mm": {"mm", "thickness"},
        "days": {"days", "day"},
        "years": {"year", "age"},
        "percent": {"percent", "pct"},
    }
    for unit, keywords in keyword_units.items():
        if token_set & keywords:
            hints.append(unit)
    sample_blob = " ".join(samples[:5]).lower()
    if "%" in sample_blob:
        hints.append("percent")
    if re.search(r"\b\d+(\.\d+)?\s*kg\b", sample_blob):
        hints.append("kg")
    if re.search(r"\b\d+(\.\d+)?\s*cm\b", sample_blob):
        hints.append("cm")
    return _dedupe_list(hints)


def _infer_value_pattern(series: pd.Series, samples: List[str]) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    lowered = [str(v).strip().lower() for v in samples if str(v).strip()]
    if lowered and all(v in {"yes", "no", "true", "false", "0", "1", "present", "absent", "male", "female"} for v in lowered[: min(5, len(lowered))]):
        return "binary_flag"
    if lowered and all(_looks_like_date(v) for v in lowered[: min(5, len(lowered))]):
        return "date_like"
    if pd.api.types.is_numeric_dtype(series):
        if float(non_null.min()) >= 0:
            return "positive_numeric"
        return "numeric"
    if lowered and len(set(lowered)) <= 12:
        return "short_enum"
    if lowered and all(re.match(r"^[a-z]*\d+[a-z0-9._-]*$", v) for v in lowered[: min(5, len(lowered))]):
        return "code_like"
    avg_length = np.mean([len(v) for v in lowered[: min(10, len(lowered))]]) if lowered else 0
    if avg_length > 30:
        return "long_text"
    return "text"


def _infer_semantic_type(series: pd.Series, family: str, value_pattern: str) -> str:
    if family in {"identifier", "code"}:
        return family
    if family in {"time_date", "age"} or value_pattern == "date_like":
        return "temporal"
    if family == "size_measurement" or value_pattern in {"numeric", "positive_numeric"}:
        return "measurement"
    if value_pattern in {"binary_flag", "short_enum"} or family in {"status", "sex_gender", "race_ethnicity"}:
        return "categorical"
    if family in {"site", "country", "diagnosis", "sample_specimen", "treatment"}:
        return "semantic_text"
    return "text"


def _infer_schema_role(
    tokens: List[str],
    family: str,
    value_pattern: str,
    non_null: int,
    unique: int,
) -> str:
    token_set = set(tokens)
    compact_name = re.sub(r"[^a-z0-9]+", "", "".join(tokens))
    unique_ratio = (unique / non_null) if non_null else 0.0

    if token_set & _SCHEMA_ROLE_KEYWORDS["identifier"]:
        return "identifier"
    if token_set & _SCHEMA_ROLE_KEYWORDS["name_label"]:
        return "name_label"
    if token_set & _SCHEMA_ROLE_KEYWORDS["type_class"]:
        return "type_class"
    if token_set & _SCHEMA_ROLE_KEYWORDS["flag_status"]:
        return "flag_status"
    if token_set & _SCHEMA_ROLE_KEYWORDS["percent_rate"]:
        return "percent_rate"
    if token_set & _SCHEMA_ROLE_KEYWORDS["code"] or family == "code":
        return "code"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["name_label"]):
        return "name_label"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["type_class"]):
        return "type_class"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["flag_status"]):
        return "flag_status"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["percent_rate"]):
        return "percent_rate"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["code"]):
        return "code"
    if any(compact_name.endswith(suffix) for suffix in _ROLE_SUFFIXES["identifier"]) and len(compact_name) >= 5:
        return "identifier"
    if family == "identifier":
        return "identifier"
    if value_pattern == "binary_flag" or family == "status":
        return "flag_status"
    if value_pattern == "date_like" or family in {"time_date", "age"}:
        return "temporal"
    if family == "size_measurement" or value_pattern in {"numeric", "positive_numeric"}:
        return "amount_measure"
    if value_pattern == "code_like" and unique_ratio >= 0.8 and non_null >= 3:
        return "identifier"
    return "unknown"


def _compact_signature(name: str, role: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    for suffix in _ROLE_SUFFIXES.get(role, []):
        if compact.endswith(suffix) and len(compact) - len(suffix) >= 4:
            compact = compact[: -len(suffix)]
            break
    return compact


def _infer_group_signature(name: str, expanded_texts: List[str], role: str) -> str:
    candidate_tokens: List[str] = []
    for text in expanded_texts[:2]:
        candidate_tokens.extend(text.split())
    filtered = [token for token in candidate_tokens if token not in _ROLE_STOPWORDS]
    if filtered:
        return "".join(filtered[:4])
    compact = _compact_signature(name, role)
    return compact if len(compact) >= 4 else re.sub(r"[^a-z0-9]+", "", name.lower())


def _roles_are_complementary(role_a: str, role_b: str) -> bool:
    return frozenset({role_a, role_b}) in _ROLE_PAIR_SETS


def build_column_profile(col_name: str, series: pd.Series) -> Dict[str, Any]:
    samples = [str(v) for v in series.dropna().head(5).tolist()]
    expanded = _name_candidate_variants(col_name)
    tokens = _split_identifier(col_name)
    primary_text = expanded[1]["text"] if len(expanded) > 1 else expanded[0]["text"]
    semantic_tokens = tokens + primary_text.split()
    family = _infer_attribute_family(semantic_tokens)
    value_pattern = _infer_value_pattern(series, samples)
    semantic_type = _infer_semantic_type(series, family, value_pattern)
    non_null = int(series.notna().sum())
    unique = int(series.nunique())
    schema_role = _infer_schema_role(semantic_tokens, family, value_pattern, non_null, unique)
    abbreviation_signatures = _generate_abbreviation_signatures(tokens, [c["text"] for c in expanded])
    return {
        "column": col_name,
        "tokens": tokens,
        "expanded_name_candidates": expanded,
        "expanded_name_texts": [c["text"] for c in expanded],
        "abbreviation_signatures": abbreviation_signatures,
        "semantic_type": semantic_type,
        "attribute_family": family,
        "schema_role": schema_role,
        "group_signature": _infer_group_signature(col_name, [c["text"] for c in expanded], schema_role),
        "unit_hint": _detect_unit_hints(tokens, samples),
        "value_pattern": value_pattern,
        "time_anchor": _infer_time_anchor(tokens + primary_text.split()),
        "granularity": _infer_granularity(tokens + primary_text.split()),
        "non_null": non_null,
        "unique": unique,
        "sample_values": samples,
    }


def build_schema_card(
    table_type: str,
    profile: Dict[str, Any],
    local_context: Dict[str, Any],
    enrichment: Optional[str],
    *,
    identifier_like: bool,
    name_like: bool,
) -> Dict[str, Any]:
    expanded_text = (
        profile["expanded_name_candidates"][0]["text"]
        if profile.get("expanded_name_candidates")
        else profile["column"].replace("_", " ")
    )
    entity_tokens = [
        tok for tok in expanded_text.split()
        if tok not in ROLE_STOPWORDS and tok not in {"label", "value", "code"}
    ]
    likely_entity = " ".join(entity_tokens[:3]) if entity_tokens else profile["column"].replace("_", " ")

    if enrichment:
        definition_summary = enrichment[:200]
    else:
        role_text = profile["schema_role"].replace("_", " ")
        family_text = profile["attribute_family"].replace("_", " ")
        definition_summary = (
            f"Likely {role_text} field for {likely_entity}; "
            f"family={family_text}, pattern={profile['value_pattern']}"
        )

    return {
        "column": profile["column"],
        "likely_entity": likely_entity,
        "likely_role": profile["schema_role"],
        "likely_family": profile["attribute_family"],
        "semantic_type": profile["semantic_type"],
        "value_pattern": profile["value_pattern"],
        "identifier_like": identifier_like,
        "name_like": name_like,
        "unit_hint": profile["unit_hint"],
        "group_signature": profile["group_signature"],
        "paired_columns": local_context["paired_columns"],
        "sibling_columns": local_context["sibling_columns"],
        "definition_summary": definition_summary,
    }


def profile_is_identifier_like(profile: Dict[str, Any]) -> bool:
    if profile.get("schema_role") in {"identifier", "code"}:
        return True
    if profile.get("semantic_type") in {"identifier", "code"}:
        return True
    if profile.get("value_pattern") == "code_like":
        return True
    non_null = max(1, int(profile.get("non_null", 0) or 0))
    unique = int(profile.get("unique", 0) or 0)
    return non_null >= 3 and (unique / non_null) >= 0.85 and profile.get("value_pattern") in {"code_like", "short_enum"}


def profile_is_name_like(profile: Dict[str, Any]) -> bool:
    return profile.get("schema_role") == "name_label" or profile.get("semantic_type") in {"text", "semantic_text"}

"""Candidate parsing and submitted score validation."""
from difflib import SequenceMatcher
from typing import Any, Dict, List
import json
import math
import re


def submit_candidates_from_arguments(arguments: Dict[str, Any]):
    candidates = arguments.get("ranked_candidates", [])
    if not candidates and arguments.get("target_column"):
        candidates = arguments.get("target_column")
    if not candidates and arguments.get("target"):
        candidates = arguments.get("target")
    if not candidates and arguments.get("match"):
        candidates = arguments.get("match")
    return candidates


def coerce_candidates(raw) -> list:
    """Best-effort extraction of candidate column names from various formats."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    names = re.findall(r"<name>\s*(.+?)\s*</name>", raw)
    if names:
        return names
    quoted = re.findall(r'["\']([^"\']+)["\']', raw)
    if quoted:
        return quoted
    ticked = re.findall(r"`([^`]+)`", raw)
    if ticked:
        return ticked
    if "<" not in raw and len(raw) < 200:
        return [c.strip() for c in raw.split(",") if c.strip()]
    return []


def fuzzy_match_columns(names: List[str], target_columns: List[str], threshold: float = 0.8) -> List[str]:
    """Try to fuzzy-match invalid column names to actual target columns."""
    matched = []
    for name in names:
        name_lower = name.strip().lower().replace(" ", "_")
        best_score, best_col = 0.0, None
        for col in target_columns:
            col_lower = col.lower()
            score = SequenceMatcher(None, name_lower, col_lower).ratio()
            if score > best_score:
                best_score, best_col = score, col
        if best_score >= threshold and best_col and best_col not in matched:
            matched.append(best_col)
    return matched


def normalize_submit_candidates(
    ranked_candidates: List[Any],
    target_columns,
    topk: int,
) -> tuple[List[str], List[str]]:
    valid_candidates: List[str] = []
    invalid_names: List[str] = []
    target_column_set = set(target_columns)
    for col in ranked_candidates:
        if not isinstance(col, str):
            continue
        col = col.strip()
        if col.lower() == "none" or not col:
            continue
        if col in target_column_set:
            if col not in valid_candidates:
                valid_candidates.append(col)
        else:
            invalid_names.append(col)

    if not valid_candidates and invalid_names:
        valid_candidates = fuzzy_match_columns(invalid_names, list(target_columns))

    return valid_candidates[:topk], invalid_names


def validate_match_scores(candidates, scores, target_columns):
    """Validate the full submitted list before top-k truncation.

    Exact names avoid attaching a model score to a fuzzy-repaired target.
    The legacy rank-scoring path retains its original permissive normalization.
    """
    if not candidates or any(not isinstance(c, str) for c in candidates):
        raise ValueError("ranked_candidates must contain target column names")
    if len(candidates) == 1 and candidates[0].strip().lower() == "none":
        if scores != {}:
            raise ValueError("explicit no-match requires candidate_scores: {}")
        return {}
    if len(set(candidates)) != len(candidates):
        raise ValueError("ranked_candidates must not contain duplicates")
    if any(c.lower() == "none" or c not in target_columns for c in candidates):
        raise ValueError("use exact target column names; do not mix None with matches")
    if not isinstance(scores, dict) or set(scores) != set(candidates):
        raise ValueError("candidate_scores keys must exactly match ranked_candidates")
    result = {}
    previous = math.inf
    for name in candidates:
        value = scores[name]
        if type(value) not in (float, int) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("each candidate score must be a finite number in [0, 1]")
        if value > previous:
            raise ValueError("candidate_scores must be non-increasing in ranked_candidates order")
        result[name] = float(value)
        previous = value
    return result


def match_score_prompt(prompt, mode):
    """Add a submission-only instruction, retaining the five cognitive stages."""
    if mode == "rank":
        return prompt
    if mode != "llm":
        raise ValueError("unknown match score mode")
    anchor = '- `reasoning` (str): brief explanation of the matching rationale'
    if prompt.count(anchor) != 1:
        raise ValueError("submit prompt layout changed; inspect before applying score instruction")
    example = '<submit>{"ranked_candidates": ["best_match", "second_best"], "reasoning": "Based on value overlap and semantic similarity"}</submit>'
    scored_example = '<submit>{"ranked_candidates": ["best_match", "second_best"], "candidate_scores": {"best_match": 0.86, "second_best": 0.64}, "reasoning": "Based on value overlap and semantic similarity"}</submit>'
    if prompt.count(example) != 1:
        raise ValueError("submit example changed; inspect before applying score instruction")
    instruction = """
- candidate_scores: a complete mapping from each exact submitted name to a finite
  number in [0, 1], non-increasing in ranked_candidates order. No duplicates;
  ties keep the submitted order. For ["None"], use {}.
Scores express evidence support for correspondence, not calibrated probabilities.
Use a common scale across source fields; consider meaning, definitions, units and
encodings. Contradictions lower support; absent evidence is not positive support.
Do not derive scores from rank, normalize within a field, force the first to 1,
or require a sum of 1. Multiple valid matches may score highly. Example numbers
illustrate format only.
"""
    return prompt.replace(example, scored_example).replace(anchor, anchor + '\n' + instruction)

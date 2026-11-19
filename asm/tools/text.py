"""Text normalization and similarity helpers for matching tools."""
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd


_ABBREVIATION_MAP = {
    "src": ["source"],
    "pt": ["patient"],
    "pat": ["patient", "pathology"],
    "dx": ["diagnosis"],
    "diag": ["diagnosis"],
    "dob": ["date of birth"],
    "dobs": ["date of birth"],
    "wt": ["weight"],
    "ht": ["height"],
    "len": ["length"],
    "wid": ["width"],
    "qty": ["quantity"],
    "amt": ["amount"],
    "cnt": ["count"],
    "num": ["number"],
    "pct": ["percent"],
    "perc": ["percent"],
    "vol": ["volume"],
    "proc": ["procedure"],
    "path": ["pathologic", "pathology"],
    "clin": ["clinical"],
    "fu": ["follow up"],
    "fup": ["follow up"],
    "res": ["residence", "result"],
    "enroll": ["enrollment"],
    "orig": ["origin"],
    "org": ["organ"],
    "loc": ["location"],
}


def _split_identifier(name: str) -> List[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    normalized = normalized.replace(".", "_").replace("-", "_").replace("/", "_")
    return [part.lower() for part in re.split(r"[^A-Za-z0-9]+", normalized) if part]


def _dedupe_list(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _name_candidate_variants(name: str, max_candidates: int = 4) -> List[Dict[str, Any]]:
    tokens = _split_identifier(name)
    if not tokens:
        return [{"text": name.lower(), "confidence": 1.0}]

    variants: List[Tuple[str, float]] = [(" ".join(tokens), 1.0)]
    primary_tokens: List[str] = []
    ambiguous_variants: List[Tuple[str, float]] = []
    for idx, token in enumerate(tokens):
        expansions = _ABBREVIATION_MAP.get(token)
        if expansions:
            primary_tokens.extend(expansions[0].split())
            for alt in expansions[1:]:
                alt_tokens = tokens[:idx] + alt.split() + tokens[idx + 1 :]
                ambiguous_variants.append((" ".join(alt_tokens), 0.6))
        else:
            primary_tokens.append(token)
    variants.append((" ".join(primary_tokens), 0.88))
    variants.extend(ambiguous_variants[: max_candidates - 2])

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for text, confidence in variants:
        text = " ".join(text.split())
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append({"text": text, "confidence": round(confidence, 2)})
        if len(deduped) >= max_candidates:
            break
    return deduped


def _generate_abbreviation_signatures(tokens: List[str], expanded_texts: List[str]) -> List[str]:
    signatures: List[str] = []

    token_sets: List[List[str]] = [tokens]
    for text in expanded_texts[:2]:
        words = [w for w in text.split() if w]
        if words:
            token_sets.append(words)

    for words in token_sets:
        if not words:
            continue
        signatures.append("".join(w[0] for w in words if w))
        signatures.append("".join(w[:2] for w in words if w))
        signatures.append("".join((w[:2] if i < len(words) - 1 else w[:1]) for i, w in enumerate(words) if w))
        signatures.append("".join((w[:2] if i == 0 else w[:1]) for i, w in enumerate(words) if w))

    cleaned = []
    seen = set()
    for sig in signatures:
        compact = re.sub(r"[^a-z0-9]+", "", sig.lower())
        if len(compact) >= 2 and compact not in seen:
            seen.add(compact)
            cleaned.append(compact)
    return cleaned


def _abbreviation_similarity(source_signatures: List[str], target_signatures: List[str]) -> float:
    if not source_signatures or not target_signatures:
        return 0.0
    best = 0.0
    for src in source_signatures:
        for tgt in target_signatures:
            if src == tgt:
                return 1.0
            if src in tgt or tgt in src:
                best = max(best, min(len(src), len(tgt)) / max(len(src), len(tgt)))
            else:
                best = max(best, SequenceMatcher(None, src, tgt).ratio())
    return round(float(best), 4)


def _build_column_text(
    col_name: str,
    series: pd.Series,
    max_samples: int = 5,
    enrichment: Optional[str] = None,
) -> str:
    """Build a textual representation of a column for embedding."""
    parts = [col_name.replace("_", " ")]
    if enrichment:
        parts.append(enrichment)
    dtype_str = str(series.dtype)
    parts.append(f"type:{dtype_str}")
    non_null = series.dropna()
    if len(non_null) > 0:
        samples = non_null.head(max_samples).astype(str).tolist()
        parts.append("values: " + ", ".join(samples))
    return " | ".join(parts)


def _best_text_similarity(source_texts: List[str], target_texts: List[str]) -> float:
    if not source_texts or not target_texts:
        return 0.0
    best = 0.0
    for src in source_texts:
        src_tokens = set(src.split())
        for tgt in target_texts:
            tgt_tokens = set(tgt.split())
            if not src_tokens or not tgt_tokens:
                continue
            token_jaccard = len(src_tokens & tgt_tokens) / len(src_tokens | tgt_tokens)
            sequence = SequenceMatcher(None, src, tgt).ratio()
            best = max(best, 0.6 * token_jaccard + 0.4 * sequence)
    return round(float(best), 4)


def _signature_similarity(source_sig: str, target_sig: str) -> float:
    if not source_sig or not target_sig:
        return 0.35
    if source_sig == target_sig:
        return 1.0
    return round(float(SequenceMatcher(None, source_sig, target_sig).ratio()), 4)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and a matrix of vectors."""
    return np.dot(b, a)

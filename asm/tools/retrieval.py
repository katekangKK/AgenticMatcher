"""Embedding and string retrieval for candidate shortlisting."""
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence
import os

import numpy as np

from asm.tools.text import _build_column_text, _cosine_similarity


class _EmbeddingModel:
    """Lazy-loaded sentence-transformer for shortlist_match and doc_query.

    Uses BGE-large for better retrieval quality. Query texts get an instruction
    prefix as recommended by the BGE authors.
    """

    _instance = None

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    ALLOW_REMOTE_ENV = "ASM_ALLOW_REMOTE_EMBEDDING_DOWNLOAD"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._load_attempted = False
        return cls._instance

    def _load(self):
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            allow_remote = os.getenv(self.ALLOW_REMOTE_ENV, "").lower() in {"1", "true", "yes"}
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                local_files_only=not allow_remote,
            )
        except ImportError:
            self._model = None
        except Exception:
            self._model = None

    def encode(self, texts: List[str], *, is_query: bool = False) -> Optional[np.ndarray]:
        self._load()
        if self._model is None:
            return None
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


_embedding_model = _EmbeddingModel()


def detect_table_prefixed_columns(cols: Optional[Sequence[str]]) -> Optional[Dict[str, List[int]]]:
    """Group TABLE.column names by table when table prefixes are dominant."""
    if not cols:
        return None

    dotted = sum(1 for col in cols if "." in col)
    if dotted < len(cols) * 0.5:
        return None

    groups: Dict[str, List[int]] = {}
    for idx, col in enumerate(cols):
        table = col.split(".")[0] if "." in col else "__default__"
        groups.setdefault(table, []).append(idx)
    return groups


def table_prefilter(
    src_emb: np.ndarray,
    target_col_embeddings: Optional[np.ndarray],
    target_col_names: Optional[Sequence[str]],
    top_j: int = 3,
) -> Optional[List[int]]:
    """Return column indices from the top-J most relevant prefixed tables."""
    groups = detect_table_prefixed_columns(target_col_names)
    if groups is None or len(groups) <= top_j:
        return None
    if target_col_embeddings is None:
        return None

    table_names = list(groups.keys())
    table_embs = np.stack([
        target_col_embeddings[groups[table]].mean(axis=0)
        for table in table_names
    ])
    table_embs = table_embs / (np.linalg.norm(table_embs, axis=1, keepdims=True) + 1e-9)
    sims = _cosine_similarity(src_emb, table_embs)
    top_tables = np.argsort(-sims)[:top_j]

    allowed_indices: List[int] = []
    for table_idx in top_tables:
        allowed_indices.extend(groups[table_names[table_idx]])
    return sorted(allowed_indices)


def combine_similarity_scores(
    name_sims: np.ndarray,
    value_sims: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Combine name and value similarities using the shortlist strategy."""
    if strategy == "name_only":
        return name_sims
    if strategy == "value_only":
        return value_sims
    return 0.4 * name_sims + 0.6 * value_sims


def apply_allowed_index_mask(scores: np.ndarray, allowed_indices: Optional[Sequence[int]]) -> np.ndarray:
    """Mask scores outside allowed indices with negative infinity."""
    if allowed_indices is None:
        return scores

    masked = np.full(len(scores), -np.inf)
    for idx in allowed_indices:
        masked[idx] = scores[idx]
    return masked


def select_pool_indices(scores: np.ndarray, top_k: int) -> List[int]:
    """Select the shortlist candidate pool indices from ranked scores."""
    pool_size = min(max(top_k * 4, 20), len(scores))
    return list(np.argsort(-scores)[:pool_size])


def build_shortlist_candidate(
    col_name: str,
    col_series: Any,
    target_profile: Dict[str, Any],
    match_details: Dict[str, Any],
    name_similarity: float,
    value_similarity: float,
    target_local_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the public shortlist candidate payload."""
    non_null = int(col_series.notna().sum())
    total = len(col_series)
    return {
        "column": col_name,
        "score": match_details["final_score"],
        "name_similarity": round(float(name_similarity), 4),
        "value_similarity": round(float(value_similarity), 4),
        "expanded_name_match": match_details["expansion_match"],
        "abbreviation_match": match_details["abbreviation_match"],
        "semantic_type_compatibility": match_details["semantic_type_compatibility"],
        "attribute_family_compatibility": match_details["attribute_family_compatibility"],
        "role_compatibility": match_details["role_compatibility"],
        "local_context_compatibility": match_details["local_context_compatibility"],
        "pair_consistency_bonus": match_details["pair_consistency_bonus"],
        "structural_signal_bonus": match_details["structural_signal_bonus"],
        "structural_signal_penalty": match_details["structural_signal_penalty"],
        "structural_subtype_match": match_details["structural_subtype_match"],
        "target_structural_subtype": match_details["target_structural_subtype"],
        "schema_confidence": match_details["schema_confidence"],
        "unit_compatibility": match_details["unit_compatibility"],
        "value_pattern_compatibility": match_details["value_pattern_compatibility"],
        "why_matched": match_details["why_matched"],
        "normalized_evidence": {
            "target_expansions": target_profile["expanded_name_candidates"],
            "target_semantic_type": target_profile["semantic_type"],
            "target_attribute_family": target_profile["attribute_family"],
            "target_schema_role": target_profile["schema_role"],
            "target_schema_card": target_profile.get("schema_card"),
            "target_group_signature": target_profile["group_signature"],
            "target_unit_hint": target_profile["unit_hint"],
            "target_value_pattern": target_profile["value_pattern"],
            "target_local_context": target_local_context,
            "possible_transformations": match_details["possible_transformations"],
        },
        "non_null": non_null,
        "unique": int(col_series.nunique()),
        "fill_rate": f"{non_null/total:.0%}" if total else "0%",
    }


def shortlist_match(env, arguments: Dict[str, Any], embedding_model) -> Dict[str, Any]:
    """Embedding-based candidate retrieval with table pre-filtering."""
    if env._source_df is None or env._target_df is None:
        return {"error": "No task loaded"}

    top_k = arguments.get("top_k", 10)
    strategy = arguments.get("strategy", "hybrid")

    src_col = env._source_column
    src_series = env._source_df[src_col]
    source_profile = env._get_profile("source", src_col)

    src_name_text = src_col.replace("_", " ")
    src_enrichment = env._get_column_enrichment(src_col)
    src_value_text = _build_column_text(src_col, src_series, enrichment=src_enrichment)

    env._compute_target_embeddings()

    if env._target_col_embeddings is not None:
        # Encode source as query
        tgt_name_texts = [c.replace("_", " ") for c in env._target_col_names]
        tgt_name_embs = embedding_model.encode(tgt_name_texts, is_query=False)
        src_name_emb = embedding_model.encode([src_name_text], is_query=True)[0]
        name_sims = _cosine_similarity(src_name_emb, tgt_name_embs)

        src_value_emb = embedding_model.encode([src_value_text], is_query=True)[0]
        value_sims = _cosine_similarity(src_value_emb, env._target_col_embeddings)

        combined = combine_similarity_scores(name_sims, value_sims, strategy)

        # P2: table-level pre-filter for large multi-table targets
        allowed_indices = env._table_prefilter(src_value_emb, top_j=3)
        combined = apply_allowed_index_mask(combined, allowed_indices)

        pool_indices = select_pool_indices(combined, top_k)
        scored_candidates = []
        for idx in pool_indices:
            if combined[idx] == -np.inf:
                continue
            col_name = env._target_col_names[idx]
            col_series = env._target_df[col_name]
            target_profile = env._get_profile("target", col_name)
            match_details = env._pairwise_match_details(
                source_profile,
                target_profile,
                base_score=float(combined[idx]),
            )
            scored_candidates.append(build_shortlist_candidate(
                col_name,
                col_series,
                target_profile,
                match_details,
                name_sims[idx],
                value_sims[idx],
                env._get_local_context("target", col_name),
            ))
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = scored_candidates[:top_k]
        env._last_shortlist_candidates = candidates
        must_verify_candidates = env._select_must_verify_candidates(candidates)
        env._last_must_verify_candidates = must_verify_candidates

        note_parts = []
        if allowed_indices is not None:
            groups = env._detect_table_prefixed_columns()
            n_tables = len(groups) if groups else 0
            note_parts.append(f"table-filter: top-3 of {n_tables} tables")

        result = {
            "success": True,
            "source_column": src_col,
            "strategy": strategy,
            "source_interpretation": env._build_interpretation_summary(source_profile, "source"),
            "candidates": candidates,
            "total_target_columns": len(env._target_col_names),
        }
        if len(candidates) >= 2:
            gap = round(candidates[0]["score"] - candidates[1]["score"], 4)
            result["top_gap"] = gap
            if gap <= 0.08:
                result["recommended_next_step"] = "differential_verify"
        if must_verify_candidates:
            result["must_verify_candidates"] = must_verify_candidates
        if note_parts:
            result["note"] = "; ".join(note_parts)
        return result

    src_name_lower = src_col.lower().replace("_", " ")
    scores = []
    for c in env._target_df.columns:
        tgt_name_lower = c.lower().replace("_", " ")
        sim = SequenceMatcher(None, src_name_lower, tgt_name_lower).ratio()
        scores.append((c, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = []
    for c, s in scores[: max(top_k * 4, 20)]:
        col_series = env._target_df[c]
        target_profile = env._get_profile("target", c)
        match_details = env._pairwise_match_details(
            source_profile,
            target_profile,
            base_score=(2 * s) - 1,
        )
        candidates.append(build_shortlist_candidate(
            c,
            col_series,
            target_profile,
            match_details,
            s,
            0.0,
            env._get_local_context("target", c),
        ))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_k]
    env._last_shortlist_candidates = candidates
    must_verify_candidates = env._select_must_verify_candidates(candidates)
    env._last_must_verify_candidates = must_verify_candidates
    result = {
        "success": True,
        "source_column": src_col,
        "source_interpretation": env._build_interpretation_summary(source_profile, "source"),
        "strategy": "fallback_string_similarity",
        "candidates": candidates,
        "total_target_columns": len(env._target_df.columns),
        "note": "sentence-transformers not available, using string similarity fallback",
    }
    if must_verify_candidates:
        result["must_verify_candidates"] = must_verify_candidates
    return result

"""
Schema Matching Environment – manages data, tool execution, and evaluation.

Implements the ASM tool layer:
- shortlist_match  (LOW cost)   – embedding-based candidate retrieval
- get_sample_data  (FREE)       – column sample & statistics
- execute_python   (HIGH cost)  – budgeted subprocess execution
- doc_query        (MEDIUM cost) – domain dictionary semantic search
- submit           (FREE)       – final answer submission
"""

import copy
import math
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from asm.tools.python import CodeExecutor
from asm.tools.profiling import _roles_are_complementary
from asm.tools.text import _signature_similarity
from asm.tools.text import _build_column_text
from asm.tools.profiling import build_column_profile
from asm.tools.profiling import build_schema_card
from asm.tools.retrieval import _embedding_model
from asm.tools.retrieval import detect_table_prefixed_columns, table_prefilter
from asm.tools.compatibility import pairwise_match_details
from asm.tools.compatibility import build_interpretation_summary
from asm.submission import coerce_candidates, normalize_submit_candidates, submit_candidates_from_arguments
from asm.submission import validate_match_scores
from asm.tools.sampling import get_sample_data
from asm.tools.python import execute_python
from asm.tools.documents import doc_query, prepare_documentation
from asm.tools.compatibility import differential_verify
from asm.tools.retrieval import shortlist_match
from asm.tools.profiling import profile_is_identifier_like, profile_is_name_like


# ---------------------------------------------------------------------------
# Tool cost constants for trajectory analysis
# ---------------------------------------------------------------------------

TOOL_COSTS = {
    "shortlist_match": "low",
    "differential_verify": "low",
    "get_sample_data": "free",
    "execute_python": "high",
    "doc_query": "medium",
    "submit": "free",
}


@dataclass
class MatchResult:
    """Result of a single schema matching task."""
    task_name: str
    source_column: str
    predicted_column: Optional[str] = None
    predicted_candidates: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None
    num_rounds: int = 0
    truncation_count: int = 0
    conversation_history: List[Dict] = field(default_factory=list)
    trajectory: List[Dict] = field(default_factory=list)
    evidence_trace: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    termination_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class SchemaMatchingEnvironment:
    """
    ASM Schema Matching Environment.

    This class is the orchestration layer. Domain logic lives in tools,
    profiling, validation, and submission modules; the environment
    owns task state, caches, tool dispatch, and result bookkeeping.
    """

    def __init__(
        self,
        sample_size: int = 10,
        topk: int = 20,
        workspace_dir: str = None,
        doc_index: Optional[Dict[str, List[Dict[str, str]]]] = None,
        match_score_mode: str = "rank",
    ):
        """
        Args:
            sample_size: number of sample rows returned by get_sample_data
            topk: max candidates accepted by submit
            workspace_dir: working directory for code executor temp files
            doc_index: optional pre-built domain documentation index
                       Format: {"domain_name": [{"term": ..., "definition": ..., "text": ...}, ...]}
        """
        if match_score_mode not in ("rank", "llm"):
            raise ValueError("match_score_mode must be rank or llm")
        self.match_score_mode = match_score_mode
        self.sample_size = sample_size
        self.topk = topk
        self.executor = CodeExecutor(workspace_dir=workspace_dir)
        self.doc_index = prepare_documentation(doc_index)
        self._doc_embeddings: Optional[Dict[str, Any]] = None

        # Current task state
        self._source_df: Optional[pd.DataFrame] = None
        self._target_df: Optional[pd.DataFrame] = None
        self._source_column: Optional[str] = None
        self._task_name: Optional[str] = None

        # Cached embeddings for current task
        self._target_col_embeddings: Optional[np.ndarray] = None
        self._target_col_names: Optional[List[str]] = None
        self._target_col_texts: Optional[List[str]] = None
        self._profile_cache: Dict[str, Dict[str, Dict[str, Any]]] = {"source": {}, "target": {}}
        self._local_context_cache: Dict[str, Dict[str, Dict[str, Any]]] = {"source": {}, "target": {}}
        self._last_shortlist_candidates: List[Dict[str, Any]] = []
        self._last_must_verify_candidates: List[str] = []
        self._last_doc_current_target_hits: List[str] = []
        self._last_differential_result: Optional[Dict[str, Any]] = None
        self._verification_since_shortlist: bool = False
        self._execute_python_calls_used: int = 0
        self._max_execute_python_calls: Optional[int] = None
        self._doc_query_calls_used: int = 0
        self._max_doc_query_calls: Optional[int] = None
        self._shortlist_top_k_override: Optional[int] = None

        # Submission state
        self._current_proposal = None
        self._current_proposal_scores = {}
        self._submitted_candidates: List[str] = []
        self._submitted_reasoning: Optional[str] = None
        self._submitted_match: Optional[str] = None
        self._submitted_confidence: float = 0.0
        self._submitted_score_map: Dict[str, float] = {}
        self._submitted_trace: List[Dict[str, Any]] = []
        self._evidence_trace: Dict[str, Any] = self._new_evidence_trace()
        self._evidence_event_counter: int = 0
        self._tool_observation_counter: int = 0
        self._execute_python_budget_override: Optional[int] = None
        self._doc_query_budget_override: Optional[int] = None

    # ------------------------------------------------------------------
    # Task lifecycle and state reset
    # ------------------------------------------------------------------


    def configure_tool_budgets(
        self,
        *,
        execute_python_budget: Optional[int] = None,
        doc_query_budget: Optional[int] = None,
    ) -> None:
        """Set optional per-run tool budget overrides applied after every task load."""
        if execute_python_budget is not None and execute_python_budget < 0:
            raise ValueError("execute_python_budget must be >= 0")
        if doc_query_budget is not None and doc_query_budget < 0:
            raise ValueError("doc_query_budget must be >= 0")
        self._execute_python_budget_override = execute_python_budget
        self._doc_query_budget_override = doc_query_budget
        if execute_python_budget is not None:
            self._max_execute_python_calls = execute_python_budget
        if doc_query_budget is not None:
            self._max_doc_query_calls = doc_query_budget

    def configure_shortlist_top_k_override(
        self,
        top_k: Optional[int] = None,
    ) -> None:
        """Force one generic shortlist size for an explicit parameter ablation."""
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("shortlist_top_k_override must be a positive integer")
        self._shortlist_top_k_override = top_k


    def load_task(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_column: str,
        task_name: str = "",
    ) -> bool:
        """Load a schema matching task."""
        try:
            if source_column not in source_df.columns:
                print(f"Error: source column '{source_column}' not found in source table")
                return False

            self._source_df = source_df
            self._target_df = target_df
            self._source_column = source_column
            self._task_name = task_name
            self._execute_python_calls_used = 0
            self._max_execute_python_calls = (
                self._execute_python_budget_override
                if self._execute_python_budget_override is not None
                else 2
            )
            self._doc_query_calls_used = 0
            self._max_doc_query_calls = (
                self._doc_query_budget_override
                if self._doc_query_budget_override is not None
                else None
            )

            # Reset submission and any unaccepted proposal for this attribute.
            self._current_proposal = None
            self._current_proposal_scores = {}
            self._submitted_match = None
            self._submitted_confidence = 0.0
            self._submitted_score_map = {}
            self._submitted_reasoning = None
            self._submitted_candidates = []
            self._submitted_trace = []
            self._evidence_trace = self._new_evidence_trace()
            self._evidence_event_counter = 0
            self._tool_observation_counter = 0

            # Invalidate cached embeddings
            self._target_col_embeddings = None
            self._target_col_names = None
            self._target_col_texts = None
            self._profile_cache = {"source": {}, "target": {}}
            self._local_context_cache = {"source": {}, "target": {}}
            self._last_shortlist_candidates = []
            self._last_must_verify_candidates = []
            self._last_doc_current_target_hits = []
            self._last_differential_result = None
            self._verification_since_shortlist = False

            return True
        except Exception as e:
            print(f"Unable to load task ({type(e).__name__}). Check input tables and configuration.")
            return False

    def _new_evidence_trace(self) -> Dict[str, Any]:
        return {
            "schema_version": 3,
            "task_name": self._task_name,
            "source_column": self._source_column,
            "candidate_proposals": [],
            "candidate_proposal_history": [],
            "tool_observations": [],
            "evidence_events": [],
            "pre_submit_snapshot": {},
            "candidate_evidence": [],
            "final_decision": {},
        }

    def _capture_pre_submit_snapshot(
        self,
        raw_ranked_candidates: Any,
        ranked_candidates: List[str],
        reasoning: str,
    ) -> None:
        """Capture the evidence boundary before the first submit attempt."""
        if not self._evidence_trace or self._evidence_trace.get("pre_submit_snapshot"):
            return

        proposals = copy.deepcopy(self._evidence_trace.get("candidate_proposals") or [])
        proposal_history = copy.deepcopy(
            self._evidence_trace.get("candidate_proposal_history") or []
        )
        observations = copy.deepcopy(self._evidence_trace.get("tool_observations") or [])
        events = copy.deepcopy(self._evidence_trace.get("evidence_events") or [])
        target_columns = set(self._target_df.columns) if self._target_df is not None else set()

        candidate_pool: List[str] = []
        seen = set()
        candidate_names = [row.get("target") for row in proposals]
        candidate_names.extend(event.get("target") for event in events)
        candidate_names.extend(ranked_candidates)
        for candidate in candidate_names:
            if (
                isinstance(candidate, str)
                and candidate in target_columns
                and candidate not in seen
            ):
                seen.add(candidate)
                candidate_pool.append(candidate)

        self._evidence_trace["pre_submit_snapshot"] = {
            "schema_version": 2,
            "capture_stage": "before_first_submit_attempt",
            "raw_ranked_candidates": copy.deepcopy(raw_ranked_candidates),
            "ranked_candidates": list(ranked_candidates),
            "raw_reasoning": str(reasoning or ""),
            "explicit_none": bool(
                ranked_candidates
                and all(
                    isinstance(candidate, str)
                    and candidate.strip().lower() == "none"
                    for candidate in ranked_candidates
                )
            ),
            "candidate_pool": candidate_pool,
            "candidate_proposals": proposals,
            "candidate_proposal_history": proposal_history,
            "tool_observations": observations,
            "evidence_events": events,
            "tool_observation_count": len(observations),
            "evidence_event_count": len(events),
            "candidate_proposal_history_count": len(proposal_history),
        }

    def _next_evidence_event_id(self) -> str:
        self._evidence_event_counter += 1
        return f"e{self._evidence_event_counter}"

    def _next_tool_observation_id(self) -> str:
        self._tool_observation_counter += 1
        return f"o{self._tool_observation_counter}"

    @staticmethod
    def _event_strength(value: Any, default: float = 0.5) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return default

    def _add_evidence_event(
        self,
        target: Optional[str],
        event_type: str,
        polarity: str,
        strength: float,
        provenance: str,
        *,
        round_index: Optional[int] = None,
        observation_id: Optional[str] = None,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._evidence_trace:
            return
        event: Dict[str, Any] = {
            "event_id": self._next_evidence_event_id(),
            "target": target,
            "event_type": event_type,
            "polarity": polarity,
            "strength": self._event_strength(strength),
            "provenance": provenance,
        }
        if round_index is not None:
            event["round"] = round_index
        if observation_id is not None:
            event["observation_id"] = observation_id
        if summary:
            event["summary"] = summary[:240]
        if metadata:
            event["metadata"] = metadata
        self._evidence_trace["evidence_events"].append(event)

    def _record_tool_observation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        round_index: Optional[int] = None,
    ) -> None:
        if not self._evidence_trace:
            return
        observation_id = self._next_tool_observation_id()
        observation: Dict[str, Any] = {
            "observation_id": observation_id,
            "tool": tool_name,
            "success": bool(result.get("success")) and not bool(result.get("error")),
            "error": result.get("error"),
        }
        if tool_name == "get_sample_data":
            # Correct the execution record without changing the tool payload
            # seen by the model. Frozen scorers require an explicit legacy
            # projection; keep both meanings rather than rewriting old traces.
            observation["observation_contract_version"] = 2
            observation["legacy_success_for_scoring"] = observation["success"]
            observation["success"] = (not bool(result.get("error"))
                                      and "sample_data" in result
                                      and "statistics" in result)
        if round_index is not None:
            observation["round"] = round_index

        def add_evidence_event(*args, **kwargs) -> None:
            kwargs["round_index"] = round_index
            kwargs["observation_id"] = observation_id
            self._add_evidence_event(*args, **kwargs)

        if tool_name == "shortlist_match":
            candidates = result.get("candidates") or []
            proposal_rows = []
            for idx, candidate in enumerate(candidates, 1):
                target = candidate.get("column")
                if not target:
                    continue
                row = {
                    "target": target,
                    "rank": idx,
                    "proposal_score": self._event_strength(candidate.get("score", 0.0), 0.0),
                    "schema_confidence": self._event_strength(candidate.get("schema_confidence", 0.0), 0.0),
                    "role_compatibility": self._event_strength(candidate.get("role_compatibility", 0.0), 0.0),
                    "local_context_compatibility": self._event_strength(
                        candidate.get("local_context_compatibility", 0.0),
                        0.0,
                    ),
                    "source": "shortlist_match",
                }
                proposal_rows.append(row)
                event_type = "shortlist_top_support" if idx == 1 else "shortlist_candidate_support"
                add_evidence_event(
                    target,
                    event_type,
                    "support",
                    row["proposal_score"],
                    "shortlist_match",
                    summary=f"shortlist rank {idx}",
                    metadata={"rank": idx},
                )
            self._evidence_trace["candidate_proposals"] = proposal_rows
            proposal_observation = {
                "observation_id": observation_id,
                "strategy": result.get("strategy"),
                "top_gap": result.get("top_gap"),
                "candidates": copy.deepcopy(proposal_rows),
            }
            if round_index is not None:
                proposal_observation["round"] = round_index
            self._evidence_trace["candidate_proposal_history"].append(
                proposal_observation
            )
            observation.update({
                "strategy": result.get("strategy"),
                "candidate_count": len(candidates),
                "requested_top_k": result.get("_requested_top_k"),
                "effective_top_k": result.get("_effective_top_k"),
                "top_k_overridden": bool(result.get("_top_k_overridden")),
                "top_gap": result.get("top_gap"),
                "must_verify_candidates": result.get("must_verify_candidates", []),
            })
            if result.get("recommended_next_step") == "differential_verify":
                add_evidence_event(
                    None,
                    "multiple_plausible_targets",
                    "conflict",
                    0.6,
                    "shortlist_match",
                    summary="shortlist top gap recommends differential verification",
                )
        elif tool_name == "doc_query":
            support_targets = list(result.get("current_target_hits") or [])[:5]
            seen = set(support_targets)
            # Record every returned target, while retaining existing event counts
            # for repeated definitions and adding each newly attributed field once.
            for entry in result.get("results") or []:
                target = entry.get("term")
                if (isinstance(target, str) and target in self._target_df.columns
                        and target not in seen):
                    seen.add(target)
                    support_targets.append(target)
            hits = list(dict.fromkeys(support_targets))
            absent = list(result.get("absent_from_current_target") or [])
            observation.update({
                "query": arguments.get("query", ""),
                "result_count": len(result.get("results") or []),
                "current_target_hits": hits,
                "absent_from_current_target": absent,
            })
            for target in support_targets:
                add_evidence_event(
                    target,
                    "doc_current_target_hit",
                    "support",
                    0.8,
                    "doc_query",
                    summary="documentation returned a field present in current target table",
                )
            if absent and not hits:
                add_evidence_event(
                    None,
                    "doc_absent_current_target",
                    "oppose",
                    0.65,
                    "doc_query",
                    summary="documentation returned only fields absent from current target table",
                    metadata={"absent_terms": absent[:5]},
                )
            if not result.get("results"):
                add_evidence_event(
                    None,
                    "doc_query_empty",
                    "missing",
                    0.5,
                    "doc_query",
                    summary="doc_query returned no useful entries",
                )
        elif tool_name == "differential_verify":
            preferred = result.get("preferred")
            candidates = [
                arguments.get("candidate_a"),
                arguments.get("candidate_b"),
            ]
            if not all(candidates) and isinstance(arguments.get("candidates"), list):
                candidates = list(arguments.get("candidates")[:2])
            # Evidence attribution is restricted to actual target attributes.
            # The trajectory retains the original arguments and tool failure.
            candidates = [
                candidate if isinstance(candidate, str)
                and candidate in self._target_df.columns else None
                for candidate in candidates
            ]
            observation.update({
                "candidate_a": candidates[0] if candidates else None,
                "candidate_b": candidates[1] if len(candidates) > 1 else None,
                "preferred": preferred,
                "confidence": result.get("confidence"),
                "margin": result.get("margin"),
                "conflict_type": result.get("conflict_type"),
                "deciding_signal": result.get("deciding_signal"),
                "needs_execute_python": result.get("needs_execute_python"),
            })
            if preferred:
                add_evidence_event(
                    preferred,
                    "differential_preferred",
                    "support",
                    result.get("confidence", 0.6),
                    "differential_verify",
                    summary=f"preferred by differential verification via {result.get('deciding_signal')}",
                    metadata={
                        "conflict_type": result.get("conflict_type"),
                        "deciding_signal": result.get("deciding_signal"),
                        "margin": result.get("margin"),
                    },
                )
                for target in candidates:
                    if target and target != preferred:
                        add_evidence_event(
                            target,
                            "differential_not_preferred",
                            "oppose",
                            result.get("confidence", 0.6),
                            "differential_verify",
                            summary="not preferred by differential verification",
                        )
            if result.get("needs_execute_python"):
                add_evidence_event(
                    None,
                    "candidate_not_compared",
                    "missing",
                    0.55,
                    "differential_verify",
                    summary="differential verification suggested execute_python before final commitment",
                )
        elif tool_name == "get_sample_data":
            profile = result.get("profile") or {}
            observation.update({
                "table_type": result.get("table_type"),
                "column": result.get("column"),
                "value_pattern": profile.get("value_pattern"),
                "fill_rate": profile.get("fill_rate"),
                "unique_count": (result.get("statistics") or {}).get("unique_count"),
            })
            target = result.get("column") if result.get("table_type") == "target" else None
            add_evidence_event(
                target,
                "value_profile_observed",
                "neutral",
                0.5,
                "get_sample_data",
                summary=f"value pattern={profile.get('value_pattern')}, fill_rate={profile.get('fill_rate')}",
            )
        elif tool_name == "execute_python":
            observation.update({
                "output_compressed": bool(result.get("output_compressed")),
                "has_error": bool(result.get("error")),
                "budget": result.get("execute_python_budget"),
            })
            add_evidence_event(
                None,
                "execute_python_observed",
                "neutral" if result.get("success") else "missing",
                0.5,
                "execute_python",
                summary="execute_python completed" if result.get("success") else "execute_python failed or was blocked",
            )
        elif tool_name == "submit":
            ranked = coerce_candidates(arguments.get("ranked_candidates", []))
            observation.update({
                "ranked_candidates": ranked,
                "submitted_candidates": result.get("submitted_candidates", []),
                "top_candidate": result.get("top_candidate"),
                "recommended_next_step": result.get("recommended_next_step"),
                "submit_trace": result.get("submit_trace", []),
            })
            if result.get("success"):
                if result.get("top_candidate") is None:
                    add_evidence_event(
                        None,
                        "explicit_none_submission",
                        "support",
                        1.0,
                        "submit",
                        summary="agent committed to no-match",
                    )
                for idx, target in enumerate(result.get("submitted_candidates") or [], 1):
                    add_evidence_event(
                        target,
                        "submitted_candidate",
                        "support",
                        max(0.0, 1.0 - 0.05 * (idx - 1)),
                        "submit",
                        summary="candidate submitted as final answer",
                        metadata={"rank": idx},
                    )
            else:
                for target in ranked[:5]:
                    if isinstance(target, str) and target.lower() != "none":
                        add_evidence_event(
                            target,
                            "submit_blocked",
                            "conflict",
                            0.7,
                            "submit",
                            summary=str(result.get("error") or result.get("message") or "submit blocked")[:180],
                            metadata={"recommended_next_step": result.get("recommended_next_step")},
                        )
        self._evidence_trace["tool_observations"].append(observation)


    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _tool_registry(self):
        return {
            "shortlist_match": lambda arguments: self._tool_shortlist_match(arguments),
            "get_sample_data": lambda arguments: self._tool_get_sample_data(
                arguments.get("table_type", "source"),
                arguments.get("column", ""),
            ),
            "differential_verify": lambda arguments: self._tool_differential_verify(arguments),
            "execute_python": lambda arguments: self._tool_execute_python(arguments.get("code", "")),
            "doc_query": lambda arguments: self._tool_doc_query(arguments.get("query", "")),
            "submit": lambda arguments: self._tool_submit(
                submit_candidates_from_arguments(arguments),
                arguments.get("reasoning", ""),
                candidate_scores=arguments.get("candidate_scores"),
            ),
        }

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        round_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a tool call and return the result with timing metadata."""
        t0 = time.time()

        handler = self._tool_registry().get(tool_name)
        if handler is not None:
            result = handler(arguments)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        elapsed = time.time() - t0
        result["_tool_name"] = tool_name
        result["_cost"] = TOOL_COSTS.get(tool_name, "unknown")
        result["_elapsed_seconds"] = round(elapsed, 3)
        self._record_tool_observation(
            tool_name,
            arguments,
            result,
            round_index=round_index,
        )
        return result

    # ------------------------------------------------------------------
    # Retrieval support for shortlist_match
    # ------------------------------------------------------------------

    def _compute_target_embeddings(self):
        """Pre-compute embeddings for all target columns (cached per task).

        If a supplied ``doc_index`` is available, each
        target column text is enriched with its definition/description so that
        embedding similarity can bridge vocabulary gaps.
        """
        if self._target_col_embeddings is not None:
            return

        cols = list(self._target_df.columns)
        texts = []
        for c in cols:
            enrichment = self._get_column_enrichment(c)
            texts.append(_build_column_text(c, self._target_df[c], enrichment=enrichment))
        embeddings = _embedding_model.encode(texts, is_query=False)

        self._target_col_names = cols
        self._target_col_texts = texts
        self._target_col_embeddings = embeddings

    # ------------------------------------------------------------------
    # Retrieval pre-filtering
    # ------------------------------------------------------------------

    def _detect_table_prefixed_columns(self) -> Optional[Dict[str, List[int]]]:
        """If target columns use TABLE.column format, group them by table.

        Returns ``{table_name: [col_index, ...]}`` or *None* when no table
        prefix is detected.
        """
        return detect_table_prefixed_columns(self._target_col_names)

    def _table_prefilter(self, src_emb: np.ndarray, top_j: int = 3) -> Optional[List[int]]:
        """Return column indices belonging to the top-J most relevant tables.

        Uses average embedding of each table's columns as the table
        representation, then selects the top-J tables closest to *src_emb*.
        """
        return table_prefilter(
            src_emb,
            self._target_col_embeddings,
            self._target_col_names,
            top_j=top_j,
        )

    # ------------------------------------------------------------------
    # Column enrichment and profile cache
    # ------------------------------------------------------------------

    def _get_column_enrichment(self, col_name: str) -> Optional[str]:
        """Read a supplied dictionary definition using exact or bare field names."""
        bare = col_name.split(".")[-1]
        for entries in (self.doc_index or {}).values():
            for name in (col_name.lower(), bare.lower().replace("_", " "), bare.lower()):
                for entry in entries:
                    if entry["term"].lower() == name:
                        definition = entry.get("definition", entry.get("text", ""))
                        if definition:
                            return str(definition)[:220]
        return None

    def _build_column_profile(self, col_name: str, series: pd.Series) -> Dict[str, Any]:
        return build_column_profile(col_name, series)

    def _build_schema_card(self, table_type: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        local_context = self._get_local_context(table_type, profile["column"])
        enrichment = self._get_column_enrichment(profile["column"])
        return build_schema_card(
            table_type,
            profile,
            local_context,
            enrichment,
            identifier_like=profile_is_identifier_like(profile),
            name_like=profile_is_name_like(profile),
        )

    def _get_base_profile(self, table_type: str, col_name: str) -> Dict[str, Any]:
        cache = self._profile_cache[table_type]
        if col_name in cache:
            return cache[col_name]
        df = self._source_df if table_type == "source" else self._target_df
        profile = self._build_column_profile(col_name, df[col_name])
        cache[col_name] = profile
        return profile

    def _get_profile(self, table_type: str, col_name: str) -> Dict[str, Any]:
        profile = self._get_base_profile(table_type, col_name)
        if "schema_card" not in profile:
            profile["schema_card"] = self._build_schema_card(table_type, profile)
        if "structural_subtype" not in profile:
            profile["structural_subtype"] = None
        return profile

    def _get_local_context(self, table_type: str, col_name: str) -> Dict[str, Any]:
        cache = self._local_context_cache[table_type]
        if col_name in cache:
            return cache[col_name]

        df = self._source_df if table_type == "source" else self._target_df
        profile = self._get_base_profile(table_type, col_name)
        # Seed the cache before scanning sibling profiles to avoid deep
        # schema_card/local_context recursion on very wide tables.
        context = {
            "group_signature": profile["group_signature"],
            "sibling_columns": [],
            "paired_columns": [],
            "sibling_roles": [],
            "paired_roles": [],
        }
        cache[col_name] = context

        siblings: List[str] = []
        paired_columns: List[str] = []
        sibling_roles = set()
        paired_roles = set()
        signature = profile["group_signature"]

        for other in df.columns:
            if other == col_name:
                continue
            other_profile = self._get_base_profile(table_type, other)
            same_group = _signature_similarity(signature, other_profile["group_signature"]) >= 0.72
            lexical_overlap = bool(set(profile["tokens"]) & set(other_profile["tokens"]))
            if same_group or lexical_overlap:
                siblings.append(other)
                sibling_roles.add(other_profile["schema_role"])
            if same_group and _roles_are_complementary(profile["schema_role"], other_profile["schema_role"]):
                paired_columns.append(other)
                paired_roles.add(other_profile["schema_role"])

        context = {
            "group_signature": signature,
            "sibling_columns": siblings[:5],
            "paired_columns": paired_columns[:3],
            "sibling_roles": sorted(role for role in sibling_roles if role != "unknown"),
            "paired_roles": sorted(role for role in paired_roles if role != "unknown"),
        }
        cache[col_name] = context
        return context

    # ------------------------------------------------------------------
    # Structural context protocol
    # ------------------------------------------------------------------


    def _structural_alignment_signal(self, source_profile, target_profile) -> Dict[str, Any]:
        # Retain the neutral feature contract for the frozen scoring models.
        return {"bonus": 0.0, "penalty": 0.0, "exact_match": 0.0,
                "source_subtype": None, "target_subtype": None, "evidence": []}

    # ------------------------------------------------------------------
    # Pairwise compatibility scoring protocol
    # ------------------------------------------------------------------

    def _pairwise_match_details(
        self,
        source_profile: Dict[str, Any],
        target_profile: Dict[str, Any],
        base_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        source_context = self._get_local_context("source", source_profile["column"])
        target_context = self._get_local_context("target", target_profile["column"])
        return pairwise_match_details(
            source_profile,
            target_profile,
            source_context,
            target_context,
            self._structural_alignment_signal,
            base_score,
        )

    def _build_interpretation_summary(self, profile: Dict[str, Any], table_type: str) -> Dict[str, Any]:
        return build_interpretation_summary(profile, table_type, self._get_local_context)

    # ------------------------------------------------------------------
    # Tool adapters
    #
    # These methods keep the tool dispatch API stable while delegating the
    # implementation to asm.tools and asm.submission.
    # ------------------------------------------------------------------

    # shortlist_match  [LOW cost]
    # ------------------------------------------------------------------

    def _tool_shortlist_match(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        requested_top_k = arguments.get("top_k", 10)
        effective_arguments = dict(arguments)
        if self._shortlist_top_k_override is not None:
            effective_arguments["top_k"] = self._shortlist_top_k_override
        result = shortlist_match(self, effective_arguments, _embedding_model)
        result["_requested_top_k"] = requested_top_k
        result["_effective_top_k"] = effective_arguments.get("top_k", 10)
        result["_top_k_overridden"] = self._shortlist_top_k_override is not None
        if result.get("success"):
            self._verification_since_shortlist = False
        return result

    # ------------------------------------------------------------------
    # get_sample_data  [FREE]
    # ------------------------------------------------------------------

    def _tool_get_sample_data(self, table_type: str, column: str) -> Dict[str, Any]:
        result = get_sample_data(
            self._source_df,
            self._target_df,
            table_type,
            column,
            self.sample_size,
            self._get_profile,
        )
        if not result.get("error"):
            self._verification_since_shortlist = True
        return result

    # ------------------------------------------------------------------
    # differential_verify  [LOW cost]
    # ------------------------------------------------------------------

    def _tool_differential_verify(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = differential_verify(self, arguments)
        if result.get("success"):
            self._verification_since_shortlist = True
        return result

    @staticmethod
    def _select_must_verify_candidates(candidates: List[Dict[str, Any]]) -> List[str]:
        must_verify: List[str] = []
        for idx, candidate in enumerate(candidates[:5]):
            abbreviation = float(candidate.get("abbreviation_match", 0.0))
            expansion = float(candidate.get("expanded_name_match", 0.0))
            role = float(candidate.get("role_compatibility", 0.0))
            local_context = float(candidate.get("local_context_compatibility", 0.0))
            schema_confidence = float(candidate.get("schema_confidence", 0.0))
            if abbreviation >= 0.88 and expansion >= 0.72 and role >= 0.55 and schema_confidence >= 0.58:
                must_verify.append(candidate["column"])
            elif idx < 3 and abbreviation >= 0.84 and schema_confidence >= 0.70 and max(role, local_context) >= 0.55:
                must_verify.append(candidate["column"])
            elif idx == 0 and abbreviation >= 0.92 and schema_confidence >= 0.62:
                must_verify.append(candidate["column"])
        return must_verify[:2]

    # ------------------------------------------------------------------
    # execute_python  [HIGH cost]
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_tool_budget(
        result: Dict[str, Any],
        tool_name: str,
        used: int,
        max_calls: Optional[int],
    ) -> None:
        result[f"{tool_name}_budget"] = {
            "used": used,
            "max": max_calls,
            "remaining": None if max_calls is None else max(0, max_calls - used),
        }

    @staticmethod
    def _compress_execute_python_output(
        result: Dict[str, Any],
        max_lines: int = 40,
        max_chars: int = 2500,
    ) -> None:
        output = result.get("output")
        if not isinstance(output, str):
            return
        if len(output) <= max_chars and output.count("\n") < max_lines:
            return
        lines = output.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        compressed = "\n".join(lines)
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars]
        result["output"] = (
            compressed.rstrip()
            + "\n... [output compressed; print fewer columns/rows if more detail is needed]"
        )
        result["output_compressed"] = True

    def _tool_execute_python(self, code: str) -> Dict[str, Any]:
        result, self._execute_python_calls_used = execute_python(
            self.executor,
            code,
            self._source_df,
            self._target_df,
            self._source_column,
            self._execute_python_calls_used,
            self._max_execute_python_calls,
        )
        self._annotate_tool_budget(
            result,
            "execute_python",
            self._execute_python_calls_used,
            self._max_execute_python_calls,
        )
        self._compress_execute_python_output(result)
        if result.get("success"):
            self._verification_since_shortlist = True
        return result

    # ------------------------------------------------------------------
    # doc_query  [MEDIUM cost]
    # ------------------------------------------------------------------

    def _annotate_doc_query_results(self, result: Dict[str, Any]) -> None:
        if self._target_df is None:
            return
        targets = set(self._target_df.columns)
        present, absent = [], []
        for entry in result.get("results") or []:
            term = entry.get("term")
            if not isinstance(term, str):
                continue
            entry["present_in_current_target"] = term in targets
            (present if term in targets else absent).append(term)
        result["current_target_hits"] = present[:5]
        if absent:
            result["absent_from_current_target"] = absent[:5]
        result["quality_hint"] = (
            "Dictionary definitions are evidence, not candidate labels. "
            "Check the source meaning and submit only fields in the current target table."
        )
        self._last_doc_current_target_hits = present[:5]


    def _tool_doc_query(self, query: str) -> Dict[str, Any]:
        if (
            self._max_doc_query_calls is not None
            and self._doc_query_calls_used >= self._max_doc_query_calls
        ):
            return {
                "success": False,
                "error": (
                    f"doc_query budget exhausted for this task "
                    f"({self._doc_query_calls_used}/{self._max_doc_query_calls}). "
                    "Use shortlist_match, differential_verify, get_sample_data, execute_python, or submit."
                ),
                "recommended_next_step": "current_target_search_or_submit",
                "doc_query_budget": {
                    "used": self._doc_query_calls_used,
                    "max": self._max_doc_query_calls,
                    "remaining": 0,
                },
            }
        if self._doc_embeddings is None:
            self._doc_embeddings = {}
        result = doc_query(
            query,
            self.doc_index,
            self._doc_embeddings,
            _embedding_model,
        )
        result["query"] = query
        if result.get("success"):
            self._doc_query_calls_used += 1
        self._annotate_doc_query_results(result)
        self._annotate_tool_budget(
            result,
            "doc_query",
            self._doc_query_calls_used,
            self._max_doc_query_calls,
        )
        if not result.get("error"):
            self._verification_since_shortlist = True
        return result

    # ------------------------------------------------------------------
    # submit  [FREE]
    # ------------------------------------------------------------------

    @staticmethod
    def _build_submit_trace_entry(
        stage: str,
        candidates: List[str],
        score_map: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "stage": stage,
            "top_candidate": candidates[0] if candidates else None,
            "candidates": list(candidates),
        }
        if score_map:
            entry["score_map"] = {
                col: float(score_map[col])
                for col in candidates
                if col in score_map
            }
        if extra:
            entry["extra"] = extra
        return entry


    def _tool_submit(self, ranked_candidates, reasoning: str, candidate_scores=None) -> Dict[str, Any]:
        first_attempt = not self._evidence_trace.get("pre_submit_snapshot")
        raw_ranked_candidates = copy.deepcopy(ranked_candidates)
        ranked_candidates = coerce_candidates(ranked_candidates)
        self._capture_pre_submit_snapshot(
            raw_ranked_candidates,
            ranked_candidates,
            reasoning,
        )
        # Keep a schema-normalized proposal even when a later admission gate
        # rejects it. It remains available if the episode exhausts its budget.
        proposal, _ = normalize_submit_candidates(ranked_candidates, self._target_df.columns, self.topk)
        explicit_none = bool(ranked_candidates) and all(
            isinstance(c, str) and c.strip().lower() == "none" for c in ranked_candidates
        )
        if proposal or explicit_none:
            self._current_proposal = proposal
            self._current_proposal_scores = {}
            for i, c in enumerate(proposal):
                value = candidate_scores.get(c) if isinstance(candidate_scores, dict) else None
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                    value = max(0.0, 1.0 - i * 0.05)
                self._current_proposal_scores[c] = float(value)
        model_scores = None
        if self.match_score_mode == "llm":
            if first_attempt:
                self._evidence_trace["pre_submit_snapshot"].update({
                    "match_score_mode": "llm",
                    "raw_candidate_scores": copy.deepcopy(candidate_scores),
                })
            try:
                model_scores = validate_match_scores(
                    ranked_candidates, candidate_scores, self._target_df.columns,
                )
            except ValueError as exc:
                return {"success": False, "error": str(exc),
                        "error_code": "invalid_candidate_scores",
                        "recommended_next_step": "correct_submit_scores"}
        submit_trace: List[Dict[str, Any]] = []

        if not ranked_candidates:
            return {
                "success": False,
                "error": "ranked_candidates cannot be empty. Submit at least one candidate or ['None'] for no match.",
            }

        explicit_none = all(
            isinstance(c, str) and c.strip().lower() == "none" for c in ranked_candidates
        )
        if explicit_none:
            self._submitted_candidates = []
            self._submitted_reasoning = reasoning
            self._submitted_match = None
            self._submitted_confidence = 0.0
            self._submitted_score_map = {}
            self._submitted_trace = [
                self._build_submit_trace_entry("explicit_none", [])
            ]
            return {
                "success": True,
                "message": "No match submitted (explicit None).",
                "submitted_candidates": [],
                "top_candidate": None,
            }

        valid_candidates, invalid_names = normalize_submit_candidates(
            ranked_candidates,
            self._target_df.columns,
            self.topk,
        )
        submit_trace.append(
            self._build_submit_trace_entry(
                "normalized",
                valid_candidates,
                extra={"invalid_names": invalid_names[:3]},
            )
        )
        if invalid_names and not valid_candidates:
            return {
                "success": False,
                "error": (
                    f"Submitted column(s) {invalid_names[:3]} are not present in the current target table. "
                    "Submit only columns from the target table, or submit ['None'] if no current target-table "
                    "column is a valid semantic match."
                ),
                "invalid_names": invalid_names[:3],
                "recommended_next_step": "submit_current_target_column_or_none",
                "submit_trace": submit_trace,
            }
        score_map = {
            col: (model_scores[col] if model_scores is not None
                  else round(max(0.0, 1.0 - (idx * 0.05)), 4))
            for idx, col in enumerate(valid_candidates)
        }
        if model_scores is not None:
            submit_trace.append(self._build_submit_trace_entry(
                "model_scores_validated", valid_candidates,
                extra={"candidate_scores": dict(score_map), "match_score_mode": "llm"},
            ))

        self._submitted_candidates = valid_candidates
        self._submitted_reasoning = reasoning
        self._submitted_match = valid_candidates[0] if valid_candidates else None
        self._submitted_confidence = score_map.get(valid_candidates[0], 0.0) if valid_candidates else 0.0
        self._submitted_score_map = dict(score_map)
        self._submitted_trace = submit_trace

        result: Dict[str, Any] = {
            "success": True,
            "message": f"Submitted {len(valid_candidates)} candidate(s)",
            "submitted_candidates": valid_candidates,
            "top_candidate": valid_candidates[0] if valid_candidates else None,
            "top_score": score_map.get(valid_candidates[0], 0.0) if valid_candidates else 0.0,
        }
        if model_scores is not None:
            result.update(match_score_mode="llm", candidate_scores=dict(score_map))
        if submit_trace:
            result["submit_trace"] = submit_trace
        if invalid_names and valid_candidates:
            result["warning"] = f"Some names auto-corrected: {invalid_names[:3]}"
        return result

    # ------------------------------------------------------------------
    # Submission result and evaluation
    # ------------------------------------------------------------------

    def has_submitted(self) -> bool:
        return len(self._submitted_candidates) > 0 or self._submitted_match is not None

    def _build_candidate_evidence_summary(self) -> List[Dict[str, Any]]:
        events = self._evidence_trace.get("evidence_events", []) if self._evidence_trace else []
        targets = {
            event.get("target")
            for event in events
            if event.get("target")
        }
        targets.update(self._submitted_candidates)
        proposal_map = {
            row.get("target"): row
            for row in self._evidence_trace.get("candidate_proposals", [])
            if row.get("target")
        } if self._evidence_trace else {}

        summaries: List[Dict[str, Any]] = []
        for target in sorted(targets):
            target_events = [event for event in events if event.get("target") == target]
            by_polarity = {
                polarity: [event for event in target_events if event.get("polarity") == polarity]
                for polarity in ("support", "oppose", "conflict", "missing", "neutral")
            }
            proposal = proposal_map.get(target, {})
            summaries.append({
                "target": target,
                "proposal_rank": proposal.get("rank"),
                "proposal_score": proposal.get("proposal_score"),
                "support_count": len(by_polarity["support"]),
                "oppose_count": len(by_polarity["oppose"]),
                "conflict_count": len(by_polarity["conflict"]),
                "missing_evidence_count": len(by_polarity["missing"]),
                "neutral_count": len(by_polarity["neutral"]),
                "tool_support": sorted({
                    event.get("provenance")
                    for event in by_polarity["support"]
                    if event.get("provenance")
                }),
                "tool_oppose": sorted({
                    event.get("provenance")
                    for event in by_polarity["oppose"]
                    if event.get("provenance")
                }),
                "unresolved_conflicts": [
                    event.get("event_type")
                    for event in by_polarity["conflict"]
                    if event.get("event_type")
                ],
                "decision": (
                    "submit"
                    if target in self._submitted_candidates
                    else "not_submitted"
                ),
            })
        summaries.sort(
            key=lambda row: (
                row["decision"] != "submit",
                row.get("proposal_rank") if row.get("proposal_rank") is not None else 999,
                row["target"],
            )
        )
        return summaries

    def _finalize_evidence_trace(self, predicted_column: Optional[str]) -> Dict[str, Any]:
        if not self._evidence_trace:
            return {}
        trace = {
            **self._evidence_trace,
            "task_name": self._task_name,
            "source_column": self._source_column,
        }
        trace["candidate_evidence"] = self._build_candidate_evidence_summary()
        decision_type = (
            "submit_none"
            if not self._submitted_candidates
            else ("submit_one" if len(self._submitted_candidates) == 1 else "submit_multiple")
        )
        trace["final_decision"] = {
            "decision_type": decision_type,
            "submitted_candidates": list(self._submitted_candidates),
            "top_candidate": predicted_column,
            "explicit_none": not self._submitted_candidates and any(
                entry.get("stage") == "explicit_none"
                for entry in self._submitted_trace
            ),
            "submit_trace": list(self._submitted_trace),
        }
        if self.match_score_mode == "llm":
            trace["final_decision"].update({
                "match_score_mode": "llm",
                "candidate_scores": dict(self._submitted_score_map),
                "score_semantics": "evidence_support_not_calibrated_probability",
            })
        return trace

    def finalize_budget(self, result: MatchResult) -> MatchResult:
        """Return the last proposal or no-match without inventing a submission."""
        names = self._current_proposal or []
        result.predicted_candidates = [(c, self._current_proposal_scores[c]) for c in names]
        result.predicted_column = names[0] if names else None
        result.confidence = result.predicted_candidates[0][1] if names else 0.0
        result.error = None
        result.termination_reason = "budget_exhausted_proposal" if names else "budget_exhausted_no_match"
        result.evidence_trace["budget_fallback"] = {"reason": result.termination_reason,
                                                    "semantic_no_match": False}
        result.evidence_trace["final_decision"].update({
            "decision_type": result.termination_reason,
            "returned_candidates": list(names),
            "top_candidate": result.predicted_column,
            "candidate_scores": dict(result.predicted_candidates),
            "explicit_none": False,
        })
        return result

    def evaluate(self) -> MatchResult:
        candidates_with_scores: List[Tuple[str, float]] = []
        n = len(self._submitted_candidates)
        for i, col in enumerate(self._submitted_candidates):
            score = self._submitted_score_map.get(col)
            if score is None:
                score = (n - i) / n if n > 0 else 0.0
            candidates_with_scores.append((col, score))

        predicted_column = (
            self._submitted_candidates[0]
            if self._submitted_candidates
            else self._submitted_match
        )

        result = MatchResult(
            task_name=self._task_name,
            source_column=self._source_column,
            predicted_column=predicted_column,
            predicted_candidates=candidates_with_scores,
            confidence=self._submitted_confidence,
            reasoning=self._submitted_reasoning,
            evidence_trace=self._finalize_evidence_trace(predicted_column),
        )

        if predicted_column is None:
            explicit_none_submitted = any(
                entry.get("stage") == "explicit_none"
                for entry in self._submitted_trace
            )
            if not explicit_none_submitted:
                result.error = "No answer submitted"

        return result

    def reset(self):
        self._current_proposal = None
        self._current_proposal_scores = {}
        self._source_df = None
        self._target_df = None
        self._source_column = None
        self._task_name = None
        self._submitted_candidates = []
        self._submitted_reasoning = None
        self._submitted_match = None
        self._submitted_confidence = 0.0
        self._submitted_score_map = {}
        self._submitted_trace = []
        self._evidence_trace = self._new_evidence_trace()
        self._evidence_event_counter = 0
        self._target_col_embeddings = None
        self._target_col_names = None
        self._target_col_texts = None
        self._profile_cache = {"source": {}, "target": {}}
        self._local_context_cache = {"source": {}, "target": {}}

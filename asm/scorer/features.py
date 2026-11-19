"""Evidence and first-submission features for the final scorer."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import copy
import math
import statistics

import numpy as np


FEATURE_SCHEMA_VERSION = "section6_candidate_features_v1"
CAPTURE_STAGE = "before_first_submit_attempt"
FORBIDDEN_KEYS = {
    "correct",
    "evaluation",
    "final_decision",
    "ground_truth",
    "is_gt",
    "label",
    "metrics",
}
POLARITIES = ("support", "oppose", "conflict", "missing", "neutral")
PROVENANCE_GROUPS = (
    "shortlist_match",
    "differential_verify",
    "doc_query",
    "get_sample_data",
    "execute_python",
    "other",
)
TOOL_GROUPS = (
    "shortlist_match",
    "differential_verify",
    "doc_query",
    "get_sample_data",
    "execute_python",
)


def _feature_columns() -> Tuple[str, ...]:
    columns = [
        "pool_size",
        "pool_rank",
        "pool_rank_reciprocal",
        "core_proposal_count",
        "core_proposed",
        "core_rank",
        "core_rank_reciprocal",
        "explicit_none",
        "proposal_present",
        "proposal_rank",
        "proposal_rank_reciprocal",
        "proposal_score",
        "schema_confidence",
        "role_compatibility",
        "local_context_compatibility",
        "top_proposal_score",
        "second_proposal_score",
        "proposal_top_gap",
        "candidate_score_margin_to_top",
        "candidate_event_count",
        "candidate_unique_observations",
        "candidate_unique_rounds",
        "candidate_first_round",
        "candidate_last_round",
        "candidate_round_span",
        "candidate_round_missing",
        "history_appearance_count",
        "history_first_rank",
        "history_last_rank",
        "history_best_rank",
        "history_worst_rank",
        "history_rank_change",
        "history_score_mean",
        "history_score_last",
        "history_score_std",
        "history_missing",
        "tool_observation_count",
        "tool_success_count",
        "tool_error_count",
        "unique_tool_count",
        "shared_event_count",
    ]
    for polarity in POLARITIES:
        columns.extend((
            f"candidate_{polarity}_count",
            f"candidate_{polarity}_strength_sum",
            f"candidate_{polarity}_strength_max",
            f"shared_{polarity}_count",
            f"shared_{polarity}_strength_sum",
        ))
    columns.extend(f"candidate_{group}_event_count" for group in PROVENANCE_GROUPS)
    columns.extend(f"source_{tool}_call_count" for tool in TOOL_GROUPS)
    return tuple(columns)


FEATURE_COLUMNS = _feature_columns()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_positive_int(value: Any) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _deduplicate_candidates(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate or candidate.lower() == "none" or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _forbidden_paths(value: Any, path: str = "snapshot") -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}"
            # Only this map is keyed by user-defined target names, not metadata.
            # Still inspect its values recursively; nested label fields are invalid.
            if path != "snapshot.raw_candidate_scores" and str(key).strip().lower() in FORBIDDEN_KEYS:
                paths.append(child)
            paths.extend(_forbidden_paths(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_forbidden_paths(nested, f"{path}[{index}]"))
    return paths


def validate_pre_submit_snapshot(snapshot: Dict[str, Any]) -> None:
    """Fail closed on leakage or post-submit evidence."""
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("pre_submit_snapshot is missing")
    if snapshot.get("capture_stage") != CAPTURE_STAGE:
        raise ValueError("pre_submit_snapshot has an invalid capture stage")
    forbidden = _forbidden_paths(snapshot)
    if forbidden:
        raise ValueError(f"pre_submit_snapshot contains forbidden fields: {forbidden[:3]}")
    for observation in snapshot.get("tool_observations") or []:
        if isinstance(observation, dict) and observation.get("tool") == "submit":
            raise ValueError("pre_submit_snapshot contains a submit observation")
    for event in snapshot.get("evidence_events") or []:
        if not isinstance(event, dict):
            continue
        if event.get("provenance") == "submit" or str(
            event.get("event_type", "")
        ).startswith("submitted_"):
            raise ValueError("pre_submit_snapshot contains a post-submit event")


def _candidate_universe(snapshot: Dict[str, Any]) -> List[str]:
    values: List[Any] = list(snapshot.get("candidate_pool") or [])
    values.extend(snapshot.get("ranked_candidates") or [])
    values.extend(
        row.get("target")
        for row in snapshot.get("candidate_proposals") or []
        if isinstance(row, dict)
    )
    for history_row in snapshot.get("candidate_proposal_history") or []:
        if not isinstance(history_row, dict):
            continue
        values.extend(
            row.get("target")
            for row in history_row.get("candidates") or []
            if isinstance(row, dict)
        )
    values.extend(
        event.get("target")
        for event in snapshot.get("evidence_events") or []
        if isinstance(event, dict)
    )
    return _deduplicate_candidates(values)


def _polarity_features(
    events: Sequence[Dict[str, Any]],
    prefix: str,
    *,
    include_max: bool,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for polarity in POLARITIES:
        selected = [event for event in events if event.get("polarity") == polarity]
        strengths = [_as_float(event.get("strength")) for event in selected]
        features[f"{prefix}_{polarity}_count"] = len(selected)
        features[f"{prefix}_{polarity}_strength_sum"] = sum(strengths)
        if include_max:
            features[f"{prefix}_{polarity}_strength_max"] = max(strengths, default=0.0)
    return features


def extract_candidate_features(
    snapshot: Dict[str, Any],
    *,
    trace_schema_version: int = 0,
) -> List[Dict[str, Any]]:
    """Return candidate identity separately from model-ready numeric features."""
    validate_pre_submit_snapshot(snapshot)
    candidates = _candidate_universe(snapshot)
    pool_rank = {candidate: index for index, candidate in enumerate(candidates, 1)}
    core_candidates = _deduplicate_candidates(
        (snapshot.get("submit_normalization") or {}).get(
            "candidates", snapshot.get("ranked_candidates") or []))
    core_rank = {candidate: index for index, candidate in enumerate(core_candidates, 1)}

    proposals = [
        row for row in snapshot.get("candidate_proposals") or []
        if isinstance(row, dict) and isinstance(row.get("target"), str)
    ]
    proposal_map = {row["target"]: row for row in proposals}
    ordered_proposal_scores = sorted(
        (_as_float(row.get("proposal_score")) for row in proposals),
        reverse=True,
    )
    top_score = ordered_proposal_scores[0] if ordered_proposal_scores else 0.0
    second_score = ordered_proposal_scores[1] if len(ordered_proposal_scores) > 1 else 0.0

    events = [
        event for event in snapshot.get("evidence_events") or []
        if isinstance(event, dict)
    ]
    shared_events = [event for event in events if not event.get("target")]
    observations = [
        row for row in snapshot.get("tool_observations") or []
        if isinstance(row, dict)
    ]
    tool_counts = Counter(str(row.get("tool") or "") for row in observations)
    shared_features = _polarity_features(
        shared_events,
        "shared",
        include_max=False,
    )

    history_by_candidate: Dict[str, List[Dict[str, Any]]] = {
        candidate: [] for candidate in candidates
    }
    for history_row in snapshot.get("candidate_proposal_history") or []:
        if not isinstance(history_row, dict):
            continue
        round_index = _as_positive_int(history_row.get("round"))
        for row in history_row.get("candidates") or []:
            if not isinstance(row, dict) or row.get("target") not in history_by_candidate:
                continue
            history_by_candidate[row["target"]].append({
                "round": round_index,
                "rank": _as_positive_int(row.get("rank")),
                "score": _as_float(row.get("proposal_score")),
            })

    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        proposal = proposal_map.get(candidate, {})
        proposal_rank = _as_positive_int(proposal.get("rank"))
        candidate_events = [event for event in events if event.get("target") == candidate]
        event_rounds = [
            _as_positive_int(event.get("round")) for event in candidate_events
        ]
        event_rounds = [round_index for round_index in event_rounds if round_index]
        observation_ids = {
            event.get("observation_id")
            for event in candidate_events
            if isinstance(event.get("observation_id"), str)
        }
        provenance_counts = Counter(
            event.get("provenance")
            if event.get("provenance") in PROVENANCE_GROUPS[:-1]
            else "other"
            for event in candidate_events
        )
        history = history_by_candidate[candidate]
        history_ranks = [row["rank"] for row in history if row["rank"]]
        history_scores = [row["score"] for row in history]
        first_round = min(event_rounds) if event_rounds else 0
        last_round = max(event_rounds) if event_rounds else 0
        candidate_pool_rank = pool_rank[candidate]
        candidate_core_rank = core_rank.get(candidate, 0)
        features: Dict[str, float] = {
            "pool_size": len(candidates),
            "pool_rank": candidate_pool_rank,
            "pool_rank_reciprocal": 1.0 / candidate_pool_rank,
            "core_proposal_count": len(core_candidates),
            "core_proposed": int(candidate in core_rank),
            "core_rank": candidate_core_rank,
            "core_rank_reciprocal": 1.0 / candidate_core_rank if candidate_core_rank else 0.0,
            "explicit_none": int(bool(snapshot.get("explicit_none"))),
            "proposal_present": int(bool(proposal)),
            "proposal_rank": proposal_rank,
            "proposal_rank_reciprocal": 1.0 / proposal_rank if proposal_rank else 0.0,
            "proposal_score": _as_float(proposal.get("proposal_score")),
            "schema_confidence": _as_float(proposal.get("schema_confidence")),
            "role_compatibility": _as_float(proposal.get("role_compatibility")),
            "local_context_compatibility": _as_float(
                proposal.get("local_context_compatibility")
            ),
            "top_proposal_score": top_score,
            "second_proposal_score": second_score,
            "proposal_top_gap": top_score - second_score,
            "candidate_score_margin_to_top": (
                _as_float(proposal.get("proposal_score")) - top_score
                if proposal
                else -top_score
            ),
            "candidate_event_count": len(candidate_events),
            "candidate_unique_observations": len(observation_ids),
            "candidate_unique_rounds": len(set(event_rounds)),
            "candidate_first_round": first_round,
            "candidate_last_round": last_round,
            "candidate_round_span": last_round - first_round if event_rounds else 0,
            "candidate_round_missing": int(bool(candidate_events) and not event_rounds),
            "history_appearance_count": len(history),
            "history_first_rank": history_ranks[0] if history_ranks else 0,
            "history_last_rank": history_ranks[-1] if history_ranks else 0,
            "history_best_rank": min(history_ranks) if history_ranks else 0,
            "history_worst_rank": max(history_ranks) if history_ranks else 0,
            "history_rank_change": (
                history_ranks[-1] - history_ranks[0] if history_ranks else 0
            ),
            "history_score_mean": statistics.fmean(history_scores) if history_scores else 0.0,
            "history_score_last": history_scores[-1] if history_scores else 0.0,
            "history_score_std": (
                statistics.pstdev(history_scores) if len(history_scores) > 1 else 0.0
            ),
            "history_missing": int(not history),
            "tool_observation_count": len(observations),
            "tool_success_count": sum(bool(row.get("success")) for row in observations),
            "tool_error_count": sum(bool(row.get("error")) for row in observations),
            "unique_tool_count": len({row.get("tool") for row in observations}),
            "shared_event_count": len(shared_events),
            **shared_features,
            **_polarity_features(candidate_events, "candidate", include_max=True),
        }
        features.update({
            f"candidate_{group}_event_count": provenance_counts[group]
            for group in PROVENANCE_GROUPS
        })
        features.update({
            f"source_{tool}_call_count": tool_counts[tool]
            for tool in TOOL_GROUPS
        })
        missing_columns = set(FEATURE_COLUMNS) - set(features)
        extra_columns = set(features) - set(FEATURE_COLUMNS)
        if missing_columns or extra_columns:
            raise AssertionError(
                f"feature schema mismatch: missing={sorted(missing_columns)}, "
                f"extra={sorted(extra_columns)}"
            )
        rows.append({
            "candidate_id": candidate,
            "features": {column: features[column] for column in FEATURE_COLUMNS},
        })
    return rows


EPSILON = 1e-8

def target_names(values):
    if not isinstance(values,list):
        raise ValueError('Recorded candidate list must be a list')
    if any(not isinstance(x,str) or not x.strip() for x in values):
        raise ValueError('Invalid recorded candidate')
    if any(x.lower()=='none' for x in values):
        if len(values)!=1 or values[0].lower()!='none':
            raise ValueError('Mixed None and target list')
        return []
    # Preserve names/order/multiplicity: no schema normalization or deduplication.
    return list(values)

def first_list(snapshot):
    validate_pre_submit_snapshot(snapshot)
    values=snapshot.get('raw_ranked_candidates',snapshot.get('ranked_candidates'))
    return target_names(values)

def eligibility(snapshot, final_targets):
    first=first_list(snapshot)
    if not final_targets:
        return False,'no_match'
    if first!=list(final_targets):
        return False,'first_request_final_list_mismatch'
    return True,'eligible'

def _unique(values):
    return list(dict.fromkeys(x for x in values if isinstance(x,str) and x and x.lower()!='none'))

def _extract_evidence_features(snapshot):
    validate_pre_submit_snapshot(snapshot)
    first=first_list(snapshot)
    view=copy.deepcopy(snapshot)
    view.pop('submit_normalization',None)
    view['ranked_candidates']=first
    latest=[r.get('target') for r in view.get('candidate_proposals',[])]
    history={r.get('observation_id'):r.get('candidates',[]) for r in view.get('candidate_proposal_history',[])}
    events=view.get('evidence_events',[])
    observations=view.get('tool_observations',[])
    ids=[o['observation_id'] for o in observations]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate observation IDs')
    observed=[]
    for o in observations:
        oid=o['observation_id']
        observed.extend(r.get('target') for r in history.get(oid,[]))
        observed.extend(o.get('current_target_hits') or [])
        observed.extend([o.get('candidate_a'),o.get('candidate_b')])
        if o.get('tool')=='get_sample_data' and o.get('table_type')=='target':
            observed.append(o.get('column'))
        observed.extend(e.get('target') for e in events if e.get('observation_id')==oid)
        if (o.get('tool')=='get_sample_data' and not o.get('error')
            and o.get('table_type') in ('source','target')
            and o.get('column') and o.get('unique_count') is not None):
            o['success']=True
    for e in events:
        if e.get('observation_id') not in set(ids):raise ValueError('Event outside recorded observations')
        strength=float(e.get('strength',0))
        if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid evidence strength')
    pool=_unique(latest+observed+first)
    if not pool:raise ValueError('Empty pool cannot supply source training vector')
    view['candidate_pool']=pool
    rows=extract_candidate_features(view)
    if [r['candidate_id'] for r in rows]!=pool:
        raise ValueError('Unexplained candidate outside observed/proposed pool')
    ranks={}
    for i,name in enumerate(first,1):ranks.setdefault(name,i)
    for r in rows:
        f=r['features'];name=r['candidate_id'];rank=ranks.get(name,0)
        # Numeric ranking context follows the literal recorded positions.
        f['core_proposal_count']=len(first)
        f['core_proposed']=int(name in ranks)
        f['core_rank']=rank
        f['core_rank_reciprocal']=1/rank if rank else 0.
        if not f['proposal_present']:f['candidate_score_margin_to_top']=0.
    return rows

def project(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid scores')
    out=a.copy()
    for i in range(1,len(out)):out[i]=max(0.,min(float(a[i]),float(out[i-1])-EPSILON))
    return out


SCORE_COLUMNS = (
    'first_submission_score_present', 'first_submission_score',
    'first_top_score_available', 'first_top_score', 'first_score_margin_to_first_top',
)
COLUMNS = tuple(FEATURE_COLUMNS) + SCORE_COLUMNS


def extract(snapshot):
    rows = _extract_evidence_features(snapshot)
    first = first_list(snapshot)
    scores = snapshot.get('raw_candidate_scores')
    available = bool(first and isinstance(scores, dict) and first[0] in scores)
    top = float(scores[first[0]]) if available else 0.
    if not math.isfinite(top) or not 0 <= top <= 1:
        raise ValueError('Invalid first-request top score')
    for row in rows:
        features = row['features']
        candidate = row['candidate_id']
        present = isinstance(scores, dict) and candidate in scores
        score = float(scores[candidate]) if present else 0.
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid first-request score')
        features.update(
            first_submission_score_present=float(present),
            first_submission_score=score,
            first_top_score_available=float(available),
            first_top_score=top,
            first_score_margin_to_first_top=score - top if available and present else 0.,
        )
        assert set(features) == set(COLUMNS)
    return rows

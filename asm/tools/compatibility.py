"""Profile compatibility scores and candidate comparison."""
from typing import Any, Callable, Dict, List, Optional, Tuple

from asm.tools.text import _abbreviation_similarity, _best_text_similarity, _signature_similarity


def family_compatibility(source_family: str, target_family: str) -> float:
    if source_family == target_family:
        return 1.0
    if "unknown" in {source_family, target_family}:
        return 0.55
    compatible_pairs = {
        ("age", "time_date"),
        ("time_date", "age"),
        ("site", "diagnosis"),
        ("diagnosis", "site"),
        ("status", "categorical"),
    }
    if (source_family, target_family) in compatible_pairs:
        return 0.7
    return 0.0


def semantic_compatibility(source_type: str, target_type: str) -> float:
    if source_type == target_type:
        return 1.0
    if "text" in {source_type, target_type}:
        return 0.55
    if "categorical" in {source_type, target_type} and "semantic_text" in {source_type, target_type}:
        return 0.5
    if "measurement" in {source_type, target_type} and "temporal" in {source_type, target_type}:
        return 0.2
    if "code" in {source_type, target_type} and "identifier" in {source_type, target_type}:
        return 0.4
    return 0.0


def role_assessment(source_role: str, target_role: str) -> Tuple[float, float]:
    if source_role == target_role:
        return 1.0, 0.0
    if "unknown" in {source_role, target_role}:
        return 0.55, 0.0
    soft_pairs = {
        frozenset({"identifier", "code"}): 0.82,
        frozenset({"name_label", "type_class"}): 0.35,
        frozenset({"percent_rate", "amount_measure"}): 0.25,
    }
    role_pair = frozenset({source_role, target_role})
    if role_pair in soft_pairs:
        return soft_pairs[role_pair], 0.0
    hard_mismatches = {
        frozenset({"identifier", "name_label"}),
        frozenset({"identifier", "flag_status"}),
        frozenset({"identifier", "percent_rate"}),
        frozenset({"code", "name_label"}),
        frozenset({"code", "percent_rate"}),
        frozenset({"flag_status", "amount_measure"}),
        frozenset({"flag_status", "temporal"}),
        frozenset({"percent_rate", "flag_status"}),
    }
    if role_pair in hard_mismatches:
        return 0.0, 1.0
    return 0.15, 0.4


def effective_role(profile: Dict[str, Any]) -> str:
    schema_role = profile.get("schema_role", "unknown")
    if schema_role != "unknown":
        return schema_role
    card = profile.get("schema_card") or {}
    if card.get("identifier_like") and not card.get("name_like"):
        return "identifier"
    if card.get("name_like") and not card.get("identifier_like"):
        return "name_label"
    return schema_role


def schema_card_role_bonus(
    source_profile: Dict[str, Any],
    target_profile: Dict[str, Any],
) -> Tuple[float, List[str]]:
    source_card = source_profile.get("schema_card") or {}
    target_card = target_profile.get("schema_card") or {}
    evidence: List[str] = []
    bonus = 0.0

    if source_card.get("identifier_like") and not source_card.get("name_like"):
        if target_card.get("identifier_like") and not target_card.get("name_like"):
            bonus += 0.08
            evidence.append("schema-card role: source and target both look identifier-like")
        if target_card.get("name_like") and not target_card.get("identifier_like"):
            bonus -= 0.08
    elif source_card.get("name_like") and not source_card.get("identifier_like"):
        if target_card.get("name_like") and not target_card.get("identifier_like"):
            bonus += 0.05
            evidence.append("schema-card role: source and target both look name-like")
        if target_card.get("identifier_like") and not target_card.get("name_like"):
            bonus -= 0.05

    return round(float(bonus), 4), evidence


def pattern_compatibility(source_pattern: str, target_pattern: str) -> float:
    if source_pattern == target_pattern:
        return 1.0
    if "empty" in {source_pattern, target_pattern}:
        return 0.35
    compatible_pairs = {
        ("positive_numeric", "numeric"),
        ("numeric", "positive_numeric"),
        ("short_enum", "binary_flag"),
        ("binary_flag", "short_enum"),
        ("code_like", "text"),
        ("text", "code_like"),
    }
    if (source_pattern, target_pattern) in compatible_pairs:
        return 0.65
    return 0.0


def unit_compatibility(source_units: List[str], target_units: List[str]) -> float:
    if not source_units or not target_units:
        return 0.5
    if set(source_units) & set(target_units):
        return 1.0
    if {"cm", "mm"} == set(source_units[:1] + target_units[:1]) or {"kg", "lb"} == set(source_units[:1] + target_units[:1]):
        return 0.8
    return 0.0


def time_anchor_compatibility(source_anchor: str, target_anchor: str) -> float:
    if source_anchor == target_anchor:
        return 1.0 if source_anchor != "unknown" else 0.6
    if "unknown" in {source_anchor, target_anchor}:
        return 0.45
    return 0.0


def granularity_compatibility(source_granularity: str, target_granularity: str) -> float:
    if source_granularity == target_granularity:
        return 1.0
    if "generic" in {source_granularity, target_granularity}:
        return 0.55
    if {source_granularity, target_granularity} == {"location", "organ_or_origin"}:
        return 0.75
    if "event_specific" in {source_granularity, target_granularity}:
        return 0.15
    return 0.3


def local_context_compatibility(
    source_context: Dict[str, Any],
    target_context: Dict[str, Any],
) -> float:
    sig = _signature_similarity(source_context["group_signature"], target_context["group_signature"])

    src_paired = set(source_context.get("paired_roles", []))
    tgt_paired = set(target_context.get("paired_roles", []))
    if src_paired and tgt_paired:
        paired = 1.0 if src_paired & tgt_paired else 0.35
    elif source_context.get("paired_columns") or target_context.get("paired_columns"):
        paired = 0.45
    else:
        paired = 0.55

    src_sibling = set(source_context.get("sibling_roles", []))
    tgt_sibling = set(target_context.get("sibling_roles", []))
    if src_sibling and tgt_sibling:
        sibling = len(src_sibling & tgt_sibling) / len(src_sibling | tgt_sibling)
    else:
        sibling = 0.5

    return round(float(0.6 * sig + 0.25 * paired + 0.15 * sibling), 4)


def pairwise_match_details(
    source_profile: Dict[str, Any],
    target_profile: Dict[str, Any],
    source_context: Dict[str, Any],
    target_context: Dict[str, Any],
    structural_alignment: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    base_score: Optional[float] = None,
) -> Dict[str, Any]:
    expansion_match = _best_text_similarity(
        source_profile["expanded_name_texts"],
        target_profile["expanded_name_texts"],
    )
    abbreviation_match = _abbreviation_similarity(
        source_profile["abbreviation_signatures"],
        target_profile["abbreviation_signatures"],
    )
    family = family_compatibility(
        source_profile["attribute_family"],
        target_profile["attribute_family"],
    )
    semantic = semantic_compatibility(
        source_profile["semantic_type"],
        target_profile["semantic_type"],
    )
    pattern = pattern_compatibility(
        source_profile["value_pattern"],
        target_profile["value_pattern"],
    )
    unit = unit_compatibility(
        source_profile["unit_hint"],
        target_profile["unit_hint"],
    )
    time_anchor = time_anchor_compatibility(
        source_profile["time_anchor"],
        target_profile["time_anchor"],
    )
    granularity = granularity_compatibility(
        source_profile["granularity"],
        target_profile["granularity"],
    )
    effective_source_role = effective_role(source_profile)
    effective_target_role = effective_role(target_profile)
    role, role_penalty = role_assessment(
        effective_source_role,
        effective_target_role,
    )
    local_context = local_context_compatibility(source_context, target_context)
    pair_bonus, pair_evidence = 0.0, []
    card_role_bonus, card_role_evidence = schema_card_role_bonus(
        source_profile,
        target_profile,
    )
    structural_signal = structural_alignment(source_profile, target_profile)
    abbreviation_bonus = (
        0.06
        if abbreviation_match >= 0.85
        else (0.03 if abbreviation_match >= 0.75 else 0.0)
    )
    schema_confidence = (
        0.30 * role
        + 0.22 * max(expansion_match, abbreviation_match)
        + 0.18 * local_context
        + 0.15 * family
        + 0.15 * semantic
        + pair_bonus
        + max(0.0, card_role_bonus)
        + structural_signal["bonus"]
        - 0.5 * structural_signal["penalty"]
    )
    core_score = (
        0.16 * expansion_match
        + 0.12 * abbreviation_match
        + 0.14 * family
        + 0.12 * semantic
        + 0.17 * role
        + 0.12 * local_context
        + 0.10 * pattern
        + 0.08 * unit
        + 0.07 * time_anchor
        + 0.05 * granularity
        + pair_bonus
        + card_role_bonus
        + structural_signal["bonus"]
        + abbreviation_bonus
        - 0.08 * role_penalty
        - structural_signal["penalty"]
    )
    if base_score is not None:
        base_norm = max(min((float(base_score) + 1.0) / 2.0, 1.0), 0.0)
        final_score = 0.4 * base_norm + 0.6 * core_score
    else:
        final_score = core_score

    why_matched: List[str] = []
    if expansion_match >= 0.72:
        why_matched.append("name expansions align closely")
    if abbreviation_match >= 0.78:
        why_matched.append("abbreviation pattern aligns closely")
    if role >= 0.95 and effective_source_role != "unknown":
        why_matched.append(f"schema role aligns ({effective_source_role})")
    if family >= 0.95 and source_profile["attribute_family"] != "unknown":
        why_matched.append(
            f"both columns look like {source_profile['attribute_family']} fields"
        )
    if semantic >= 0.95 and source_profile["semantic_type"] != "text":
        why_matched.append(f"semantic type is compatible ({source_profile['semantic_type']})")
    if local_context >= 0.8 and target_context["group_signature"]:
        why_matched.append("local schema context is consistent")
    why_matched.extend(structural_signal["evidence"])
    if unit >= 0.95 and source_profile["unit_hint"]:
        why_matched.append(f"unit hints overlap ({', '.join(source_profile['unit_hint'])})")
    if pattern >= 0.95 and source_profile["value_pattern"] != "text":
        why_matched.append(f"value patterns match ({source_profile['value_pattern']})")
    if time_anchor >= 0.95 and source_profile["time_anchor"] != "unknown":
        why_matched.append(f"time anchor aligns around {source_profile['time_anchor']}")
    why_matched.extend(pair_evidence)
    why_matched.extend(card_role_evidence)
    if not why_matched:
        why_matched.append("retrieval score is stronger than nearby alternatives")

    transformations: List[str] = []
    if source_profile["unit_hint"] and target_profile["unit_hint"] and unit < 1.0 and unit > 0.0:
        transformations.append("unit_conversion")
    if {source_profile["value_pattern"], target_profile["value_pattern"]} == {
        "code_like",
        "text",
    }:
        transformations.append("codebook_lookup")
    if {source_profile["value_pattern"], target_profile["value_pattern"]} <= {
        "binary_flag",
        "short_enum",
    }:
        transformations.append("enum_normalization")
    if {source_profile["value_pattern"], target_profile["value_pattern"]} == {
        "positive_numeric",
        "numeric",
    }:
        transformations.append("numeric_normalization")

    return {
        "final_score": round(float(final_score), 4),
        "expansion_match": expansion_match,
        "abbreviation_match": abbreviation_match,
        "attribute_family_compatibility": round(float(family), 4),
        "semantic_type_compatibility": round(float(semantic), 4),
        "role_compatibility": round(float(role), 4),
        "role_conflict_penalty": round(float(role_penalty), 4),
        "effective_source_role": effective_source_role,
        "effective_target_role": effective_target_role,
        "schema_card_role_bonus": round(float(card_role_bonus), 4),
        "local_context_compatibility": round(float(local_context), 4),
        "pair_consistency_bonus": round(float(pair_bonus), 4),
        "structural_signal_bonus": structural_signal["bonus"],
        "structural_signal_penalty": structural_signal["penalty"],
        "structural_subtype_match": structural_signal["exact_match"],
        "source_structural_subtype": structural_signal["source_subtype"],
        "target_structural_subtype": structural_signal["target_subtype"],
        "abbreviation_bonus": round(float(abbreviation_bonus), 4),
        "schema_confidence": round(float(min(1.0, schema_confidence)), 4),
        "value_pattern_compatibility": round(float(pattern), 4),
        "unit_compatibility": round(float(unit), 4),
        "time_anchor_compatibility": round(float(time_anchor), 4),
        "granularity_compatibility": round(float(granularity), 4),
        "why_matched": why_matched[:3],
        "possible_transformations": transformations,
    }


def build_interpretation_summary(
    profile: Dict[str, Any],
    table_type: str,
    get_local_context: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    local_context = get_local_context(table_type, profile["column"])
    return {
        "expanded_name_candidates": profile["expanded_name_candidates"],
        "semantic_type": profile["semantic_type"],
        "attribute_family": profile["attribute_family"],
        "schema_role": profile["schema_role"],
        "unit_hint": profile["unit_hint"],
        "value_pattern": profile["value_pattern"],
        "time_anchor": profile["time_anchor"],
        "granularity": profile["granularity"],
        "group_signature": profile["group_signature"],
        "schema_card": profile.get("schema_card"),
        "local_context": {
            "paired_columns": local_context["paired_columns"],
            "sibling_columns": local_context["sibling_columns"],
        },
    }


def differential_verify(env, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if env._source_df is None or env._target_df is None:
        return {"error": "No task loaded"}
    a, b = arguments.get("candidate_a"), arguments.get("candidate_b")
    pair = arguments.get("candidates", [])
    if (not a or not b) and isinstance(pair, list) and len(pair) >= 2:
        a, b = pair[:2]
    if not isinstance(a, str) or not isinstance(b, str) or a == b:
        return {"error": "Provide two distinct target candidate names."}
    if a not in env._target_df.columns or b not in env._target_df.columns:
        return {"error": "Both candidates must exist in the target table."}
    source = env._get_profile("source", env._source_column)
    pa, pb = env._get_profile("target", a), env._get_profile("target", b)
    da, db = env._pairwise_match_details(source, pa), env._pairwise_match_details(source, pb)
    delta = round(da["final_score"] - db["final_score"], 4)
    preferred = a if delta > 0 else b if delta < 0 else None
    components = ("role_compatibility", "attribute_family_compatibility",
                  "time_anchor_compatibility", "granularity_compatibility",
                  "value_pattern_compatibility", "local_context_compatibility")
    evidence = [f"{key}: {a}={da[key]:.4f}, {b}={db[key]:.4f}"
                for key in components if key in da and key in db and da[key] != db[key]]
    result = {
        "success": True, "source_column": env._source_column,
        "preferred": preferred, "tie": delta == 0,
        "confidence": round(min(0.95, 0.5 + abs(delta)), 2),
        "margin": delta, "margin_magnitude": abs(delta),
        "conflict_type": "profile_comparison", "deciding_signal": "profile_score",
        "comparison": {
            "source": env._build_interpretation_summary(source, "source"),
            a: {**env._build_interpretation_summary(pa, "target"), "match_score": da["final_score"]},
            b: {**env._build_interpretation_summary(pb, "target"), "match_score": db["final_score"]},
        },
        "decisive_evidence": evidence or ["Equal profile scores." if delta == 0 else "Aggregate profile score difference."],
        "local_validation": None, "needs_execute_python": False,
        "suggested_next_step": "gather_more_evidence" if delta == 0 else "interpret_comparison",
    }
    env._last_differential_result = result
    return result

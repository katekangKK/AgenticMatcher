"""Documentation lookup and field definitions."""
from typing import Any, Dict, List
import json
import re

import numpy as np

from asm.tools.text import _cosine_similarity


def prepare_documentation(doc_index):
    """Accept field definitions only; container names are not model evidence."""
    if doc_index is None:
        return None
    if not isinstance(doc_index, dict):
        raise ValueError("Documentation must map groups to field definitions.")
    allowed = {"term", "definition", "text", "entity", "data_type", "enum_sample"}
    prepared = {}
    for index, entries in enumerate(doc_index.values()):
        if not isinstance(entries, list):
            raise ValueError("Documentation groups must contain lists of field definitions.")
        clean = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - allowed:
                raise ValueError("Documentation supports field definitions only; remove unrelated metadata.")
            if not isinstance(entry.get("term"), str) or not entry["term"].strip() or not isinstance(entry.get("text"), str) or not entry["text"].strip():
                raise ValueError("Documentation entries require nonempty term and text.")
            if any(key in entry and not isinstance(entry[key], str) for key in ("definition", "entity", "data_type")):
                raise ValueError("Definition, entity and data_type must be text.")
            sample = entry.get("enum_sample")
            if sample is not None and (not isinstance(sample, list) or any(type(v) not in (str, int, float, bool, type(None)) for v in sample)):
                raise ValueError("enum_sample must be a list of scalar values.")
            clean.append(entry.copy())
        prepared[f"dictionary_{index + 1}"] = clean
    return prepared


def _definition_key(entry):
    # Equal wording is not enough to merge distinct entity contexts.
    return (entry.get('domain'), entry.get('term'), entry.get('entity'),
            entry.get('definition'), entry.get('text'),
            json.dumps(entry.get('enum_sample'), sort_keys=True))


def _cover_named_definitions(query, ranked, legacy):
    """Reserve space for explicitly named fields, keeping entity alternatives.

    A bounded result is explicit about omissions. Dictionary text and enum
    examples remain verbatim; no GT, task names, or generated definitions.
    """
    stop = {'a','an','the','in','of','and','or','for','from','to','is','are','as','on','by','with','at'}
    terms = {str(e.get('term','')) for e in ranked} - {''}
    positions = {}
    for term in terms:
        if term.lower() in stop:
            continue
        hit = re.search(r'(?<![\w])' + re.escape(term) + r'(?![\w])', query, re.IGNORECASE)
        if hit:
            positions[term] = hit.start()
    named = sorted(positions, key=lambda t:(positions[t],t))
    choices = []
    for term in named:
        choices.extend(e for e in ranked if e.get('term') == term)
    # Preserve normal semantic retrieval after explicit lookup. Do not raise
    # vector relevance scores to 1 or make name matches imply correspondence.
    choices.extend(legacy)
    result = []; seen = set()
    for e in choices:
        key = _definition_key(e)
        if key in seen:
            continue
        if len(result) >= 8:
            break
        seen.add(key)
        item = e.copy()
        item.pop('domain', None)
        if item.get('enum_sample') is not None:
            item['enum_sample_note'] = 'Illustrative examples, not a guarantee of the full allowed value set; absence from this sample is not a value incompatibility test.'
        result.append(item)
    coverage = []
    for term in named:
        available = {_definition_key(e) for e in ranked if e.get('term') == term}
        included = {key for key in seen if key[1] == term}
        coverage.append({'term':term,'available_definitions':len(available),
                         'returned_definitions':len(included),'complete':available <= included})
    return {'success':True,'results':result,'definition_coverage':coverage,
            'definition_contract_version':2,
            'definition_note':'A field name may have different meanings for different entities. Compare entity, definition and units; listed alternatives do not all imply valid matches.'}


def doc_query(
    query: str,
    doc_index: Dict[str, List[Dict[str, Any]]],
    doc_embeddings: Dict[str, Dict[str, Any]],
    embedding_model,
) -> Dict[str, Any]:
    if not query.strip():
        return {"error": "Query cannot be empty"}

    if doc_index is None or len(doc_index) == 0:
        return {
            "success": True,
            "results": [],
            "message": "No field documentation supplied. "
                       "Rely on column names, data samples, and value analysis instead.",
        }

    if not doc_embeddings:
        for domain, entries in doc_index.items():
            texts = [e.get("text", e.get("term", "")) for e in entries]
            embs = embedding_model.encode(texts, is_query=False)
            if embs is not None:
                doc_embeddings[domain] = {
                    "entries": entries,
                    "embeddings": embs,
                }

    if not doc_embeddings:
        results = []
        all_entries = []
        query_lower = query.lower()
        for domain, entries in doc_index.items():
            for entry in entries:
                entry = dict(entry, domain=domain)
                all_entries.append(entry)
                text = entry.get("text", entry.get("term", "")).lower()
                if any(w in text for w in query_lower.split()):
                    results.append(entry)
        return _cover_named_definitions(query, all_entries, results[:5])

    query_emb = embedding_model.encode([query], is_query=True)
    if query_emb is None:
        return {"success": False, "error": "Embedding model not available"}
    query_emb = query_emb[0]

    all_results = []
    ranked = []
    for domain, data in doc_embeddings.items():
        sims = _cosine_similarity(query_emb, data["embeddings"])
        top_idx = np.argsort(-sims)[:3]
        for idx in np.argsort(-sims):
            entry = data["entries"][idx].copy()
            entry["relevance_score"] = round(float(sims[idx]), 4)
            entry["domain"] = domain
            ranked.append(entry)
            if idx in top_idx:
                all_results.append(entry)

    all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    ranked.sort(key=lambda x:x.get('relevance_score',0), reverse=True)
    return _cover_named_definitions(query, ranked, all_results[:5])

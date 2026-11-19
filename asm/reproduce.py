"""Run the GDC benchmark with frozen task-held-out scorers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

import pandas as pd

from asm import __version__
from asm.agent import AgentConfig, SchemaMatchingAgent
from asm.cli import write_output
from asm.environment import SchemaMatchingEnvironment
from asm.llm import ConfigurationError, LLMClient
from asm.scorer import Scorer
from asm.tools.retrieval import _embedding_model
from asm.prompts import get_system_prompt
from asm.submission import match_score_prompt


PACKAGE = Path(__file__).resolve().parent


class BenchmarkError(ValueError):
    """An input error whose message contains no user-supplied path or value."""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root):
    """Validate the benchmark before starting any paid inference."""
    manifest = json.loads((PACKAGE / "resources" / "gdc.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise BenchmarkError("GDC input missing or changed. Use the original benchmark CSV files and set ASM_GDC_DATA_DIR to their dataset root.")
    target = pd.read_csv(root / "target-tables" / "gdc_unique_columns_concat_values.csv", low_memory=False)
    tasks = []
    for entry in manifest["tasks"]:
        name = entry["name"]
        source = pd.read_csv(root / "source-tables" / (name + ".csv"))
        source.columns = source.columns.str.strip()
        if len(source.columns) != entry["source_columns"] or not source.columns.is_unique:
            raise BenchmarkError("Unexpected GDC source schema.")
        # Labels stay in the evaluator; neither the agent nor the scorer receives them.
        truth = pd.read_csv(root / "ground-truth" / (name + ".csv")).dropna()
        if truth.shape[1] != 2:
            raise BenchmarkError("Ground truth must contain two columns.")
        pairs = set(truth.itertuples(index=False, name=None))
        if not pairs or any(s not in source or t not in target for s, t in pairs):
            raise BenchmarkError("Ground-truth columns do not agree with the benchmark schemas.")
        tasks.append((name, source, pairs))
    dictionary = PACKAGE / "resources" / "gdc_dictionary.json"
    if sha256(dictionary) != manifest["dictionary_sha256"]:
        raise BenchmarkError("GDC dictionary checksum mismatch.")
    docs = json.loads(dictionary.read_text())
    if not isinstance(docs, dict) or not docs or any(
        not isinstance(entries, list) or not entries or any(
            not isinstance(e.get("term"), str) or not isinstance(e.get("text"), str) or not e["text"].strip()
            for e in entries
        ) for entries in docs.values()
    ):
        raise BenchmarkError("Invalid GDC dictionary.")
    return manifest, target, tasks, docs


def load_scorers(task_names):
    scorers = {}
    for name in task_names:
        scorer = Scorer(PACKAGE / "models" / "gdc" / name)
        if scorer.manifest.get("held_out_task") != name or set(scorer.manifest.get("training_tasks", [])) != set(task_names) - {name}:
            raise BenchmarkError("A GDC scorer does not match its held-out task.")
        scorers[name] = scorer
    return scorers


def evaluate(predictions, truth, candidate_key):
    """Source MRR (zero ties keep list order) and tie-aware global Recall@GT."""
    by_source = {}
    for source, target in truth:
        by_source.setdefault(source, set()).add(target)
    pairs = {}
    reciprocal = 0.0
    seen = set()
    for row in predictions:
        source = row["source_column"]
        if source in seen:
            raise BenchmarkError("Duplicate source prediction.")
        seen.add(source)
        if row.get("error"):
            continue
        candidates = row[candidate_key]
        if len({c for c, _ in candidates}) != len(candidates) or any(not math.isfinite(v) or not 0 <= v <= 1 for _, v in candidates):
            raise BenchmarkError("Invalid candidate scores.")
        pairs.update({(source, c): float(v) for c, v in candidates})
        ordered = sorted(candidates, key=lambda item: -item[1])
        index = 0
        while index < len(ordered):
            above = index
            score = ordered[index][1]
            group = []
            while index < len(ordered) and ordered[index][1] == score:
                group.append(ordered[index][0])
                index += 1
            relevant = len(set(group) & by_source.get(source, set()))
            if relevant:
                # The paper preserves returned-list order for zero-score ties.
                if score == 0:
                    position = next(i for i, target in enumerate(group, 1)
                                    if target in by_source[source])
                    reciprocal += 1 / (above + position)
                    break
                size = len(group)
                reciprocal += sum(
                    math.comb(size - position, relevant - 1) / math.comb(size, relevant) / (above + position)
                    for position in range(1, size - relevant + 2)
                )
                break
    cutoff = len(truth)
    ranked_scores = sorted(pairs.values(), reverse=True)
    if len(pairs) <= cutoff:
        credit = len(set(pairs) & truth)
    else:
        boundary = ranked_scores[cutoff - 1]
        above = {p for p, score in pairs.items() if score > boundary}
        tied = {p for p, score in pairs.items() if score == boundary}
        credit = len(above & truth) + (cutoff - len(above)) * len(tied & truth) / len(tied)
    return {"mrr": reciprocal / len(by_source), "recall_at_gt": credit / cutoff}


def run_task(name, source, target, docs, client, scorer, output, save_evidence):
    predictions = []
    journal = output / (name + ".partial.jsonl")
    fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as progress, tempfile.TemporaryDirectory(prefix="agenticmatcher-") as workspace:
        env = SchemaMatchingEnvironment(topk=20, workspace_dir=workspace, doc_index=docs, match_score_mode="llm")
        env.configure_tool_budgets(execute_python_budget=2, doc_query_budget=3)
        agent = SchemaMatchingAgent(client, env, AgentConfig())
        for column in source.columns:
            try:
                if not env.load_task(source, target, column, task_name=name):
                    raise ValueError("Unable to load source column.")
                result = agent.run(column)
                snapshot = result.evidence_trace.get("pre_submit_snapshot")
                rescored = scorer.score(snapshot, result.predicted_candidates) if not result.error else {
                    "applied": False, "reason": "episode_error", "candidates": result.predicted_candidates}
                row = {
                    "source_column": column, "core_candidates": result.predicted_candidates,
                    "predicted_candidates": rescored.pop("candidates"), "rescoring": rescored,
                    "error": result.error, "termination_reason": result.termination_reason,
                    "num_rounds": result.num_rounds, "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens, "total_tokens": result.total_tokens,
                }
                if save_evidence:
                    row["pre_submit_snapshot"] = snapshot
            except Exception:
                row = {"source_column": column, "core_candidates": [], "predicted_candidates": [],
                       "rescoring": {"applied": False, "reason": "episode_error"},
                       "error": "Attribute failed during matching or scoring.", "termination_reason": "execution_error"}
            predictions.append(row)
            progress.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
    return predictions, journal


def run(args):
    manifest, target, tasks, docs = load_inputs(Path(args.data_dir))
    scorers = load_scorers([name for name, _, _ in tasks])
    if args.check_only:
        print("GDC inputs, dictionary and ten held-out scorers verified (569 source attributes). No API calls made.")
        return 0
    client = LLMClient()
    # Benchmark runs must not silently switch to string-based retrieval.
    _embedding_model._load()
    if _embedding_model._model is None:
        raise BenchmarkError("GDC reproduction requires the BGE encoder. Install the embeddings extra and cache the encoder or enable ASM_ALLOW_REMOTE_EMBEDDING_DOWNLOAD.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir) / run_id
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    config = {"dataset": "gdc", "release_version": __version__, "run_id": run_id, "model": client.model,
              "max_rounds": 25, "top_k": 20, "temperature": 0.7, "top_p": 0.9,
              "max_tokens": 8192, "execute_python_budget": 2, "doc_query_budget": 3,
              "scoring": "frozen_leave_one_task_out", "input_sha256": manifest["files"],
              "dictionary_sha256": manifest["dictionary_sha256"],
              "system_prompt_sha256": hashlib.sha256(match_score_prompt(get_system_prompt(), "llm").encode()).hexdigest()}
    # No endpoint, credentials, input directory or workspace paths in saved config.
    write_output(output / "config.json", config)
    results = []
    for name, source, truth in tasks:
        print("Running GDC task " + name, flush=True)
        predictions, journal = run_task(name, source, target, docs, client, scorers[name], output, args.save_evidence)
        result = {"task": name, "source_attributes": len(source.columns), "reference_pairs": len(truth),
                  "errors": sum(bool(row.get("error")) for row in predictions),
                  "core": evaluate(predictions, truth, "core_candidates"),
                  "rescored": evaluate(predictions, truth, "predicted_candidates")}
        write_output(output / (name + ".json"), {**result, "predictions": predictions})
        journal.unlink()
        results.append(result)
    errors = sum(row["errors"] for row in results)
    summary = {"dataset": "gdc", "run_id": run_id, "errors": errors, "complete": errors == 0,
               "tasks": results, "macro": {
                   method: {metric: sum(row[method][metric] for row in results) / len(results)
                            for metric in ("mrr", "recall_at_gt")}
                   for method in ("core", "rescored")}}
    write_output(output / "summary.json", summary)
    print(json.dumps({"run_id": run_id, "errors": errors, "macro": summary["macro"]}, indent=2))
    return 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the GDC matching experiment with frozen held-out scorers.")
    parser.add_argument("--dataset", choices=["gdc"], required=True)
    parser.add_argument("--data-dir", default=os.getenv("ASM_GDC_DATA_DIR") or "data/gdc", help="GDC dataset root; otherwise use ASM_GDC_DATA_DIR or the project data directory.")
    parser.add_argument("--output-dir", default=os.getenv("ASM_OUTPUT_DIR") or "outputs/gdc", help="Experiment output directory; each invocation creates a separate run.")
    parser.add_argument("--check-only", action="store_true", help="Validate input checksums and held-out models without API or embedding calls.")
    parser.add_argument("--save-evidence", action="store_true", help="Save new pre-submit snapshots; these can contain input data.")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (BenchmarkError, ConfigurationError) as exc:
        print(str(exc), file=sys.stderr)
    except Exception as exc:
        print("Unable to run GDC experiment (" + type(exc).__name__ + "). Check inputs and configuration.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

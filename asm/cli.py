"""Portable CSV matching and offline rescoring commands."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import pandas as pd
from asm import __version__
from asm.agent import AgentConfig, SchemaMatchingAgent
from asm.environment import SchemaMatchingEnvironment
from asm.llm import ConfigurationError, LLMClient
from asm.scorer import Scorer


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return value


def write_output(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse replacement and make output private to the current user.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_table(path):
    # Inspect the raw header because pandas otherwise silently rewrites duplicates.
    import csv
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or len(columns) != len(set(columns)) or any(not c.strip() for c in columns):
        raise ValueError("CSV headers must be nonempty and unique.")
    if any(c.strip().lower() == "none" for c in columns):
        raise ValueError("The reserved no-match name 'None' cannot be a CSV header.")
    return pd.read_csv(path, encoding="utf-8-sig")


def match(args):
    if Path(args.output).exists():
        raise FileExistsError("Output already exists; choose a new file.")
    source, target = read_table(args.source), read_table(args.target)
    columns = args.columns or list(source.columns)
    if len(columns) != len(set(columns)) or any(c not in source for c in columns):
        raise ValueError("Requested source columns must be unique existing headers.")
    docs = json.loads(Path(args.docs).read_text()) if args.docs else None
    if docs is not None and (not isinstance(docs, dict) or any(
        not isinstance(entries, list) or any(not isinstance(e, dict) or not isinstance(e.get('term'), str) or not isinstance(e.get('text'), str) or not e['text'].strip() for e in entries)
        for entries in docs.values())):
        raise ValueError("Documentation entries require term and nonempty text fields; include the definition in text for retrieval.")
    client = LLMClient(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env)
    scorer = None if args.core_only else Scorer()
    predictions = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_name(output.name + ".partial.jsonl")
    # Reserve an owner-only journal before starting calls. Never replace old work.
    fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as progress, tempfile.TemporaryDirectory(prefix="agenticmatcher-") as workspace:
        env = SchemaMatchingEnvironment(topk=args.top_k, workspace_dir=workspace, doc_index=docs, match_score_mode="llm")
        if args.disable_python:
            env.configure_tool_budgets(execute_python_budget=0)
        agent = SchemaMatchingAgent(client, env, AgentConfig(max_rounds=args.max_rounds, temperature=args.temperature))
        for column in columns:
            try:
                if not env.load_task(source, target, column, task_name="custom"):
                    raise ValueError("Unable to load source column.")
                result = agent.run(column)
                snapshot = result.evidence_trace.get("pre_submit_snapshot")
                rescored = scorer.score(snapshot, result.predicted_candidates) if scorer and not result.error else {
                    "applied": False, "reason": "core_only" if scorer is None else "episode_error", "candidates": result.predicted_candidates}
                record = {
                    "source_column": column, "predicted_candidates": rescored.pop("candidates"),
                    "core_candidates": result.predicted_candidates, "rescoring": rescored,
                    "termination_reason": result.termination_reason, "error": result.error,
                    "num_rounds": result.num_rounds,
                }
                if args.save_evidence:
                    record["pre_submit_snapshot"] = snapshot
            except Exception:
                # A failed attribute is an error, never a successful no-match.
                # Avoid exposing exception bodies, file paths or provider details.
                record = {
                    "source_column": column, "predicted_candidates": [], "core_candidates": [],
                    "rescoring": {"applied": False, "reason": "episode_error"},
                    "termination_reason": "execution_error",
                    "error": "Attribute failed during tool execution or scoring.", "num_rounds": None,
                }
            predictions.append(record)
            progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
    payload = {"schema_version": 1, "release_version": __version__, "predictions": predictions}
    write_output(args.output, payload)
    journal.unlink()
    return 1 if any(row["error"] for row in predictions) else 0



def rescore(args):
    if Path(args.output).exists():
        raise FileExistsError("Output already exists; choose a new file.")
    data = json.loads(Path(args.input).read_text())
    scorer = Scorer()
    for row in data["predictions"]:
        candidates = row.get("core_candidates", row["predicted_candidates"])
        if row.get("error"):
            result = {"applied": False, "reason": "episode_error", "candidates": candidates}
        else:
            snapshot = row.get("pre_submit_snapshot") or (row.get("evidence_trace") or {}).get("pre_submit_snapshot")
            result = scorer.score(snapshot, candidates)
        row["core_candidates"] = candidates
        row["predicted_candidates"] = result.pop("candidates")
        row["rescoring"] = result
    data["release_version"] = __version__
    write_output(args.output, data)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agenticmatcher")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("match", help="Match source CSV columns against a target CSV using your API endpoint.")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--columns", nargs="+")
    p.add_argument("--model", help="Model name; otherwise set ASM_MODEL. No default model.")
    p.add_argument("--base-url", help="Compatible API base URL; otherwise set ASM_BASE_URL. No default endpoint.")
    p.add_argument("--api-key-env", help="Environment variable NAME, never the key itself. Defaults to ASM_API_KEY.")
    p.add_argument("--docs", help="Optional public target documentation JSON.")
    p.add_argument("--max-rounds", type=positive, default=25)
    p.add_argument("--top-k", type=positive, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--core-only", action="store_true")
    p.add_argument("--disable-python", action="store_true")
    p.add_argument("--save-evidence", action="store_true", help="Save pre-submit evidence for replay; it can include input data.")
    p.set_defaults(func=match)
    p = sub.add_parser("rescore", help="Offline replay; no API key or network access required.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=rescore)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileExistsError:
        print("Output already exists; choose a new file.", file=sys.stderr)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
    except Exception as exc:
        # No traceback, credentials, prompt body or local path in CLI diagnostics.
        print(f"Unable to complete command ({type(exc).__name__}). Check arguments, input files and configuration.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

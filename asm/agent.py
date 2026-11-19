"""
ASM Schema Matching Agent

Implements the cognitive-action decoupled agent loop:
1. Parse cognitive tags (PLAN / SHORTLIST / DEEPEN / VERIFY / ANSWER)
2. Parse tool tags (shortlist_match / differential_verify / get_sample_data / execute_python / doc_query / submit)
3. Record structured evidence for offline rescoring
"""

import json
import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from asm.environment import SchemaMatchingEnvironment, MatchResult, TOOL_COSTS
from asm.prompts import get_system_prompt, get_user_task_prompt
from asm.submission import match_score_prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COGNITIVE_TAGS = {"PLAN", "SHORTLIST", "DEEPEN", "VERIFY", "ANSWER"}

TOOL_TAGS = {"shortlist_match", "differential_verify", "get_sample_data", "execute_python", "doc_query", "submit"}

ALL_TOOL_TAGS = TOOL_TAGS

SUBMIT_TOOLS = {"submit"}


@dataclass
class AgentConfig:
    """Agent configuration."""
    max_rounds: int = 25
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    verbose: bool = False


@dataclass
class TrajectoryStep:
    """A single step in the agent trajectory."""
    step: int
    cognitive_tag: Optional[str] = None
    cognitive_content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[Dict] = None
    tool_success: Optional[bool] = None
    tool_cost: Optional[str] = None
    raw_response: str = ""
    format_violation: Optional[str] = None
    format_recovery: Optional[str] = None
    timestamp: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        d = {
            "step": self.step,
            "cognitive_tag": self.cognitive_tag,
            "cognitive_content": self.cognitive_content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result if include_raw else None,
            "tool_success": self.tool_success,
            "tool_cost": self.tool_cost,
            "raw_response": self.raw_response if include_raw else None,
            "format_violation": self.format_violation,
            "timestamp": self.timestamp,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.format_recovery:
            d["format_recovery"] = self.format_recovery
        return d


class SchemaMatchingAgent:
    """
    ASM Schema Matching Agent.

    Drives a multi-turn conversation with cognitive-action decoupled XML tags.
    Records a structured trajectory for each task.
    """

    def __init__(
        self,
        llm,  # LLMClient or compatible object exposing .chat()
        environment: SchemaMatchingEnvironment,
        config: Optional[AgentConfig] = None,
    ):
        self.llm = llm
        self.env = environment
        self.config = config or AgentConfig()
        self.messages: List[Dict[str, Any]] = []
        self.trajectory: List[TrajectoryStep] = []

    # ------------------------------------------------------------------
    # Parsing: cognitive tags
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cognitive_tag(text: str) -> Optional[Dict[str, str]]:
        """Extract the first cognitive tag from model output.

        Returns {"tag": "PLAN", "content": "..."} or None.
        """
        for tag in COGNITIVE_TAGS:
            pattern = rf"<{tag}>(.*?)</{tag}>"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return {"tag": tag, "content": m.group(1).strip()}
        return None

    # ------------------------------------------------------------------
    # Parsing: tool tags  (new ASM format + legacy <tool_call> format)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml_args(body: str) -> Dict[str, Any]:
        """Fallback: extract arguments from XML-style nested tags.

        Handles cases where the model outputs e.g.
        ``<query>some text</query><top_k>5</top_k>``
        instead of the expected JSON ``{"query": "some text", "top_k": 5}``.
        """
        args: Dict[str, Any] = {}
        for m in re.finditer(r"<(\w+)>(.*?)</\1>", body, re.DOTALL):
            key, val = m.group(1), m.group(2).strip()
            try:
                parsed = json.loads(val)
                args[key] = parsed
            except (json.JSONDecodeError, ValueError):
                args[key] = val
        return args

    @staticmethod
    def _action_text(text: str) -> str:
        """Ignore inline-code examples before an action, preserving tool bodies.

        Cognitive tags may contain real actions in legacy traces. Only quoted
        inline examples are excluded; once an unquoted tool starts, JSON/code
        bytes are left untouched (including backticks within Python strings).
        """
        action = re.compile(r'<(?:' + '|'.join(sorted(ALL_TOOL_TAGS)) + r'|tool_call)>')
        quoted = re.compile(r'(?<!`)`(?!`)[^`\n]*`(?!`)')
        parts, offset = [], 0
        for span in quoted.finditer(text):
            tool = action.search(text, offset, span.start())
            if tool is not None:
                break
            parts.append(text[offset:span.start()])
            offset = span.end()
        parts.append(text[offset:])
        return ''.join(parts)

    @staticmethod
    def _find_tool_tags(text: str) -> List[str]:
        text = SchemaMatchingAgent._action_text(text)
        matches: List[Tuple[int, str]] = []
        for tag in ALL_TOOL_TAGS:
            pattern = rf"<{tag}>\s*.*?\s*</{tag}>"
            for m in re.finditer(pattern, text, re.DOTALL):
                matches.append((m.start(), tag))
        matches.sort(key=lambda item: item[0])
        return [tag for _pos, tag in matches]

    @staticmethod
    def _recover_python_json_body(body: str) -> Optional[Dict[str, str]]:
        """Unwrap strict JSON followed by known redundant closing markers.

        Preserve the code string exactly. Never repair code, incomplete JSON,
        duplicate keys, additional arguments, or arbitrary trailing text.
        """
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("Nonstandard JSON constant")

        try:
            text = body.strip()
            decoder = json.JSONDecoder(object_pairs_hook=unique_object,
                                       parse_constant=reject_constant)
            args, end = decoder.raw_decode(text)
            # Consume the JSON object first: its own closing brace is never
            # part of a redundant tail. Do not repair incomplete JSON/code.
            tail = text[end:].strip()
            if tail not in ('>', '}>', '}', '"}', '</code>', '</code>}',
                            '</code>>', '</arg_value>}'):
                return None
        except (ValueError, TypeError):
            return None
        if isinstance(args, dict) and set(args) == {"code"} and isinstance(args["code"], str):
            return args
        return None

    @staticmethod
    def _recover_multiline_python_json(body: str) -> Optional[Dict[str, str]]:
        """Unwrap one complete code field containing literal line breaks.

        Two encodings are possible: JSON escapes, or an unescaped raw Python
        body. Accept only when syntax validation leaves exactly one distinct
        code string. Never execute here or modify either candidate's bytes.
        """
        try:
            json.loads(body)
            return None
        except json.JSONDecodeError as exc:
            if not exc.msg.startswith("Invalid control character"):
                return None
        if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in body):
            return None

        def unique_object(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate key")
                obj[key] = value
            return obj

        try:
            args = json.loads(body, strict=False, object_pairs_hook=unique_object)
        except (ValueError, TypeError):
            return None
        if not isinstance(args, dict) or set(args) != {"code"} or not isinstance(args["code"], str):
            return None
        wrapper = re.fullmatch(r'\s*\{\s*"code"\s*:\s*"(.*)"\s*\}\s*', body, re.DOTALL)
        if wrapper is None:
            return None
        valid = set()
        for code in (args["code"], wrapper.group(1)):
            try:
                compile(code, "<tool_argument_validation>", "exec")
            except (SyntaxError, ValueError, OverflowError):
                continue
            valid.add(code)
        return {"code": valid.pop()} if len(valid) == 1 else None

    @staticmethod
    def _parse_tool_tag(text: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call from model output.

        Supports two formats:
        1. ASM format:  <tool_name>{...json...}</tool_name>
        2. Fallback format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

        Returns {"name": str, "arguments": dict} or None.
        """
        text = SchemaMatchingAgent._action_text(text)
        # --- ASM format: <tool_name>{...}</tool_name> ---
        for tag in ALL_TOOL_TAGS:
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                body = m.group(1).strip()
                try:
                    args = json.loads(body)
                    if isinstance(args, list):
                        args = {"ranked_candidates": args} if tag == "submit" else {}
                    elif not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    if tag == "execute_python":
                        recovered = SchemaMatchingAgent._recover_python_json_body(body)
                        if recovered is not None:
                            return {"name": tag, "arguments": recovered,
                                    "format_recovery": ("redundant_argument_closing_tag" if body.endswith("</arg_value>}")
                                                        else "redundant_code_closing_tag_and_angle" if body.endswith("</code>>")
                                                        else "redundant_json_closing_brace_and_angle" if body.endswith("}>")
                                                        else "redundant_code_closing_tag_and_json_brace" if body.endswith("</code>}")
                                                        else "redundant_code_closing_tag" if body.endswith("</code>")
                                                        else "redundant_json_closing_quote_and_brace" if body.endswith('"}')
                                                        else "redundant_json_closing_brace")}
                        recovered = SchemaMatchingAgent._recover_multiline_python_json(body)
                        if recovered is not None:
                            return {"name": tag, "arguments": recovered,
                                    "format_recovery": "literal_multiline_code_wrapper"}
                        args = {"code": body}
                    else:
                        args = SchemaMatchingAgent._parse_xml_args(body)
                        if not args and body:
                            if tag == "submit":
                                args = {"ranked_candidates": body}
                            elif tag == "doc_query":
                                args = {"query": body}
                            elif tag == "get_sample_data":
                                args = {"column": body}
                return {"name": tag, "arguments": args}

        # --- Legacy format: <tool_call>{"name": ..., "arguments": ...}</tool_call> ---
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
            try:
                data = json.loads(json_str)
                if "name" in data:
                    return {
                        "name": data["name"],
                        "arguments": data.get("arguments", {}),
                    }
            except json.JSONDecodeError:
                pass

        # --- Fallback: bare JSON with "name" field ---
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        m = re.search(json_pattern, text)
        if m:
            try:
                data = json.loads(m.group(0))
                if "name" in data:
                    return {"name": data["name"], "arguments": data.get("arguments", {})}
            except json.JSONDecodeError:
                pass

        # --- Fallback: find first balanced JSON object ---
        try:
            start = text.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            data = json.loads(text[start : i + 1])
                            if "name" in data:
                                return {
                                    "name": data["name"],
                                    "arguments": data.get("arguments", {}),
                                }
                            break
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def reset(self):
        self.messages = []
        self.trajectory = []

    def run(self, source_column: str) -> MatchResult:
        """Run one schema matching task and return the result."""
        self.reset()

        user_prompt = get_user_task_prompt(
            source_column=source_column,
            source_df=self.env._source_df,
            target_df=self.env._target_df,
        )

        self.messages = [
            {"role": "system", "content": (match_score_prompt(get_system_prompt(), "llm") if getattr(self.env, "match_score_mode", "rank") == "llm" else get_system_prompt())},
            {"role": "user", "content": user_prompt},
        ]

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"Task: match source column '{source_column}'")
            print(f"{'=' * 60}")

        round_count = 0
        truncation_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        run_start_time = time.time()

        while round_count < self.config.max_rounds:
            round_count += 1
            step = TrajectoryStep(step=round_count, timestamp=time.time())

            if self.config.verbose:
                print(f"\n--- Round {round_count} ---")

            # Call LLM
            try:
                chat_result = self.llm.chat(
                    messages=self.messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            except Exception as e:
                if self.config.verbose:
                    print("LLM call failed; check provider configuration.")
                result = self.env.evaluate()
                result.error = "LLM call failed; check provider configuration."
                result.termination_reason = "provider_error"
                result.num_rounds = round_count
                result.truncation_count = truncation_count
                result.prompt_tokens = total_prompt_tokens
                result.completion_tokens = total_completion_tokens
                result.total_tokens = total_tokens
                result.elapsed_seconds = round(time.time() - run_start_time, 2)
                result.conversation_history = self.messages.copy()
                result.trajectory = [s.to_dict() for s in self.trajectory]
                return result

            response = chat_result.content
            finish_reason = chat_result.finish_reason
            step.raw_response = response

            step.prompt_tokens = chat_result.prompt_tokens
            step.completion_tokens = chat_result.completion_tokens
            step.total_tokens = chat_result.total_tokens
            total_prompt_tokens += chat_result.prompt_tokens
            total_completion_tokens += chat_result.completion_tokens
            total_tokens += chat_result.total_tokens

            self.messages.append({"role": "assistant", "content": response})

            if finish_reason == "length":
                truncation_count += 1

            if self.config.verbose:
                display = response[:500] + "..." if len(response) > 500 else response
                print(f"Assistant: {display}")
                if finish_reason == "length":
                    print(f"  [truncated, count={truncation_count}]")

            # --- Parse cognitive tag ---
            cog = self._parse_cognitive_tag(response)
            if cog:
                step.cognitive_tag = cog["tag"]
                step.cognitive_content = cog["content"][:500]
                if self.config.verbose:
                    print(f"  Cognitive: <{cog['tag']}>")
            else:
                step.format_violation = "missing_cognitive_tag"

            # --- Parse tool tag ---
            tool_tags_found = self._find_tool_tags(response)
            if len(tool_tags_found) > 1:
                step.format_violation = "multiple_tool_tags"
                step.tool_success = False
                self.trajectory.append(step)
                prompt_msg = (
                    "Your previous response contained multiple tool tags "
                    f"({', '.join(tool_tags_found)}). Use exactly ONE tool tag per response. "
                    "Choose the single next action and resend with one cognitive tag followed by one tool call."
                )
                self.messages.append({"role": "user", "content": prompt_msg})
                if self.config.verbose:
                    print(f"  Multiple tool tags detected: {tool_tags_found}")
                continue

            tool_call = self._parse_tool_tag(response)

            if tool_call:
                step.format_recovery = tool_call.get("format_recovery")
                tool_name = tool_call["name"]
                arguments = tool_call["arguments"]
                step.tool_name = tool_name
                step.tool_args = arguments
                step.tool_cost = TOOL_COSTS.get(tool_name, "unknown")

                if self.config.verbose:
                    print(f"  Tool: {tool_name} [cost: {step.tool_cost}]")
                    if tool_name == "execute_python":
                        code = arguments.get("code", "")
                        code_display = code[:200] + "..." if len(code) > 200 else code
                        print(f"    code: {code_display}")

                # Execute tool
                tool_result = self.env.execute_tool(
                    tool_name,
                    arguments,
                    round_index=round_count,
                )
                step.tool_result = tool_result
                step.tool_success = not bool(tool_result.get("error"))

                if self.config.verbose:
                    result_str = json.dumps(tool_result, ensure_ascii=False)
                    if len(result_str) > 300:
                        result_str = result_str[:300] + "..."
                    print(f"    result: {result_str}")

                # Build observation message (strip internal metadata)
                obs = {k: v for k, v in tool_result.items() if not k.startswith("_")}
                self.messages.append({
                    "role": "user",
                    "content": f"Tool result:\n{json.dumps(obs, ensure_ascii=False, indent=2)}",
                })

                # Record trajectory step
                self.trajectory.append(step)

                # Check termination
                if tool_name in SUBMIT_TOOLS and tool_result.get("success", False):
                    result = self.env.evaluate()
                    result.termination_reason = "submitted"
                    result.num_rounds = round_count
                    result.truncation_count = truncation_count
                    result.prompt_tokens = total_prompt_tokens
                    result.completion_tokens = total_completion_tokens
                    result.total_tokens = total_tokens
                    result.elapsed_seconds = round(time.time() - run_start_time, 2)
                    result.conversation_history = self.messages.copy()
                    result.trajectory = [s.to_dict() for s in self.trajectory]

                    if self.config.verbose:
                        print(f"  Predicted: {result.predicted_column}")
                        if result.predicted_candidates:
                            print(f"  Candidates: {len(result.predicted_candidates)}")
                    return result
            else:
                # No tool call found
                self.trajectory.append(step)

                if self.config.verbose:
                    print("  No tool call detected, prompting...")

                if finish_reason == "length":
                    prompt_msg = (
                        "Your previous response was truncated due to length limits. "
                        "Please be more concise. Wrap your reasoning in a cognitive tag "
                        "(e.g., <PLAN>...</PLAN>) and call a tool."
                    )
                else:
                    prompt_msg = (
                        "Please structure your response with a cognitive tag "
                        "(e.g., <PLAN>, <SHORTLIST>, <DEEPEN>, <VERIFY>, or <ANSWER>) "
                        "followed by a tool call. If you are done, use <ANSWER>...</ANSWER> "
                        "with a <submit>...</submit> tool call."
                    )

                self.messages.append({"role": "user", "content": prompt_msg})

        # Exceeded max rounds
        if self.config.verbose:
            print(f"\nMax rounds reached ({self.config.max_rounds})")

        result = self.env.evaluate()
        result.num_rounds = round_count
        result.truncation_count = truncation_count
        result.prompt_tokens = total_prompt_tokens
        result.completion_tokens = total_completion_tokens
        result.total_tokens = total_tokens
        result.elapsed_seconds = round(time.time() - run_start_time, 2)
        result.conversation_history = self.messages.copy()
        result.trajectory = [s.to_dict() for s in self.trajectory]
        result = self.env.finalize_budget(result)
        return result

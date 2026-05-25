# -*- coding: utf-8 -*-
"""Generic StageRunner: a mini-ReAct engine for a single pipeline stage.

Each stage gets its own system prompt, tool subset, and max rounds.
The StageRunner handles:
- Message construction (system + image + input_context + recent rounds)
- LLM calling with retry
- Action parsing (tool_call or output)
- Tool execution
- Output schema validation (pydantic)
- Force-output when max rounds exhausted
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from src.orchestrator.llm_backend import LLMBackend, LLMResponse
from src.tools.base import BaseTool
from src.tools.vision_utils import image_to_data_url


@dataclass
class StageStep:
    """A single step within a stage's execution."""

    round: int = 0
    thought: str = ""
    action_type: str = ""  # "tool_call"|"output"|"format_error"
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    output: Optional[Dict[str, Any]] = None
    tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})


class StageRunner:
    """Generic mini-ReAct engine for a single pipeline stage.

    Simpler than HarnessAgentLoop:
    - No AgentState (lightweight)
    - No phase transitions
    - No ToolLayer (dedup handled at orchestrator level)
    - Pydantic schema validation for output
    - Sliding window: system + image + input_context + last N rounds
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str,
        tools: List[BaseTool],
        output_schema: Optional[Type[BaseModel]] = None,
        max_rounds: int = 5,
        image_path: str = "",
        stage_name: str = "",
        recent_rounds_to_keep: int = 2,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.tools_list = tools
        self.output_schema = output_schema
        self.max_rounds = max_rounds
        self.image_path = image_path
        self.stage_name = stage_name
        self.recent_rounds_to_keep = recent_rounds_to_keep

    async def run(self, input_context: str) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run the mini-ReAct loop for this stage.

        Args:
            input_context: Text context from previous stages (e.g., PerceptionReport JSON).

        Returns:
            (parsed_output, steps) — parsed_output is None if stage failed to produce valid output.
        """
        steps: List[StageStep] = []
        messages: List[Dict[str, Any]] = []

        # Build initial messages
        system_msg = {"role": "system", "content": self._build_system_content()}
        user_msg = self._build_user_message(input_context)

        # Shadow history for sliding window
        shadow: List[Dict[str, Any]] = [system_msg, user_msg]

        # Rolling summary: accumulated evidence from all tool calls so far
        # Each entry is a 1-line summary of a tool result
        evidence_so_far: List[str] = []

        consecutive_errors = 0

        for round_num in range(1, self.max_rounds + 1):
            # Build messages for this round (with evidence summary injected)
            messages = self._build_round_messages(system_msg, user_msg, shadow, evidence_so_far)

            # Call LLM
            response = await self._call_llm(messages)
            content = response.text

            step = StageStep(
                round=round_num,
                tokens={"prompt": response.prompt_tokens, "completion": response.completion_tokens},
            )

            if not content or not content.strip():
                step.action_type = "format_error"
                step.thought = "(empty response)"
                steps.append(step)
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    break
                continue

            # Parse action
            thought = self._extract_think(content)
            step.thought = thought

            # Check for output — either <output> tags or bare JSON (for no-tool stages)
            if "<output>" in content and "</output>" in content:
                # If tools are available but none have been called yet, reject early output
                tool_calls_so_far = sum(1 for s in steps if s.action_type == "tool_call")
                if self.tools_list and tool_calls_so_far == 0 and round_num <= self.max_rounds - 1:
                    step.action_type = "format_error"
                    step.thought = "(tried to output without calling any tools)"
                    steps.append(step)
                    consecutive_errors += 1
                    shadow.append({"role": "assistant", "content": content})
                    shadow.append({
                        "role": "user",
                        "content": "你必须先使用工具收集信息，不能直接输出结论。请调用工具。",
                    })
                    continue

                output_json = self._extract_output(content)
                if output_json is not None:
                    step.action_type = "output"
                    step.output = output_json
                    steps.append(step)

                    # Validate against schema
                    parsed = self._validate_output(output_json)
                    if parsed is not None:
                        return parsed, steps
                    # Schema validation failed — try to use raw dict
                    # Still return it, orchestrator can handle partial data
                    if self.output_schema:
                        try:
                            return self.output_schema.model_validate(output_json), steps
                        except Exception:
                            pass
                    return None, steps

                step.action_type = "format_error"
                steps.append(step)
                consecutive_errors += 1
                shadow.append({"role": "assistant", "content": content})
                shadow.append({"role": "user", "content": "Output JSON 格式错误，请重新输出。"})
                continue

            # Check for tool_call
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_name, tool_args = self._parse_tool_call(content)
                if tool_name and tool_name in self.tools:
                    step.action_type = "tool_call"
                    step.tool_name = tool_name
                    step.tool_args = tool_args

                    # Execute tool
                    result = await self._execute_tool(tool_name, tool_args)
                    step.tool_result = result

                    steps.append(step)
                    consecutive_errors = 0

                    # Extract 1-line summary and add to rolling evidence
                    summary = self._summarize_tool_result(tool_name, tool_args, result)
                    if summary:
                        evidence_so_far.append(summary)

                    # Add to shadow history
                    shadow.append({"role": "assistant", "content": content})
                    shadow.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})
                    continue
                else:
                    # Unknown tool
                    step.action_type = "format_error"
                    step.thought = f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
                    steps.append(step)
                    shadow.append({"role": "assistant", "content": content})
                    shadow.append({
                        "role": "user",
                        "content": f"工具 '{tool_name}' 不存在。可用工具: {list(self.tools.keys())}",
                    })
                    consecutive_errors += 1
                    continue

            # Neither output nor tool_call — try bare JSON (for no-tool stages)
            if not self.tools_list:
                bare_json = self._try_parse_bare_json(content)
                if bare_json is not None:
                    step.action_type = "output"
                    step.output = bare_json
                    steps.append(step)
                    parsed = self._validate_output(bare_json)
                    if parsed is not None:
                        return parsed, steps
                    return None, steps

            # Format error
            step.action_type = "format_error"
            steps.append(step)
            consecutive_errors += 1
            shadow.append({"role": "assistant", "content": content})
            shadow.append({
                "role": "user",
                "content": "格式错误。请输出 <tool_call>...</tool_call> 或 <output>...</output>。",
            })

            if consecutive_errors >= 3:
                break

        # Max rounds exhausted — force output
        forced = await self._force_output(system_msg, user_msg, shadow, evidence_so_far)
        if forced is not None:
            steps.append(StageStep(
                round=len(steps) + 1,
                action_type="output",
                thought="(forced output)",
                output=forced.model_dump() if isinstance(forced, BaseModel) else forced,
            ))
            return forced, steps

        return None, steps

    def _build_system_content(self) -> str:
        """Build system prompt with tool descriptions."""
        if not self.tools_list:
            return self.system_prompt

        tools_desc = "\n\n## Available Tools\n\n"
        for tool in self.tools_list:
            tools_desc += f"### {tool.name}\n{tool.description}\n"
            props = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            if props:
                tools_desc += "Parameters:\n"
                for pname, pinfo in props.items():
                    req_mark = " (required)" if pname in required else ""
                    tools_desc += f"  - {pname}: {pinfo.get('description', '')}{req_mark}\n"
            tools_desc += "\n"

        output_format = """
## Output Format

Each round output exactly ONE of:
(1) <think>your reasoning</think>
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
OR
(2) <think>your reasoning</think>
    <output>{...JSON matching the required schema...}</output>

IMPORTANT: Only ONE action per round. Always include <think> first.
"""
        return self.system_prompt + tools_desc + output_format

    def _build_user_message(self, input_context: str) -> Dict[str, Any]:
        """Build the initial user message with image and context."""
        content_parts: List[Dict[str, Any]] = []

        # Add image if available and file exists
        if self.image_path:
            try:
                data_url = image_to_data_url(self.image_path)
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            except (FileNotFoundError, OSError):
                pass  # Skip image if not accessible

        # Add input context
        content_parts.append({"type": "text", "text": input_context})

        return {"role": "user", "content": content_parts}

    def _build_round_messages(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        shadow: List[Dict[str, Any]],
        evidence_so_far: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for this round with sliding window + evidence summary.

        Structure:
        [system]
        [user: image + input_context]
        [assistant: evidence summary (if any)]  ← injected rolling summary
        [recent N rounds: assistant + user pairs]
        """
        messages = [system_msg, user_msg]

        # Inject evidence summary as an assistant message (so model sees accumulated knowledge)
        if evidence_so_far:
            summary_text = "## Evidence collected so far\n" + "\n".join(
                f"  {i+1}. {e}" for i, e in enumerate(evidence_so_far)
            )
            messages.append({"role": "assistant", "content": f"<think>Let me review what I've found so far.</think>\n{summary_text}"})
            messages.append({"role": "user", "content": "继续验证。根据已收集的证据，决定下一步行动。"})

        # Add recent rounds from shadow (skip system and initial user)
        recent = shadow[2:]  # Everything after system + user
        if len(recent) > self.recent_rounds_to_keep * 2:
            recent = recent[-(self.recent_rounds_to_keep * 2):]

        # Ensure recent starts with assistant (not user) to avoid consecutive user messages
        while recent and recent[0].get("role") != "assistant":
            recent = recent[1:]

        messages.extend(recent)
        return messages

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """Call LLM with retry on empty response."""
        response = await self.llm.get_response(messages)
        if not response.text or not response.text.strip():
            import asyncio
            await asyncio.sleep(2)
            response = await self.llm.get_response(messages)
        return response

    def _extract_think(self, content: str) -> str:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_output(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from <output> block."""
        match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        if not match:
            return None
        text = match.group(1).strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import json5
                return json5.loads(text)
            except Exception:
                return None

    def _parse_tool_call(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Extract tool name and args from <tool_call> block."""
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not match:
            return "", {}
        text = match.group(1).strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed.get("name", ""), parsed.get("arguments", {})
        except json.JSONDecodeError:
            return "", {}

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool and return result as string.

        Handles both sync and async tools:
        - If tool has call_async(), await it directly
        - Otherwise run sync call() in a thread to avoid blocking the event loop
        """
        tool = self.tools[tool_name]

        # Inject image_input if the tool needs it and it's not a valid path
        if "image_input" in tool.parameters.get("properties", {}):
            provided = tool_args.get("image_input", "")
            # Override if empty, or if it doesn't look like a real file path
            if not provided or ("/" not in provided and "\\" not in provided):
                tool_args["image_input"] = self.image_path

        # Some tools (visual_anomaly, compare_reference) use self.image_path attribute
        if hasattr(tool, "image_path") and self.image_path:
            tool.image_path = self.image_path

        try:
            if hasattr(tool, "call_async"):
                # Async tool — await directly
                result = await tool.call_async(tool_args)
            else:
                # Sync tool — run in thread to avoid blocking event loop
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, tool_args)

            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    def _validate_output(self, output_json: Dict[str, Any]) -> Optional[BaseModel]:
        """Validate output against pydantic schema."""
        if self.output_schema is None:
            return None
        try:
            return self.output_schema.model_validate(output_json)
        except Exception:
            return None

    def _try_parse_bare_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to parse bare JSON from content (for no-tool stages that skip tags).

        Handles cases where the model outputs JSON directly without <output> tags,
        possibly wrapped in markdown fences or preceded by thinking text.
        """
        # Strip thinking tags if present
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

        # Try to find a JSON object in the text
        # Look for the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                import json5
                return json5.loads(json_str)
            except Exception:
                return None

    def _summarize_tool_result(self, tool_name: str, tool_args: Dict[str, Any], result: str) -> str:
        """Extract a 1-line summary from a tool result (rule-based, no LLM call).

        This feeds the rolling evidence summary so the model remembers
        what it found in earlier rounds without needing the full raw output.
        """
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            data = None

        # --- Search tools: extract top result ---
        if tool_name in ("text_search", "news_search"):
            query = tool_args.get("query", "")
            if data and isinstance(data, dict):
                results = data.get("results", data.get("organic", []))
                if isinstance(results, list) and results:
                    first = results[0]
                    snippet = first.get("snippet", first.get("title", ""))[:100]
                    return f"[{tool_name}] query='{query}' → {snippet}"
                elif data.get("status") == "error":
                    return f"[{tool_name}] query='{query}' → error: {data.get('error', 'unknown')}"
            return f"[{tool_name}] query='{query}' → no results"

        # --- Reverse image search ---
        if tool_name == "reverse_image_search":
            if data and isinstance(data, dict):
                matches = data.get("results", data.get("matches", []))
                if isinstance(matches, list) and matches:
                    first = matches[0]
                    title = first.get("title", first.get("source", ""))[:80]
                    return f"[reverse_image_search] → found: {title}"
            return f"[reverse_image_search] → no matches"

        # --- Visit ---
        if tool_name == "visit":
            url = tool_args.get("url", "")[:50]
            content_preview = result[:100] if result else "empty"
            return f"[visit] {url} → {content_preview}"

        # --- VLM analysis tools ---
        if tool_name == "check_consistency":
            aspect = tool_args.get("aspect", "all")
            if data and isinstance(data, dict):
                consistent = data.get("consistent", True)
                details = data.get("details", "")[:80]
                status = "consistent" if consistent else "INCONSISTENT"
                return f"[check_consistency:{aspect}] → {status}. {details}"
            return f"[check_consistency:{aspect}] → unknown"

        if tool_name == "analyze_visual_anomalies":
            if data and isinstance(data, dict):
                auth = data.get("overall_authenticity", "uncertain")
                anomalies = data.get("anomalies", [])
                n = len(anomalies) if isinstance(anomalies, list) else 0
                return f"[visual_anomalies] → {auth}, {n} anomalies found"
            # String result (from call_async)
            if "authenticity" in result.lower():
                return f"[visual_anomalies] → {result[:100]}"
            return f"[visual_anomalies] → analysis done"

        if tool_name == "crop_and_inspect":
            question = tool_args.get("focus_question", "")[:40]
            if data and isinstance(data, dict):
                answer = data.get("answer", "")[:80]
                return f"[crop_and_inspect] '{question}' → {answer}"
            return f"[crop_and_inspect] '{question}' → done"

        if tool_name == "count_objects":
            target = tool_args.get("target_object", "")
            if data and isinstance(data, dict):
                count = data.get("count", "?")
                conf = data.get("confidence", 0)
                return f"[count_objects] '{target}' → count={count} (conf={conf:.1f})"
            return f"[count_objects] '{target}' → done"

        if tool_name == "compare_with_reference":
            if data and isinstance(data, dict):
                diffs = data.get("differences", [])
                n = len(diffs) if isinstance(diffs, list) else 0
                return f"[compare_reference] → {n} differences found"
            return f"[compare_reference] → comparison done"

        if tool_name == "verify_face_identity":
            if data and isinstance(data, dict):
                verdict = data.get("verdict", "unknown")
                sim = data.get("similarity", 0)
                return f"[verify_face_identity] → {verdict} (similarity={sim})"
            return f"[verify_face_identity] → done"

        # --- Perception tools ---
        if tool_name == "perceive_scene":
            if data and isinstance(data, dict):
                n_ent = data.get("total_entities", 0)
                scene = data.get("scene_description", "")[:60]
                return f"[perceive_scene] → {n_ent} entities. {scene}"
            return f"[perceive_scene] → done"

        if tool_name == "ocr_with_position":
            if data and isinstance(data, dict):
                n = data.get("total_regions", 0)
                full_text = data.get("full_text", "")[:60]
                return f"[ocr] → {n} regions. text: {full_text}"
            return f"[ocr] → done"

        if tool_name == "face_detect":
            if data and isinstance(data, dict):
                n = data.get("total_faces", 0)
                return f"[face_detect] → {n} faces detected"
            return f"[face_detect] → done"

        # Fallback
        return f"[{tool_name}] → completed"

    async def _force_output(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        shadow: List[Dict[str, Any]],
        evidence_so_far: Optional[List[str]] = None,
    ) -> Optional[BaseModel]:
        """Force the model to produce output when rounds are exhausted."""
        # Include evidence summary so the model has full context for final output
        evidence_block = ""
        if evidence_so_far:
            evidence_block = (
                "\n\n## Evidence collected so far\n"
                + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(evidence_so_far))
                + "\n\n"
            )

        force_prompt = (
            f"{evidence_block}"
            "你已经没有更多工具调用机会了。请立即输出 <output>...</output>，"
            "基于目前已收集的信息给出最佳结果。如果信息不足，在相应字段标注不确定。"
        )

        messages = [system_msg, user_msg]
        # Add last few rounds for context
        recent = shadow[2:]
        if len(recent) > 4:
            recent = recent[-4:]
        messages.extend(recent)
        messages.append({"role": "user", "content": force_prompt})

        response = await self.llm.get_response(messages)
        if response.text:
            output_json = self._extract_output(response.text)
            if output_json is None:
                output_json = self._try_parse_bare_json(response.text)
            if output_json and self.output_schema:
                try:
                    return self.output_schema.model_validate(output_json)
                except Exception:
                    pass
        return None

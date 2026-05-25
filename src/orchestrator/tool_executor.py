# -*- coding: utf-8 -*-
"""Async tool executor and registry for ReAct agent.

Wraps existing sync tools from src/tools/ into an async interface,
and provides a ToolRegistry for the agent to discover and call tools.
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class AsyncTool(ABC):
    """Base class for async tools (mirrors Gen-DeepResearch's DeepResearchTool)."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class SyncToolWrapper(AsyncTool):
    """Wraps an existing sync tool (from src/tools/) into async interface.

    The sync tool is expected to have an `invoke(**kwargs)` method that returns
    a dict with 'status', 'tool_output', 'error' keys.
    """

    # Tools whose results benefit from reranking
    RERANK_TOOLS = ("text_search", "news_search")

    def __init__(self, sync_tool: Any, executor: ThreadPoolExecutor, rerank_client: Optional[Any] = None, rerank_top_k: int = 5):
        self.sync_tool = sync_tool
        self.executor = executor
        self.rerank_client = rerank_client
        self.rerank_top_k = rerank_top_k
        self.name = getattr(sync_tool, "name", "unknown")
        self.description = getattr(sync_tool, "description", "")
        self.parameters = getattr(sync_tool, "parameters", {"type": "object", "properties": {}})

    async def call(self, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.executor, lambda: self._invoke_sync(kwargs)
        )
        return result

    def _invoke_sync(self, kwargs: Dict[str, Any]) -> str:
        """Call the underlying sync tool and format the result as string."""
        try:
            # The existing tools have a `call(params: dict)` method
            if hasattr(self.sync_tool, "call"):
                result = self.sync_tool.call(kwargs)
            elif hasattr(self.sync_tool, "invoke"):
                result = self.sync_tool.invoke(**kwargs)
            elif hasattr(self.sync_tool, "run"):
                result = self.sync_tool.run(**kwargs)
            elif callable(self.sync_tool):
                result = self.sync_tool(**kwargs)
            else:
                return f"Error: Tool '{self.name}' has no callable method"

            # Apply rerank for search tools before formatting
            if self.rerank_client and self.name in self.RERANK_TOOLS:
                result = self._apply_rerank(result, kwargs)

            if isinstance(result, dict):
                if result.get("status") == "success":
                    output = result.get("tool_output", "")
                    return self._format_output(output)
                elif "error" in result:
                    return f"Error: {result.get('error', 'Unknown error')}"
                else:
                    return self._format_output(result)
            elif isinstance(result, str):
                return result
            elif isinstance(result, list):
                return self._format_output(result)
            else:
                return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"

    def _apply_rerank(self, result: Any, kwargs: Dict[str, Any]) -> Any:
        """Apply reranking to search results.

        Extracts query from kwargs, builds documents from result snippets,
        reranks, and returns only top-k results.
        """
        # Extract query for reranking
        queries = kwargs.get("queries", [])
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return result

        try:
            if isinstance(result, list):
                # Multiple search responses (one per query)
                reranked = []
                for i, item in enumerate(result):
                    query = queries[i] if i < len(queries) else queries[0]
                    reranked.append(self._rerank_single_response(item, query))
                return reranked
            elif isinstance(result, dict) and "results" in result:
                query = queries[0]
                return self._rerank_single_response(result, query)
        except Exception:
            # If rerank fails, return original results
            pass
        return result

    def _rerank_single_response(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Rerank a single search response dict."""
        results = response.get("results", [])
        if not results or len(results) <= self.rerank_top_k:
            return response

        # Build documents from snippets
        documents = []
        for r in results:
            doc = f"{r.get('title', '')}. {r.get('snippet', '')}"
            documents.append(doc)

        # Rerank
        ranked = self.rerank_client.rerank(query=query, documents=documents, top_k=self.rerank_top_k)

        # Reorder results based on rerank scores
        reranked_results = [results[r["index"]] for r in ranked if r["index"] < len(results)]

        # Return modified response
        reranked_response = dict(response)
        reranked_response["results"] = reranked_results
        reranked_response["reranked"] = True
        reranked_response["rerank_top_k"] = self.rerank_top_k
        return reranked_response

    def _format_output(self, output: Any) -> str:
        """Format tool output into a readable string for the agent."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # Handle hybrid reverse image search results
            if "lens_results" in output or "semantic_results" in output:
                return self._format_reverse_search(output)
            # Handle single search response dict (with "results" key)
            if "results" in output:
                return self._format_search_response(output)
            # Generic dict
            return json.dumps(output, ensure_ascii=False, default=str)[:3000]
        if isinstance(output, list):
            # Search results list — each item is a search response dict
            formatted_parts = []
            for item in output:
                if isinstance(item, dict):
                    formatted_parts.append(self._format_search_response(item))
                else:
                    formatted_parts.append(str(item))
            return "\n\n".join(formatted_parts) if formatted_parts else "(no results)"
        return json.dumps(output, ensure_ascii=False, default=str)[:3000]

    def _format_reverse_search(self, output: dict) -> str:
        """Format hybrid reverse image search results (Lens + semantic)."""
        parts = []

        # Lens results (true visual reverse search)
        lens_results = output.get("lens_results", [])
        lens_error = output.get("lens_error")
        if lens_error:
            parts.append(f"[Lens Error] {lens_error}")
        elif lens_results:
            parts.append("=== Google Lens (Visual Match) ===")
            for i, r in enumerate(lens_results[:8], 1):
                if isinstance(r, dict):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    url = r.get("url", "")
                    source = r.get("source", "")
                    image_url = r.get("image_url", "")
                    line = f"[Lens-{i}] {title}"
                    if source:
                        line += f" - {source}"
                    if snippet:
                        line += f"\n    {snippet}"
                    if url:
                        line += f"\n    URL: {url}"
                    if image_url:
                        line += f"\n    Image: {image_url}"
                    parts.append(line)
        else:
            parts.append("[Lens] No visual matches found.")

        # Semantic results (VLM query → image search)
        vlm_query = output.get("vlm_query", "")
        semantic_results = output.get("semantic_results", [])
        vlm_error = output.get("vlm_error")
        if vlm_error:
            parts.append(f"\n[VLM Query Error] {vlm_error}")
        elif semantic_results:
            parts.append(f"\n=== Semantic Search (query: \"{vlm_query}\") ===")
            for i, r in enumerate(semantic_results[:5], 1):
                if isinstance(r, dict):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    image_url = r.get("image_url", "")
                    source = r.get("source", "")
                    line = f"[Sem-{i}] {title}"
                    if source:
                        line += f" - {source}"
                    if url:
                        line += f"\n    URL: {url}"
                    if image_url:
                        line += f"\n    Image: {image_url}"
                    parts.append(line)

        return "\n\n".join(parts) if parts else "(no reverse search results)"

    def _format_search_response(self, item: dict) -> str:
        """Format a single search response dict (text/news search result)."""
        formatted_parts = []

        # Handle answer_box (direct answer from Google)
        answer_box = item.get("answer_box")
        if answer_box and isinstance(answer_box, dict):
            ab_title = answer_box.get("title", "")
            ab_answer = answer_box.get("answer", "") or answer_box.get("snippet", "")
            if ab_answer:
                formatted_parts.append(f"[Direct Answer] {ab_title}: {ab_answer}")

        # Handle knowledge_graph
        kg = item.get("knowledge_graph")
        if kg and isinstance(kg, dict):
            kg_title = kg.get("title", "")
            kg_type = kg.get("type", "")
            kg_desc = kg.get("description", "")
            if kg_title:
                kg_line = f"[Knowledge Graph] {kg_title}"
                if kg_type:
                    kg_line += f" ({kg_type})"
                if kg_desc:
                    kg_line += f": {kg_desc}"
                formatted_parts.append(kg_line)

        # Handle organic results
        results = item.get("results", [item])
        for r in results[:10]:
            if isinstance(r, dict) and r.get("title"):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                url = r.get("url", "")
                date = r.get("date", "")
                source = r.get("source", "")
                image_url = r.get("image_url", "")
                line = f"[{len(formatted_parts)+1}] {title}"
                if date:
                    line += f" ({date})"
                if source:
                    line += f" - {source}"
                if snippet:
                    line += f"\n    {snippet}"
                if url:
                    line += f"\n    URL: {url}"
                if image_url:
                    line += f"\n    Image: {image_url}"
                formatted_parts.append(line)

        return "\n\n".join(formatted_parts) if formatted_parts else "(no results)"


class ToolRegistry:
    """Registry of async tools with execution and schema generation."""

    def __init__(self, max_workers: int = 8, default_timeout: float = 60.0, rerank_client: Optional[Any] = None, rerank_top_k: int = 5):
        self.tools: Dict[str, AsyncTool] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout
        self.rerank_client = rerank_client
        self.rerank_top_k = rerank_top_k

    def register(self, tool: AsyncTool) -> None:
        """Register an async tool."""
        self.tools[tool.name] = tool

    def wrap_and_register(self, sync_tool: Any) -> None:
        """Wrap a sync tool and register it."""
        wrapped = SyncToolWrapper(sync_tool, self.executor, rerank_client=self.rerank_client, rerank_top_k=self.rerank_top_k)
        self.register(wrapped)

    async def execute(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with timeout."""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"
        try:
            return await asyncio.wait_for(
                tool.call(**kwargs), timeout=self.default_timeout
            )
        except asyncio.TimeoutError:
            return f"Error: Tool '{tool_name}' timed out after {self.default_timeout}s"
        except Exception as e:
            return f"Error executing '{tool_name}': {type(e).__name__}: {str(e)}"

    def get_tool_schemas_xml(self) -> str:
        """Generate <tools> XML block for the system prompt."""
        schemas = []
        for tool in self.tools.values():
            schemas.append(json.dumps(tool.get_schema(), ensure_ascii=False))
        return "<tools>\n" + "\n".join(schemas) + "\n</tools>"

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=False)

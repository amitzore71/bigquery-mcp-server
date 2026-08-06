"""Gemini chat with tool-calling protocol (ported from client/app.py)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional

from bqsaas.ai.prompts import FOLLOW_UP_PROMPT, SYSTEM_PROMPT, tools_description_text
from bqsaas.config import get_settings
from bqsaas.mcp.tools import TOOL_SPECS, call_tool

logger = logging.getLogger(__name__)


def _get_gemini_client(api_key: Optional[str] = None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or get_settings().gemini_api_key
    if not key:
        return None
    try:
        from google import genai

        return genai.Client(api_key=key)
    except Exception as e:
        logger.error("Failed to create Gemini client: %s", e)
        return None


def extract_response_text(response) -> Optional[str]:
    try:
        if hasattr(response, "text"):
            return response.text
    except (ValueError, AttributeError):
        pass

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = candidate.content.parts
                if parts:
                    return "".join(
                        p.text for p in parts if hasattr(p, "text") and p.text
                    )
    except (ValueError, AttributeError, IndexError):
        pass

    return None


def extract_tool_call(message: str) -> Optional[dict]:
    patterns = [
        (r"```tool_call\s*\n?(.*?)\n?```", 1),
        (r"```json\s*\n?(.*?)\n?```", 1),
        (r"```\s*\n?(.*?)\n?```", 1),
        (r'(\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\})', 1),
    ]

    for pattern, group in patterns:
        match = re.search(pattern, message, re.DOTALL)
        if match:
            candidate = match.group(group).strip()
            if '"tool"' in candidate and '"arguments"' in candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    return None


def process_with_gemini(
    user_message: str,
    *,
    history: Optional[list[dict[str, str]]] = None,
    client: Any = None,
    project_id: str = "",
    dataset_id: str = "",
    tool_executor: Optional[Callable[[str, dict], dict]] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run one turn of Gemini + optional tool call.

    Returns::
        {
          "status": "success"|"error",
          "content": str,          # assistant reply
          "tool_call": dict|None,
          "tool_result": dict|None,
        }
    """
    settings = get_settings()
    gemini = _get_gemini_client(api_key)
    if gemini is None:
        return {
            "status": "error",
            "content": (
                "Gemini API key not configured. Set GEMINI_API_KEY environment "
                "variable or gemini_api_key in settings."
            ),
            "tool_call": None,
            "tool_result": None,
        }

    model = model_id or settings.gemini_model
    history = history or []
    project_id = project_id or settings.gcp_project_id
    dataset_id = dataset_id or settings.dataset_id

    tools_desc = tools_description_text(TOOL_SPECS)
    system = SYSTEM_PROMPT.format(
        tools_description=tools_desc,
        project_id=project_id,
        dataset_id=dataset_id,
    )

    history_text = ""
    for msg in history[-6:]:
        role = msg.get("role", "user").title()
        history_text += f"\n{role}: {msg.get('content', '')}"

    full_prompt = system
    if history_text:
        full_prompt += f"\n\nPrevious conversation:{history_text}"
    full_prompt += f"\n\nUser: {user_message}"

    def default_executor(name: str, arguments: dict) -> dict:
        if client is None:
            return {
                "status": "error",
                "message": "No BigQuery client available for tool execution",
            }
        return call_tool(name, arguments, client, project_id, dataset_id)

    executor = tool_executor or default_executor

    try:
        response = gemini.models.generate_content(model=model, contents=full_prompt)
        assistant_message = extract_response_text(response)
        if not assistant_message:
            return {
                "status": "error",
                "content": "I couldn't process that request. Please try rephrasing.",
                "tool_call": None,
                "tool_result": None,
            }

        tool_call = extract_tool_call(assistant_message)
        tool_result = None

        if tool_call:
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments") or {}
            if tool_name:
                tool_result = executor(tool_name, arguments)
                follow_up = FOLLOW_UP_PROMPT.format(
                    tool_result_json=json.dumps(tool_result, indent=2, default=str),
                    user_message=user_message,
                )
                follow_up_response = gemini.models.generate_content(
                    model=model, contents=follow_up
                )
                follow_up_text = extract_response_text(follow_up_response)
                if follow_up_text:
                    assistant_message = follow_up_text
                else:
                    assistant_message = (
                        "I executed the query and received:\n\n```json\n"
                        f"{json.dumps(tool_result, indent=2, default=str)}\n```"
                    )

        return {
            "status": "success",
            "content": assistant_message,
            "tool_call": tool_call,
            "tool_result": tool_result,
        }
    except Exception as e:
        logger.exception("Gemini processing failed")
        return {
            "status": "error",
            "content": f"Error processing request: {e}",
            "tool_call": None,
            "tool_result": None,
        }

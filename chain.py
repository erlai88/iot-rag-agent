"""
Main RAG pipeline.

Flow:
1. Retrieve Top-K document chunks.
2. Ask the LLM with function calling enabled.
3. Execute tool calls if requested.
4. Return the final answer and log the interaction.
"""

from __future__ import annotations

import json
import os
from getpass import getpass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from logger import log_interaction
from retriever import Retriever
from tools import TOOLS_SCHEMA, handle_tool_call


load_dotenv()

TOP_K = 5


def get_llm_model() -> str:
    """Resolve the configured chat model."""
    return os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))


def get_llm_base_url() -> str:
    """Resolve the configured OpenAI-compatible LLM base URL."""
    return os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).strip()


def get_embedding_api_key(api_key: str | None = None) -> str:
    """Get the embedding API key."""
    if api_key:
        return api_key

    env_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if env_key:
        return env_key

    input_key = getpass("Please enter the embedding API key: ").strip()
    if not input_key:
        raise ValueError("Missing embedding API key.")
    return input_key


def get_llm_api_key(api_key: str | None = None) -> str:
    """Get the LLM API key."""
    if api_key:
        return api_key

    env_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if env_key:
        return env_key

    input_key = getpass("Please enter the LLM API key: ").strip()
    if not input_key:
        raise ValueError("Missing LLM API key.")
    return input_key


def build_llm_client(api_key: str) -> OpenAI:
    """Build an OpenAI-compatible LLM client."""
    base_url = get_llm_base_url()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def build_system_prompt() -> str:
    """Create the system prompt."""
    return (
        "You are an IoT support expert. Answer only from the provided documents. "
        "If the documents do not contain the answer, clearly say you do not know. "
        "Do not invent details that are not present in the documents."
    )


def format_context(contexts: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a prompt-friendly block."""
    if not contexts:
        return "No relevant documents were retrieved."

    blocks = []
    for index, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"source: {item.get('source')}",
                    f"page: {item.get('page')}",
                    f"score: {item.get('score')}",
                    f"text: {item.get('text')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Convert SDK tool calls into plain dicts."""
    serialized = []
    for tool_call in tool_calls or []:
        serialized.append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return serialized


def parse_tool_result(tool_output: str) -> dict[str, Any] | None:
    """Parse tool output if it is JSON."""
    try:
        parsed = json.loads(tool_output)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def ask(
    query: str,
    embedding_api_key: str | None = None,
    llm_api_key: str | None = None,
) -> dict[str, Any]:
    """Run one full RAG + tool-calling interaction."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    resolved_embedding_key = get_embedding_api_key(embedding_api_key)
    resolved_llm_key = get_llm_api_key(llm_api_key)

    retriever = Retriever(api_key=resolved_embedding_key)
    client = build_llm_client(resolved_llm_key)

    contexts = retriever.search(query=query, k=TOP_K)
    context_text = format_context(contexts)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": f"Document context:\n{context_text}\n\nUser question:\n{query}",
        },
    ]

    first_response = client.chat.completions.create(
        model=get_llm_model(),
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message

    used_tool = bool(first_message.tool_calls)
    tool_result: dict[str, Any] | None = None

    if used_tool:
        messages.append(
            {
                "role": "assistant",
                "content": first_message.content or "",
                "tool_calls": serialize_tool_calls(first_message.tool_calls),
            }
        )

        for tool_call in first_message.tool_calls or []:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool arguments are not valid JSON: {tool_call.function.arguments}") from exc

            tool_output = handle_tool_call(tool_name, tool_args)
            parsed_output = parse_tool_result(tool_output)
            if parsed_output is not None:
                tool_result = parsed_output

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

        final_response = client.chat.completions.create(
            model=get_llm_model(),
            messages=messages,
        )
        answer = final_response.choices[0].message.content or ""
    else:
        answer = first_message.content or ""

    result = {
        "answer": answer,
        "sources": contexts,
        "used_tool": used_tool,
        "tool_result": tool_result,
    }

    interaction_id = log_interaction(
        query=query,
        answer=answer,
        sources=contexts,
        used_tool=used_tool,
        feedback=None,
        tool_result=tool_result,
    )
    result["interaction_id"] = interaction_id
    return result


def run_agent(
    question: str,
    device_id: str | None = None,
    embedding_api_key: str | None = None,
    llm_api_key: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper."""
    enriched_question = question.strip()
    if device_id:
        enriched_question = f"{enriched_question}\n\nDevice ID: {device_id}"
    return ask(
        enriched_question,
        embedding_api_key=embedding_api_key,
        llm_api_key=llm_api_key,
    )

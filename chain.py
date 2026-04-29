"""
Main RAG pipeline.

Features:
1. Hybrid retrieval with dense search + BM25 reranking
2. Optional tool calling
3. Streaming answer generation
4. Conversation history support
5. User-friendly exception mapping
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from getpass import getpass
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from logger import log_interaction
from retriever import Retriever
from tools import TOOLS_SCHEMA, handle_tool_call


load_dotenv()

TOP_K = 5
MAX_HISTORY_MESSAGES = 6


class UserFacingError(RuntimeError):
    """Raised when a friendly message should be shown to end users."""


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
        raise UserFacingError("Embedding API key is missing.")
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
        raise UserFacingError("LLM API key is missing.")
    return input_key


def build_llm_client(api_key: str) -> OpenAI:
    """Build an OpenAI-compatible LLM client."""
    base_url = get_llm_base_url()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def map_llm_exception(exc: Exception) -> UserFacingError:
    """Map provider exceptions to user-friendly messages."""
    if isinstance(exc, AuthenticationError):
        return UserFacingError("Model provider authentication failed. Please check the deployment secrets.")
    if isinstance(exc, RateLimitError):
        return UserFacingError("The model provider is rate-limiting requests right now. Please try again shortly.")
    if isinstance(exc, APITimeoutError):
        return UserFacingError("The model provider timed out. Please try again in a moment.")
    if isinstance(exc, APIConnectionError):
        return UserFacingError("The app could not reach the model provider. Please try again later.")
    if isinstance(exc, APIStatusError):
        return UserFacingError(f"Model provider returned an error ({exc.status_code}). Please try again later.")
    if isinstance(exc, UserFacingError):
        return exc
    return UserFacingError("The assistant could not finish this request. Please try again.")


def build_system_prompt() -> str:
    """Create the system prompt."""
    return (
        "You are an IoT support expert. Answer only from the provided documents. "
        "If the documents do not contain the answer, clearly say you do not know. "
        "Do not invent details that are not present in the documents. "
        "When useful, cite the document source and page in your answer."
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


def build_user_prompt(query: str, contexts: list[dict[str, Any]]) -> str:
    """Build the user prompt for the current turn."""
    return f"Document context:\n{format_context(contexts)}\n\nUser question:\n{query}"


def normalize_chat_history(chat_history: Iterable[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep only recent user/assistant turns for the model."""
    if not chat_history:
        return []

    normalized = []
    for item in chat_history:
        role = item.get("role")
        content = item.get("content", "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized[-MAX_HISTORY_MESSAGES:]


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


@dataclass
class PreparedInteraction:
    """Prepared state for both blocking and streaming answer generation."""

    client: OpenAI
    model: str
    query: str
    contexts: list[dict[str, Any]]
    final_messages: list[dict[str, Any]]
    used_tool: bool
    tool_result: dict[str, Any] | None


def prepare_interaction(
    query: str,
    embedding_api_key: str | None = None,
    llm_api_key: str | None = None,
    chat_history: Iterable[dict[str, str]] | None = None,
) -> PreparedInteraction:
    """Prepare retrieval, tool calls, and final messages for response generation."""
    if not query.strip():
        raise UserFacingError("Question cannot be empty.")

    resolved_embedding_key = get_embedding_api_key(embedding_api_key)
    resolved_llm_key = get_llm_api_key(llm_api_key)

    retriever = Retriever(api_key=resolved_embedding_key)
    client = build_llm_client(resolved_llm_key)

    contexts = retriever.search_with_rerank(query=query, k=TOP_K)
    history_messages = normalize_chat_history(chat_history)
    base_messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
    base_messages.extend(history_messages)
    base_messages.append({"role": "user", "content": build_user_prompt(query, contexts)})

    try:
        first_response = client.chat.completions.create(
            model=get_llm_model(),
            messages=base_messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
    except Exception as exc:
        raise map_llm_exception(exc) from exc

    first_message = first_response.choices[0].message
    used_tool = bool(first_message.tool_calls)
    tool_result: dict[str, Any] | None = None
    final_messages = list(base_messages)

    if used_tool:
        final_messages.append(
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
                raise UserFacingError("The tool call returned invalid JSON arguments.") from exc

            tool_output = handle_tool_call(tool_name, tool_args)
            parsed_output = parse_tool_result(tool_output)
            if parsed_output is not None:
                tool_result = parsed_output

            final_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    return PreparedInteraction(
        client=client,
        model=get_llm_model(),
        query=query,
        contexts=contexts,
        final_messages=final_messages,
        used_tool=used_tool,
        tool_result=tool_result,
    )


@dataclass
class AnswerStream:
    """Streaming wrapper that collects final answer metadata."""

    interaction: PreparedInteraction
    result: dict[str, Any] | None = field(default=None, init=False)

    def __iter__(self):
        answer_parts: list[str] = []
        try:
            stream = self.interaction.client.chat.completions.create(
                model=self.interaction.model,
                messages=self.interaction.final_messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                answer_parts.append(delta)
                yield delta
        except Exception as exc:
            raise map_llm_exception(exc) from exc

        answer = "".join(answer_parts)
        self.result = {
            "answer": answer,
            "sources": self.interaction.contexts,
            "used_tool": self.interaction.used_tool,
            "tool_result": self.interaction.tool_result,
        }
        interaction_id = log_interaction(
            query=self.interaction.query,
            answer=answer,
            sources=self.interaction.contexts,
            used_tool=self.interaction.used_tool,
            feedback=None,
            tool_result=self.interaction.tool_result,
        )
        self.result["interaction_id"] = interaction_id


def stream_answer(
    query: str,
    embedding_api_key: str | None = None,
    llm_api_key: str | None = None,
    chat_history: Iterable[dict[str, str]] | None = None,
) -> AnswerStream:
    """Return a streaming answer object for Streamlit rendering."""
    interaction = prepare_interaction(
        query=query,
        embedding_api_key=embedding_api_key,
        llm_api_key=llm_api_key,
        chat_history=chat_history,
    )
    return AnswerStream(interaction=interaction)


def ask(
    query: str,
    embedding_api_key: str | None = None,
    llm_api_key: str | None = None,
    chat_history: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one full RAG + tool-calling interaction."""
    interaction = prepare_interaction(
        query=query,
        embedding_api_key=embedding_api_key,
        llm_api_key=llm_api_key,
        chat_history=chat_history,
    )

    try:
        final_response = interaction.client.chat.completions.create(
            model=interaction.model,
            messages=interaction.final_messages,
        )
    except Exception as exc:
        raise map_llm_exception(exc) from exc

    answer = final_response.choices[0].message.content or ""
    result = {
        "answer": answer,
        "sources": interaction.contexts,
        "used_tool": interaction.used_tool,
        "tool_result": interaction.tool_result,
    }

    interaction_id = log_interaction(
        query=query,
        answer=answer,
        sources=interaction.contexts,
        used_tool=interaction.used_tool,
        feedback=None,
        tool_result=interaction.tool_result,
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

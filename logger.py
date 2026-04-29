"""
交互日志工具。

日志文件：logs/interactions.jsonl

每条记录的核心字段：
- timestamp
- query
- answer
- sources
- used_tool
- score
- feedback
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


LOG_FILE = Path(__file__).parent / "logs" / "interactions.jsonl"


def ensure_log_file() -> None:
    """确保日志目录和文件存在。"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)


def _read_jsonl() -> list[dict[str, Any]]:
    """读取全部 JSONL 记录。"""
    ensure_log_file()
    records: list[dict[str, Any]] = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def _write_jsonl(records: list[dict[str, Any]]) -> None:
    """覆盖写回 JSONL 文件。"""
    ensure_log_file()
    with LOG_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mean_similarity(sources: list[dict[str, Any]]) -> float:
    """计算来源片段相似度均值。"""
    scores = []
    for source in sources:
        score = source.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def log_interaction(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    used_tool: bool,
    feedback: str | None = None,
    interaction_id: str | None = None,
    tool_result: dict[str, Any] | None = None,
) -> str:
    """追加一条问答记录到 JSONL，并返回 interaction_id。"""
    ensure_log_file()
    interaction_id = interaction_id or str(uuid4())

    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "sources": sources,
        "used_tool": used_tool,
        "score": _mean_similarity(sources),
        "feedback": feedback,
        "tool_result": tool_result,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return interaction_id


def read_interactions() -> list[dict[str, Any]]:
    """读取全部交互记录。"""
    return _read_jsonl()


def update_feedback(interaction_id: str, feedback: str) -> bool:
    """把指定交互标记为 good 或 bad。"""
    if feedback not in {"good", "bad"}:
        raise ValueError("feedback 只能是 `good` 或 `bad`。")

    records = _read_jsonl()
    updated = False

    for record in records:
        if record.get("interaction_id") == interaction_id:
            record["feedback"] = feedback
            updated = True
            break

    if updated:
        _write_jsonl(records)

    return updated


def build_bad_case_report() -> str:
    """导出 bad case 的 Markdown 报告内容。"""
    records = [record for record in _read_jsonl() if record.get("feedback") == "bad"]

    lines = [
        "# Bad Case Report",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"总 bad case 数量: {len(records)}",
        "",
    ]

    if not records:
        lines.append("当前没有 bad case 记录。")
        return "\n".join(lines)

    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## Case {index}",
                "",
                f"- interaction_id: {record.get('interaction_id')}",
                f"- timestamp: {record.get('timestamp')}",
                f"- query: {record.get('query')}",
                f"- used_tool: {record.get('used_tool')}",
                f"- score: {record.get('score')}",
                "",
                "### Answer",
                "",
                record.get("answer", ""),
                "",
                "### Sources",
                "",
            ]
        )

        sources = record.get("sources") or []
        if sources:
            for source in sources:
                lines.append(
                    f"- {source.get('source')} (page {source.get('page')}, score {source.get('score')})"
                )
        else:
            lines.append("- 无")

        tool_result = record.get("tool_result")
        if tool_result:
            lines.extend(
                [
                    "",
                    "### Tool Result",
                    "",
                    "```json",
                    json.dumps(tool_result, ensure_ascii=False, indent=2),
                    "```",
                ]
            )

        lines.append("")

    return "\n".join(lines)

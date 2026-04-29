"""
评估脚本。

读取 logs/interactions.jsonl，计算：
1. 平均检索相似度分
2. bad case 比例
3. 工具调用频率
4. 按 source 统计文档被引用次数

输出：evaluation_report.md
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger import LOG_FILE, read_interactions


REPORT_FILE = Path(__file__).parent / "evaluation_report.md"


def _safe_ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """计算评估指标。"""
    total = len(records)
    avg_similarity = (
        sum(float(record.get("score", 0.0)) for record in records) / total if total else 0.0
    )
    bad_cases = sum(1 for record in records if record.get("feedback") == "bad")
    tool_calls = sum(1 for record in records if record.get("used_tool") is True)

    source_counter: Counter[str] = Counter()
    for record in records:
        for source in record.get("sources") or []:
            source_name = source.get("source")
            if source_name:
                source_counter[source_name] += 1

    return {
        "total": total,
        "avg_similarity": avg_similarity,
        "bad_case_ratio": _safe_ratio(bad_cases, total),
        "tool_call_frequency": _safe_ratio(tool_calls, total),
        "source_counter": source_counter,
    }


def build_report(metrics: dict[str, Any]) -> str:
    """生成 Markdown 报告内容。"""
    lines = [
        "# Evaluation Report",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        f"日志文件: {LOG_FILE}",
        "",
        "## Summary",
        "",
        f"- 总问答数: {metrics['total']}",
        f"- 平均检索相似度分: {metrics['avg_similarity']:.4f}",
        f"- bad case 比例: {metrics['bad_case_ratio']:.2%}",
        f"- 工具调用频率: {metrics['tool_call_frequency']:.2%}",
        "",
        "## Source Citation Count",
        "",
    ]

    source_counter: Counter[str] = metrics["source_counter"]
    if not source_counter:
        lines.append("当前没有来源文档引用记录。")
        return "\n".join(lines)

    lines.extend(
        [
            "| Source | Citation Count |",
            "| --- | ---: |",
        ]
    )
    for source, count in source_counter.most_common():
        lines.append(f"| {source} | {count} |")

    return "\n".join(lines)


def evaluate() -> Path:
    """读取日志、生成评估报告并返回文件路径。"""
    records = read_interactions()
    metrics = compute_metrics(records)
    report = build_report(metrics)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main() -> None:
    """命令行入口。"""
    report_path = evaluate()
    print(f"Evaluation report generated: {report_path}")


if __name__ == "__main__":
    main()

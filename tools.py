"""
工具模块。

当前提供一个模拟设备状态查询工具，并暴露给 OpenAI function calling 使用。
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any


DEVICE_STATUSES = ["online", "offline", "error"]
ERROR_CODES = [f"E{index:03d}" for index in range(1, 11)]


def get_device_status(device_id: str) -> dict[str, Any]:
    """模拟返回设备状态。"""
    status = random.choice(DEVICE_STATUSES)
    error_code = random.choice(ERROR_CODES) if status == "error" else None

    return {
        "device_id": device_id,
        "status": status,
        "signal_strength": random.randint(0, 100),
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "error_code": error_code,
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": "查询指定 IoT 设备的当前状态，包括在线状态、信号强度、最近在线时间和错误码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "设备唯一标识，例如 device-001。",
                    }
                },
                "required": ["device_id"],
                "additionalProperties": False,
            },
        },
    }
]


def handle_tool_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    """统一处理工具调用入口，返回 JSON 字符串。"""
    if tool_name == "get_device_status":
        device_id = tool_args.get("device_id")
        if not device_id:
            raise ValueError("缺少必填参数 `device_id`。")
        result = get_device_status(device_id=device_id)
        return json.dumps(result, ensure_ascii=False)

    raise ValueError(f"未知工具: {tool_name}")

"""算法参数：集中定义可在页面上调整的运行参数。

参数仍存放在 config.json 的原有位置（顶层或 chat 段），页面保存时回写，
服务启动时照旧由 main.py 读取。大部分参数可通过 update_tuning 热更新，
标记 restart 的参数需要重启进程才能生效。
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Optional

from .paths import app_dir

logger = logging.getLogger(__name__)

CONFIG_PATH = app_dir() / "config.json"

# group 决定页面上的分组；restart 表示修改后需重启服务生效
SPECS: list[dict[str, Any]] = [
    {
        "key": "quality",
        "group": "视频",
        "label": "JPEG 画质",
        "type": "int",
        "default": 80,
        "min": 20,
        "max": 100,
        "step": 1,
        "unit": "",
        "restart": False,
        "desc": "视频帧压缩质量，越高越清晰，带宽与 CPU 占用也越高。",
    },
    {
        "key": "face_interval",
        "group": "人脸识别",
        "label": "识别间隔",
        "type": "float",
        "default": 1.0,
        "min": 0.2,
        "max": 5.0,
        "step": 0.1,
        "unit": "秒",
        "restart": False,
        "desc": "每隔多久检测一次人脸；调小反应更快，但 CPU 占用更高。",
    },
    {
        "key": "say_cooldown",
        "group": "人脸识别",
        "label": "播报冷却",
        "type": "float",
        "default": 5.0,
        "min": 0,
        "max": 300,
        "step": 1,
        "unit": "秒",
        "restart": False,
        "desc": "同一个人被识别后，多久内不再重复播报。",
    },
    {
        "key": "face_det_size",
        "group": "人脸识别",
        "label": "检测尺寸",
        "type": "int",
        "default": 480,
        "min": 160,
        "max": 1024,
        "step": 32,
        "unit": "px",
        "restart": True,
        "desc": "人脸检测模型输入分辨率；越大越准，但更慢。",
    },
    {
        "key": "face_threads",
        "group": "人脸识别",
        "label": "推理线程",
        "type": "int",
        "default": 2,
        "min": 1,
        "max": 16,
        "step": 1,
        "unit": "",
        "restart": True,
        "desc": "人脸推理使用的 CPU 线程数；调大更快，但更占 CPU。",
    },
    {
        "key": "vad_rms",
        "group": "语音对话",
        "label": "唤醒灵敏度",
        "type": "int",
        "default": 500,
        "min": 50,
        "max": 8000,
        "step": 10,
        "unit": "RMS",
        "restart": False,
        "desc": "声音能量超过该值才认为在说话；环境嘈杂时调大，说话声小可调小。",
    },
    {
        "key": "silence_ms",
        "group": "语音对话",
        "label": "静音判定",
        "type": "int",
        "default": 600,
        "min": 200,
        "max": 3000,
        "step": 50,
        "unit": "ms",
        "restart": False,
        "desc": "说话停顿超过该时长即认为一句话结束。",
    },
    {
        "key": "max_utterance_ms",
        "group": "语音对话",
        "label": "单句最长",
        "type": "int",
        "default": 8000,
        "min": 1000,
        "max": 30000,
        "step": 500,
        "unit": "ms",
        "restart": False,
        "desc": "一句话最长录制时长，超过后强制截断送去识别。",
    },
    {
        "key": "session_idle_s",
        "group": "语音对话",
        "label": "对话静默超时",
        "type": "float",
        "default": 10.0,
        "min": 3,
        "max": 600,
        "step": 1,
        "unit": "秒",
        "restart": False,
        "desc": "唤醒后超过该时长没有新的语音输入，自动结束对话（需重新说唤醒词）。",
    },
]

_CHAT_KEYS = {"vad_rms", "silence_ms", "max_utterance_ms", "session_idle_s"}

_lock = threading.Lock()
# 进程启动时的参数值，用于判断哪些「重启生效」参数还没真正生效
_baseline: Optional[dict] = None


def specs() -> list[dict]:
    """返回所有参数定义（供页面渲染）。"""
    return [dict(spec) for spec in SPECS]


def defaults() -> dict:
    """返回出厂默认值（「还原默认值」按钮使用）。"""
    return {spec["key"]: spec["default"] for spec in SPECS}


def _read_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取配置文件失败 %s: %s", CONFIG_PATH, exc)
        return {}


def _coerce(spec: dict, raw: Any) -> Any:
    """把页面输入转成合法数值：类型转换 + 范围裁剪。"""
    try:
        num = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"「{spec['label']}」必须是数字") from None
    if not math.isfinite(num):
        raise ValueError(f"「{spec['label']}」必须是有限数字")
    if spec["type"] == "int":
        num = int(round(num))
    return max(spec["min"], min(spec["max"], num))


def current_values() -> dict:
    """从 config.json 读出当前参数（缺失或非法时回退默认值）。"""
    cfg = _read_config()
    chat = cfg.get("chat") if isinstance(cfg.get("chat"), dict) else {}
    values: dict[str, Any] = {}
    for spec in SPECS:
        key = spec["key"]
        raw = chat.get(key, spec["default"]) if key in _CHAT_KEYS else cfg.get(key, spec["default"])
        try:
            values[key] = _coerce(spec, raw)
        except ValueError:
            values[key] = spec["default"]
    return values


def _write_values(cfg: dict, values: dict) -> dict:
    """把参数写回配置结构（顶层或 chat 段），保留其他配置项。"""
    chat = cfg.get("chat") if isinstance(cfg.get("chat"), dict) else {}
    for spec in SPECS:
        key = spec["key"]
        if key in _CHAT_KEYS:
            chat[key] = values[key]
        else:
            cfg[key] = values[key]
    if chat:
        cfg["chat"] = chat
    return cfg


def _persist(values: dict) -> None:
    """回写 config.json（调用方需持有 _lock）。"""
    cfg = _write_values(_read_config(), values)
    try:
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(f"写入配置文件失败: {exc}") from exc


def _baseline_values() -> dict:
    """返回进程启动时的参数值（首次调用时从配置快照）。"""
    global _baseline
    with _lock:
        if _baseline is None:
            _baseline = current_values()
        return dict(_baseline)


def _restart_pending(values: dict, baseline: dict) -> list[str]:
    return [
        spec["key"]
        for spec in SPECS
        if spec["restart"] and values[spec["key"]] != baseline[spec["key"]]
    ]


def _apply(stream, chat, values: dict) -> tuple[list[str], list[str]]:
    """把参数推到运行中的对象上，返回 (已生效, 需重启)。"""
    applied: list[str] = []

    update = getattr(stream, "update_tuning", None)
    if callable(update):
        update(
            jpeg_quality=values["quality"],
            face_interval=values["face_interval"],
            say_cooldown=values["say_cooldown"],
        )
        applied += ["quality", "face_interval", "say_cooldown"]

    if chat is not None:
        update = getattr(chat, "update_tuning", None)
        if callable(update):
            update(
                vad_rms=values["vad_rms"],
                silence_ms=values["silence_ms"],
                max_utterance_ms=values["max_utterance_ms"],
                session_idle_s=values["session_idle_s"],
            )
            applied += ["vad_rms", "silence_ms", "max_utterance_ms", "session_idle_s"]

    return applied, _restart_pending(values, _baseline_values())


def snapshot() -> dict:
    """返回页面渲染所需的完整参数快照。"""
    values = current_values()
    return {
        "specs": specs(),
        "values": values,
        "defaults": defaults(),
        "restart_pending": _restart_pending(values, _baseline_values()),
    }


def update(stream, chat, raw: dict) -> dict:
    """保存页面提交的参数并尽量热更新，返回最新快照。"""
    _baseline_values()  # 写盘前固定启动基线
    values = current_values()
    for spec in SPECS:
        if spec["key"] in raw:
            values[spec["key"]] = _coerce(spec, raw[spec["key"]])
    with _lock:
        _persist(values)
    applied, restart = _apply(stream, chat, values)
    return {**snapshot(), "applied": applied, "restart_required": restart}


def reset(stream, chat) -> dict:
    """恢复出厂默认值并尽量热更新，返回最新快照。"""
    _baseline_values()
    values = defaults()
    with _lock:
        _persist(values)
    applied, restart = _apply(stream, chat, values)
    return {**snapshot(), "applied": applied, "restart_required": restart}

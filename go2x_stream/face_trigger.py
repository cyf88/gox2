"""人脸触发：只要画面中出现人脸（无需识别身份）就播报语音并执行动作。

配置单独保存在 <faces_dir>/face_trigger.json，支持运行时开关。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FILE_NAME = "face_trigger.json"
STOP_FILE_NAME = "face_stop.json"
MAX_SAY_LEN = 200

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "say": "",
    "steps": [],
    "cooldown": 10.0,
    # 未认出来时先坐下，让画面稳定后再重新识别/执行动作
    "sit_first": True,
    "settle_s": 2.0,
}


class FaceTriggerStore:
    """人脸触发配置的持久化存储。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / FILE_NAME
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._data["steps"] = []
        self._load()

    # ─── 对外接口 ───────────────────────────────────────────────

    def get(self) -> dict:
        with self._lock:
            return {
                "enabled": bool(self._data.get("enabled")),
                "say": self._data.get("say", ""),
                "steps": list(self._data.get("steps") or []),
                "cooldown": float(self._data.get("cooldown", 10.0)),
                "sit_first": bool(self._data.get("sit_first", True)),
                "settle_s": float(self._data.get("settle_s", 2.0)),
            }

    def save(self, payload: dict) -> dict:
        """保存配置（播报文本与动作至少填一项才能开启）。"""
        say = str(payload.get("say", "") or "").strip()
        if len(say) > MAX_SAY_LEN:
            raise ValueError(f"播报文本不能超过 {MAX_SAY_LEN} 个字符")
        steps = payload.get("steps")
        steps = list(steps) if isinstance(steps, list) else []
        try:
            cooldown = float(payload.get("cooldown", 10.0))
        except (TypeError, ValueError):
            cooldown = 10.0
        cooldown = max(0.0, min(3600.0, cooldown))
        try:
            settle_s = float(payload.get("settle_s", 2.0))
        except (TypeError, ValueError):
            settle_s = 2.0
        settle_s = max(0.0, min(30.0, settle_s))
        sit_first = bool(payload.get("sit_first", True))
        enabled = bool(payload.get("enabled", False))
        if enabled and not say and not steps:
            raise ValueError("开启前请至少填写播报文本或添加动作")

        with self._lock:
            self._data = {
                "enabled": enabled,
                "say": say,
                "steps": steps,
                "cooldown": cooldown,
                "sit_first": sit_first,
                "settle_s": settle_s,
            }
            self._save_locked()
        logger.info(
            "人脸触发已保存（%s，%d 步，冷却 %.1fs，%s）",
            "开启" if enabled else "关闭", len(steps), cooldown,
            f"先坐下稳定 {settle_s:.1f}s" if sit_first else "直接执行",
        )
        return self.get()

    def set_enabled(self, enabled: bool) -> dict:
        with self._lock:
            data = dict(self._data)
        data["enabled"] = bool(enabled)
        return self.save(data)

    # ─── 内部工具 ───────────────────────────────────────────────

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载人脸触发配置失败 %s: %s", self._path, exc)
            return
        if isinstance(data, dict):
            self._data = {
                "enabled": bool(data.get("enabled", False)),
                "say": str(data.get("say", "") or ""),
                "steps": list(data.get("steps") or []),
                "cooldown": float(data.get("cooldown", 10.0) or 10.0),
                "sit_first": bool(data.get("sit_first", True)),
                "settle_s": float(data.get("settle_s", 2.0) or 0.0),
            }
            logger.info(
                "已加载人脸触发配置（%s）",
                "开启" if self._data["enabled"] else "关闭",
            )

    def _save_locked(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("保存人脸触发配置失败 %s: %s", self._path, exc)


class FaceStopStore:
    """检测到人脸时暂停移动的配置（时长可调）。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / STOP_FILE_NAME
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"enabled": True, "seconds": 3.0}
        self._load()

    def get(self) -> dict:
        with self._lock:
            return {
                "enabled": bool(self._data.get("enabled")),
                "seconds": float(self._data.get("seconds", 3.0)),
            }

    def save(self, payload: dict) -> dict:
        try:
            seconds = float(payload.get("seconds", 3.0))
        except (TypeError, ValueError):
            seconds = 3.0
        seconds = max(0.0, min(60.0, seconds))
        data = {"enabled": bool(payload.get("enabled", True)), "seconds": seconds}
        with self._lock:
            self._data = data
            self._save_locked()
        logger.info(
            "人脸暂停移动：%s（%.1f 秒）",
            "开启" if data["enabled"] else "关闭", seconds,
        )
        return self.get()

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载人脸暂停配置失败 %s: %s", self._path, exc)
            return
        if isinstance(data, dict):
            self._data = {
                "enabled": bool(data.get("enabled", True)),
                "seconds": float(data.get("seconds", 3.0) or 3.0),
            }

    def _save_locked(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("保存人脸暂停配置失败 %s: %s", self._path, exc)

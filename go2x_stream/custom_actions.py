"""自定义动作：一组动作序列 + 自定义播报文本，可绑定给已注册人脸。

识别到被绑定的人脸时，机器狗先播报该动作配置的文本，再依次执行动作序列。
数据保存在 <faces_dir>/custom_actions.json：

    {
      "actions": {
        "迎宾": {"say": "你好，欢迎光临", "steps": [...]},
      },
      "assignments": {"张三": "迎宾"}
    }
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FILE_NAME = "custom_actions.json"
MAX_NAME_LEN = 30
MAX_SAY_LEN = 200


class CustomActionStore:
    """自定义动作与「人脸 → 自定义动作」绑定的持久化存储。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / FILE_NAME
        self._lock = threading.Lock()
        self._actions: dict[str, dict] = {}
        self._assignments: dict[str, str] = {}
        self._load()

    # ─── 对外接口 ───────────────────────────────────────────────

    def list_actions(self) -> list[dict]:
        """返回所有自定义动作：[{name, say, steps}]。"""
        with self._lock:
            return [
                {"name": name, "say": item.get("say", ""), "steps": item.get("steps", [])}
                for name, item in self._actions.items()
            ]

    def get(self, name: str) -> Optional[dict]:
        with self._lock:
            item = self._actions.get(name)
            if item is None:
                return None
            return {"name": name, "say": item.get("say", ""), "steps": item.get("steps", [])}

    def save(self, name: str, say: str, steps: list) -> dict:
        """新建或覆盖一个自定义动作。"""
        name = str(name or "").strip()
        if not name:
            raise ValueError("名称不能为空")
        if len(name) > MAX_NAME_LEN:
            raise ValueError(f"名称不能超过 {MAX_NAME_LEN} 个字符")
        if not isinstance(steps, list) or not steps:
            raise ValueError("动作序列不能为空")
        say = str(say or "").strip()
        if len(say) > MAX_SAY_LEN:
            raise ValueError(f"播报文本不能超过 {MAX_SAY_LEN} 个字符")
        with self._lock:
            self._actions[name] = {"say": say, "steps": steps}
            self._save_locked()
        logger.info("已保存自定义动作: %s（%d 步）", name, len(steps))
        return {"name": name, "say": say, "steps": steps}

    def delete(self, name: str) -> bool:
        """删除自定义动作，同时解除相关人脸绑定。"""
        with self._lock:
            if name not in self._actions:
                return False
            self._actions.pop(name, None)
            for face in [f for f, a in self._assignments.items() if a == name]:
                self._assignments.pop(face, None)
            self._save_locked()
        logger.info("已删除自定义动作: %s", name)
        return True

    def assign(self, face_name: str, action_name: str) -> dict:
        """把人脸绑定到自定义动作；action_name 为空表示解除绑定。"""
        with self._lock:
            if action_name:
                if action_name not in self._actions:
                    raise ValueError(f"自定义动作不存在: {action_name}")
                self._assignments[face_name] = action_name
            else:
                self._assignments.pop(face_name, None)
            self._save_locked()
        logger.info("人脸 %s 绑定动作: %s", face_name, action_name or "无")
        return {"name": face_name, "action": action_name or ""}

    def action_of(self, face_name: str) -> Optional[dict]:
        """返回人脸绑定的自定义动作（未绑定返回 None）。"""
        with self._lock:
            action_name = self._assignments.get(face_name)
            if not action_name:
                return None
            item = self._actions.get(action_name)
            if item is None:
                return None
            return {
                "name": action_name,
                "say": item.get("say", ""),
                "steps": item.get("steps", []),
            }

    def assignment_of(self, face_name: str) -> str:
        with self._lock:
            return self._assignments.get(face_name, "")

    def remove_face(self, face_name: str) -> None:
        """删除人脸时同步清理绑定。"""
        with self._lock:
            if self._assignments.pop(face_name, None) is not None:
                self._save_locked()

    # ─── 内部工具 ───────────────────────────────────────────────

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载自定义动作失败 %s: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        actions = data.get("actions")
        if isinstance(actions, dict):
            self._actions = {
                str(k): {
                    "say": str((v or {}).get("say", "")),
                    "steps": list((v or {}).get("steps") or []),
                }
                for k, v in actions.items()
                if isinstance(v, dict)
            }
        assignments = data.get("assignments")
        if isinstance(assignments, dict):
            self._assignments = {str(k): str(v) for k, v in assignments.items() if v}
        logger.info(
            "已加载 %d 个自定义动作、%d 条人脸绑定", len(self._actions), len(self._assignments)
        )

    def _save_locked(self) -> None:
        payload = {"actions": self._actions, "assignments": self._assignments}
        try:
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("保存自定义动作失败 %s: %s", self._path, exc)

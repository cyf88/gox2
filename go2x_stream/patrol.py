"""巡逻路线：一串「前进/转向/动作/延时」步骤，循环执行。

配置保存在 <faces_dir>/patrol_route.json。路线由运动控制线程逐步下发，
检测到人脸时的暂停会冻结当前步的计时，恢复后继续走完剩余时间。
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FILE_NAME = "patrol_route.json"
MAX_STEPS = 50


def rectangle_steps(
    length: float = 2.0,
    width: float = 0.5,
    speed: float = 0.5,
    turn_rate: float = 0.785,
) -> list[dict]:
    """生成 长 length × 宽 width 的顺时针矩形巡逻路线（4 条边 + 4 个 90° 右转）。

    距离由「速度 × 时长」换算：直行时长 = 边长 / speed，
    转向时长 = (π/2) / turn_rate（vyaw 负值表示右转）。
    """
    turn_s = round((math.pi / 2) / turn_rate, 2)
    long_s = round(length / speed, 2)
    short_s = round(width / speed, 2)

    def straight(seconds: float) -> dict:
        return {"move": {"vx": speed, "vy": 0, "vyaw": 0}, "duration": seconds}

    def turn() -> dict:
        return {"move": {"vx": 0, "vy": 0, "vyaw": -turn_rate}, "duration": turn_s}

    return [
        straight(long_s), turn(),
        straight(short_s), turn(),
        straight(long_s), turn(),
        straight(short_s), turn(),
    ]


DEFAULT_STEPS: list[dict] = rectangle_steps()


class PatrolStore:
    """巡逻路线的持久化存储。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / FILE_NAME
        self._lock = threading.Lock()
        self._steps: list = []
        self._loop = True
        self._load()

    # ─── 对外接口 ───────────────────────────────────────────────

    def get(self) -> dict:
        with self._lock:
            return {"steps": list(self._steps), "loop": bool(self._loop)}

    def save(self, steps: list, loop: bool = True) -> dict:
        if not isinstance(steps, list) or not steps:
            raise ValueError("巡逻路线不能为空")
        if len(steps) > MAX_STEPS:
            raise ValueError(f"路线步骤不能超过 {MAX_STEPS} 步")
        with self._lock:
            self._steps = list(steps)
            self._loop = bool(loop)
            self._save_locked()
        logger.info("已保存巡逻路线（%d 步，%s）", len(steps), "循环" if loop else "单次")
        return self.get()

    # ─── 内部工具 ───────────────────────────────────────────────

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            # 首次运行：默认 2m × 0.5m 矩形巡逻轨迹
            self._steps = list(DEFAULT_STEPS)
            logger.info("未找到路线文件，使用默认矩形巡逻路线（2.0m × 0.5m，%d 步）", len(self._steps))
            return
        try:
            data: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载巡逻路线失败 %s: %s", self._path, exc)
            return
        if isinstance(data, dict):
            self._steps = list(data.get("steps") or [])
            self._loop = bool(data.get("loop", True))
            logger.info("已加载巡逻路线（%d 步）", len(self._steps))

    def _save_locked(self) -> None:
        payload = {"steps": self._steps, "loop": self._loop}
        try:
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("保存巡逻路线失败 %s: %s", self._path, exc)

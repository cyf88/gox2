"""Go2 机器狗运动控制（走 WebRTC sport 通道）。

go2x 的 WebRTC 连接除了拉视频，也承载 Unitree 的 sport 指令
（topic=rt/api/sport/request）。相比 DDS，它复用同一条已建立的 WebRTC
数据通道，不依赖组播/discovery，也不需要关闭机器狗的主运控服务。

api_id 采用 MCF 空间（固件 ≥1.1.7）；旧的普通模式 id 见
unitree_webrtc_connect.constants.SPORT_CMD。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 特技动作名 → MCF sport api_id（无参数，瞬时触发）
ACTION_MAP: dict[str, int] = {
    "hello": 1016,
    "heart": 1036,
    "balance_stand": 1002,
    "dance1": 1022,
    "dance2": 1023,
    "sit": 1009,
    "standup": 1004,
    "standdown": 1005,
    "stretch": 1017,
    "frontjump": 1031,
    "frontflip": 1030,
    "backflip": 2043,
    "handstand": 2044,
    "frontwalk": 1061,   # 静态行走
    "trot": 1062,         # 小跑
    "ecomode": 1063,      # 经济步态
}

# 移动控制 api_id
MOVE_API_ID = 1008       # Move(vx, vy, vyaw) — 持续速度指令
STOP_MOVE_API_ID = 1003  # StopMove

# 动作分类 → 动作名列表（给前端下拉用）
ACTION_CATEGORIES: dict[str, list[str]] = {
    "日常": ["hello", "heart", "dance1", "dance2"],
    "姿态": ["sit", "standup", "standdown", "stretch"],
    "步态": ["frontwalk", "trot", "ecomode"],
    "特技": ["frontjump", "frontflip", "backflip", "handstand"],
}

# 前端显示名映射
ACTION_LABELS: dict[str, str] = {
    "hello": "挥手", "heart": "比心", "dance1": "跳舞1", "dance2": "跳舞2",
    "balance_stand": "平衡站立",
    "sit": "坐下", "standup": "站立", "standdown": "趴下", "stretch": "伸懒腰",
    "frontwalk": "静态行走", "trot": "小跑", "ecomode": "经济步态",
    "frontjump": "前跳", "frontflip": "前空翻", "backflip": "后空翻", "handstand": "倒立",
}


def _parse_step(step: Any) -> Optional[tuple[int, Optional[dict], float]]:
    """把组合动作的单步解析成 (api_id, parameter, duration) 元组。

    支持的 step 格式：
      - "hello"                       → 特技动作，无参数，duration=0
      - {"action": "hello"}           → 同上
      - {"action": "hello", "duration": 3.0}
      - {"move": {"vx": 0.5, "vy": 0, "vyaw": 0}, "duration": 2.0}
      - {"vx": 0.5, "vy": 0, "vyaw": 0, "duration": 2.0}  → 简写
      - {"stop_move": true}           → 停止移动
      - {"delay": 2.0}                → 仅延时，不发指令
    """
    if isinstance(step, str):
        api_id = ACTION_MAP.get(step)
        if api_id is None:
            return None
        return (api_id, None, 0.0)

    if not isinstance(step, dict):
        return None

    # 延时步
    if "delay" in step:
        return (0, None, float(step["delay"]))

    # 停止移动
    if step.get("stop_move") is True:
        return (STOP_MOVE_API_ID, None, 0.3)

    # 移动指令
    if "move" in step or "vx" in step:
        vel = step.get("move", step)
        vx = float(vel.get("vx", 0))
        vy = float(vel.get("vy", 0))
        vyaw = float(vel.get("vyaw", 0))
        duration = float(step.get("duration", 2.0))
        return (MOVE_API_ID, {"vx": vx, "vy": vy, "vyaw": vyaw}, duration)

    # 特技动作
    action_name = step.get("action")
    if action_name:
        api_id = ACTION_MAP.get(action_name)
        if api_id is None:
            return None
        duration = float(step.get("duration", 0))
        return (api_id, None, duration)

    return None


class Go2Motion:
    """后台线程串行下发 sport 动作，实际发送委托给 WebRTC 连接。"""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = False
        self._error: Optional[str] = None
        self._stream = None
        self._seq_counter = 0  # 序列编号
        # 检测到人脸时的临时暂停：到期后恢复暂停前的速度
        self._hold_until = 0.0
        self._last_vel: Optional[dict] = None
        self._pending_vel: Optional[dict] = None
        # 巡逻：循环「前进 → 转向」，可被暂停/停止打断
        self._patrol_stop = threading.Event()
        self._patrol_thread: Optional[threading.Thread] = None

    # ─── 对外接口 ───────────────────────────────────────────────

    def attach(self, stream) -> None:
        """绑定 WebRTC 拉流对象（动作经它的数据通道下发）。"""
        self._stream = stream

    def start(self) -> None:
        """启动动作线程（幂等）。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="go2-motion", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._patrol_stop.set()
        self._queue.put(None)  # 唤醒阻塞的 get()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # --- 巡逻 ---

    def start_patrol(
        self,
        vx: float = 0.5,
        vyaw: float = 0.6,
        forward_s: float = 4.0,
        turn_s: float = 2.0,
    ) -> dict:
        """简单巡逻：前进 forward_s 秒 → 原地转 turn_s 秒，无限循环。"""
        steps = [
            {"move": {"vx": vx, "vy": 0, "vyaw": 0}, "duration": forward_s},
            {"move": {"vx": 0, "vy": 0, "vyaw": vyaw}, "duration": turn_s},
        ]
        return self.start_route(steps, loop=True)

    def start_route(self, steps: list, loop: bool = True) -> dict:
        """按自定义路线巡逻：前进/转向/动作/延时，可循环。

        检测到人脸进入暂停时，当前步的计时会被冻结，恢复后继续。
        """
        parsed: list[tuple[int, Optional[dict], float]] = []
        for i, step in enumerate(steps or []):
            item = _parse_step(step)
            if item is None:
                logger.warning("跳过无效的路线步骤 #%d: %r", i, step)
                continue
            parsed.append(item)
        if not parsed:
            raise ValueError("巡逻路线为空")

        self.stop_patrol()
        self._patrol_stop.clear()
        self._patrol_thread = threading.Thread(
            target=self._route_loop,
            args=(parsed, loop),
            name="go2-patrol",
            daemon=True,
        )
        self._patrol_thread.start()
        logger.info("开始巡逻路线（%d 步，%s）", len(parsed), "循环" if loop else "单次")
        return self.status()

    def stop_patrol(self) -> dict:
        self._patrol_stop.set()
        thread = self._patrol_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._patrol_thread = None
        self.stop_move()
        logger.info("已停止巡逻")
        return self.status()

    def _route_loop(self, steps: list, loop: bool) -> None:
        try:
            while not self._patrol_stop.is_set() and not self._stop.is_set():
                for api_id, param, duration in steps:
                    if self._patrol_stop.is_set() or self._stop.is_set():
                        return
                    if api_id == 0:
                        if self._route_wait(duration):
                            return
                        continue
                    if api_id == MOVE_API_ID:
                        # 暂停期间不发移动，等恢复后再走
                        if self._wait_until_free():
                            return
                        self.move(param.get("vx", 0), param.get("vy", 0), param.get("vyaw", 0))
                    else:
                        self._queue.put(("action_id", api_id, param))
                    if duration > 0 and self._route_wait(duration):
                        return
                if not loop:
                    break
        finally:
            if not self._stop.is_set():
                self.stop_move()
            logger.info("巡逻路线结束")

    def _wait_until_free(self) -> bool:
        """等到暂停结束；返回 True 表示被停止。"""
        while self.is_held():
            if self._patrol_stop.is_set() or self._stop.is_set():
                return True
            time.sleep(0.1)
        return False

    def _route_wait(self, seconds: float) -> bool:
        """路线单步计时；暂停期间冻结计时。返回 True 表示被停止。"""
        remaining = max(0.0, seconds)
        while remaining > 0:
            if self._patrol_stop.is_set() or self._stop.is_set():
                return True
            if self.is_held():
                time.sleep(0.1)
                continue
            time.sleep(0.1)
            remaining -= 0.1
        return False

    def _patrol_wait(self, seconds: float) -> bool:
        """兼容旧接口的等待（不带暂停冻结）。"""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self._patrol_stop.is_set() or self._stop.is_set():
                return True
            time.sleep(0.1)
        return False

    # --- 特技动作 ---

    def do_action(self, action: str) -> dict:
        """提交一个特技动作（异步），返回当前状态。"""
        if action not in ACTION_MAP:
            raise ValueError(f"不支持的动作: {action}")
        self._queue.put(("action", action))
        return self.status()

    # --- 速度控制 ---

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> dict:
        """设置持续速度（vx 前进正/后退负，vy 左正/右负，vyaw 左转正）。

        调用后机器狗持续以该速度运动，直到 stop_move() 或新的 move() 覆盖。
        暂停期间（hold）只记录速度，到期后自动执行。
        """
        # 夹紧到合理范围
        vx = max(-0.7, min(1.6, vx))
        vy = max(-0.4, min(0.4, vy))
        vyaw = max(-1.0, min(1.0, vyaw))
        vel = {"vx": vx, "vy": vy, "vyaw": vyaw}
        with self._lock:
            held = time.monotonic() < self._hold_until
            if held:
                self._pending_vel = vel
            else:
                self._last_vel = vel
        if held:
            return self.status()
        self._queue.put(("move", vel))
        return self.status()

    def hold(self, seconds: float) -> dict:
        """立即停止移动，并在 seconds 秒内忽略移动指令；到期恢复暂停前的速度。

        重复调用会刷新暂停时长（人在画面里就一直停），但只在首次进入暂停时下发 StopMove。
        """
        seconds = max(0.0, float(seconds))
        now = time.monotonic()
        with self._lock:
            already = now < self._hold_until
            self._hold_until = max(self._hold_until, now + seconds)
            if not already:
                self._pending_vel = self._last_vel
                self._last_vel = None
        if not already:
            self._queue.put(("action_id", STOP_MOVE_API_ID, None))
            logger.info("暂停移动 %.1f 秒", seconds)
        return self.status()

    def is_held(self) -> bool:
        with self._lock:
            return time.monotonic() < self._hold_until

    def stop_move(self) -> dict:
        """立即停止移动（仅停止速度控制，不影响已触发的特技动作）。"""
        with self._lock:
            self._last_vel = None
            self._pending_vel = None
            self._hold_until = 0.0
        self._queue.put(("action_id", STOP_MOVE_API_ID, None))
        return self.status()

    # --- 动作序列 ---

    def do_sequence(self, steps: list, wait: bool = False) -> dict:
        """串行执行一串动作（每个 step 可特技、移动、延时、停止）。

        示例 steps：
          ["hello", {"action": "sit", "duration": 2},
           {"vx": 0.5, "duration": 2}, {"stop_move": true},
           {"action": "dance1"}]

        wait=True 时阻塞到整串执行完毕（用于「先站立、再挥手、最后播报」这类有顺序要求的场景）。
        """
        parsed: list[tuple[int, Optional[dict], float]] = []
        for i, step in enumerate(steps):
            parsed_step = _parse_step(step)
            if parsed_step is None:
                logger.warning("跳过无效的组合步骤 #%d: %r", i, step)
                continue
            parsed.append(parsed_step)
        if not parsed:
            raise ValueError("组合动作为空")
        self._seq_counter += 1
        seq_id = self._seq_counter
        done = threading.Event() if wait else None
        self._queue.put(("sequence", seq_id, parsed, done))
        if done is not None:
            total = sum(duration for _, _, duration in parsed) + 10.0
            done.wait(total)
        return {"status": "queued", "seq_id": seq_id, "total_steps": len(parsed), **self.status()}

    # --- 查询 ---

    def list_actions(self) -> dict:
        """返回所有可用动作（给前端下拉/组合编辑器用）。"""
        flat = []
        for cat, names in ACTION_CATEGORIES.items():
            for n in names:
                flat.append({
                    "name": n,
                    "label": ACTION_LABELS.get(n, n),
                    "category": cat,
                    "api_id": ACTION_MAP.get(n),
                })
        return {
            "actions": flat,
            "move_api_id": MOVE_API_ID,
            "stop_move_api_id": STOP_MOVE_API_ID,
            "velocity_range": {"vx": [-0.7, 1.6], "vy": [-0.4, 0.4], "vyaw": [-1.0, 1.0]},
        }

    def status(self) -> dict:
        thread = self._patrol_thread
        with self._lock:
            return {
                "ready": self._ready,
                "error": self._error,
                "interface": "WebRTC",
                "domain_id": 0,
                "patrol": bool(thread and thread.is_alive()),
                "held": time.monotonic() < self._hold_until,
            }

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    # ─── 后台线程 ───────────────────────────────────────────────

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                stream = self._stream
                if stream is not None and stream.sport_ready():
                    self._set_ready(True)
                    logger.info("运动控制已就绪（WebRTC sport 通道）")
                    break
                time.sleep(0.5)
            else:
                return

            while not self._stop.is_set():
                self._check_hold()
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    continue
                self._dispatch(item)
        finally:
            self._set_ready(False)
            logger.info("运动控制线程退出")

    def _check_hold(self) -> None:
        """暂停到期后，恢复暂停前的速度（若有）。"""
        with self._lock:
            if not self._hold_until or time.monotonic() < self._hold_until:
                return
            self._hold_until = 0.0
            vel = self._pending_vel
            self._pending_vel = None
            self._last_vel = vel
        if vel:
            logger.info("暂停结束，恢复移动 %s", vel)
            self._queue.put(("move", vel))

    def _dispatch(self, item: tuple) -> None:
        tag = item[0]
        if tag == "action":
            name = item[1]
            api_id = ACTION_MAP.get(name)
            if api_id is not None:
                self._send(api_id, None, f"特技:{name}")
        elif tag == "move":
            vel = item[1]
            with self._lock:
                if time.monotonic() < self._hold_until:
                    # 暂停期间不移动，记下来等恢复
                    self._pending_vel = vel
                    return
                self._last_vel = vel
            self._send(MOVE_API_ID, vel, f"速度({vel['vx']},{vel['vy']},{vel['vyaw']})")
        elif tag == "action_id":
            api_id, param = item[1], item[2]
            self._send(api_id, param, f"指令:{api_id}")
        elif tag == "sequence":
            seq_id, steps = item[1], item[2]
            done = item[3] if len(item) > 3 else None
            logger.info("开始执行组合 #%d（%d 步）", seq_id, len(steps))
            try:
                for idx, (api_id, param, duration) in enumerate(steps):
                    if self._stop.is_set():
                        logger.info("组合 #%d 被中断", seq_id)
                        break
                    if api_id == 0:
                        # 纯延时
                        if duration > 0:
                            time.sleep(duration)
                        continue
                    if api_id == MOVE_API_ID:
                        # 暂停期间不移动：等暂停结束后再执行该步
                        while self.is_held() and not self._stop.is_set():
                            time.sleep(0.1)
                        if self._stop.is_set():
                            break
                    label = f"组合#{seq_id}[{idx}]"
                    self._send(api_id, param, label)
                    if duration > 0:
                        time.sleep(duration)
            finally:
                if done is not None:
                    done.set()
            logger.info("组合 #%d 执行完毕", seq_id)

    def _send(self, api_id: int, parameter: Optional[dict], label: str = "") -> None:
        logger.info("→ %s api_id=%s param=%s", label, api_id, parameter)
        resp = self._stream.sport_action(api_id, parameter)
        if resp is None:
            logger.warning("× 发送失败: %s", label)
            return
        code = (
            (resp.get("data") or {})
            .get("header", {})
            .get("status", {})
            .get("code")
        )
        if code != 0:
            logger.warning("× 返回错误码 %s: %s", code, label)

    # ─── 内部工具 ───────────────────────────────────────────────

    def _set_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = ready
            if ready:
                self._error = None

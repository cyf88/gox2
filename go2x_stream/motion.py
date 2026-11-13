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
from typing import Optional

logger = logging.getLogger(__name__)

# 动作名 → MCF sport api_id
ACTION_MAP = {
    "hello": 1016,
    "heart": 1036,
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
}


class Go2Motion:
    """后台线程串行下发 sport 动作，实际发送委托给 WebRTC 连接。"""

    def __init__(self) -> None:
        self._in: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = False
        self._error: Optional[str] = None
        self._stream = None

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
        self._in.put("")  # 唤醒阻塞的 get()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def do_action(self, action: str) -> dict:
        """提交一个动作（异步），返回当前状态。"""
        if action not in ACTION_MAP:
            raise ValueError(f"不支持的动作: {action}")
        self._in.put(action)
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "ready": self._ready,
                "error": self._error,
                "interface": "WebRTC",
                "domain_id": 0,
            }

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    # ─── 后台线程 ───────────────────────────────────────────────

    def _run(self) -> None:
        try:
            # 等待 WebRTC 数据通道就绪
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
                try:
                    action = self._in.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not action:
                    continue
                self._exec(action)
        finally:
            self._set_ready(False)
            logger.info("运动控制线程退出")

    def _exec(self, action: str) -> None:
        api_id = ACTION_MAP[action]
        logger.info("执行动作: %s (api_id=%s)", action, api_id)
        resp = self._stream.sport_action(api_id)
        if resp is None:
            logger.warning("动作 %s 发送失败", action)
            return
        code = (
            (resp.get("data") or {})
            .get("header", {})
            .get("status", {})
            .get("code")
        )
        if code != 0:
            logger.warning("动作 %s 返回错误码 %s", action, code)

    # ─── 内部工具 ───────────────────────────────────────────────

    def _set_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = ready
            if ready:
                self._error = None

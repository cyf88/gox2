"""通过 Unitree DDS 服务从 Go2 前置相机拉取图像，并在内存中保留最新一帧。

Go2 的相机图像由机载 `videohub` 服务（DDS RPC，API 1001）以 JPEG 形式返回，
unitree_sdk2py 的 VideoClient 封装了该请求。该方式需要 PC 有线接入
192.168.123.x 内部网络（官方仅支持有线使用 SDK2）。

DDS RPC 是同步阻塞调用，因此后台线程负责轮询，Web 服务只读取最新帧。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

from .frame import Frame, jpeg_size

logger = logging.getLogger(__name__)


class Go2DdsStream:
    """轮询 Go2 前置相机，对外只暴露"最新一帧"。"""

    def __init__(
        self,
        network_interface: Optional[str] = None,
        domain_id: int = 0,
        timeout: float = 1.0,
        max_fps: float = 30.0,
    ) -> None:
        self._interface = network_interface
        self._domain_id = domain_id
        self._timeout = timeout
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._frame_times: deque[float] = deque(maxlen=90)
        self._last_success = 0.0
        self._error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─── 对外接口 ───────────────────────────────────────────────

    def start(self) -> None:
        """启动后台拉流线程（幂等）。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="go2-video-stream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止拉流并等待线程退出。"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._timeout + 2)
            self._thread = None

    def latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def status(self) -> dict:
        with self._lock:
            frame = self._frame
            connected = bool(frame) and time.monotonic() - self._last_success < 2.0
            return {
                "connected": connected,
                "error": self._error,
                "source": f"DDS videohub · domain {self._domain_id} · "
                f"{self._interface or '自动网卡'}",
                "width": frame.width if frame else 0,
                "height": frame.height if frame else 0,
                "fps": round(self._fps_locked(), 1),
                "frames": self._seq,
            }

    # ─── 后台线程 ───────────────────────────────────────────────

    def _run(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.video.video_client import VideoClient
        except ImportError as exc:
            self._set_error(f"未安装 unitree_sdk2py：{exc}")
            logger.error("未安装 unitree_sdk2py，请先 pip install 该 SDK")
            return

        try:
            ChannelFactoryInitialize(self._domain_id, self._interface)
        except Exception as exc:
            self._set_error(f"DDS 初始化失败：{exc}")
            logger.error("DDS 初始化失败: %s", exc)
            return

        logger.info(
            "DDS 已初始化 (domain=%s, interface=%s)，开始请求 videohub 图像",
            self._domain_id,
            self._interface or "自动选择",
        )

        client = None
        last_error_log = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if client is None:
                    client = VideoClient()
                    client.SetTimeout(self._timeout)
                    client.Init()

                code, data = client.GetImageSample()
                if code != 0 or not data:
                    now = time.monotonic()
                    if now - last_error_log > 10:
                        logger.warning("获取相机图像失败，错误码 %s", code)
                        last_error_log = now
                    self._set_error(f"videohub 返回错误码 {code}")
                else:
                    self._store(bytes(data))
            except Exception as exc:
                logger.exception("请求相机图像异常")
                self._set_error(str(exc))
                client = None

            elapsed = time.monotonic() - started
            self._sleep(max(self._min_interval - elapsed, 0.02))

    def _store(self, jpeg: bytes) -> None:
        if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8":
            self._set_error("收到的数据不是 JPEG 图像")
            return
        width, height = jpeg_size(jpeg)
        with self._lock:
            self._seq += 1
            self._frame = Frame(seq=self._seq, jpeg=jpeg, width=width, height=height)
            now = time.monotonic()
            self._frame_times.append(now)
            self._last_success = now
            self._error = None

    # ─── 内部工具 ───────────────────────────────────────────────

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _fps_locked(self) -> float:
        now = time.monotonic()
        recent = [t for t in self._frame_times if now - t <= 3.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def _sleep(self, seconds: float) -> None:
        """可被 stop() 提前打断的等待。"""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)

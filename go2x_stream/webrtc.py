"""通过 WebRTC 从宇树 Go2 机器狗拉取视频流，并在内存中保留最新一帧。

Go2 的相机视频由官方 WebRTC 通道下发 H.264，unitree_webrtc_connect 负责
信令与 ICE/DTLS，aiortc 自动解码成视频帧。适用于 PC 连接机器狗 WiFi 热点的场景。
为避免阻塞 Web 服务，连接与接收都在独立线程的事件循环中运行。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
from aiortc.mediastreams import MediaStreamError
from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

from .frame import Frame
from .tts import synthesize_wav, write_temp_wav

logger = logging.getLogger(__name__)


class Go2WebRtcStream:
    """维护与 Go2 的 WebRTC 连接，对外只暴露"最新一帧"。

    - 不传 ``ip`` / ``sn`` 时按 AP 直连（192.168.12.1）连接机器狗热点；
    - 传 ``ip`` 或 ``sn`` 时按 STA 模式连接同一局域网内的机器狗；
    - 断线或视频中断后自动重连。
    """

    def __init__(
        self,
        ip: Optional[str] = None,
        serial_number: Optional[str] = None,
        aes_key: Optional[str] = None,
        jpeg_quality: int = 80,
        reconnect_delay: float = 3.0,
        stall_timeout: float = 10.0,
    ) -> None:
        if ip is None and serial_number is None:
            self._method = WebRTCConnectionMethod.LocalAP
            self._target = "192.168.12.1 (AP 直连)"
        else:
            self._method = WebRTCConnectionMethod.LocalSTA
            self._target = ip or f"SN={serial_number} (局域网发现)"

        self._ip = ip
        self._serial_number = serial_number
        self._aes_key = aes_key
        self._jpeg_quality = jpeg_quality
        self._reconnect_delay = reconnect_delay
        self._stall_timeout = stall_timeout

        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._frame_times: deque[float] = deque(maxlen=90)
        self._connected = False
        self._connected_at = 0.0
        self._error: Optional[str] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._conn: Optional[UnitreeWebRTCConnection] = None

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
        loop, conn = self._loop, self._conn
        if loop and conn and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(conn.disconnect(), loop)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def status(self) -> dict:
        with self._lock:
            frame = self._frame
            return {
                "connected": self._connected,
                "error": self._error,
                "source": f"WebRTC · {self._target}",
                "width": frame.width if frame else 0,
                "height": frame.height if frame else 0,
                "fps": round(self._fps_locked(), 1),
                "frames": self._seq,
            }

    def say(self, text: str) -> dict:
        """把文字合成语音并让机器狗播报（阻塞，直到提交完成）。"""
        loop = self._loop
        conn = self._conn
        if loop is None or loop.is_closed():
            raise RuntimeError("拉流线程未运行")
        if conn is None or not conn.isConnected:
            raise RuntimeError("机器狗未连接")
        future = asyncio.run_coroutine_threadsafe(self._say_async(text), loop)
        return future.result(timeout=90)

    async def _say_async(self, text: str) -> dict:
        conn = self._conn
        # TTS 合成是阻塞调用，放到线程池避免卡住事件循环
        wav = await asyncio.to_thread(synthesize_wav, text)
        path = write_temp_wav(wav)
        try:
            unique_id = await self._upload_and_play(conn, path)
            logger.info("已触发播报, unique_id=%s", unique_id)
            return {"ok": True, "unique_id": unique_id}
        finally:
            try:
                from pathlib import Path

                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    async def _upload_and_play(self, conn, path: str) -> str:
        """上传 WAV 到机器狗 audiohub 并触发播放，返回唯一 ID。

        直接调用数据通道 pub_sub，避免引入 webrtc_audiohub 的 pydub 依赖
        （pydub 在 Python 3.13 下因 audioop 被移除而无法导入）。
        """
        from pathlib import Path

        pub_sub = conn.datachannel.pub_sub
        audio_data = Path(path).read_bytes()
        file_md5 = hashlib.md5(audio_data).hexdigest()
        b64_data = base64.b64encode(audio_data).decode("utf-8")
        chunks = [b64_data[i : i + 61440] for i in range(0, len(b64_data), 61440)]
        total_chunks = len(chunks)
        file_name = Path(path).stem

        last_response = None
        for i, chunk in enumerate(chunks, 1):
            parameter = {
                "file_name": file_name,
                "file_type": "wav",
                "file_size": len(audio_data),
                "current_block_index": i,
                "total_block_number": total_chunks,
                "block_content": chunk,
                "current_block_size": len(chunk),
                "file_md5": file_md5,
                "create_time": int(time.time() * 1000),
            }
            last_response = await pub_sub.publish_request_new(
                "rt/api/audiohub/request",
                {"api_id": 2001, "parameter": json.dumps(parameter, ensure_ascii=True)},
            )
            await asyncio.sleep(0.02)

        # 上传响应里只有 current_block_index，不包含文件唯一 ID；
        # 需要再查询音频列表，按文件名匹配出刚上传的那条记录。
        list_response = await pub_sub.publish_request_new(
            "rt/api/audiohub/request",
            {"api_id": 1001, "parameter": json.dumps({})},
        )
        logger.info("音频列表响应: %s", list_response)
        unique_id = self._find_audio_id(list_response, file_name)
        if not unique_id:
            unique_id = self._extract_unique_id(last_response)
        if not unique_id:
            raise RuntimeError(f"未能找到音频文件的 unique_id（file_name={file_name}）")

        await pub_sub.publish_request_new(
            "rt/api/audiohub/request",
            {"api_id": 1002, "parameter": json.dumps({"unique_id": unique_id})},
        )
        return unique_id

    @staticmethod
    def _find_audio_id(list_response, file_name: str) -> Optional[str]:
        """从音频列表响应中，按文件名匹配出对应的 unique_id。

        机器狗返回的列表结构为
        ``data.data -> {"audio_list": [{"UNIQUE_ID": ..., "CUSTOM_NAME": ...}]}``，
        字段名为大写，且 data 字段是嵌套的 JSON 字符串。
        """
        data = list_response
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None

        audio_list = Go2WebRtcStream._locate_audio_list(data)
        if not audio_list:
            return None

        for item in audio_list:
            if not isinstance(item, dict):
                continue
            uid = item.get("UNIQUE_ID") or item.get("unique_id") or item.get("uuid") or item.get("id")
            name = item.get("CUSTOM_NAME") or item.get("custom_name") or item.get("file_name") or item.get("name")
            if uid and (name is None or name == file_name):
                return str(uid)
        return None

    @staticmethod
    def _locate_audio_list(node, depth: int = 0):
        """递归查找 audio_list 数组，处理 data 字段为 JSON 字符串的嵌套。"""
        if depth > 6 or node is None:
            return None
        if isinstance(node, str):
            try:
                node = json.loads(node)
            except json.JSONDecodeError:
                return None
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            if isinstance(node.get("audio_list"), list):
                return node["audio_list"]
            for key in ("data", "list", "files", "audio_list"):
                if key in node:
                    result = Go2WebRtcStream._locate_audio_list(node[key], depth + 1)
                    if result:
                        return result
        return None

    @staticmethod
    def _extract_unique_id(response) -> Optional[str]:
        """从机器狗返回的上传响应中提取唯一 ID（递归查找）。"""
        if isinstance(response, str):
            import json

            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return None
        if not isinstance(response, dict):
            return None
        stack = [response]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ("unique_id", "uuid") and isinstance(value, str) and value:
                        return value
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
        return None

    # ─── 后台线程 ───────────────────────────────────────────────

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except Exception:
            logger.exception("拉流线程异常退出")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                self._loop = None

    async def _main(self) -> None:
        logger.info("视频源: %s", self._target)
        while not self._stop.is_set():
            conn: Optional[UnitreeWebRTCConnection] = None
            try:
                conn = UnitreeWebRTCConnection(
                    self._method,
                    ip=self._ip,
                    serialNumber=self._serial_number,
                    aes_128_key=self._aes_key,
                )
                self._conn = conn
                logger.info("正在连接机器狗 ...")
                await conn.connect()
                self._set_connected(True, None)
                logger.info("WebRTC 已连接，开启视频通道")

                conn.video.add_track_callback(self._receive_video)
                conn.video.switchVideoChannel(True)

                while not self._stop.is_set():
                    if not conn.isConnected:
                        logger.warning("WebRTC 连接已断开")
                        break
                    if self._video_stalled():
                        logger.warning(
                            "超过 %.0f 秒未收到视频帧，重连", self._stall_timeout
                        )
                        break
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("拉流失败: %s", exc)
                self._set_connected(False, str(exc))
            finally:
                if conn is not None:
                    try:
                        await conn.disconnect()
                    except Exception:
                        pass
                self._conn = None
                with self._lock:
                    self._connected = False

            if self._stop.is_set():
                break
            logger.info("%.0f 秒后重连", self._reconnect_delay)
            await self._sleep(self._reconnect_delay)

    async def _receive_video(self, track) -> None:
        """持续读取解码后的视频帧，编码为 JPEG 供显示层使用。"""
        logger.info("开始接收视频帧")
        try:
            while not self._stop.is_set():
                frame = await track.recv()
                self._store(frame)
        except MediaStreamError:
            logger.warning("视频轨道已结束")
        except asyncio.CancelledError:
            raise

    def _store(self, frame) -> None:
        array = frame.to_ndarray(format="bgr24")
        ok, buf = cv2.imencode(
            ".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:
            logger.warning("JPEG 编码失败，丢弃该帧")
            return
        with self._lock:
            self._seq += 1
            self._frame = Frame(
                seq=self._seq,
                jpeg=buf.tobytes(),
                width=frame.width,
                height=frame.height,
            )
            self._frame_times.append(time.monotonic())

    # ─── 内部工具 ───────────────────────────────────────────────

    def _set_connected(self, connected: bool, error: Optional[str]) -> None:
        with self._lock:
            self._connected = connected
            if connected:
                self._error = None
                self._frame_times.clear()
                self._frame = None
                self._connected_at = time.monotonic()
            elif error is not None:
                self._error = error

    def _video_stalled(self) -> bool:
        with self._lock:
            base = self._frame_times[-1] if self._frame_times else self._connected_at
            return base > 0 and time.monotonic() - base > self._stall_timeout

    def _fps_locked(self) -> float:
        now = time.monotonic()
        recent = [t for t in self._frame_times if now - t <= 3.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    async def _sleep(self, seconds: float) -> None:
        """可被 stop() 提前打断的等待。"""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.2)

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
import socket
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
from unitree_webrtc_connect.constants import RTC_TOPIC
from unitree_webrtc_connect.multicast_scanner import discover_ip_sn

from .frame import Frame
from .tts import synthesize_wav, write_temp_wav

logger = logging.getLogger(__name__)

# 自动模式下的候选地址：机器狗 AP 热点、有线直连的内置电脑
_AUTO_CANDIDATES = ("192.168.12.1", "192.168.123.161")


def _port_open(ip: str, port: int = 9991, timeout: float = 0.6) -> bool:
    """探测机器狗 WebRTC 信令端口是否可达。"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


class Go2WebRtcStream:
    """维护与 Go2 的 WebRTC 连接，对外只暴露"最新一帧"。

    - 不传 ``ip`` / ``sn`` 时自动探测：AP 热点 → 有线直连 → 组播发现；
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
        face_recognizer=None,
        say_cooldown: float = 30.0,
        face_interval: float = 1.0,
        motion=None,
    ) -> None:
        self._auto = ip is None and serial_number is None
        if self._auto:
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

        # 人脸识别（可选，PC 本地运行）
        self._face_recognizer = face_recognizer
        self._say_cooldown = say_cooldown
        self._face_interval = face_interval
        self._last_said: dict[str, float] = {}
        self._face_stop = threading.Event()
        self._face_thread: Optional[threading.Thread] = None

        # 运动控制（可选，WebRTC sport 通道）
        self._motion = motion

        if self._auto:
            self._apply_auto_target()

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

        if self._face_recognizer is not None:
            self._face_stop.clear()
            self._face_thread = threading.Thread(
                target=self._face_loop, name="go2-face-recognize", daemon=True
            )
            self._face_thread.start()

    def stop(self) -> None:
        """停止拉流并等待线程退出。"""
        self._stop.set()
        self._face_stop.set()
        if self._motion is not None:
            self._motion.stop()
        loop, conn = self._loop, self._conn
        if loop and conn and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(conn.disconnect(), loop)
            except RuntimeError:
                pass
        if self._face_thread:
            self._face_thread.join(timeout=5)
            self._face_thread = None
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
                "ip": self._ip or "",
                "width": frame.width if frame else 0,
                "height": frame.height if frame else 0,
                "fps": round(self._fps_locked(), 1),
                "frames": self._seq,
            }

    def connect(self, ip: str, aes_key: Optional[str] = None) -> dict:
        """运行时更新机器狗 IP（可选 AES 密钥）并触发重连。"""
        if not ip:
            raise ValueError("IP 不能为空")
        self._auto = False
        self._ip = ip
        self._serial_number = None
        self._method = WebRTCConnectionMethod.LocalSTA
        self._target = ip
        if aes_key:
            self._aes_key = aes_key
        with self._lock:
            self._error = None
        # 断开当前连接以触发重连
        loop, conn = self._loop, self._conn
        if loop and conn and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(conn.disconnect(), loop)
            except RuntimeError:
                pass
        return self.status()

    def sport_ready(self) -> bool:
        """WebRTC 数据通道是否可用（运动指令依赖它）。"""
        conn = self._conn
        return bool(conn is not None and conn.isConnected)

    def sport_action(self, api_id: int, timeout: float = 8.0) -> Optional[dict]:
        """经 WebRTC 数据通道下发一条 sport 指令（线程安全，阻塞到收到应答）。"""
        loop, conn = self._loop, self._conn
        if conn is None or loop is None or loop.is_closed():
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": api_id, "parameter": {}},
                ),
                loop,
            )
            return future.result(timeout=timeout)
        except Exception as exc:
            logger.warning("sport 指令 %d 发送失败: %s", api_id, exc)
            return None

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

    def register_face(self, name: str, image_bytes: bytes, category: str = "") -> dict:
        """从上传的图片字节注册人脸（需已启用人脸识别）。"""
        if self._face_recognizer is None:
            raise RuntimeError("未启用人脸识别（启动服务时需加 --face）")
        import numpy as np

        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("无法解码图片，请上传 JPG/PNG 格式")
        self._face_recognizer.register_image(name, img, category)
        return {"ok": True, "name": name, "category": category}

    def registered_faces(self) -> list[dict]:
        """返回已注册的人脸列表（含姓名与分类）。"""
        if self._face_recognizer is None:
            return []
        return [
            {"name": n, "category": self._face_recognizer.category_of(n)}
            for n in self._face_recognizer.registered_names()
        ]

    def do_action(self, action: str) -> dict:
        """触发机器狗动作（需已启用运动控制）。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.do_action(action)

    def motion_status(self) -> dict:
        """返回运动控制状态。"""
        if self._motion is None:
            return {"ready": False, "error": "未启用运动控制", "interface": "", "domain_id": 0}
        return self._motion.status()

    def face_image_path(self, name: str):
        """返回注册人脸的缩略图路径（不存在则返回 None）。"""
        if self._face_recognizer is None:
            return None
        return self._face_recognizer.face_image_path(name)

    def delete_face(self, name: str) -> dict:
        """删除一个已注册的人脸。"""
        if self._face_recognizer is None:
            raise RuntimeError("未启用人脸识别（启动服务时需加 --face）")
        if not self._face_recognizer.delete(name):
            raise RuntimeError(f"未找到已注册人脸: {name}")
        return {"ok": True, "name": name}

    def detect_video(self, video_bytes: bytes, filename: str = "video.mp4") -> dict:
        """对上传的离线视频做人脸检测识别，返回识别结果汇总。"""
        if self._face_recognizer is None:
            raise RuntimeError("未启用人脸识别（启动服务时需加 --face）")
        import tempfile
        from pathlib import Path

        suffix = Path(filename).suffix or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="go2x_detect_")
        with open(fd, "wb") as f:
            f.write(video_bytes)
        try:
            return self._face_recognizer.detect_video(path)
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    def detect_frame(self, frame) -> list:
        """检测单帧图像，返回 [(姓名, 相似度, bbox)]（供画框展示）。"""
        if self._face_recognizer is None:
            return []
        return self._face_recognizer.detect_with_boxes(frame)

    # ─── 人脸识别 ───────────────────────────────────────────────

    def _face_loop(self) -> None:
        """人脸识别循环：定时从最新帧检测并识别，识别到人时去重播报。"""
        import numpy as np

        while not self._face_stop.is_set():
            frame = self.latest_frame()
            if frame is not None:
                try:
                    img = cv2.imdecode(
                        np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        results = self._face_recognizer.recognize(img)
                        if results:
                            name, sim = results[0]
                            now = time.monotonic()
                            if now - self._last_said.get(name, 0.0) >= self._say_cooldown:
                                self._last_said[name] = now
                                self._announce(name, sim)
                except Exception as exc:
                    logger.warning("人脸识别失败: %s", exc)
            self._face_stop.wait(self._face_interval)

    def _announce(self, name: str, sim: float) -> None:
        """识别到人后触发机器狗播报，并按该人的分类触发对应动作。"""
        text = f"识别到{name}"
        logger.info("识别到 %s（相似度 %.2f），播报: %s", name, sim, text)
        try:
            self.say(text)
        except Exception as exc:
            logger.warning("播报失败: %s", exc)

        # 按人脸分类触发动作（分类为空则不动作）
        if self._motion is not None and self._face_recognizer is not None:
            category = self._face_recognizer.category_of(name)
            if category:
                try:
                    self._motion.do_action(category)
                except Exception as exc:
                    logger.warning("触发动作 %s 失败: %s", category, exc)

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

    def _apply_auto_target(self) -> None:
        """自动选择链路：AP 热点 → 有线直连 → 组播发现 → 回退 AP 等待。"""
        for candidate in _AUTO_CANDIDATES:
            if _port_open(candidate):
                self._method = WebRTCConnectionMethod.LocalSTA
                self._ip = candidate
                self._target = f"{candidate} (自动发现)"
                logger.info("自动发现机器狗: %s", candidate)
                return
        try:
            found = discover_ip_sn(timeout=2)
            if found:
                ip = next(iter(found.values()))
                self._method = WebRTCConnectionMethod.LocalSTA
                self._ip = ip
                self._target = f"{ip} (组播发现)"
                logger.info("组播发现机器狗: %s", ip)
                return
        except Exception as exc:
            logger.debug("组播发现失败: %s", exc)
        self._method = WebRTCConnectionMethod.LocalAP
        self._ip = None
        self._target = "192.168.12.1 (AP 直连，等待连接)"

    def _run(self) -> None:
        # Windows 默认的 Proactor 事件循环在子线程里会触发 aiortc 的
        # __connect 竞态（remoteDescription 尚未设置就调用 __remoteRtp），
        # 导致连接卡死。改用 Selector 事件循环可稳定建立连接。
        loop = asyncio.SelectorEventLoop()
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
            if self._auto:
                self._apply_auto_target()
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

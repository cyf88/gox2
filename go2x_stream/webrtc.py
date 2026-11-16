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
from .tts import prepare_wav, prewarm

logger = logging.getLogger(__name__)

# 人脸判定：强匹配直接认定；弱匹配需要在时间窗内命中两次（应对走动时的模糊/侧脸）
_FACE_STRONG_SIM = 0.40
_FACE_WEAK_SIM = 0.33
_FACE_VOTE_WINDOW = 2.0
# 检测到人脸但未认出时，先等这么久（给识别留时间），再触发通用播报/动作
_UNKNOWN_GRACE = 1.0

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
        custom_actions=None,
        face_trigger=None,
        face_stop_store=None,
        patrol_store=None,
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
        # 人脸检测结果缓存（供 MJPEG 叠加框复用，避免重复推理）
        self._face_lock = threading.Lock()
        self._face_results: list = []
        self._face_seq = -1

        # 音频采集（机器狗麦克风 → WebSocket 推给网页播放）
        self._audio_lock = threading.Lock()
        self._audio_chunks: deque[bytes] = deque(maxlen=100)
        self._audio_seq = 0
        self._audio_rate = 48000
        self._audio_clients = 0
        self._audio_logged = False
        self._audio_sink_lock = threading.Lock()
        self._audio_sinks: list = []
        self._audio_channel_on = False

        # 运动控制（可选，WebRTC sport 通道）
        self._motion = motion

        # 自定义动作（可选，绑定到人脸后自动执行）
        self._custom_actions = custom_actions

        # 人脸触发（检测到任意人脸即播报 + 动作，无需识别身份）
        self._face_trigger = face_trigger
        # 检测到人脸时暂停移动的配置
        self._face_stop_store = face_stop_store
        # 巡逻路线
        self._patrol_store = patrol_store
        self._face_retry_left = 2  # 未认出时还能补检几次（换人/认出后重置）
        self._face_hits: dict[str, list[float]] = {}  # 姓名 -> 最近命中时间（弱匹配投票用）
        self._unknown_since = 0.0  # 检测到「未认出的人脸」的起始时间
        self._ft_last_trigger = 0.0

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

        # 后台预生成已有播报文本的语音（不阻塞启动）
        if self._face_recognizer is not None or self._custom_actions is not None:
            threading.Thread(
                target=self._prewarm_tts, name="go2-tts-prewarm", daemon=True
            ).start()

    def _prewarm_tts(self) -> None:
        """预生成所有可能用到的播报语音，首次识别时无需等待合成。"""
        try:
            texts: list[str] = []
            if self._custom_actions is not None:
                for item in self._custom_actions.list_actions():
                    say = (item.get("say") or "").strip()
                    if say:
                        texts.append(say)
            if self._face_recognizer is not None:
                texts.extend(f"识别到{n}" for n in self._face_recognizer.registered_names())
            if self._face_trigger is not None:
                ft_say = (self._face_trigger.get().get("say") or "").strip()
                if ft_say:
                    texts.append(ft_say)
            if not texts:
                return
            done = 0
            for text in texts:
                if self._stop.is_set():
                    break
                if prewarm(text):
                    done += 1
            logger.info("已预生成 %d 条播报语音", done)
        except Exception as exc:
            logger.warning("预生成语音失败: %s", exc)

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

    def sport_action(self, api_id: int, parameter: Optional[dict] = None, timeout: float = 8.0) -> Optional[dict]:
        """经 WebRTC 数据通道下发一条 sport 指令（线程安全，阻塞到收到应答）。"""
        return self.raw_request(RTC_TOPIC["SPORT_MOD"], api_id, parameter, timeout)

    def raw_request(
        self,
        topic: str,
        api_id: int,
        parameter: Optional[dict] = None,
        timeout: float = 8.0,
        priority: bool = False,
    ) -> Optional[dict]:
        """向任意 rt/api/*/request 主题下发一条请求（调试用）。"""
        loop, conn = self._loop, self._conn
        if conn is None or loop is None or loop.is_closed():
            return None
        payload = {"api_id": api_id, "parameter": parameter or {}}
        if priority:
            payload["priority"] = 1
        try:
            future = asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(topic, payload),
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

    def register_face(self, name: str, image_bytes: bytes) -> dict:
        """从上传的图片字节注册人脸（需已启用人脸识别）。"""
        if self._face_recognizer is None:
            raise RuntimeError("未启用人脸识别（启动服务时需加 --face）")
        import numpy as np

        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("无法解码图片，请上传 JPG/PNG 格式")
        self._face_recognizer.register_image(name, img)
        # 预生成默认播报语音，识别时无需再合成
        prewarm(f"识别到{name}")
        return {"ok": True, "name": name}

    def registered_faces(self) -> list[dict]:
        """返回已注册的人脸列表（含姓名与绑定的自定义动作名）。"""
        if self._face_recognizer is None:
            return []
        faces = []
        for n in self._face_recognizer.registered_names():
            action = self._custom_actions.assignment_of(n) if self._custom_actions else ""
            faces.append({"name": n, "action": action})
        return faces

    def do_action(self, action: str) -> dict:
        """触发机器狗动作（需已启用运动控制）。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.do_action(action)

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> dict:
        """设置持续速度指令（前进/后退/侧移/旋转）。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.move(vx, vy, vyaw)

    def stop_move(self) -> dict:
        """停止移动（仅速度控制）。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.stop_move()

    def do_sequence(self, steps: list) -> dict:
        """串行执行一组动作/移动/延时。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.do_sequence(steps)

    def patrol_start(self, vx: float = 0.5, vyaw: float = 0.6,
                     forward_s: float = 4.0, turn_s: float = 2.0) -> dict:
        """开始巡逻（前进 + 转向循环）；检测到人脸会自动暂停。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.start_patrol(vx, vyaw, forward_s, turn_s)

    def patrol_stop(self) -> dict:
        """停止巡逻。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        return self._motion.stop_patrol()

    # ─── 巡逻路线 ───────────────────────────────────────────────

    def patrol_route_get(self) -> dict:
        if self._patrol_store is None:
            return {"steps": [], "loop": True}
        return self._patrol_store.get()

    def patrol_route_save(self, steps: list, loop: bool = True) -> dict:
        if self._patrol_store is None:
            raise RuntimeError("巡逻路线未启用")
        return self._patrol_store.save(steps, loop)

    def patrol_route_start(self) -> dict:
        """按已保存的路线开始巡逻。"""
        if self._motion is None:
            raise RuntimeError("未启用运动控制")
        cfg = self.patrol_route_get()
        steps = cfg.get("steps") or []
        if not steps:
            raise RuntimeError("巡逻路线为空，请先编排并保存")
        return self._motion.start_route(steps, bool(cfg.get("loop", True)))

    def list_actions(self) -> dict:
        """返回所有可用动作定义（给前端组合编辑器）。"""
        if self._motion is None:
            return {"actions": [], "velocity_range": {}}
        return self._motion.list_actions()

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
        """删除一个已注册的人脸，并清理其自定义动作绑定。"""
        if self._face_recognizer is None:
            raise RuntimeError("未启用人脸识别（启动服务时需加 --face）")
        if not self._face_recognizer.delete(name):
            raise RuntimeError(f"未找到已注册人脸: {name}")
        if self._custom_actions is not None:
            self._custom_actions.remove_face(name)
        return {"ok": True, "name": name}

    # ─── 自定义动作 ─────────────────────────────────────────────

    def custom_actions_list(self) -> list[dict]:
        """返回所有自定义动作。"""
        if self._custom_actions is None:
            return []
        return self._custom_actions.list_actions()

    def save_custom_action(self, name: str, say: str, steps: list) -> dict:
        """新建或覆盖一个自定义动作（动作序列 + 播报文本）。"""
        if self._custom_actions is None:
            raise RuntimeError("未启用自定义动作存储")
        result = self._custom_actions.save(name, say, steps)
        # 保存时预生成语音，识别到人时直接上传，无需现场合成
        if (say or "").strip():
            result["tts_ready"] = prewarm(say)
        return result

    def delete_custom_action(self, name: str) -> dict:
        """删除一个自定义动作。"""
        if self._custom_actions is None:
            raise RuntimeError("未启用自定义动作存储")
        if not self._custom_actions.delete(name):
            raise RuntimeError(f"未找到自定义动作: {name}")
        return {"ok": True, "name": name}

    def assign_face_action(self, face_name: str, action_name: str) -> dict:
        """把已注册人脸绑定到某个自定义动作（传空字符串解除绑定）。"""
        if self._custom_actions is None:
            raise RuntimeError("未启用自定义动作存储")
        if self._face_recognizer is not None and face_name not in self._face_recognizer.registered_names():
            raise RuntimeError(f"未找到已注册人脸: {face_name}")
        return self._custom_actions.assign(face_name, action_name or "")

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
        """检测单帧图像，返回 [(姓名, 相似度, bbox)]（供离线视频检测使用）。"""
        if self._face_recognizer is None:
            return []
        return self._face_recognizer.detect_with_boxes(frame)

    def latest_detections(self) -> tuple[int, list]:
        """返回最近一次人脸检测的 (帧序号, [(姓名, 相似度, bbox)])。

        画面叠加直接复用该缓存，避免每个 MJPEG 客户端各自跑一遍推理。
        """
        with self._face_lock:
            return self._face_seq, list(self._face_results)

    # ─── 人脸识别 ───────────────────────────────────────────────

    def _face_loop(self) -> None:
        """人脸识别循环：定时检测一次，结果缓存给画面叠加并触发播报。"""
        import numpy as np

        while not self._face_stop.is_set():
            frame = self.latest_frame()
            if frame is not None:
                try:
                    img = cv2.imdecode(
                        np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        candidates, face_count = self._face_recognizer.match_faces(img)
                        results = self._accept_faces(candidates)
                        if face_count <= 0 or results:
                            self._face_retry_left = 2
                        # 有脸但仍未判定：换后面的帧补检（每次出现最多补 2 次）
                        retries = 0
                        while face_count > 0 and not results and self._face_retry_left > 0:
                            retries += 1
                            self._face_retry_left -= 1
                            if self._face_stop.wait(0.15):
                                break
                            nxt = self.latest_frame()
                            if nxt is None:
                                break
                            nimg = cv2.imdecode(
                                np.frombuffer(nxt.jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            if nimg is None:
                                break
                            candidates, face_count = self._face_recognizer.match_faces(nimg)
                            results = self._accept_faces(candidates)
                            if results:
                                self._face_retry_left = 2
                        if results and retries:
                            logger.info("人脸补检 %d 次后识别成功", retries)
                        with self._face_lock:
                            self._face_results = results
                            self._face_seq = frame.seq
                        self._handle_face_presence(face_count, len(results))
                        if results:
                            name, sim, _ = max(results, key=lambda r: r[1])
                            now = time.monotonic()
                            if now - self._last_said.get(name, 0.0) >= self._say_cooldown:
                                self._last_said[name] = now
                                self._announce(name, sim)
                except Exception as exc:
                    logger.warning("人脸识别失败: %s", exc)
            self._face_stop.wait(self._face_interval)

    def _accept_faces(
        self, candidates: list[tuple[Optional[str], float, list[int]]]
    ) -> list[tuple[str, float, list[int]]]:
        """把每张脸的最佳匹配转成判定结果。

        相似度 >= 0.40 直接认定；0.33~0.40 之间需要在 2 秒内再命中一次，避免走动模糊时漏检。
        """
        now = time.monotonic()
        accepted: list[tuple[str, float, list[int]]] = []
        for name, sim, bbox in candidates:
            if not name or sim < _FACE_WEAK_SIM:
                continue
            hits = [t for t in self._face_hits.get(name, []) if now - t <= _FACE_VOTE_WINDOW]
            hits.append(now)
            self._face_hits[name] = hits
            if sim >= _FACE_STRONG_SIM or len(hits) >= 2:
                accepted.append((name, round(sim, 2), bbox))
        for name in list(self._face_hits):
            recent = [t for t in self._face_hits[name] if now - t <= _FACE_VOTE_WINDOW]
            if recent:
                self._face_hits[name] = recent
            else:
                self._face_hits.pop(name, None)
        return accepted

    def _announce(self, name: str, sim: float) -> None:
        """识别到人后播报并按绑定的自定义动作执行动作序列。"""
        custom = self._custom_actions.action_of(name) if self._custom_actions else None
        text = (custom or {}).get("say") or f"识别到{name}"
        logger.info("识别到 %s（相似度 %.2f），播报: %s", name, sim, text)
        try:
            self.say(text)
        except Exception as exc:
            logger.warning("播报失败: %s", exc)

        if custom and custom.get("steps") and self._motion is not None:
            try:
                self._motion.do_sequence(custom.get("steps") or [])
            except Exception as exc:
                logger.warning("执行自定义动作 %s 失败: %s", custom.get("name"), exc)

    def _handle_face_presence(self, face_count: int, recognized: int = 0) -> None:
        """有人脸就暂停移动；只有「没认出是谁」时才触发通用播报/动作。

        认出是谁时改由 _announce 播个性化内容；未认出时先等 _UNKNOWN_GRACE 秒，
        避免识别慢半拍时又重复播通用内容。
        """
        if face_count <= 0:
            self._unknown_since = 0.0
            return
        self._hold_on_face()
        if recognized:
            self._unknown_since = 0.0
            return
        now = time.monotonic()
        if self._unknown_since <= 0.0:
            self._unknown_since = now
            return
        if now - self._unknown_since >= _UNKNOWN_GRACE:
            self._unknown_since = 0.0
            self._trigger_face_action()

    def _hold_on_face(self) -> None:
        """检测到人脸时暂停移动一段时间（配置可关闭）。"""
        if self._face_stop_store is None or self._motion is None:
            return
        cfg = self._face_stop_store.get()
        if not cfg.get("enabled"):
            return
        try:
            self._motion.hold(float(cfg.get("seconds", 3.0)))
        except Exception as exc:
            logger.warning("暂停移动失败: %s", exc)

    def _trigger_face_action(self) -> None:
        """按「人脸触发」配置播报并执行动作（不需要识别身份）。"""
        if self._face_trigger is None:
            return
        cfg = self._face_trigger.get()
        if not cfg.get("enabled"):
            return
        now = time.monotonic()
        cooldown = float(cfg.get("cooldown", 10.0))
        if now - self._ft_last_trigger < cooldown:
            return
        self._ft_last_trigger = now

        say_text = (cfg.get("say") or "").strip()
        steps = cfg.get("steps") or []
        logger.info(
            "检测到未识别的人脸，触发通用动作（播报: %s，%d 步）",
            say_text or "无", len(steps),
        )

        def _say() -> None:
            try:
                self.say(say_text)
            except Exception as exc:
                logger.warning("人脸触发播报失败: %s", exc)

        def _run() -> None:
            motion = self._motion
            if motion is not None:
                try:
                    # 先站立，站起来之后再播报/做动作
                    motion.do_sequence(
                        [{"action": "standup", "duration": 2.5}], wait=True
                    )
                except Exception as exc:
                    logger.warning("人脸触发起立失败: %s", exc)
            if say_text:
                # 播报与动作并行：音频走 audiohub，动作走 sport 通道，互不阻塞
                threading.Thread(
                    target=_say, name="go2-face-say", daemon=True
                ).start()
            if steps and motion is not None:
                try:
                    motion.do_sequence(steps)
                except Exception as exc:
                    logger.warning("人脸触发动作失败: %s", exc)

        threading.Thread(target=_run, name="go2-face-trigger", daemon=True).start()

    # ─── 人脸触发 / 暂停移动配置 ─────────────────────────────────

    def face_trigger_get(self) -> dict:
        if self._face_trigger is None:
            return {"enabled": False, "say": "", "steps": [], "cooldown": 10.0}
        return self._face_trigger.get()

    def face_trigger_save(self, payload: dict) -> dict:
        if self._face_trigger is None:
            raise RuntimeError("人脸触发未启用（启动服务时需加 --face）")
        result = self._face_trigger.save(payload)
        say = (result.get("say") or "").strip()
        if say:
            result["tts_ready"] = prewarm(say)
        return result

    def face_stop_get(self) -> dict:
        if self._face_stop_store is None:
            return {"enabled": False, "seconds": 3.0}
        return self._face_stop_store.get()

    def face_stop_save(self, payload: dict) -> dict:
        if self._face_stop_store is None:
            raise RuntimeError("人脸暂停移动未启用（启动服务时需加 --face）")
        return self._face_stop_store.save(payload)

    async def _say_async(self, text: str) -> dict:
        conn = self._conn
        # 已预生成则直接复用缓存文件；否则现场合成（阻塞调用放线程池）
        path, is_temp = await asyncio.to_thread(prepare_wav, text)
        try:
            unique_id = await self._upload_and_play(conn, path)
            logger.info("已触发播报, unique_id=%s", unique_id)
            return {"ok": True, "unique_id": unique_id}
        finally:
            if is_temp:
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
        # 注意：WebRTC 数据通道单条消息上限 65536 字节，块不能超过该值
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
            if i < total_chunks:
                await asyncio.sleep(0.005)

        # 上传响应里通常只有 current_block_index，需要查列表按文件名匹配 unique_id
        list_response = await pub_sub.publish_request_new(
            "rt/api/audiohub/request",
            {"api_id": 1001, "parameter": json.dumps({})},
        )
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
                conn.audio.add_track_callback(self._receive_audio)
                # 只有确实需要音频（网页播放/语音对话）时才让机器狗推音频
                self._audio_channel_on = False
                self._sync_audio_channel()

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
                self._audio_channel_on = False
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

    async def _receive_audio(self, frame) -> None:
        """把机器狗麦克风音频帧转成 16bit 单声道 PCM，转给播放/对话模块。"""
        with self._audio_lock:
            want_stream = self._audio_clients > 0
        with self._audio_sink_lock:
            sinks = list(self._audio_sinks)
        if not want_stream and not sinks:
            return
        try:
            import numpy as np

            pcm = frame.to_ndarray()
            if not self._audio_logged:
                self._audio_logged = True
                logger.info(
                    "音频帧格式: rate=%s layout=%s dtype=%s shape=%s",
                    frame.sample_rate, getattr(frame.layout, "name", "?"),
                    pcm.dtype, pcm.shape,
                )
            if pcm.dtype != np.int16:
                pcm = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
            channels = int(getattr(frame.layout, "nb_channels", 1) or 1)
            if pcm.ndim == 2 and channels > 1 and pcm.shape[0] == channels:
                # 平面格式：(channels, samples)，按声道取均值
                pcm = pcm.mean(axis=0).astype(np.int16)
            else:
                flat = pcm.reshape(-1)
                if channels > 1 and flat.size and flat.size % channels == 0:
                    # 打包格式：交错存放 (samples*channels,)，重排后取均值
                    pcm = flat.reshape(-1, channels).mean(axis=1).astype(np.int16)
                else:
                    pcm = flat
        except Exception as exc:
            logger.debug("音频帧转换失败: %s", exc)
            return

        rate = int(getattr(frame, "sample_rate", 0) or self._audio_rate)
        with self._audio_lock:
            self._audio_rate = rate
            if want_stream:
                self._audio_chunks.append(pcm.tobytes())
                self._audio_seq += 1

        for callback in sinks:
            try:
                callback(pcm, rate)
            except Exception as exc:
                logger.debug("音频回调失败: %s", exc)

    # ─── 音频对外接口（供 WebSocket 使用）────────────────────────

    def audio_begin(self) -> None:
        """登记一个音频客户端；从此刻起开始缓存音频。"""
        with self._audio_lock:
            self._audio_chunks.clear()
            self._audio_clients += 1
        self._sync_audio_channel()

    def add_audio_sink(self, callback) -> None:
        """注册常驻音频消费者（如语音对话），与网页播放互不影响。"""
        with self._audio_sink_lock:
            if callback not in self._audio_sinks:
                self._audio_sinks.append(callback)
        self._sync_audio_channel()

    def remove_audio_sink(self, callback) -> None:
        with self._audio_sink_lock:
            if callback in self._audio_sinks:
                self._audio_sinks.remove(callback)
        self._sync_audio_channel()

    def _sync_audio_channel(self) -> None:
        """按需开关机器狗音频通道：无人播放且未启用对话时关闭，省 CPU。"""
        conn, loop = self._conn, self._loop
        if conn is None or loop is None or loop.is_closed():
            return
        with self._audio_lock:
            want = self._audio_clients > 0
        with self._audio_sink_lock:
            want = want or bool(self._audio_sinks)
        if want == self._audio_channel_on:
            return
        try:
            loop.call_soon_threadsafe(self._switch_audio_channel, want)
        except RuntimeError:
            pass

    def _switch_audio_channel(self, on: bool) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            conn.audio.switchAudioChannel(on)
            self._audio_channel_on = on
            logger.info("机器狗音频通道已%s", "开启" if on else "关闭")
        except Exception as exc:
            logger.debug("切换音频通道失败: %s", exc)

    def audio_end(self) -> None:
        with self._audio_lock:
            self._audio_clients = max(0, self._audio_clients - 1)
        self._sync_audio_channel()

    def audio_seq(self) -> int:
        with self._audio_lock:
            return self._audio_seq

    def audio_info(self) -> dict:
        with self._audio_lock:
            return {
                "rate": self._audio_rate,
                "channels": 1,
                "clients": self._audio_clients,
            }

    def audio_read(self, since: int) -> tuple[int, int, bytes]:
        """返回 (当前序号, 采样率, 自 since 之后的新增 PCM 字节)。"""
        with self._audio_lock:
            seq = self._audio_seq
            rate = self._audio_rate
            if since >= seq:
                return seq, rate, b""
            pending = seq - since
            chunks = list(self._audio_chunks)
            if pending > len(chunks):
                # 客户端落后过多，只给当前缓存里的全部
                pending = len(chunks)
            return seq, rate, b"".join(chunks[-pending:]) if pending else b""

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

"""语音对话：唤醒词 → 本地 ASR（faster-whisper）→ DeepSeek → 本地 TTS 播报。

音频来源是机器狗麦克风（由 WebRTC 拉流层通过 add_audio_sink 推送 PCM）。
整个链路在后台线程串行执行，不会阻塞视频与音频转发。
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WAKE_WORD = "小狗小狗"

_PUNCT_RE = re.compile(r"[\s，。！？、,.!?：:；;“”\"'（）()【】\[\]…—~]+")


def normalize_text(text: str) -> str:
    """去掉空白与标点，便于唤醒词匹配。"""
    return _PUNCT_RE.sub("", text or "")


def resample_to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """把 int16 PCM 重采样到 16kHz（整数倍用均值抽取，否则线性插值）。"""
    pcm = np.asarray(pcm).reshape(-1)
    if rate == 16000 or pcm.size == 0:
        return pcm
    if rate % 16000 == 0:
        factor = rate // 16000
        usable = (pcm.size // factor) * factor
        if usable == 0:
            return pcm
        return pcm[:usable].reshape(-1, factor).mean(axis=1).astype(np.int16)
    n_out = max(1, int(pcm.size * 16000 / rate))
    x = np.linspace(0, pcm.size - 1, n_out)
    return np.interp(x, np.arange(pcm.size), pcm).astype(np.int16)


class AsrEngine:
    """faster-whisper 本地离线识别（CPU / int8）。"""

    def __init__(
        self,
        model_size: str = "small",
        threads: int = 2,
        language: str = "zh",
    ) -> None:
        # 国内直连 HuggingFace 常常失败，默认走镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from faster_whisper import WhisperModel

        self._language = language
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=max(1, threads)
        )
        logger.info("语音识别就绪（faster-whisper %s / int8 / %d 线程）", model_size, threads)

    def transcribe(self, pcm: np.ndarray, rate: int) -> str:
        audio = resample_to_16k(pcm, rate).astype(np.float32) / 32768.0
        if audio.size < 1600:  # 少于 0.1 秒，视为无效
            return ""
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()


class ChatClient:
    """DeepSeek（OpenAI 兼容协议）多轮对话客户端。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = "",
        history_rounds: int = 6,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._system_prompt = system_prompt
        self._history_rounds = max(0, history_rounds)
        self._timeout = timeout
        self._history: list[dict] = []
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return bool(self._api_key)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def ask(self, text: str) -> str:
        if not self._api_key:
            raise RuntimeError("未配置 DeepSeek API Key（config.json → chat.api_key）")
        import requests

        with self._lock:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.extend(self._history)
            messages.append({"role": "user", "content": text})

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": False,
            },
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek 返回 {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        reply = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        reply = (reply or "").strip()
        if not reply:
            raise RuntimeError("DeepSeek 返回空回复")

        with self._lock:
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            keep = self._history_rounds * 2
            if keep and len(self._history) > keep:
                self._history = self._history[-keep:]
        return reply


class VoiceChat:
    """唤醒词触发 → 录音切片 → 识别 → 对话 → 播报。"""

    def __init__(
        self,
        say: Callable[[str], Any],
        asr: AsrEngine,
        chat: ChatClient,
        wake_word: str = DEFAULT_WAKE_WORD,
        vad_rms: float = 500.0,
        silence_ms: int = 600,
        max_utterance_ms: int = 8000,
        session_idle_s: float = 10.0,
        ack_text: str = "我在",
        on_event: Optional[Callable[[dict], None]] = None,
        on_enabled_change: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._say = say
        self._asr = asr
        self._chat = chat
        self._wake_word = wake_word or DEFAULT_WAKE_WORD
        self._wake_norm = normalize_text(self._wake_word)
        self._vad_rms = vad_rms
        self._silence_ms = silence_ms
        self._max_utterance_ms = max_utterance_ms
        self._session_idle_s = session_idle_s
        self._ack_text = ack_text
        self._on_event = on_event
        self._on_enabled_change = on_enabled_change

        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._enabled = True
        self._lock = threading.Lock()

        # 状态（供页面展示）
        self._in_session = False
        self._last_active = 0.0
        self._last_text = ""
        self._last_reply = ""
        self._last_asr = ""
        self._last_error = ""
        self._listening = False

    # ─── 生命周期 ───────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="go2x-voice-chat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ─── 对外接口 ───────────────────────────────────────────────

    @property
    def ack_text(self) -> str:
        return self._ack_text

    def feed(self, pcm: np.ndarray, rate: int) -> None:
        """由拉流层回调：送入一帧 PCM（int16 单声道）。"""
        if not self._enabled or self._stop.is_set():
            return
        try:
            self._queue.put_nowait((pcm.copy(), int(rate)))
        except queue.Full:
            pass

    def update_tuning(
        self,
        vad_rms: Optional[float] = None,
        silence_ms: Optional[int] = None,
        max_utterance_ms: Optional[int] = None,
        session_idle_s: Optional[float] = None,
    ) -> None:
        """运行时更新语音算法参数（页面「算法参数」保存时调用）。"""
        with self._lock:
            if vad_rms is not None:
                self._vad_rms = max(1.0, float(vad_rms))
            if silence_ms is not None:
                self._silence_ms = max(100, int(silence_ms))
            if max_utterance_ms is not None:
                self._max_utterance_ms = max(500, int(max_utterance_ms))
            if session_idle_s is not None:
                self._session_idle_s = max(1.0, float(session_idle_s))

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            self._in_session = False
            self._last_active = 0.0
        logger.info("语音对话已%s", "开启" if enabled else "关闭")
        if self._on_enabled_change is not None:
            try:
                self._on_enabled_change(bool(enabled))
            except Exception:
                pass
        self._emit()

    def ask_and_reply(self, text: str) -> dict:
        """直接文字提问（页面按钮/接口测试用）。"""
        text = (text or "").strip()
        if not text:
            raise ValueError("问题不能为空")
        reply = self._chat.ask(text)
        self._report(text, reply)
        try:
            self._say(reply)
        except Exception as exc:
            logger.warning("播报失败: %s", exc)
        return {"question": text, "reply": reply}

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "listening": self._enabled and not self._stop.is_set(),
                "in_session": self._in_session,
                "wake_word": self._wake_word,
                "api_ready": self._chat.ready,
                "last_text": self._last_text,
                "last_reply": self._last_reply,
                "last_asr": self._last_asr,
                "error": self._last_error,
            }

    # ─── 后台线程 ───────────────────────────────────────────────

    def _run(self) -> None:
        buf: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []
        recording = False
        silence_ms = 0.0
        total_ms = 0.0
        chunk_ms = 20.0

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                # 空闲时结束超时会话
                self._maybe_end_session()
                continue
            if item is None:
                continue
            pcm, rate = item

            block_ms = 1000.0 * len(pcm) / max(1, rate)
            chunk_ms = block_ms or chunk_ms
            rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32))))) if pcm.size else 0.0

            if not recording:
                if not self._enabled:
                    continue
                pre_roll.append(pcm)
                if rate:
                    keep = max(1, int(300 / block_ms)) if block_ms else 15
                    pre_roll = pre_roll[-keep:]
                if rms >= self._vad_rms:
                    recording = True
                    silence_ms = 0.0
                    total_ms = 0.0
                    buf = list(pre_roll)
                    pre_roll = []
                    self._set_listening(True)
                    self._touch()
                continue

            buf.append(pcm)
            total_ms += block_ms
            silence_ms = 0.0 if rms >= self._vad_rms else silence_ms + block_ms

            if silence_ms >= self._silence_ms or total_ms >= self._max_utterance_ms:
                recording = False
                self._set_listening(False)
                data = np.concatenate(buf) if buf else np.empty(0, dtype=np.int16)
                buf = []
                pre_roll = []
                self._touch()
                self._handle_utterance(data, rate)
            self._maybe_end_session()

    def _touch(self) -> None:
        """有新的语音输入时刷新会话活跃时间，避免说话途中被判超时。"""
        with self._lock:
            if self._in_session:
                self._last_active = time.monotonic()

    def _maybe_end_session(self) -> None:
        with self._lock:
            if not self._in_session:
                return
            if time.monotonic() - self._last_active <= self._session_idle_s:
                return
            self._in_session = False
            idle = self._session_idle_s
        logger.info("对话会话超时结束（%.0f 秒无语音输入）", idle)
        self._chat.reset()
        self._emit()

    def _handle_utterance(self, pcm: np.ndarray, rate: int) -> None:
        duration_ms = 1000.0 * len(pcm) / max(1, rate)
        try:
            text = self._asr.transcribe(pcm, rate)
        except Exception as exc:
            logger.warning("语音识别失败: %s", exc)
            self._set_error(f"语音识别失败: {exc}")
            return
        if not text:
            logger.info("语音识别结果为空（音频 %.0f ms）", duration_ms)
            return
        logger.info("识别到语音: %s", text)
        with self._lock:
            self._last_asr = text
        self._emit()

        norm = normalize_text(text)
        with self._lock:
            in_session = self._in_session

        if in_session:
            query = text
            if self._wake_norm and self._wake_norm in norm:
                # 会话中重复唤醒词，直接去掉
                query = norm.split(self._wake_norm, 1)[1] or ""
                query = query or text
            self._handle_query(query)
            return

        if not self._wake_norm or self._wake_norm not in norm:
            logger.info("未命中唤醒词「%s」，忽略: %s", self._wake_word, text)
            return  # 未唤醒

        remainder = norm.split(self._wake_norm, 1)[1]
        with self._lock:
            self._in_session = True
            self._last_active = time.monotonic()
        logger.info("唤醒词命中")
        if remainder:
            self._handle_query(remainder)
        else:
            self._speak(self._ack_text)
            self._report(text, "")
            self._emit()

    def _handle_query(self, query: str) -> None:
        query = (query or "").strip()
        if not query:
            return
        try:
            reply = self._chat.ask(query)
        except Exception as exc:
            logger.warning("大模型调用失败: %s", exc)
            self._set_error(str(exc))
            self._speak("抱歉，我现在连不上大脑了")
            return
        with self._lock:
            self._last_active = time.monotonic()
            self._in_session = True
        self._report(query, reply)
        self._speak(reply)

    def _speak(self, text: str) -> None:
        try:
            self._say(text)
        except Exception as exc:
            logger.warning("播报失败: %s", exc)
            self._set_error(f"播报失败: {exc}")

    def _report(self, text: str, reply: str) -> None:
        with self._lock:
            self._last_text = text
            self._last_reply = reply
            self._last_error = ""
        logger.info("对话: %s → %s", text, reply)
        self._emit()

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        self._emit()

    def _set_listening(self, value: bool) -> None:
        with self._lock:
            changed = self._listening != value
            self._listening = value
        if changed:
            self._emit()

    def _emit(self) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(self.status())
        except Exception:
            pass

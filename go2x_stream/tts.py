"""本地文字转语音：调用 Windows SAPI（中文 Huihui）离线合成 WAV。

使用 comtypes 在进程内调用 SAPI，并通过常驻线程复用语音引擎，
避免每次播报都启动 PowerShell 进程（约 4 秒）的开销。

采样率用 16kHz 单声道：语音足够清晰，WAV 体积只有 48kHz 的 1/3，
上传到机器狗的耗时显著缩短；相同文本还会命中缓存直接复用。
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import comtypes.client

logger = logging.getLogger(__name__)

# SAPI 格式枚举：SAFT16kHz16BitMono（16000Hz / 单声道 / 16bit）
_SAFT_16K16BIT_MONO = 18
_VOICE_HINT = "Huihui"

# 合成结果缓存：相同文本直接复用，省掉重复合成与文件读写
_CACHE_MAX = 32
_cache: "OrderedDict[str, bytes]" = OrderedDict()
_cache_lock = threading.Lock()


class _TtsWorker:
    """常驻线程持有 SpVoice 对象，串行合成语音（保证 COM 线程亲和）。"""

    def __init__(self) -> None:
        self._in = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="go2x-tts", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        comtypes.CoInitialize()
        try:
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
            tokens = voice.GetVoices()
            for i in range(tokens.Count):
                if _VOICE_HINT in tokens.Item(i).GetDescription():
                    voice.Voice = tokens.Item(i)
                    break
        except Exception as exc:  # 初始化失败时后续调用都返回错误
            self._voice_error = str(exc)
            voice = None
        else:
            self._voice_error = None

        try:
            while True:
                item = self._in.get()
                if item is None:
                    break
                text, result_q = item
                try:
                    if voice is None:
                        raise RuntimeError(self._voice_error or "TTS 引擎初始化失败")
                    result_q.put(("ok", self._synthesize(voice, text)))
                except Exception as exc:
                    result_q.put(("err", str(exc)))
        finally:
            comtypes.CoUninitialize()

    def _synthesize(self, voice, text: str) -> bytes:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="go2x_tts_")
        os.close(fd)
        try:
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            fmt = comtypes.client.CreateObject("SAPI.SpAudioFormat")
            fmt.Type = _SAFT_16K16BIT_MONO
            stream.Format = fmt
            stream.Open(path, 3)  # SSFMCreateForWrite
            voice.AudioOutputStream = stream
            try:
                voice.Speak(text)
            finally:
                voice.AudioOutputStream = None
                stream.Close()
            return Path(path).read_bytes()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def synthesize(self, text: str) -> bytes:
        result_q = queue.Queue()
        self._in.put((text, result_q))
        status, data = result_q.get()
        if status == "err":
            raise RuntimeError(data)
        return data


_worker = None
_worker_lock = threading.Lock()


def synthesize_wav(text: str, voice: str = "Microsoft Huihui Desktop") -> bytes:
    """把文字合成为 16000Hz / 单声道 / 16bit 的 WAV 字节（带缓存）。"""
    if not text.strip():
        raise ValueError("播报文字不能为空")

    with _cache_lock:
        cached = _cache.get(text)
        if cached is not None:
            _cache.move_to_end(text)
            return cached

    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _TtsWorker()
    wav = _worker.synthesize(text)

    with _cache_lock:
        _cache[text] = wav
        _cache.move_to_end(text)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return wav


def write_temp_wav(wav: bytes) -> str:
    """把 WAV 字节写入临时文件并返回路径，供上传接口使用。"""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="go2x_say_")
    with open(fd, "wb") as f:
        f.write(wav)
    return path


# ─── 磁盘缓存：编辑时预生成，播报时直接读文件 ────────────────────

_cache_dir: Optional[Path] = None


def configure_cache(directory) -> None:
    """设置预生成语音的存放目录（通常在 faces/tts_cache）。"""
    global _cache_dir
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    _cache_dir = p
    logger.info("语音缓存目录: %s", p)


def _disk_path(text: str) -> Optional[Path]:
    if _cache_dir is None:
        return None
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return _cache_dir / f"{digest}.wav"


def cached_path(text: str) -> Optional[str]:
    """返回已预生成的语音文件路径（不存在返回 None）。"""
    p = _disk_path(text)
    return str(p) if p is not None and p.exists() else None


def prepare_wav(text: str) -> tuple[str, bool]:
    """返回可直接上传的 WAV 路径与「是否为临时文件」。

    优先使用磁盘缓存（编辑时已预生成）；没有则现场合成并写入缓存。
    """
    existing = cached_path(text)
    if existing:
        return existing, False

    wav = synthesize_wav(text)
    target = _disk_path(text)
    if target is None:
        return write_temp_wav(wav), True
    try:
        target.write_bytes(wav)
        return str(target), False
    except OSError as exc:
        logger.warning("写入语音缓存失败 %s: %s", target, exc)
        return write_temp_wav(wav), True


def prewarm(text: str) -> bool:
    """预先合成并落盘（供保存自定义动作、注册人脸时调用）。"""
    text = (text or "").strip()
    if not text:
        return False
    try:
        prepare_wav(text)
        return True
    except Exception as exc:
        logger.warning("预生成语音失败「%s」: %s", text, exc)
        return False

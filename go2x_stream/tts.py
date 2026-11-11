"""本地文字转语音：调用 Windows SAPI（中文 Huihui）离线合成 WAV。

使用 comtypes 在进程内调用 SAPI，并通过常驻线程复用语音引擎，
避免每次播报都启动 PowerShell 进程（约 4 秒）的开销。
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
from pathlib import Path

import comtypes.client

logger = logging.getLogger(__name__)

# SAPI 格式枚举：SAFT44kHz16BitMono（44100Hz / 单声道 / 16bit）
_SAFT_44K16BIT_MONO = 34
_VOICE_HINT = "Huihui"


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
            fmt.Type = _SAFT_44K16BIT_MONO
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
    """把文字合成为 44100Hz / 单声道 / 16bit 的 WAV 字节。"""
    if not text.strip():
        raise ValueError("播报文字不能为空")

    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _TtsWorker()
    return _worker.synthesize(text)


def write_temp_wav(wav: bytes) -> str:
    """把 WAV 字节写入临时文件并返回路径，供上传接口使用。"""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="go2x_say_")
    with open(fd, "wb") as f:
        f.write(wav)
    return path

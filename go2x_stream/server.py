"""本地 Web 服务：把 Go2 的最新视频帧以 MJPEG 形式推送到浏览器。"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .frame import Frame
from .paths import resource_dir
from . import params as algo_params

_INDEX_HTML = (resource_dir() / "go2x_stream" / "web" / "index.html").read_text(encoding="utf-8")

# 上传的视频临时缓存：video_id -> 文件路径
_VIDEOS: dict[str, str] = {}

# 人脸框叠加结果缓存：同一帧被多个客户端拉取时只解码/画框/编码一次
_OVERLAY_LOCK = threading.Lock()
_OVERLAY_SRC: Optional[bytes] = None
_OVERLAY_JPEG: Optional[bytes] = None


class VideoSource(Protocol):
    """Web 层只依赖这两个方法，WebRTC / DDS 两种实现都可传入。"""

    def latest_frame(self) -> Optional[Frame]: ...

    def status(self) -> dict: ...


class SayRequest(BaseModel):
    text: str


class ConnectRequest(BaseModel):
    ip: str
    aes_key: str = ""


class ActionRequest(BaseModel):
    action: str


class CustomActionRequest(BaseModel):
    name: str
    say: str = ""
    steps: list


class AssignActionRequest(BaseModel):
    action: str = ""


class ChatAskRequest(BaseModel):
    text: str


class ChatToggleRequest(BaseModel):
    enabled: bool


class FaceTriggerRequest(BaseModel):
    enabled: bool = False
    say: str = ""
    steps: list = []
    cooldown: float = 10.0
    sit_first: bool = True
    settle_s: float = 2.0


class FaceStopRequest(BaseModel):
    enabled: bool = True
    seconds: float = 3.0


class PatrolRequest(BaseModel):
    action: str  # start / stop
    mode: str = "route"  # route（按已保存路线）/ simple（参数化前进+转向）
    vx: float = 0.5
    vyaw: float = 0.6
    forward_s: float = 4.0
    turn_s: float = 2.0


class PatrolRouteRequest(BaseModel):
    steps: list = []
    loop: bool = True


class MoveRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


class SequenceRequest(BaseModel):
    steps: list


class RawSportRequest(BaseModel):
    topic: str = "rt/api/sport/request"
    api_id: int
    parameter: dict = {}
    priority: bool = False


def create_app(stream: VideoSource, chat=None) -> FastAPI:
    app = FastAPI(title="Go2X 视频流")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(stream.status())

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        """返回页面配置所需的机器狗 IP（AES 密钥不返回，仅 IP）。"""
        st = stream.status()
        return JSONResponse({"ip": st.get("ip", "")})

    # ─── 算法参数 ───────────────────────────────────────────────

    @app.get("/api/params")
    async def params_get() -> JSONResponse:
        """返回算法参数定义、当前值与默认值。"""
        return JSONResponse(algo_params.snapshot())

    @app.post("/api/params")
    async def params_save(request: Request) -> JSONResponse:
        """保存页面提交的算法参数（尽量热更新，部分需重启生效）。"""
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
        try:
            result = await asyncio.to_thread(algo_params.update, stream, chat, payload)
            return JSONResponse(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/params/reset")
    async def params_reset() -> JSONResponse:
        """把算法参数恢复为默认值。"""
        try:
            result = await asyncio.to_thread(algo_params.reset, stream, chat)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/connect")
    async def connect(req: ConnectRequest) -> JSONResponse:
        connect_fn = getattr(stream, "connect", None)
        if not callable(connect_fn):
            raise HTTPException(status_code=400, detail="当前拉流方式不支持运行时切换 IP（仅 WebRTC 模式可用）")
        try:
            result = await asyncio.to_thread(
                connect_fn, req.ip, req.aes_key or None
            )
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/say")
    async def say(req: SayRequest) -> JSONResponse:
        say_fn = getattr(stream, "say", None)
        if not callable(say_fn):
            raise HTTPException(status_code=400, detail="当前拉流方式不支持语音播报（仅 WebRTC 模式可用）")
        try:
            result = await asyncio.to_thread(say_fn, req.text)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/motion")
    async def motion_status() -> JSONResponse:
        fn = getattr(stream, "motion_status", None)
        return JSONResponse(fn() if callable(fn) else {"ready": False, "error": "未启用运动控制", "interface": "", "domain_id": 0})

    @app.post("/api/action")
    async def do_action(req: ActionRequest) -> JSONResponse:
        """触发机器狗动作（如 hello/heart/dance1）。"""
        fn = getattr(stream, "do_action", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
        try:
            result = await asyncio.to_thread(fn, req.action)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/raw_sport")
    async def raw_sport(req: RawSportRequest) -> JSONResponse:
        """调试：向任意 rt/api/*/request 主题下发一条请求。"""
        fn = getattr(stream, "raw_request", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持数据通道指令")
        try:
            result = await asyncio.to_thread(
                fn, req.topic, req.api_id, req.parameter, 8.0, req.priority
            )
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/actions")
    async def list_actions() -> JSONResponse:
        """返回所有可用动作列表（给前端组合编辑器）。"""
        fn = getattr(stream, "list_actions", None)
        if not callable(fn):
            return JSONResponse({"actions": [], "velocity_range": {}})
        return JSONResponse(fn())

    @app.post("/api/move")
    async def move(req: MoveRequest) -> JSONResponse:
        """设置持续速度指令。"""
        fn = getattr(stream, "move", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
        try:
            result = await asyncio.to_thread(fn, req.vx, req.vy, req.vyaw)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/stop_move")
    async def stop_move() -> JSONResponse:
        """立即停止移动。"""
        fn = getattr(stream, "stop_move", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
        try:
            result = await asyncio.to_thread(fn)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/patrol")
    async def patrol(req: PatrolRequest) -> JSONResponse:
        """开始/停止巡逻（巡逻期间检测到人脸会自动暂停）。"""
        action = (req.action or "").lower()
        if action == "start":
            if (req.mode or "route").lower() == "route":
                fn = getattr(stream, "patrol_route_start", None)
                if not callable(fn):
                    raise HTTPException(status_code=400, detail="当前模式不支持巡逻路线")
                try:
                    result = await asyncio.to_thread(fn)
                    return JSONResponse(result)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
            fn = getattr(stream, "patrol_start", None)
            if not callable(fn):
                raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
            try:
                result = await asyncio.to_thread(
                    fn, req.vx, req.vyaw, req.forward_s, req.turn_s
                )
                return JSONResponse(result)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
        fn = getattr(stream, "patrol_stop", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
        try:
            result = await asyncio.to_thread(fn)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/patrol_route")
    async def patrol_route_get() -> JSONResponse:
        fn = getattr(stream, "patrol_route_get", None)
        if not callable(fn):
            return JSONResponse({"steps": [], "loop": True})
        return JSONResponse(fn())

    @app.post("/api/patrol_route")
    async def patrol_route_save(req: PatrolRouteRequest) -> JSONResponse:
        fn = getattr(stream, "patrol_route_save", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持巡逻路线")
        try:
            result = await asyncio.to_thread(fn, req.steps, req.loop)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sequence")
    async def do_sequence(req: SequenceRequest) -> JSONResponse:
        """串行执行一串动作/移动/延时。"""
        fn = getattr(stream, "do_sequence", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持运动控制")
        try:
            result = await asyncio.to_thread(fn, req.steps)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/register_face")
    async def register_face(request: Request):
        """注册/追加人脸样本；支持一次上传多张（字段名 images，兼容旧的 image）。"""
        register_fn = getattr(stream, "register_faces", None)
        if not callable(register_fn):
            raise HTTPException(status_code=400, detail="当前模式不支持人脸注册（启动服务时需加 --face）")
        form = await request.form()
        name = str(form.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="请填写姓名")
        # Starlette 的 form() 返回的是它自己的 UploadFile，用「非字符串」判断最稳
        files = [v for _, v in form.multi_items() if not isinstance(v, str)]
        if not files:
            raise HTTPException(status_code=400, detail="请至少上传一张人脸图片")
        payload = [await f.read() for f in files]
        try:
            result = await asyncio.to_thread(register_fn, name, payload)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/faces")
    async def faces() -> JSONResponse:
        fn = getattr(stream, "registered_faces", None)
        return JSONResponse(fn() if callable(fn) else [])

    # ─── 自定义动作 ─────────────────────────────────────────────

    @app.get("/api/custom_actions")
    async def custom_actions() -> JSONResponse:
        fn = getattr(stream, "custom_actions_list", None)
        return JSONResponse(fn() if callable(fn) else [])

    @app.post("/api/custom_actions")
    async def save_custom_action(req: CustomActionRequest) -> JSONResponse:
        fn = getattr(stream, "save_custom_action", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持自定义动作")
        try:
            result = await asyncio.to_thread(fn, req.name, req.say, req.steps)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/custom_actions/{name}")
    async def delete_custom_action(name: str) -> JSONResponse:
        fn = getattr(stream, "delete_custom_action", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持自定义动作")
        try:
            result = await asyncio.to_thread(fn, name)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/face/{name}/action")
    async def assign_face_action(name: str, req: AssignActionRequest) -> JSONResponse:
        fn = getattr(stream, "assign_face_action", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持自定义动作")
        try:
            result = await asyncio.to_thread(fn, name, req.action)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ─── 人脸触发（检测到任意人脸即播报 + 动作）──────────────────

    @app.get("/api/face_trigger")
    async def face_trigger_get() -> JSONResponse:
        fn = getattr(stream, "face_trigger_get", None)
        if not callable(fn):
            return JSONResponse({
                "enabled": False, "say": "", "steps": [], "cooldown": 10.0,
                "sit_first": True, "settle_s": 2.0,
            })
        return JSONResponse(fn())

    @app.post("/api/face_trigger")
    async def face_trigger_save(req: FaceTriggerRequest) -> JSONResponse:
        fn = getattr(stream, "face_trigger_save", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持人脸触发（启动服务时需加 --face）")
        try:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            result = await asyncio.to_thread(fn, payload)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/face_stop")
    async def face_stop_get() -> JSONResponse:
        fn = getattr(stream, "face_stop_get", None)
        if not callable(fn):
            return JSONResponse({"enabled": False, "seconds": 3.0})
        return JSONResponse(fn())

    @app.post("/api/face_stop")
    async def face_stop_save(req: FaceStopRequest) -> JSONResponse:
        fn = getattr(stream, "face_stop_save", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持人脸暂停移动")
        try:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            result = await asyncio.to_thread(fn, payload)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/face_image/{name}")
    async def face_image(name: str):
        fn = getattr(stream, "face_image_path", None)
        path = fn(name) if callable(fn) else None
        if not path:
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(path, media_type="image/jpeg")

    @app.delete("/api/face/{name}")
    async def delete_face(name: str):
        fn = getattr(stream, "delete_face", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持删除人脸")
        try:
            result = await asyncio.to_thread(fn, name)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/detect_video")
    async def detect_video(video: UploadFile = File(...)):
        fn = getattr(stream, "detect_video", None)
        if not callable(fn):
            raise HTTPException(status_code=400, detail="当前模式不支持视频检测（启动服务时需加 --face）")
        data = await video.read()
        try:
            result = await asyncio.to_thread(fn, data, video.filename or "video.mp4")
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/upload_video")
    async def upload_video(video: UploadFile = File(...)):
        data = await video.read()
        vid = uuid.uuid4().hex
        suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="go2x_video_")
        with open(fd, "wb") as f:
            f.write(data)
        _VIDEOS[vid] = path
        return JSONResponse({"video_id": vid})

    @app.get("/api/detect_stream/{video_id}")
    async def detect_stream(video_id: str) -> StreamingResponse:
        path = _VIDEOS.get(video_id)
        if not path:
            raise HTTPException(status_code=404, detail="视频不存在或已过期")
        return StreamingResponse(
            _detect_video_stream(stream, path),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/video")
    async def video() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg(stream),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    # ─── 语音对话 ───────────────────────────────────────────────

    @app.get("/api/chat")
    async def chat_status() -> JSONResponse:
        if chat is None:
            return JSONResponse({"enabled": False, "error": "未启用语音对话（config.json → chat.enabled）"})
        return JSONResponse(chat.status())

    @app.post("/api/chat/toggle")
    async def chat_toggle(req: ChatToggleRequest) -> JSONResponse:
        if chat is None:
            raise HTTPException(status_code=400, detail="未启用语音对话")
        chat.set_enabled(req.enabled)
        return JSONResponse(chat.status())

    @app.post("/api/chat/ask")
    async def chat_ask(req: ChatAskRequest) -> JSONResponse:
        if chat is None:
            raise HTTPException(status_code=400, detail="未启用语音对话")
        try:
            result = await asyncio.to_thread(chat.ask_and_reply, req.text)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.websocket("/ws/audio")
    async def audio_stream(ws: WebSocket):
        """把机器狗麦克风的 PCM 音频推给网页，由浏览器实时播放。"""
        begin = getattr(stream, "audio_begin", None)
        read = getattr(stream, "audio_read", None)
        if not callable(begin) or not callable(read):
            await ws.close(code=1003)
            return
        await ws.accept()
        begin()
        try:
            info_fn = getattr(stream, "audio_info", None)
            info = info_fn() if callable(info_fn) else {}
            seq_fn = getattr(stream, "audio_seq", None)
            seq = seq_fn() if callable(seq_fn) else 0
            sent_rate = info.get("rate", 48000)
            await ws.send_json({"type": "info", "rate": sent_rate, "channels": 1})
            while True:
                new_seq, rate, data = read(seq)
                # 实际采样率变化时通知浏览器重建 AudioContext
                if rate and rate != sent_rate:
                    sent_rate = rate
                    await ws.send_json({"type": "info", "rate": rate, "channels": 1})
                if data:
                    seq = new_seq
                    await ws.send_bytes(data)
                else:
                    await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            end_fn = getattr(stream, "audio_end", None)
            if callable(end_fn):
                end_fn()

    return app


async def _mjpeg(stream: VideoSource):
    """每来一帧新画面就推送一次，无新帧时短暂等待。

    人脸框复用后台识别线程缓存的检测结果（latest_detections），
    这里不再单独跑推理；叠加后的 JPEG 同一帧只编码一次，供多个客户端共用。
    """
    global _OVERLAY_SRC, _OVERLAY_JPEG

    import cv2
    import numpy as np

    from .face import draw_boxes

    latest_det = getattr(stream, "latest_detections", None)
    has_detect = callable(latest_det)
    last_seq = -1
    while True:
        frame = stream.latest_frame()
        if frame is None or frame.seq == last_seq:
            await asyncio.sleep(0.02)
            continue
        last_seq = frame.seq
        jpeg = frame.jpeg

        if has_detect:
            cached = False
            with _OVERLAY_LOCK:
                if _OVERLAY_SRC is frame.jpeg and _OVERLAY_JPEG is not None:
                    jpeg = _OVERLAY_JPEG
                    cached = True
            if not cached:
                det_seq, results = latest_det()
                # 检测结果过于陈旧（超过 60 帧）就不再叠加，避免框错位
                if results and 0 <= frame.seq - det_seq <= 60:
                    try:
                        img = cv2.imdecode(
                            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if img is not None:
                            img = draw_boxes(img, results)
                            ok, buf = cv2.imencode(
                                ".jpg", img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 75],
                            )
                            if ok:
                                jpeg = buf.tobytes()
                                with _OVERLAY_LOCK:
                                    _OVERLAY_SRC = frame.jpeg
                                    _OVERLAY_JPEG = jpeg
                    except Exception:
                        pass

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


async def _detect_video_stream(stream: VideoSource, path: str):
    """逐帧读取视频、检测并画框、以 MJPEG 流推送。"""
    import cv2

    from .face import draw_boxes

    detect_fn = getattr(stream, "detect_frame", None)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if callable(detect_fn):
                try:
                    results = detect_fn(frame)
                    frame = draw_boxes(frame, results)
                except Exception:
                    pass
            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(buf)).encode()
                + b"\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(1.0 / fps)
    finally:
        cap.release()
        for vid, p in list(_VIDEOS.items()):
            if p == path:
                _VIDEOS.pop(vid, None)
                break
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

"""本地 Web 服务：把 Go2 的最新视频帧以 MJPEG 形式推送到浏览器。"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .frame import Frame
from .paths import resource_dir

_INDEX_HTML = (resource_dir() / "go2x_stream" / "web" / "index.html").read_text(encoding="utf-8")

# 上传的视频临时缓存：video_id -> 文件路径
_VIDEOS: dict[str, str] = {}


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


def create_app(stream: VideoSource) -> FastAPI:
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

    @app.get("/api/actions")
    async def list_actions() -> JSONResponse:
        """返回所有可用动作列表（给前端组合编辑器）。"""
        fn = getattr(stream, "list_actions", None)
        if not callable(fn):
            return JSONResponse({"actions": [], "velocity_range": {}})
        return JSONResponse(fn())

    class MoveRequest(BaseModel):
        vx: float = 0.0
        vy: float = 0.0
        vyaw: float = 0.0

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

    class SequenceRequest(BaseModel):
        steps: list

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
    async def register_face(
        name: str = Form(...),
        image: UploadFile = File(...),
    ):
        register_fn = getattr(stream, "register_face", None)
        if not callable(register_fn):
            raise HTTPException(status_code=400, detail="当前模式不支持人脸注册（启动服务时需加 --face）")
        data = await image.read()
        try:
            result = await asyncio.to_thread(register_fn, name, data)
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

    return app


async def _mjpeg(stream: VideoSource):
    """每来一帧新画面就推送一次，无新帧时短暂等待。

    若支持人脸识别（detect_frame），按节流间隔检测并叠加人脸框，
    检测结果在间隔内复用，避免每帧全量检测拖慢视频流。
    """
    import time

    import cv2
    import numpy as np

    from .face import draw_boxes

    detect_fn = getattr(stream, "detect_frame", None)
    has_detect = callable(detect_fn)
    last_seq = -1
    last_detect = 0.0
    detect_interval = 0.5  # 秒
    cached_results: list = []
    while True:
        frame = stream.latest_frame()
        if frame is None or frame.seq == last_seq:
            await asyncio.sleep(0.01)
            continue
        last_seq = frame.seq
        jpeg = frame.jpeg

        if has_detect:
            now = time.monotonic()
            if now - last_detect >= detect_interval:
                last_detect = now
                try:
                    img = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        cached_results = await asyncio.to_thread(detect_fn, img)
                except Exception:
                    cached_results = []
            if cached_results:
                try:
                    img = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        img = draw_boxes(img, cached_results)
                        ok, buf = cv2.imencode(
                            ".jpg", img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                        )
                        if ok:
                            jpeg = buf.tobytes()
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

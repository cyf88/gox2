"""Go2X 机器狗视频拉流服务入口。

默认通过 WebRTC 拉流（适合 PC 连接机器狗 WiFi 热点或同一局域网）；
也可用 --source dds 走有线 DDS 的 videohub 服务。

示例:
    python main.py                              # WebRTC，AP 直连 192.168.12.1
    python main.py --ip 192.168.31.20           # WebRTC，按局域网 IP 连接
    python main.py --sn B42D2000XXXXXXXX        # WebRTC，按序列号发现
    python main.py --source dds --interface 192.168.123.99   # 有线 DDS
    python main.py --display window             # 本机窗口显示，q 退出
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

# 打包后的 exe 控制台默认 GBK 编码，unitree 库会打印 emoji，需强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from go2x_stream.dds import Go2DdsStream
from go2x_stream.paths import app_dir, resource_dir
from go2x_stream.server import create_app
from go2x_stream.webrtc import Go2WebRtcStream

CONFIG_PATH = app_dir() / "config.json"


def load_config() -> dict:
    """读取 config.json，缺失或损坏时返回空配置。"""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("读取配置文件失败 %s: %s", CONFIG_PATH, exc)
        return {}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="宇树 Go2 机器狗视频拉流并显示")
    parser.add_argument(
        "--source",
        choices=["webrtc", "dds"],
        default=cfg.get("source", "webrtc"),
        help="拉流方式：webrtc（默认，走 WiFi）或 dds（需有线接入 192.168.123.x）",
    )

    webrtc = parser.add_argument_group("WebRTC")
    webrtc.add_argument(
        "--ip", default=cfg.get("ip"), help="机器狗在局域网内的 IP；不填 --ip/--sn 时按 AP 直连 192.168.12.1"
    )
    webrtc.add_argument("--sn", default=cfg.get("sn"), help="机器狗序列号，用于局域网内自动发现")
    webrtc.add_argument(
        "--aes-key", default=cfg.get("aes_key"), help="新版固件局域网连接所需的 AES 密钥（可选）"
    )
    webrtc.add_argument(
        "--quality", type=int, default=cfg.get("quality", 80), help="JPEG 编码质量 1-100，默认 80"
    )

    dds = parser.add_argument_group("DDS")
    dds.add_argument(
        "--interface",
        help="DDS 绑定的网卡名或本机 IP（如 \"以太网\"、192.168.123.99），缺省自动选择",
    )
    dds.add_argument("--domain-id", type=int, default=0, help="DDS domain id，默认 0")
    dds.add_argument(
        "--timeout", type=float, default=1.0, help="单次图像请求超时秒数，默认 1.0"
    )

    parser.add_argument(
        "--display",
        choices=["web", "window"],
        default="web",
        help="显示方式：web 浏览器页面（默认）或 window 本机窗口",
    )
    parser.add_argument("--host", default=cfg.get("host", "127.0.0.1"), help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=cfg.get("port", 8000), help="Web 服务端口")

    face = parser.add_argument_group("人脸识别")
    face.add_argument(
        "--face", action="store_true", default=cfg.get("face", False),
        help="启用人脸识别（识别到注册人脸时播报）"
    )
    face.add_argument(
        "--say-cooldown", type=float, default=cfg.get("say_cooldown", 30.0),
        help="同一人播报冷却时间（秒），默认 30",
    )
    face.add_argument(
        "--face-interval", type=float, default=cfg.get("face_interval", 1.0),
        help="人脸识别间隔（秒），默认 1；调大可显著降低 CPU",
    )
    face.add_argument(
        "--face-det-size", type=int, default=cfg.get("face_det_size", 480),
        help="人脸检测模型输入尺寸，默认 480（越小越快，640 更准）",
    )
    face.add_argument(
        "--face-threads", type=int, default=cfg.get("face_threads", 2),
        help="人脸推理线程数，默认 2；设 1 最省 CPU",
    )

    motion = parser.add_argument_group("运动控制（WebRTC sport）")
    motion.add_argument(
        "--motion", action="store_true", default=cfg.get("motion", False),
        help="启用运动控制（识别到人按分类做动作，也可页面手动触发）"
    )

    parser.add_argument(
        "--log-level",
        default=cfg.get("log_level", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()


def run_window(stream) -> None:
    """用 OpenCV 窗口直接显示画面，按 q 或 Esc 退出。"""
    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        placeholder, "Waiting for Go2 video...", (40, 350),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA,
    )
    cv2.putText(
        placeholder, "Press q to quit", (40, 400),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1, cv2.LINE_AA,
    )

    last_seq = -1
    image = placeholder
    while True:
        frame = stream.latest_frame()
        if frame is not None and frame.seq != last_seq:
            last_seq = frame.seq
            image = cv2.imdecode(
                np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        cv2.imshow("Go2X", image)
        if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def create_stream(args: argparse.Namespace):
    face_recognizer = None
    custom_actions = None
    if args.face:
        from go2x_stream.face import FaceRecognizer

        face_recognizer = FaceRecognizer(
            model_root=resource_dir(),
            faces_dir=app_dir() / "faces",
            det_size=args.face_det_size,
            num_threads=args.face_threads,
        )

    if args.face or args.motion:
        from go2x_stream.custom_actions import CustomActionStore

        custom_actions = CustomActionStore(data_dir=app_dir() / "faces")

    motion = None
    if args.motion:
        from go2x_stream.motion import Go2Motion

        motion = Go2Motion()

    if args.source == "dds":
        return Go2DdsStream(
            network_interface=args.interface,
            domain_id=args.domain_id,
            timeout=args.timeout,
        )

    stream = Go2WebRtcStream(
        ip=args.ip,
        serial_number=args.sn,
        aes_key=args.aes_key,
        jpeg_quality=args.quality,
        face_recognizer=face_recognizer,
        say_cooldown=args.say_cooldown,
        face_interval=args.face_interval,
        motion=motion,
        custom_actions=custom_actions,
    )
    if motion is not None:
        motion.attach(stream)
        motion.start()
    return stream


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # aiortc/aioice 会对每个不可用的本地地址打印 INFO，噪声较大
    for name in ("aiortc", "aioice", "aioice.ice"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 库对每个音频帧都会 logging.info("Receiving audio frame")，过滤掉避免刷屏
    class _DropAudioFrameLog(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage() != "Receiving audio frame"

    logging.getLogger().addFilter(_DropAudioFrameLog())

    # 语音预生成缓存目录：编辑自定义动作/注册人脸时落盘，播报时直接复用
    from go2x_stream.tts import configure_cache

    configure_cache(app_dir() / "faces" / "tts_cache")

    stream = create_stream(args)
    stream.start()

    try:
        if args.display == "window":
            run_window(stream)
        else:
            import uvicorn

            shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
            print(f"浏览器打开 http://{shown_host}:{args.port}", flush=True)
            uvicorn.run(
                create_app(stream),
                host=args.host,
                port=args.port,
                log_level="warning",
            )
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

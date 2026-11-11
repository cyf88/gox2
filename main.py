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
import logging
import sys

import cv2
import numpy as np

from go2x_stream.dds import Go2DdsStream
from go2x_stream.server import create_app
from go2x_stream.webrtc import Go2WebRtcStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="宇树 Go2 机器狗视频拉流并显示")
    parser.add_argument(
        "--source",
        choices=["webrtc", "dds"],
        default="webrtc",
        help="拉流方式：webrtc（默认，走 WiFi）或 dds（需有线接入 192.168.123.x）",
    )

    webrtc = parser.add_argument_group("WebRTC")
    webrtc.add_argument(
        "--ip", help="机器狗在局域网内的 IP；不填 --ip/--sn 时按 AP 直连 192.168.12.1"
    )
    webrtc.add_argument("--sn", help="机器狗序列号，用于局域网内自动发现")
    webrtc.add_argument(
        "--aes-key", help="新版固件局域网连接所需的 AES 密钥（可选）"
    )
    webrtc.add_argument(
        "--quality", type=int, default=80, help="JPEG 编码质量 1-100，默认 80"
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
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务端口")
    parser.add_argument(
        "--log-level",
        default="INFO",
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
    if args.source == "dds":
        return Go2DdsStream(
            network_interface=args.interface,
            domain_id=args.domain_id,
            timeout=args.timeout,
        )
    return Go2WebRtcStream(
        ip=args.ip,
        serial_number=args.sn,
        aes_key=args.aes_key,
        jpeg_quality=args.quality,
    )


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

"""人脸注册脚本：从一张照片提取人脸特征，加入识别库。

用法:
    python register_face.py --name 张三 --image photo.jpg
    python register_face.py --name 李四 --image li.jpg --threshold 0.4

注册后的人脸特征保存在 faces/ 目录下（每人的姓名.npy）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from go2x_stream.face import FaceRecognizer

_BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注册人脸到识别库")
    parser.add_argument("--name", required=True, help="姓名（识别到时会播报这个名字）")
    parser.add_argument("--image", required=True, help="包含人脸的图片路径")
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="识别相似度阈值，默认 0.4"
    )
    parser.add_argument(
        "--faces-dir", default=str(_BASE_DIR / "faces"), help="人脸特征库目录"
    )
    parser.add_argument(
        "--model-root", default=str(_BASE_DIR), help="InsightFace 模型根目录"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recognizer = FaceRecognizer(
        model_root=Path(args.model_root),
        faces_dir=Path(args.faces_dir),
        threshold=args.threshold,
    )
    recognizer.register(args.image, args.name)
    print(f"已注册: {args.name}")
    print(f"当前识别库: {recognizer.registered_names()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""基于 InsightFace（ArcFace）的本地人脸检测与识别。

使用 insightface 的 buffalo_l 模型包：SCRFD 检测 + ArcFace 512 维特征，
在 PC 本地离线运行，识别结果交由上层触发机器狗播报。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "buffalo_l"

# ArcFace 余弦相似度阈值：相似度 >= 该值判定为同一人
_DEFAULT_THRESHOLD = 0.4


class FaceRecognizer:
    """检测并识别画面中的人脸，返回识别结果。"""

    def __init__(
        self,
        model_root: Path,
        faces_dir: Path,
        threshold: float = _DEFAULT_THRESHOLD,
        det_size: int = 640,
    ) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name=_MODEL_NAME, root=str(model_root))
        # ctx_id=-1 使用 CPU（普通 onnxruntime）；如需 GPU 改 0 并安装 onnxruntime-gpu
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self._faces_dir = Path(faces_dir)
        self._faces_dir.mkdir(parents=True, exist_ok=True)
        self._threshold = threshold
        self._registry: dict[str, np.ndarray] = {}
        self._load_registry()

    # ─── 人脸库管理 ─────────────────────────────────────────────

    def _load_registry(self) -> None:
        self._registry.clear()
        for path in self._faces_dir.glob("*.npy"):
            name = path.stem
            try:
                self._registry[name] = np.load(path)
            except Exception as exc:
                logger.warning("加载人脸特征失败 %s: %s", path, exc)
        logger.info("已加载 %d 个注册人脸: %s", len(self._registry), list(self._registry))

    def register(self, image_path: str, name: str) -> None:
        """从图片文件路径提取人脸特征并注册。"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        self.register_image(name, img)

    def register_image(self, name: str, img: np.ndarray) -> None:
        """从图像数据（BGR ndarray）提取人脸特征并注册。"""
        faces = self._app.get(img)
        if not faces:
            raise ValueError("未在图片中检测到人脸，请换一张正脸清晰的照片")
        face = max(faces, key=lambda f: f.det_score)
        np.save(self._faces_dir / f"{name}.npy", face.normed_embedding)
        self._save_thumbnail(name, img, face)
        self._registry[name] = face.normed_embedding
        logger.info("已注册人脸: %s", name)

    def _save_thumbnail(self, name: str, img: np.ndarray, face) -> None:
        """裁剪人脸区域并保存为缩略图，供列表展示。"""
        try:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h, w = img.shape[:2]
            fw, fh = x2 - x1, y2 - y1
            pad_x, pad_y = int(fw * 0.35), int(fh * 0.35)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return
            side = max(crop.shape[0], crop.shape[1])
            square = np.full((side, side, 3), 0, dtype=np.uint8)
            square[: crop.shape[0], : crop.shape[1]] = crop
            thumb = cv2.resize(square, (160, 160))
            # 用 imencode + Python 写文件，避免 cv2.imwrite 对中文文件名编码出错
            ok, buf = cv2.imencode(".jpg", thumb,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                (self._faces_dir / f"{name}.jpg").write_bytes(buf.tobytes())
        except Exception as exc:
            logger.warning("保存人脸缩略图失败 %s: %s", name, exc)

    def face_image_path(self, name: str) -> Optional[str]:
        """返回注册人脸的缩略图路径（不存在则返回 None）。"""
        p = self._faces_dir / f"{name}.jpg"
        return str(p) if p.exists() else None

    def registered_names(self) -> list[str]:
        return list(self._registry)

    def delete(self, name: str) -> bool:
        """删除一个已注册的人脸（特征 + 缩略图）。"""
        if name not in self._registry:
            return False
        for ext in (".npy", ".jpg"):
            p = self._faces_dir / f"{name}{ext}"
            try:
                if p.exists():
                    p.unlink()
            except OSError as exc:
                logger.warning("删除人脸文件失败 %s: %s", p, exc)
        self._registry.pop(name, None)
        logger.info("已删除注册人脸: %s", name)
        return True

    # ─── 识别 ───────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> list[tuple[str, float]]:
        """识别画面中的所有已知人脸，返回 [(姓名, 相似度)]。

        相似度为余弦相似度（0~1，越大越像）。未注册的人脸不会被返回。
        """
        if not self._registry:
            return []
        faces = self._app.get(frame)
        results: list[tuple[str, float]] = []
        for face in faces:
            name, sim = self._match(face.normed_embedding)
            if name is not None:
                results.append((name, sim))
        return results

    def detect_with_boxes(self, frame: np.ndarray) -> list[tuple[str, float, list[int]]]:
        """检测并识别画面中的已知人脸，返回 [(姓名, 相似度, bbox)]。

        bbox 为 [x1, y1, x2, y2]（整数像素坐标）。
        """
        if not self._registry:
            return []
        faces = self._app.get(frame)
        results: list[tuple[str, float, list[int]]] = []
        for face in faces:
            name, sim = self._match(face.normed_embedding)
            if name is not None:
                bbox = [int(v) for v in face.bbox]
                results.append((name, round(sim, 2), bbox))
        return results

    def detect_video(self, video_path: str, frame_step: int = 10) -> dict:
        """对离线视频隔帧检测并识别，返回识别结果汇总。

        返回结构:
            {"detected": [{"name", "time", "sim"}...], "duration", "total_frames",
             "processed_frames"}
        其中 time 为首次识别到该人的秒数，detected 按首次出现顺序排列。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        detected: dict[str, dict] = {}
        frame_idx = 0
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    processed += 1
                    t = frame_idx / fps
                    for name, sim in self.recognize(frame):
                        if name not in detected:
                            detected[name] = {
                                "time": round(t, 1),
                                "sim": round(sim, 2),
                            }
                frame_idx += 1
        finally:
            cap.release()

        return {
            "detected": [
                {"name": name, "time": info["time"], "sim": info["sim"]}
                for name, info in detected.items()
            ],
            "duration": round(frame_idx / fps, 1),
            "total_frames": frame_idx,
            "processed_frames": processed,
        }

    # ─── 内部工具 ───────────────────────────────────────────────

    def _match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """与注册人脸比对，返回最匹配的 (姓名, 相似度) 或 (None, 相似度)。"""
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, ref in self._registry.items():
            sim = float(np.dot(emb, ref))  # 已归一化，点积即余弦相似度
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= self._threshold:
            return best_name, best_sim
        return None, best_sim


def draw_boxes(frame: np.ndarray, results: list) -> np.ndarray:
    """在图像上画出人脸框和姓名标签（支持中文），返回处理后的 BGR 图像。

    results 为 detect_with_boxes 的返回值：[(姓名, 相似度, bbox)]。
    """
    if not results:
        return frame
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    for name, sim, bbox in results:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"{name} {sim:.2f}"
        ty = y1 - 30 if y1 - 30 > 0 else y1 + 2
        try:
            box = draw.textbbox((x1, ty), label, font=font)
            draw.rectangle(box, fill=(0, 200, 0))
            draw.text((x1, ty), label, font=font, fill=(0, 0, 0))
        except Exception:
            draw.text((x1, ty), label, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

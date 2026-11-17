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

# 多模板注册
MAX_TEMPLATES = 5        # 每人最多保留的样本数
_MIN_DET_SCORE = 0.5     # 注册时要求的最低检测置信度
_MIN_FACE_SIZE = 80      # 注册时人脸框最短边的最小像素
_DUP_SIM = 0.95          # 与已有样本这么像就当作重复，跳过


class FaceRecognizer:
    """检测并识别画面中的人脸，返回识别结果。"""

    def __init__(
        self,
        model_root: Path,
        faces_dir: Path,
        threshold: float = _DEFAULT_THRESHOLD,
        det_size: int = 480,
        num_threads: int = 2,
        max_detect_width: int = 640,
    ) -> None:
        import onnxruntime
        from insightface.app import FaceAnalysis

        # 限制推理线程数：默认 ONNX 会用满全部核心，导致 CPU 打满
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = max(1, int(num_threads))
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._app = FaceAnalysis(
            name=_MODEL_NAME, root=str(model_root), sess_options=sess_options
        )
        # ctx_id=-1 使用 CPU（普通 onnxruntime）；如需 GPU 改 0 并安装 onnxruntime-gpu
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self._faces_dir = Path(faces_dir)
        self._faces_dir.mkdir(parents=True, exist_ok=True)
        self._threshold = threshold
        self._max_detect_width = max_detect_width
        self._registry: dict[str, list[np.ndarray]] = {}
        self._load_registry()
        logger.info(
            "人脸识别就绪（det_size=%d，推理线程=%d，检测图最大宽度=%d）",
            det_size, num_threads, max_detect_width,
        )

    # ─── 人脸库管理 ─────────────────────────────────────────────

    def _load_registry(self) -> None:
        """加载人脸库：新格式为 faces/<姓名>/<序号>.npy（多模板），兼容旧的 faces/<姓名>.npy。"""
        self._registry.clear()
        for path in self._faces_dir.glob("*.npy"):
            try:
                self._registry[path.stem] = [np.load(path)]
            except Exception as exc:
                logger.warning("加载人脸特征失败 %s: %s", path, exc)
        for folder in sorted(p for p in self._faces_dir.iterdir() if p.is_dir()):
            templates: list[np.ndarray] = []
            for path in sorted(folder.glob("*.npy")):
                try:
                    templates.append(np.load(path))
                except Exception as exc:
                    logger.warning("加载人脸特征失败 %s: %s", path, exc)
            if templates:
                self._registry[folder.name] = templates
        total = sum(len(v) for v in self._registry.values())
        logger.info(
            "已加载 %d 个注册人脸（共 %d 张模板）: %s",
            len(self._registry), total, list(self._registry),
        )

    def register(self, image_path: str, name: str) -> dict:
        """从图片文件路径提取人脸特征并注册。"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return self.register_image(name, img)

    def register_image(self, name: str, img: np.ndarray) -> dict:
        """从一张图像注册/追加一个人脸模板，返回 {"templates", "added", "detail"}。

        同一个人可以存多张不同角度/光照的样本（最多 MAX_TEMPLATES 张），
        识别时取与该人所有模板的最大相似度；名额满后替换掉最相似（最冗余）的那张。
        """
        faces = self._app.get(img)
        if not faces:
            raise ValueError("未在图片中检测到人脸，请换一张正脸清晰的照片")
        face = max(faces, key=lambda f: f.det_score)
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        side = min(x2 - x1, y2 - y1)
        if face.det_score < _MIN_DET_SCORE or side < _MIN_FACE_SIZE:
            raise ValueError(
                f"人脸质量不足（置信度 {face.det_score:.2f}，边长 {int(side)}px），"
                "请上传更清晰、正脸且更大一些的照片"
            )

        emb = face.normed_embedding
        templates = list(self._registry.get(name, []))
        if templates:
            sims = [float(np.dot(emb, t)) for t in templates]
            if max(sims) >= _DUP_SIM:
                return {
                    "templates": len(templates),
                    "added": False,
                    "detail": "与已有样本几乎相同，已跳过",
                }

        replaced = False
        if len(templates) >= MAX_TEMPLATES:
            sims = [float(np.dot(emb, t)) for t in templates]
            templates[int(np.argmax(sims))] = emb
            replaced = True
        else:
            templates.append(emb)

        self._persist_templates(name, templates)
        self._registry[name] = templates
        if not (self._faces_dir / f"{name}.jpg").exists():
            self._save_thumbnail(name, img, face)
        logger.info("已保存人脸样本: %s（共 %d 张）", name, len(templates))
        return {
            "templates": len(templates),
            "added": not replaced,
            "detail": "名额已满，已替换最相似的一张" if replaced else "",
        }

    def _persist_templates(self, name: str, templates: list[np.ndarray]) -> None:
        """把某人的全部模板重写到 faces/<姓名>/1.npy ... N.npy。"""
        folder = self._faces_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        legacy = self._faces_dir / f"{name}.npy"
        if legacy.exists():
            legacy.unlink()
        for path in folder.glob("*.npy"):
            path.unlink()
        for idx, emb in enumerate(templates, 1):
            np.save(folder / f"{idx}.npy", emb)

    def template_count(self, name: str) -> int:
        return len(self._registry.get(name, []))

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
        """删除一个已注册的人脸（全部模板 + 缩略图）。"""
        if name not in self._registry:
            return False
        folder = self._faces_dir / name
        if folder.is_dir():
            for path in folder.glob("*"):
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("删除人脸文件失败 %s: %s", path, exc)
            try:
                folder.rmdir()
            except OSError:
                pass
        for ext in (".npy", ".jpg"):
            path = self._faces_dir / f"{name}{ext}"
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("删除人脸文件失败 %s: %s", path, exc)
        self._registry.pop(name, None)
        logger.info("已删除注册人脸: %s", name)
        return True

    # ─── 识别 ───────────────────────────────────────────────────

    def detect_and_identify(self, frame: np.ndarray) -> list[tuple[str, float, list[int]]]:
        """检测 + 识别一次推理，返回 [(姓名, 相似度, bbox)]，bbox 为原始帧坐标。

        输入会先缩到 max_detect_width 宽再推理，显著降低 CPU 开销。
        """
        return self.detect_all(frame)[0]

    def detect_all(self, frame: np.ndarray) -> tuple[list[tuple[str, float, list[int]]], int]:
        """一次推理同时返回「已识别的人」和「检测到的人脸总数」。

        即使没有人脸库也会执行检测，便于实现「检测到人脸即触发」的功能。
        """
        faces, scale = self._infer(frame)
        results: list[tuple[str, float, list[int]]] = []
        for face in faces:
            if not self._registry:
                break
            name, sim = self._match(face.normed_embedding)
            if name is None:
                continue
            bbox = [int(v / scale) for v in face.bbox]
            results.append((name, round(sim, 2), bbox))
        return results, len(faces)

    def match_faces(
        self, frame: np.ndarray
    ) -> tuple[list[tuple[Optional[str], float, list[int]]], int]:
        """返回每张脸「最相似的人（不做阈值过滤）」和相似度，供上层做时间窗投票。

        结果形如 [(姓名或 None, 相似度, bbox)]；姓名为 None 表示人脸库里没有人脸。
        """
        faces, scale = self._infer(frame)
        out: list[tuple[Optional[str], float, list[int]]] = []
        for face in faces:
            if not self._registry:
                break
            name, sim = self._best_match(face.normed_embedding)
            out.append((name, sim, [int(v / scale) for v in face.bbox]))
        return out, len(faces)

    def _infer(self, frame: np.ndarray):
        """按最大宽度缩放后跑一次检测，返回 (faces, 缩放比例)。"""
        h, w = frame.shape[:2]
        scale = 1.0
        img = frame
        if self._max_detect_width and w > self._max_detect_width:
            scale = self._max_detect_width / float(w)
            img = cv2.resize(
                frame, (self._max_detect_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return self._app.get(img), scale

    def recognize(self, frame: np.ndarray) -> list[tuple[str, float]]:
        """识别画面中的所有已知人脸，返回 [(姓名, 相似度)]。"""
        return [(name, sim) for name, sim, _ in self.detect_and_identify(frame)]

    def detect_with_boxes(self, frame: np.ndarray) -> list[tuple[str, float, list[int]]]:
        """检测并识别画面中的已知人脸，返回 [(姓名, 相似度, bbox)]。

        bbox 为 [x1, y1, x2, y2]（整数像素坐标）。
        """
        return self.detect_and_identify(frame)

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
        best_name, best_sim = self._best_match(emb)
        if best_name is not None and best_sim >= self._threshold:
            return best_name, best_sim
        return None, best_sim

    def _best_match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """返回最相似的 (姓名, 相似度)，不做过阈值过滤；多人多模板时取全局最大值。"""
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, templates in self._registry.items():
            for ref in templates:
                sim = float(np.dot(emb, ref))  # 已归一化，点积即余弦相似度
                if sim > best_sim:
                    best_name, best_sim = name, sim
        return best_name, best_sim


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

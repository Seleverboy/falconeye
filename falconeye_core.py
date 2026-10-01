# =====================================================================
# PART 19 — Base Detector Abstract
# =====================================================================
from abc import ABC, abstractmethod
from contextlib import contextmanager
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import math
import os
import sys
import time
import platform
import threading
import logging

import numpy as np

from falconeye import SystemConfig, Database, EvidenceConfig
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

# logger
logger = logging.getLogger("falconeye")


@dataclass
class Detection:
    """تمثيل موحّد لكشف واحد"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int = -1
    keypoints: Optional[List[List[float]]] = None
    embedding: Optional[np.ndarray] = None
    face_name: Optional[str] = None
    face_distance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def aspect(self) -> float:
        h = self.height
        return self.width / h if h > 0 else 0.0

    def iou(self, other: "Detection") -> float:
        return compute_iou(self.bbox, other.bbox)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "center": list(self.center),
            "face_name": self.face_name,
        }


class BaseDetector(ABC):
    """واجهة موحّدة لجميع المحللات"""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._lock = threading.RLock()
        self._stats = {
            "total_calls": 0,
            "total_detections": 0,
            "total_time_ms": 0.0,
            "last_run": 0.0,
        }

    def _record_call(self, detections: int, elapsed_ms: float) -> None:
        with self._lock:
            self._stats["total_calls"] += 1
            self._stats["total_detections"] += detections
            self._stats["total_time_ms"] += elapsed_ms
            self._stats["last_run"] = time.time()

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            if stats["total_calls"]:
                stats["avg_time_ms"] = round(
                    stats["total_time_ms"] / stats["total_calls"], 2
                )
                stats["avg_detections"] = round(
                    stats["total_detections"] / stats["total_calls"], 2
                )
            return stats

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...

    def warmup(self, frame_shape: Tuple[int, ...] = (720, 1280, 3)) -> None:
        dummy = np.zeros(frame_shape, dtype=np.uint8)
        self.detect(dummy)


# =====================================================================
# PART 20 — Geometry Utilities
# =====================================================================
def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_giou(box1: Tuple[int, int, int, int],
                 box2: Tuple[int, int, int, int]) -> float:
    """Generalized IoU"""
    iou = compute_iou(box1, box2)
    cx1 = min(box1[0], box2[0])
    cy1 = min(box1[1], box2[1])
    cx2 = max(box1[2], box2[2])
    cy2 = max(box1[3], box2[3])
    c_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    if c_area == 0:
        return iou
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area1 + area2 - inter
    return iou - (c_area - union) / c_area


def compute_diou(box1: Tuple[int, int, int, int],
                 box2: Tuple[int, int, int, int]) -> float:
    """Distance IoU"""
    iou = compute_iou(box1, box2)
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    ex1 = min(box1[0], box2[0])
    ey1 = min(box1[1], box2[1])
    ex2 = max(box1[2], box2[2])
    ey2 = max(box1[3], box2[3])
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if diag_sq == 0:
        return iou
    return iou - center_dist_sq / diag_sq


def box_containment(inner: Tuple[int, int, int, int],
                    outer: Tuple[int, int, int, int]) -> float:
    """نسبة احتواء الصندوق الداخلي داخل الخارجي"""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0.0


def box_center_distance(box1: Tuple[int, int, int, int],
                        box2: Tuple[int, int, int, int]) -> float:
    """المسافة بين مركزي صندوقين"""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def expand_box(box: Tuple[int, int, int, int],
               frac: float, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """توسيع الصندوق بنسبة معينة"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    px = int(w * frac)
    py = int(h * frac)
    H, W = frame_shape[:2]
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(W, x2 + px),
        min(H, y2 + py),
    )


def is_box_inside_frame(box: Tuple[int, int, int, int],
                        frame_shape: Tuple[int, ...]) -> bool:
    H, W = frame_shape[:2]
    return 0 <= box[0] < W and 0 <= box[1] < H and box[2] > 0 and box[3] > 0


# =====================================================================
# PART 21 — YOLO-World Detector
# =====================================================================
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False


DEFAULT_WORLD_CLASSES = [
    # أشخاص
    "person",
    # أسلحة نارية
    "gun", "pistol", "rifle", "handgun", "revolver",
    # أسلحة بيضاء
    "knife", "machete", "axe", "hammer", "baseball bat",
    "sword", "dagger", "box cutter",
    # متفجرات
    "grenade", "bomb", "explosive", "dynamite",
    # صواعق
    "taser", "stun gun",
    # تخفي
    "mask", "balaclava", "ski mask", "hood", "helmet",
    # أجهزة تسجيل
    "cell phone", "smartphone", "laptop", "camera",
    "dslr", "video camera", "drone", "quadcopter",
    "tablet", "gopro", "webcam",
    # حقائب
    "backpack", "handbag", "suitcase", "briefcase",
    "duffel bag", "messenger bag",
    # أدوات أخرى
    "handcuffs", "syringe", "rope", "crowbar", "screwdriver",
    "flashlight", "binoculars", "wallet", "pen", "bottle",
    "cup", "scissors",
]


class YOLOWorldDetector(BaseDetector):
    """محلل YOLO-World للكشف المفتوح الفئات"""

    def __init__(
        self,
        model_path: str = "yolov8n-world.pt",
        classes: Optional[List[str]] = None,
        conf: float = 0.28,
        imgsz: int = 960,
        device: str = "cpu",
        enabled: bool = True,
    ):
        super().__init__("yolo_world", enabled)
        self.model_path = model_path
        self.classes = classes or DEFAULT_WORLD_CLASSES
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = None
        self._class_names: Dict[int, str] = {}
        if self.enabled and HAVE_YOLO:
            self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            self.model.set_classes(self.classes)
            try:
                self.model.to(self.device)
            except Exception:
                pass
            # تعيين أسماء الفئات
            names = self.model.names
            if isinstance(names, dict):
                self._class_names = names
            else:
                self._class_names = {i: n for i, n in enumerate(names)}
        except Exception as e:
            import logging
            logging.getLogger("falconeye").warning(
                "فشل تحميل YOLO-World: %s", e
            )
            self.model = None
            self.enabled = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled or self.model is None:
            return []
        t0 = time.perf_counter()
        try:
            import torch
            with torch.inference_mode():
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    half=(self.device != "cpu"),
                    verbose=False,
                    device=self.device,
                )
        except Exception as e:
            import logging
            logging.getLogger("falconeye").error("Detection error: %s", e)
            return []

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label = self._class_names.get(cls_id, f"class_{cls_id}")
                conf = float(boxes.conf[i].item())
                bbox = tuple(map(int, boxes.xyxy[i].tolist()))
                detections.append(Detection(
                    label=label, confidence=conf, bbox=bbox,
                    class_id=cls_id,
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record_call(len(detections), elapsed_ms)
        return detections


# =====================================================================
# PART 22 — Pose Detector
# =====================================================================
class PoseDetector(BaseDetector):
    """محلل الوضعيات البشرية — 17 نقطة مفصلية"""

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf: float = 0.5,
        device: str = "cpu",
        enabled: bool = True,
    ):
        super().__init__("yolo_pose", enabled)
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self.model = None
        if self.enabled and HAVE_YOLO:
            self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            try:
                self.model.to(self.device)
            except Exception:
                pass
        except Exception as e:
            import logging
            logging.getLogger("falconeye").warning(
                "فشل تحميل YOLO-Pose: %s", e
            )
            self.model = None
            self.enabled = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled or self.model is None:
            return []
        t0 = time.perf_counter()
        try:
            import torch
            with torch.inference_mode():
                results = self.model.predict(
                    frame, conf=self.conf, verbose=False,
                    half=(self.device != "cpu"), device=self.device,
                )
        except Exception:
            return []

        detections: List[Detection] = []
        if results and results[0].keypoints is not None:
            kps_all = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes
            if boxes is None:
                return []
            bx_all = boxes.xyxy.cpu().numpy()
            for kp, bx in zip(kps_all, bx_all):
                detections.append(Detection(
                    label="person",
                    confidence=0.9,
                    bbox=tuple(map(int, bx.tolist())),
                    keypoints=kp.tolist(),
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record_call(len(detections), elapsed_ms)
        return detections


# =====================================================================
# PART 23 — Face Recognition
# =====================================================================
class FaceRecognizer(BaseDetector):
    """التعرف على الوجوه — مع قاعدة بيانات embeddings"""

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        conf: float = 0.6,
        similarity_threshold: float = 0.55,
        enabled: bool = False,
    ):
        super().__init__("face_recognition", enabled)
        self.model_path = model_path
        self.conf = conf
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.gallery: Dict[str, np.ndarray] = {}
        if self.enabled and HAVE_YOLO:
            self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
        except Exception:
            self.model = None
            self.enabled = False

    def add_to_gallery(self, name: str, embedding: np.ndarray) -> None:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        self.gallery[name] = embedding

    def _extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """توليد embedding بسيط من الرقعة"""
        if face_crop.size == 0:
            return None
        import cv2
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        # histogram-based embedding
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-9)
        return hist.astype(np.float32)

    def _match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.gallery:
            return None, 0.0
        best_name = None
        best_score = -1.0
        for name, ref in self.gallery.items():
            score = float(np.dot(embedding, ref))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.similarity_threshold:
            return best_name, best_score
        return None, best_score

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled or self.model is None:
            return []
        t0 = time.perf_counter()
        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        except Exception:
            return []

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = tuple(map(int, boxes.xyxy[i].tolist()))
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = bbox
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                emb = self._extract_embedding(face_crop)
                name = None
                distance = None
                if emb is not None:
                    name, distance = self._match(emb)
                detections.append(Detection(
                    label="face",
                    confidence=conf,
                    bbox=bbox,
                    embedding=emb,
                    face_name=name,
                    face_distance=distance,
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record_call(len(detections), elapsed_ms)
        return detections


# =====================================================================
# PART 24 — Sound Anomaly Detector
# =====================================================================
class SoundAnomalyDetector(BaseDetector):
    """كشف الأصوات غير الطبيعية (طلقات، زجاج، صراخ)"""

    SOUND_CLASSES = ["gunshot", "glass_break", "scream", "explosion", "normal"]

    def __init__(self, enabled: bool = False, threshold: float = 0.7):
        super().__init__("sound_anomaly", enabled)
        self.threshold = threshold
        self._buffer: List[float] = []
        self._lock = threading.RLock()

    def push_audio(self, rms_value: float) -> None:
        """دفع قيمة RMS من الميكروفون"""
        with self._lock:
            self._buffer.append(rms_value)
            if len(self._buffer) > 30:
                self._buffer.pop(0)

    def detect_anomaly(self) -> Optional[str]:
        """كشف مفاجئ في مستوى الصوت"""
        with self._lock:
            if len(self._buffer) < 10:
                return None
            avg = sum(self._buffer[:-5]) / max(1, len(self._buffer) - 5)
            recent = self._buffer[-5:]
            peak = max(recent)
            if avg > 0 and peak / avg > self.threshold * 3:
                if peak > 0.5:
                    return "gunshot"
                if peak > 0.3:
                    return "glass_break"
                return "scream"
        return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


# =====================================================================
# PART 25 — Detection Fusion
# =====================================================================
class DetectionFusion:
    """دمج نتائج محللات متعددة وإزالة التكرارات"""

    def __init__(self, iou_threshold: float = 0.55):
        self.iou_threshold = iou_threshold

    def fuse(
        self,
        object_dets: List[Detection],
        pose_dets: List[Detection],
        face_dets: List[Detection],
    ) -> List[Detection]:
        """دمج الكشوفات + ربط الوجوه بالأشخاص"""
        merged: List[Detection] = []
        merged.extend(object_dets)

        # ربط الوجوه بالأشخاص
        for face in face_dets:
            best_person = None
            best_containment = 0.0
            for person in object_dets:
                if person.label != "person":
                    continue
                c = box_containment(face.bbox, person.bbox)
                if c > best_containment and c > 0.3:
                    best_containment = c
                    best_person = person
            if best_person is not None:
                best_person.metadata["face"] = {
                    "name": face.face_name,
                    "distance": face.face_distance,
                    "bbox": face.bbox,
                }
                if face.face_name:
                    best_person.metadata["identified"] = face.face_name
            else:
                merged.append(face)

        # ربط الوضعيات بالأشخاص
        for pose in pose_dets:
            for person in object_dets:
                if person.label != "person":
                    continue
                if compute_iou(pose.bbox, person.bbox) > 0.5:
                    person.keypoints = pose.keypoints
                    break

        # إزالة التكرارات
        return self._suppress_duplicates(merged)

    def _suppress_duplicates(self, detections: List[Detection]) -> List[Detection]:
        """Non-Maximum Suppression"""
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.label, []).append(d)

        result: List[Detection] = []
        for label, dets in by_class.items():
            dets.sort(key=lambda x: x.confidence, reverse=True)
            kept: List[Detection] = []
            for d in dets:
                if all(compute_iou(d.bbox, k.bbox) < self.iou_threshold for k in kept):
                    kept.append(d)
            result.extend(kept)
        return result


# =====================================================================
# PART 26 — Kalman Filter
# =====================================================================
class KalmanBoxFilter:
    """مرشح Kalman لصندوق محيط — 6 حالات (cx, cy, w, h, vx, vy)"""

    def __init__(self, cx: float, cy: float, w: float, h: float):
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6) * 30.0
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        self.Q = np.diag([0.2, 0.2, 0.2, 0.2, 1.0, 1.0])
        self.R_base = np.diag([3.0, 3.0, 3.0, 3.0])

    def predict(self) -> Tuple[float, float, float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(
        self, cx: float, cy: float, w: float, h: float, conf: float = 0.9
    ) -> Tuple[float, float, float, float]:
        sigma = 8.0 - 6.0 * max(0.0, min(1.0, conf))
        R = self.R_base * (sigma / 3.0)
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.box()

        self.x = self.x + K @ y
        self.clamp_velocity()
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.box()

    def clamp_velocity(self, max_speed: float = 180.0) -> None:
        vx, vy = self.x[4], self.x[5]
        speed = float(np.hypot(vx, vy))
        if speed > max_speed and speed > 1e-6:
            scale = max_speed / speed
            self.x[4] *= scale
            self.x[5] *= scale

    def damp_velocity(self, factor: float = 0.85) -> None:
        self.x[4] *= factor
        self.x[5] *= factor

    def box(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        w = max(4.0, w)
        h = max(4.0, h)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    def speed(self) -> float:
        return float(np.hypot(self.x[4], self.x[5]))


# =====================================================================
# PART 27 — Track Class
# =====================================================================
from collections import deque
from enum import Enum


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    ARCHIVED = "archived"


class TrackStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"


class Track:
    """تتبّع واحد لهدف واحد"""

    def __init__(self, track_id: int, detection: Detection,
                 frame_shape: Tuple[int, ...]):
        self.id = track_id
        self.state = TrackState.TENTATIVE
        self.status = TrackStatus.NORMAL
        self.label = detection.label
        self.stable_label = detection.label
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.keypoints = detection.keypoints
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.hits = 1
        self.misses = 0
        self.age_frames = 1

        # Kalman
        cx, cy = detection.center
        self.kf = KalmanBoxFilter(cx, cy, detection.width, detection.height)

        # القرارات
        self.suspicion = 0.0
        self.reason = ""
        self.soft_score = 0.0
        self.hard_score = 0.0
        self.alarm_hold_until = 0.0

        # الحركة
        self.positions = deque(maxlen=1500)
        self.positions.append((int(cx), int(cy), time.time()))
        self.motion_dir = (0.0, 0.0)

        # المنطقة
        self.inside_since: Optional[float] = None
        self.dwell = 0.0

        # بصمة لونية
        self.color_hist: Optional[np.ndarray] = None

        # سلوك
        self.photo_events = deque(maxlen=20)
        self.last_photo_event = 0.0
        self.vote_buffer: Dict[str, deque] = {}

        # مالك السلاح
        self.owner_person_id: Optional[int] = None
        self.owner_weapon_ids: set = set()

        # snapshot
        self.last_snapshot = 0.0

        # metadata
        self.metadata: Dict[str, Any] = {}

    @property
    def is_person(self) -> bool:
        return self.stable_label == "person"

    @property
    def is_weapon(self) -> bool:
        return self.stable_label in {
            "gun", "pistol", "rifle", "knife", "machete",
            "axe", "hammer", "baseball bat", "grenade",
            "taser", "bomb", "explosive",
        }

    @property
    def is_recorder(self) -> bool:
        return self.stable_label in {
            "cell phone", "camera", "dslr", "laptop",
            "drone", "quadcopter",
        }

    def update(self, detection: Detection, frame_shape: Tuple[int, ...]) -> None:
        """تحديث التتبّع بكشف جديد"""
        prev_cx, prev_cy = self.bbox[0] + (self.bbox[2] - self.bbox[0]) / 2, \
                            self.bbox[1] + (self.bbox[3] - self.bbox[1]) / 2
        new_cx, new_cy = detection.center

        # رفض القفزات الكبيرة
        jump = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))
        if jump > 0.25 * frame_shape[1]:
            return

        self.bbox = detection.bbox
        self.confidence = detection.confidence
        if detection.keypoints is not None:
            self.keypoints = detection.keypoints

        # تحديث Kalman
        self.kf.update(
            new_cx, new_cy, detection.width, detection.height,
            conf=detection.confidence,
        )

        # تحديث الحركة
        prev_pos = self.positions[-1] if self.positions else None
        if prev_pos:
            dt = max(1e-3, time.time() - prev_pos[2])
            ndx = (new_cx - prev_pos[0]) / dt
            ndy = (new_cy - prev_pos[1]) / dt
            alpha = 0.6
            self.motion_dir = (
                alpha * ndx + (1 - alpha) * self.motion_dir[0],
                alpha * ndy + (1 - alpha) * self.motion_dir[1],
            )

        self.positions.append((int(new_cx), int(new_cy), time.time()))
        self.last_seen = time.time()
        self.hits += 1
        self.misses = 0
        self.age_frames += 1

        if self.state == TrackState.TENTATIVE and self.hits >= 2:
            self.state = TrackState.CONFIRMED
        self.misses = 0
        self.age_frames += 1

    def predict(self) -> Tuple[float, float, float, float]:
        """توقّع الموضع التالي"""
        box = self.kf.predict()
        self.kf.clamp_velocity()
        self.kf.damp_velocity(0.85)
        self.bbox = tuple(map(int, box))
        self.misses += 1
        self.age_frames += 1
        return box

    def get_speed(self) -> float:
        return self.kf.speed()

    def is_loitering(self, radius: float = 60.0,
                     window_secs: float = 6.0,
                     min_samples: int = 5) -> bool:
        now = time.time()
        recent = [p for p in self.positions if now - p[2] < window_secs]
        if len(recent) < min_samples:
            return False
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        spread = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        return spread < radius

    def is_crouching(self, ratio_threshold: float = 1.2) -> bool:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        if w <= 0:
            return False
        return (h / w) < ratio_threshold

    def vote(self, key: str, value: bool,
             window: int = 5, needed: int = 3) -> bool:
        """تصويت مؤقت للسلوك"""
        if key not in self.vote_buffer:
            self.vote_buffer[key] = deque(maxlen=window)
        self.vote_buffer[key].append(bool(value))
        return sum(self.vote_buffer[key]) >= needed

    def register_photo_event(self, cooldown: float = 2.0) -> None:
        now = time.time()
        if now - self.last_photo_event < cooldown:
            return
        self.last_photo_event = now
        self.photo_events.append(now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.stable_label,
            "status": (self.status.value
            if hasattr(self.status, "value")
            else str(self.status)),
            "state": self.state.value,
            "suspicion": round(self.suspicion, 1),
            "reason": self.reason,
            "bbox": list(self.bbox),
            "age": round(time.time() - self.first_seen, 1),
            "hits": self.hits,
            "misses": self.misses,
        }


# =====================================================================
# PART 28 — Track Manager with Hungarian
# =====================================================================
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class TrackManager:
    """مدير التتبعات الكامل"""

    def __init__(
        self,
        max_coast: int = 3,
        max_coast_moving: int = 5,
        min_hits_confirm: int = 5,
        max_jump_frac: float = 0.20,
        max_track_speed: float = 180.0,
        iou_threshold: float = 0.20,
        assoc_max_cost: float = 0.55,
        reid_threshold: float = 0.80,
        reid_max_age: float = 5.0,
        history_len: int = 1500,
    ):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.lost_pool: List[Tuple[str, np.ndarray, Tuple, float, float]] = []
        self.max_coast = max_coast
        self.max_coast_moving = max_coast_moving
        self.min_hits_confirm = min_hits_confirm
        self.max_jump_frac = max_jump_frac
        self.max_track_speed = max_track_speed
        self.iou_threshold = iou_threshold
        self.assoc_max_cost = assoc_max_cost
        self.reid_threshold = reid_threshold
        self.reid_max_age = reid_max_age
        self.history_len = history_len
        self._lock = threading.RLock()

    def update(self, detections: List[Detection],
               frame_shape: Tuple[int, ...]) -> None:
        """تحديث كامل — التنبؤ → الربط → التحديث"""
        with self._lock:
            # 1) التنبؤ
            for track in self.tracks.values():
                track.predict()

            # 2) حذف المسارات الميتة
            self._purge_dead()

            # 3) الربط
            assignments = self._associate(detections, frame_shape)

            # 4) تحديث المسارات المرتبطة
            assigned_tracks = set()
            assigned_dets = set()
            for det_idx, track_id in assignments.items():
                track = self.tracks.get(track_id)
                if track is None:
                    continue
                track.update(detections[det_idx], frame_shape)
                assigned_tracks.add(track_id)
                assigned_dets.add(det_idx)

            # 5) إنشاء مسارات جديدة
            for det_idx, det in enumerate(detections):
                if det_idx in assigned_dets:
                    continue
                self._create_track(det, frame_shape)

    def _associate(
        self, detections: List[Detection], frame_shape: Tuple[int, ...]
    ) -> Dict[int, int]:
        """ربط أمثل باستخدام Hungarian"""
        if not self.tracks or not detections:
            return {}

        tids = list(self.tracks.keys())
        n_t = len(tids)
        n_d = len(detections)
        INF = 1e6
        cost = np.full((n_t, n_d), INF, dtype=np.float64)

        for i, tid in enumerate(tids):
            track = self.tracks[tid]
            for j, det in enumerate(detections):
                # رفض عدم التطابق في الفئة
                if track.stable_label != det.label:
                    if not self._label_compatible(track.stable_label, det.label):
                        continue

                # IoU
                iou = compute_iou(det.bbox, track.bbox)
                if iou < self.iou_threshold:
                    continue

                # فحص القفزة
                jump = box_center_distance(det.bbox, track.bbox)
                if jump > self.max_jump_frac * frame_shape[1]:
                    continue

                # فحص الحركة
                if not self._motion_ok(track, det):
                    continue

                # تكلفة
                label_mismatch = 0.0 if track.stable_label == det.label else 0.1
                cost[i, j] = (1 - iou) + label_mismatch

        # الحل
        assignments: Dict[int, int] = {}
        if cost.min() < INF:
            if HAVE_SCIPY:
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < self.assoc_max_cost:
                        assignments[c] = tids[r]
            else:
                # Greedy fallback
                pairs = []
                for i in range(n_t):
                    for j in range(n_d):
                        if cost[i, j] < self.assoc_max_cost:
                            pairs.append((cost[i, j], i, j))
                pairs.sort()
                used_t, used_d = set(), set()
                for _, i, j in pairs:
                    if i in used_t or j in used_d:
                        continue
                    assignments[j] = tids[i]
                    used_t.add(i)
                    used_d.add(j)

        return assignments

    def _label_compatible(self, label1: str, label2: str) -> bool:
        """هل يمكن اعتبار الفئتين متوافقتين؟"""
        confusions = {
            "gun": {"pistol", "handgun", "revolver"},
            "pistol": {"gun", "handgun"},
            "handgun": {"gun", "pistol"},
            "knife": {"dagger", "box cutter"},
            "camera": {"dslr", "video camera", "webcam"},
            "cell phone": {"smartphone", "tablet"},
            "drone": {"quadcopter"},
        }
        return label2 in confusions.get(label1, set())

    def _motion_ok(self, track: Track, det: Detection) -> bool:
        """فحص اتجاه الحركة"""
        if track.misses > 5:
            return True
        motion_mag = float(np.hypot(*track.motion_dir))
        if motion_mag < 20:
            return True
        det_dir = (
            det.center[0] - track.bbox[0] - (track.bbox[2] - track.bbox[0]) / 2,
            det.center[1] - track.bbox[1] - (track.bbox[3] - track.bbox[1]) / 2,
        )
        if float(np.hypot(*det_dir)) < 8:
            return True
        # زاوية
        dot = track.motion_dir[0] * det_dir[0] + track.motion_dir[1] * det_dir[1]
        n1 = motion_mag
        n2 = float(np.hypot(*det_dir))
        if n1 * n2 < 1e-6:
            return True
        cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle = math.degrees(math.acos(cos_a))
        if angle > 65:
            return False
        # سرعة
        det_speed = float(np.hypot(*det_dir))
        if det_speed > max(60, 2 * track.get_speed()):
            return False
        return True

    def _create_track(self, det: Detection,
                      frame_shape: Tuple[int, ...]) -> None:
        """إنشاء تتبّع جديد"""
        # محاولة Re-ID
        matched = self._try_reid(det)
        if matched is not None:
            tid = self.next_id
            self.next_id += 1
            track = Track(tid, det, frame_shape)
            track.state = TrackState.CONFIRMED
            track.hits = 100
            track.suspicion = min(matched[4], 35.0)
            self.tracks[tid] = track
            return

        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = Track(tid, det, frame_shape)

    def _try_reid(self, det: Detection) -> Optional[Tuple]:
        """محاولة إعادة التعريف"""
        if det.embedding is None:
            return None
        now = time.time()
        best = None
        best_score = self.reid_threshold
        for entry in self.lost_pool:
            label, template, box, ts, score = entry
            if label != det.label:
                continue
            if now - ts > self.reid_max_age:
                continue
            if box_center_distance(det.bbox, box) > 150:
                continue
            if template is None:
                continue
            sim = float(np.dot(det.embedding, template))
            if sim > best_score:
                best_score = sim
                best = entry
        if best:
            self.lost_pool.remove(best)
            return best
        return None

    def _purge_dead(self) -> None:
        """حذف المسارات الميتة ونقلها إلى lost_pool"""
        to_remove = []
        for tid, track in self.tracks.items():
            cap = self.max_coast
            if track.get_speed() > 40:
                cap = self.max_coast_moving
            if track.misses > cap:
                self.lost_pool.append((
                    track.stable_label, track.color_hist,
                    track.bbox, time.time(), track.suspicion,
                ))
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # تنظيف lost_pool
        now = time.time()
        self.lost_pool = [
            e for e in self.lost_pool
            if now - e[3] < self.reid_max_age + 2
        ]

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values()
                if t.state != TrackState.LOST]

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values()
                if t.state == TrackState.CONFIRMED]

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self.lost_pool.clear()
            self.next_id = 1


# =====================================================================
# PART 29 — Color Histogram Extractor
# =====================================================================
class ColorHistogram:
    """بصمة لونية HSV للمساعدة في Re-ID"""

    @staticmethod
    def compute(frame: np.ndarray,
                bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not HAVE_CV2:
            return None
        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    @staticmethod
    def similarity(h1: Optional[np.ndarray],
                   h2: Optional[np.ndarray]) -> float:
        if h1 is None or h2 is None:
            return 0.0
        try:
            return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        except Exception:
            return 0.0


# =====================================================================
# PART 30 — Frame Processor
# =====================================================================
class FrameProcessor:
    """معالج إطار واحد — ينفّذ كل خطوات التحليل"""

    def __init__(
        self,
        detector_world: YOLOWorldDetector,
        detector_pose: PoseDetector,
        detector_face: Optional[FaceRecognizer],
        track_manager: TrackManager,
        fusion: DetectionFusion,
        enable_pose: bool = True,
        enable_face: bool = False,
    ):
        self.detector_world = detector_world
        self.detector_pose = detector_pose
        self.detector_face = detector_face
        self.track_manager = track_manager
        self.fusion = fusion
        self.enable_pose = enable_pose
        self.enable_face = enable_face
        self._frame_count = 0

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """معالجة كاملة لإطار واحد"""
        t0 = time.perf_counter()
        self._frame_count += 1

        # 1) الكشف
        object_dets = self.detector_world.detect(frame)
        pose_dets = self.detector_pose.detect(frame) if self.enable_pose else []
        face_dets = (
            self.detector_face.detect(frame)
            if self.enable_face and self.detector_face else []
        )

        # 2) الدمج
        fused = self.fusion.fuse(object_dets, pose_dets, face_dets)

        # 3) استخراج البصمات اللونية
        for det in fused:
            det.embedding = ColorHistogram.compute(frame, det.bbox)

        # 4) التتبع
        self.track_manager.update(fused, frame.shape)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "detections": [d.to_dict() for d in fused],
            "tracks": [t.to_dict() for t in self.track_manager.get_active_tracks()],
            "elapsed_ms": round(elapsed, 2),
            "frame_idx": self._frame_count,
        }


# =====================================================================
# PART 31 — Multi-Camera Manager
# =====================================================================
from dataclasses import dataclass as _dataclass


@_dataclass
class CameraStream:
    """تدفق كاميرا واحدة"""
    camera_id: str
    source: Any
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True
    location: str = ""


class CameraReader:
    """قراءة إطارات من كاميرا في Thread منفصل"""

    def __init__(self, stream: CameraStream):
        self.stream = stream
        self.cap = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_actual = 0.0
        self._last_ts = 0.0
        self._frame_count = 0

    def start(self) -> bool:
        if not HAVE_CV2:
            return False
        self.cap = cv2.VideoCapture(self.stream.source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream.height)
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.stream.camera_id}", daemon=True
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.cap is None:
                time.sleep(0.1)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            if frame.shape[1] != self.stream.width or frame.shape[0] != self.stream.height:
                frame = cv2.resize(frame, (self.stream.width, self.stream.height))
            with self._lock:
                self._frame = frame
                self._frame_count += 1
                now = time.time()
                if self._last_ts:
                    dt = now - self._last_ts
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self._fps_actual = 0.9 * self._fps_actual + 0.1 * instant_fps
                self._last_ts = now

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def fps(self) -> float:
        return round(self._fps_actual, 1)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            with suppress(Exception):
                self.cap.release()


class MultiCameraManager:
    """مدير متعدد الكاميرات"""

    def __init__(self):
        self.readers: Dict[str, CameraReader] = {}
        self.processors: Dict[str, FrameProcessor] = {}
        self._lock = threading.RLock()

    def add_camera(self, stream: CameraStream,
                   processor: FrameProcessor) -> bool:
        with self._lock:
            reader = CameraReader(stream)
            if not reader.start():
                return False
            self.readers[stream.camera_id] = reader
            self.processors[stream.camera_id] = processor
            return True

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        reader = self.readers.get(camera_id)
        return reader.read() if reader else None

    def process_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        frame = self.get_frame(camera_id)
        if frame is None:
            return None
        processor = self.processors.get(camera_id)
        if processor is None:
            return None
        return processor.process(frame)

    def list_cameras(self) -> List[str]:
        return list(self.readers.keys())

    def stop_all(self) -> None:
        with self._lock:
            for reader in self.readers.values():
                reader.stop()
            self.readers.clear()
            self.processors.clear()


# =====================================================================
# PART 32 — Heatmap Generator
# =====================================================================
class HeatmapGenerator:
    """مولّد خرائط حرارية للحركة"""

    def __init__(self, width: int = 1280, height: int = 720,
                 decay: float = 0.98, blur_size: int = 31):
        self.width = width
        self.height = height
        self.decay = decay
        self.blur_size = blur_size
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self._lock = threading.RLock()

    def update(self, tracks: List[Track]) -> None:
        with self._lock:
            self.heatmap *= self.decay
            for track in tracks:
                if track.state != TrackState.CONFIRMED:
                    continue
                cx, cy = track.positions[-1][:2] if track.positions else (0, 0)
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.heatmap[cy, cx] += 1.0

    def render(self, frame: np.ndarray,
               alpha: float = 0.4) -> np.ndarray:
        """رسم الخريطة على الإطار"""
        if not HAVE_CV2:
            return frame
        with self._lock:
            if self.heatmap.max() < 1:
                return frame
            norm = cv2.normalize(
                self.heatmap, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            norm = cv2.GaussianBlur(norm, (self.blur_size, self.blur_size), 0)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
            return blended

    def reset(self) -> None:
        with self._lock:
            self.heatmap.fill(0)

    def get_peak_zones(self, top_n: int = 5,
                       min_dist: int = 50) -> List[Tuple[int, int, float]]:
        """أعلى مناطق النشاط"""
        with self._lock:
            hm = self.heatmap.copy()
        peaks: List[Tuple[int, int, float]] = []
        for _ in range(top_n):
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            cy, cx = int(idx[0]), int(idx[1])
            val = float(hm[cy, cx])
            if val < 1:
                break
            peaks.append((cx, cy, val))
            # منع التكرار
            y1 = max(0, cy - min_dist)
            y2 = min(self.height, cy + min_dist)
            x1 = max(0, cx - min_dist)
            x2 = min(self.width, cx + min_dist)
            hm[y1:y2, x1:x2] = 0
        return peaks


# =====================================================================
# PART 33 — Zone Manager
# =====================================================================
@dataclass
class Zone:
    """منطقة داخل الإطار"""
    name: str
    inner: Tuple[float, float, float, float]  # نسبية
    outer: Optional[Tuple[float, float, float, float]] = None
    color: Tuple[int, int, int] = (0, 255, 255)
    enabled: bool = True

    def get_pixels(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        H, W = frame_shape[:2]
        return (
            int(W * self.inner[0]), int(H * self.inner[1]),
            int(W * self.inner[2]), int(H * self.inner[3]),
        )

    def contains(self, center: Tuple[float, float],
                 frame_shape: Tuple[int, ...]) -> bool:
        x1, y1, x2, y2 = self.get_pixels(frame_shape)
        return x1 < center[0] < x2 and y1 < center[1] < y2


class ZoneManager:
    """إدارة مناطق محظورة متعددة"""

    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self._lock = threading.RLock()

    def add_zone(self, zone: Zone) -> None:
        with self._lock:
            self.zones[zone.name] = zone

    def remove_zone(self, name: str) -> bool:
        with self._lock:
            return self.zones.pop(name, None) is not None

    def get_zones_containing(self, center: Tuple[float, float],
                             frame_shape: Tuple[int, ...]) -> List[Zone]:
        with self._lock:
            return [
                z for z in self.zones.values()
                if z.enabled and z.contains(center, frame_shape)
            ]

    def draw_zones(self, frame: np.ndarray, alarm: bool = False) -> None:
        if not HAVE_CV2:
            return
        for zone in self.zones.values():
            if not zone.enabled:
                continue
            x1, y1, x2, y2 = zone.get_pixels(frame.shape)
            color = (0, 0, 255) if alarm else zone.color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


# =====================================================================
# PART 34 — Detection Cache
# =====================================================================
class DetectionCache:
    """ذاكرة مؤقتة لتسريع المعالجة عبر الإطارات"""

    def __init__(self, ttl_frames: int = 3):
        self.ttl_frames = ttl_frames
        self._cache: Dict[str, Tuple[List[Detection], int]] = {}
        self._lock = threading.RLock()
        self._frame_idx = 0

    def step(self) -> None:
        with self._lock:
            self._frame_idx += 1
            expired = [
                k for k, (_, fi) in self._cache.items()
                if self._frame_idx - fi > self.ttl_frames
            ]
            for k in expired:
                del self._cache[k]

    def get(self, key: str) -> Optional[List[Detection]]:
        with self._lock:
            entry = self._cache.get(key)
            return entry[0] if entry else None

    def put(self, key: str, detections: List[Detection]) -> None:
        with self._lock:
            self._cache[key] = (detections, self._frame_idx)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# =====================================================================
# PART 35 — Adaptive Resolution Controller
# =====================================================================
class AdaptiveResolution:
    """تحكم تكيفي بدقة المعالجة حسب الأداء"""

    def __init__(self, imgsz_max: int = 960, imgsz_min: int = 480,
                 target_fps: float = 15.0):
        self.imgsz_max = imgsz_max
        self.imgsz_min = imgsz_min
        self.current = imgsz_max
        self.target_fps = target_fps
        self._fps_history = deque(maxlen=30)
        self._last_adjust = time.time()

    def record_fps(self, fps: float) -> None:
        self._fps_history.append(fps)

    def adjust(self) -> int:
        """تعديل الدقة حسب الأداء"""
        if len(self._fps_history) < 10:
            return self.current
        if time.time() - self._last_adjust < 5.0:
            return self.current
        avg_fps = sum(self._fps_history) / len(self._fps_history)
        if avg_fps < self.target_fps * 0.7 and self.current > self.imgsz_min:
            self.current = max(self.imgsz_min, self.current - 160)
            self._last_adjust = time.time()
        elif avg_fps > self.target_fps * 1.3 and self.current < self.imgsz_max:
            self.current = min(self.imgsz_max, self.current + 160)
            self._last_adjust = time.time()
        return self.current

    def reset(self) -> None:
        self.current = self.imgsz_max
        self._fps_history.clear()


# =====================================================================
# PART 36 — Warmup Utilities
# =====================================================================
def warmup_all(detectors: List[BaseDetector],
               frame_shape: Tuple[int, ...] = (720, 1280, 3)) -> None:
    """تسخين جميع المحللات"""
    import logging
    log = logging.getLogger("falconeye")
    dummy = np.zeros(frame_shape, dtype=np.uint8)
    for det in detectors:
        if not det.enabled:
            continue
        try:
            t0 = time.perf_counter()
            det.detect(dummy)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("تسخين %s: %.1f ms", det.name, elapsed)
        except Exception as e:
            log.warning("فشل تسخين %s: %s", det.name, e)


# =====================================================================
# PART 37 — Device Selector
# =====================================================================
def select_device(prefer_gpu: bool = True) -> str:
    """اختيار الجهاز الأمثل"""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                return f"cuda:{idx}"
            except Exception:
                return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def device_info() -> Dict[str, Any]:
    """معلومات الجهاز"""
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    if HAVE_PSUTIL:
        info["memory_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3), 2
        )
    return info


# =====================================================================
# PART 38 — Pipeline Factory
# =====================================================================
class PipelineFactory:
    """مصنع لبناء خطوط المعالجة"""

    @staticmethod
    def build_standard(
        cfg: SystemConfig,
        db: Database,
    ) -> Dict[str, Any]:
        """بناء خط معالجة قياسي"""
        device = select_device()

        # المحللات
        world = YOLOWorldDetector(
            model_path=cfg.detection.world_model,
            conf=cfg.detection.conf_detect,
            imgsz=cfg.detection.detect_imgsz,
            device=device,
            enabled=True,
        )
        pose = PoseDetector(
            model_path=cfg.detection.pose_model,
            device=device,
            enabled=cfg.detection.enable_pose,
        )
        face = FaceRecognizer(
            model_path=cfg.detection.face_model,
            enabled=cfg.detection.enable_face_recognition,
        )

        # التسخين
        warmup_all([world, pose, face])

        # المكونات
        fusion = DetectionFusion(iou_threshold=0.55)
        tracker = TrackManager(
            max_coast=cfg.tracker.max_coast_frames,
            min_hits_confirm=cfg.tracker.min_hits_confirm,
            max_jump_frac=cfg.tracker.max_jump_frac,
            iou_threshold=cfg.tracker.min_match_iou,
            assoc_max_cost=cfg.tracker.assoc_max_cost,
            reid_threshold=cfg.tracker.reid_threshold,
            reid_max_age=cfg.tracker.reid_max_age,
        )
        processor = FrameProcessor(
            detector_world=world,
            detector_pose=pose,
            detector_face=face,
            track_manager=tracker,
            fusion=fusion,
            enable_pose=cfg.detection.enable_pose,
            enable_face=cfg.detection.enable_face_recognition,
        )
        heatmap = HeatmapGenerator()
        zones = ZoneManager()
        cache = DetectionCache(ttl_frames=2)
        adaptive = AdaptiveResolution(
            imgsz_max=cfg.detection.detect_imgsz,
            imgsz_min=cfg.detection.detect_imgsz_min,
        )

        return {
            "device": device,
            "world": world,
            "pose": pose,
            "face": face,
            "fusion": fusion,
            "tracker": tracker,
            "processor": processor,
            "heatmap": heatmap,
            "zones": zones,
            "cache": cache,
            "adaptive": adaptive,
        }


# =====================================================================
# PART 39 — Statistics Per Detector
# =====================================================================
class DetectorStats:
    """إحصائيات مدمجة لكل المحللات"""

    def __init__(self, detectors: List[BaseDetector]):
        self.detectors = detectors

    def summary(self) -> Dict[str, Any]:
        return {d.name: d.stats for d in self.detectors}

    def reset(self) -> None:
        for d in self.detectors:
            with d._lock:
                d._stats = {
                    "total_calls": 0,
                    "total_detections": 0,
                    "total_time_ms": 0.0,
                    "last_run": 0.0,
                }


# =====================================================================
# PART 40 — Performance Profiler
# =====================================================================
class PerformanceProfiler:
    """مُحلّل الأداء لكل مرحلة"""

    def __init__(self, window: int = 100):
        self.window = window
        self._stages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.RLock()

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._stages[stage].append(elapsed)

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out = {}
            for stage, values in self._stages.items():
                if not values:
                    continue
                arr = list(values)
                out[stage] = {
                    "avg_ms": round(sum(arr) / len(arr), 2),
                    "min_ms": round(min(arr), 2),
                    "max_ms": round(max(arr), 2),
                    "samples": len(arr),
                }
            return out

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()


# =====================================================================
# PART 41 — Main Integration Helper
# =====================================================================
def initialize_core(
    cfg: SystemConfig,
    db: Database,
    profiler: Optional[PerformanceProfiler] = None,
) -> Dict[str, Any]:
    """تهيئة كاملة للمكونات الأساسية"""
    log = logging.getLogger("falconeye")
    log.info("جاري بناء خط المعالجة…")
    pipeline = PipelineFactory.build_standard(cfg, db)

    log.info("الجهاز المستخدم: %s", pipeline["device"])
    info = device_info()
    log.info("معلومات النظام: %s", info)

    # إضافة المناطق الافتراضية
    for cam in cfg.cameras:
        zone = Zone(
            name=f"zone_{cam.camera_id}",
            inner=cam.zone_inner,
            outer=cam.zone_outer,
        )
        pipeline["zones"].add_zone(zone)

    return pipeline


# =====================================================================
# PART 42 — Module Exports
# =====================================================================
__all__ = [
    # Detection
    "Detection", "BaseDetector", "YOLOWorldDetector",
    "PoseDetector", "FaceRecognizer", "SoundAnomalyDetector",
    "DetectionFusion",
    # Geometry
    "compute_iou", "compute_giou", "compute_diou",
    "box_containment", "box_center_distance",
    "expand_box", "is_box_inside_frame",
    # Tracking
    "KalmanBoxFilter", "Track", "TrackState", "TrackStatus",
    "TrackManager", "ColorHistogram",
    # Processing
    "FrameProcessor", "CameraReader", "CameraStream",
    "MultiCameraManager", "HeatmapGenerator",
    "Zone", "ZoneManager", "DetectionCache",
    "AdaptiveResolution", "PerformanceProfiler",
    # Utils
    "warmup_all", "select_device", "device_info",
    "PipelineFactory", "DetectorStats", "initialize_core",
]
# =====================================================================
# PART 65 — UI Constants & Color Palette
# =====================================================================
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import cv2

logger_ui = None  # سيُعيَّن لاحقاً


# ألوان BGR
COLOR_OK = (0, 220, 0)           # أخضر
COLOR_WARN = (0, 200, 255)       # أصفر
COLOR_ALARM = (0, 0, 255)        # أحمر
COLOR_ZONE = (0, 255, 255)       # سماوي
COLOR_TEXT = (255, 255, 255)     # أبيض
COLOR_DARK = (25, 25, 25)        # داكن
COLOR_WEAPON = (255, 0, 255)     # بنفسجي
COLOR_EXPLOSIVE = (0, 128, 255)  # برتقالي
COLOR_PHOTO = (255, 128, 0)      # أزرق فاتح
COLOR_CYAN = (240, 220, 80)      # سماوي
COLOR_BG_PANEL = (20, 20, 25)
COLOR_BORDER = (80, 80, 80)
COLOR_TERRACOTTA = (79, 96, 201)  # #C9604F بلون BGR
COLOR_HIGHLIGHT = (60, 200, 240)


# أبعاد HUD
HUD_BAR_HEIGHT = 34
HUD_SAFE_TOP = 40
LABEL_BG_PAD = 3


# =====================================================================
# PART 66 — Arabic Text Renderer
# =====================================================================
class ArabicRenderer:
    """محرك رسم النصوص العربية مع RTL"""

    _FONT_CACHE: Dict[int, Any] = {}
    _MEASURE_IMG = None
    _MEASURE_DRAW = None
    _WARNED = False

    @classmethod
    def initialize(cls) -> bool:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import arabic_reshaper
            from bidi.algorithm import get_display
            cls._MEASURE_IMG = Image.new("RGB", (10, 10))
            cls._MEASURE_DRAW = ImageDraw.Draw(cls._MEASURE_IMG)
            return True
        except ImportError:
            return False

    @classmethod
    def has_arabic(cls, text: str) -> bool:
        for ch in str(text):
            if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F":
                return True
        return False

    @classmethod
    def shape(cls, text: str) -> str:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except Exception:
            return str(text)

    @classmethod
    def get_font(cls, size: int, font_path: str = "arial.ttf"):
        if size in cls._FONT_CACHE:
            return cls._FONT_CACHE[size]
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(font_path, size)
        except Exception:
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                if not cls._WARNED:
                    cls._WARNED = True
                    if logger_ui:
                        logger_ui.warning("خط عربي غير موجود — استخدام افتراضي")
            except Exception:
                font = None
        cls._FONT_CACHE[size] = font
        return font

    @classmethod
    def measure(cls, text: str, size: int, font_path: str = "arial.ttf",
                thickness: int = 1) -> Tuple[int, int]:
        if not cls.has_arabic(text):
            scale = size / 30.0
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, max(1, thickness))
            return w, h
        try:
            font = cls.get_font(size, font_path)
            shaped = cls.shape(text)
            bbox = cls._MEASURE_DRAW.textbbox(
                (0, 0), shaped, font=font,
                stroke_width=max(0, thickness - 1)
            )
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            scale = size / 30.0
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, max(1, thickness))
            return w, h

    @classmethod
    def draw(cls, frame: np.ndarray, text: str, org: Tuple[int, int],
             color: Tuple[int, int, int], size: int = 22,
             font_path: str = "arial.ttf", thickness: int = 1,
             bg: Optional[Tuple[int, int, int]] = None, pad: int = 3) -> None:
        if not cls.has_arabic(text):
            cls._draw_cv2(frame, text, org, color, size, thickness, bg, pad)
            return
        cls._draw_pil(frame, text, org, color, size, font_path, thickness, bg, pad)

    @classmethod
    def _draw_cv2(cls, frame, text, org, color, size, thickness, bg, pad):
        scale = size / 30.0
        th = max(1, thickness)
        if bg is not None:
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           scale, th)
            cv2.rectangle(frame, (org[0] - pad, org[1] - pad),
                          (org[0] + tw + pad, org[1] + tht + pad), bg, -1)
        cv2.putText(frame, text, (org[0], org[1] + size),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

    @classmethod
    def _draw_pil(cls, frame, text, org, color, size, font_path, thickness,
                  bg, pad):
        from PIL import Image, ImageDraw
        font = cls.get_font(size, font_path)
        if font is None:
            return
        shaped = cls.shape(text)
        sw = max(0, thickness - 1)
        bbox = cls._MEASURE_DRAW.textbbox(
            (0, 0), shaped, font=font, stroke_width=sw
        )
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x, y = org
        if bg is not None:
            cv2.rectangle(frame, (x - pad, y - pad),
                          (x + tw + pad, y + th + pad), bg, -1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        pos = (x - bbox[0], y - bbox[1])
        fill = (color[2], color[1], color[0])
        if sw > 0:
            draw.text(pos, shaped, font=font, fill=fill,
                      stroke_width=sw, stroke_fill=fill)
        else:
            draw.text(pos, shaped, font=font, fill=fill)
        frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# =====================================================================
# PART 67 — FrameText Batch
# =====================================================================
class FrameText:
    """دفعة نصوص تُرسم مرة واحدة (كفاءة عالية)"""

    def __init__(self, font_path: str = "arial.ttf"):
        self.items: List[Tuple] = []
        self.font_path = font_path

    def add(self, text: str, org: Tuple[int, int],
            color: Tuple[int, int, int] = COLOR_TEXT,
            font_size: int = 18,
            bg: Optional[Tuple[int, int, int]] = None,
            pad: int = 3, thickness: int = 1) -> None:
        self.items.append((text, org, color, font_size, bg, pad, thickness))

    def flush(self, frame: np.ndarray) -> None:
        if not self.items:
            return
        # فحص إن كان هناك نص عربي
        has_ar = any(ArabicRenderer.has_arabic(t) for t, *_ in self.items)
        if not has_ar:
            for item in self.items:
                self._draw_latin(frame, item)
        else:
            for item in self.items:
                self._draw_any(frame, item)
        self.items.clear()

    def _draw_latin(self, frame, item):
        text, org, color, size, bg, pad, thickness = item
        ArabicRenderer._draw_cv2(frame, text, org, color, size,
                                 thickness, bg, pad)

    def _draw_any(self, frame, item):
        text, org, color, size, bg, pad, thickness = item
        ArabicRenderer.draw(frame, text, org, color, size, self.font_path,
                            thickness, bg, pad)


# =====================================================================
# PART 68 — Draw Utilities
# =====================================================================
class Draw:
    """دوال رسم عامة"""

    @staticmethod
    def rect(frame: np.ndarray, box: Tuple[int, int, int, int],
             color: Tuple[int, int, int], thickness: int = 2) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness, cv2.LINE_AA)

    @staticmethod
    def filled_rect(frame, box, color, alpha: float = 0.3) -> None:
        x1, y1, x2, y2 = box
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def circle(frame, center, radius: int, color, thickness: int = 2) -> None:
        cv2.circle(frame, (int(center[0]), int(center[1])),
                   radius, color, thickness, cv2.LINE_AA)

    @staticmethod
    def filled_circle(frame, center, radius: int, color) -> None:
        cv2.circle(frame, (int(center[0]), int(center[1])),
                   radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def line(frame, p1, p2, color, thickness: int = 2) -> None:
        cv2.line(frame, (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)

    @staticmethod
    def arrow(frame, p1, p2, color, thickness: int = 2,
              arrow_size: int = 10) -> None:
        cv2.arrowedLine(frame, (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])), color, thickness,
                        cv2.LINE_AA, tipLength=arrow_size / 30.0)

    @staticmethod
    def corner_brackets(frame: np.ndarray, margin: int = 10,
                        size: int = 30, color=COLOR_CYAN,
                        thickness: int = 2) -> None:
        H, W = frame.shape[:2]
        pts = [
            (margin, margin, 1, 1),
            (W - margin, margin, -1, 1),
            (margin, H - margin, 1, -1),
            (W - margin, H - margin, -1, -1),
        ]
        for (x, y, dx, dy) in pts:
            cv2.line(frame, (x, y), (x + dx * size, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + dy * size), color, thickness)

    @staticmethod
    def dashed_rect(frame, box, color, thickness: int = 2,
                    dash: int = 10) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        # أعلى
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        # أسفل
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
        # يسار
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        # يمين
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    @staticmethod
    def crosshair(frame, center, size: int = 20, color=COLOR_TEXT,
                  thickness: int = 1) -> None:
        cx, cy = int(center[0]), int(center[1])
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)


# =====================================================================
# PART 69 — Status Bar / HUD
# =====================================================================
@dataclass
class HUDStats:
    """إحصائيات HUD"""
    fps: float = 0.0
    people: int = 0
    tracks: int = 0
    suspicious: int = 0
    evidence: int = 0
    alarms_total: int = 0
    mode: str = "World"
    tracker: str = "Hungarian"
    device: str = "CPU"
    imgsz: int = 960
    threat_level: str = "منخفض"
    threat_color: Tuple[int, int, int] = COLOR_OK
    unstable: bool = False
    language: str = "ar"


class HUD:
    """رأس الشاشة الرئيسي"""

    def __init__(self, font_path: str = "arial.ttf",
                 language: str = "ar"):
        self.font_path = font_path
        self.language = language

    def _t(self, ar: str, en: str) -> str:
        return ar if self.language == "ar" else en

    def render(self, frame: np.ndarray, stats: HUDStats) -> None:
        H, W = frame.shape[:2]
        # شريط علوي
        cv2.rectangle(frame, (0, 0), (W, HUD_BAR_HEIGHT), COLOR_DARK, -1)
        # خط فاصل
        cv2.line(frame, (0, HUD_BAR_HEIGHT), (W, HUD_BAR_HEIGHT),
                 COLOR_TERRACOTTA, 2)

        ft = FrameText(self.font_path)

        # النص الرئيسي — من اليسار
        parts = [
            f"{self._t('إ.إ', 'FPS')} {stats.fps:.1f}",
            f"{stats.mode}+{stats.tracker}+{stats.device.upper()}",
            f"imgsz {stats.imgsz}",
            f"{self._t('أشخاص', 'People')} {stats.people}",
            f"{self._t('تتبعات', 'Tracks')} {stats.tracks}",
            f"{self._t('مشتبه', 'Susp')} {stats.suspicious}",
            f"{self._t('أدلة', 'Evid')} {stats.evidence}",
        ]
        text = " | ".join(parts)
        if stats.unstable:
            text += f" | {self._t('غير مستقر', 'UNSTABLE')}"

        ft.add(text, (12, 5), COLOR_TEXT, font_size=16)

        # مستوى التهديد — من اليمين
        threat_text = f"{self._t('التهديد', 'THREAT')}: {stats.threat_level}"
        tw, _ = ArabicRenderer.measure(threat_text, 20, self.font_path)
        ft.add(threat_text, (W - tw - 20, 5), stats.threat_color,
               font_size=20, thickness=2)

        ft.flush(frame)


# =====================================================================
# PART 70 — Alarm Banner
# =====================================================================
class AlarmBanner:
    """شريط الإنذار الكبير"""

    def __init__(self, font_path: str = "arial.ttf"):
        self.font_path = font_path
        self._blink_phase = 0

    def render(self, frame: np.ndarray, active: bool,
               language: str = "ar") -> None:
        if not active:
            return
        H, W = frame.shape[:2]
        self._blink_phase = (self._blink_phase + 1) % 20
        # وميض خفيف
        if self._blink_phase < 10:
            bg_color = (10, 10, 30)
            text_color = COLOR_ALARM
        else:
            bg_color = (20, 20, 40)
            text_color = (50, 50, 255)

        banner = ("!!! تم اكتشاف تهديد — تم حفظ الأدلة مع بصمة التحقق !!!"
                  if language == "ar"
                  else "!!! THREAT DETECTED — EVIDENCE HASHED & SAVED !!!")
        tw, th = ArabicRenderer.measure(banner, 28, self.font_path, 3)
        bx = max(8, (W - tw) // 2)
        by = HUD_SAFE_TOP

        # خلفية
        cv2.rectangle(frame, (bx - 12, by - 6),
                      (bx + tw + 12, by + th + 12), bg_color, -1)
        cv2.rectangle(frame, (bx - 12, by - 6),
                      (bx + tw + 12, by + th + 12), COLOR_ALARM, 2)

        ArabicRenderer.draw(frame, banner, (bx, by + 6), text_color,
                            28, self.font_path, thickness=3)


# =====================================================================
# PART 71 — Track Label Renderer
# =====================================================================
class TrackLabelRenderer:
    """رسم ملصقات التتبعات"""

    def __init__(self, font_path: str = "arial.ttf",
                 language: str = "ar"):
        self.font_path = font_path
        self.language = language

    def _status_color(self, status: str) -> Tuple[int, int, int]:
        return {
            "normal": COLOR_OK,
            "warning": COLOR_WARN,
            "alarm": COLOR_ALARM,
            "acknowledged": (255, 128, 0),
            "resolved": (128, 128, 128),
        }.get(status, COLOR_OK)

    def render(self, frame: np.ndarray, track: Any,
               label_fn: Callable = None,
               reason_fn: Callable = None) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in track.bbox]

        # اللون
        status = getattr(track, "status", "normal")
        color = self._status_color(
            status.value if hasattr(status, "value") else status
        )

        # الصندوق
        thickness = 3 if status in ("alarm",) or (
            hasattr(status, "value") and status.value == "alarm"
        ) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # زوايا
        self._draw_corners(frame, (x1, y1, x2, y2), color, size=15, th=3)

        # ربط السلاح
        owner_ids = getattr(track, "owner_weapon_ids", set())
        if owner_ids:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, COLOR_WEAPON, -1)

        # المسار
        positions = getattr(track, "positions", [])
        if len(positions) > 1:
            pts = [(int(p[0]), int(p[1])) for p in positions][-30:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

        # الملصق
        stable_label = getattr(track, "stable_label", "?")
        label = self._tr_label(stable_label)
        tid = getattr(track, "id", 0)
        suspicion = getattr(track, "suspicion", 0.0)

        lbl_text = f"ID{tid} {label} {int(suspicion)}%"

        # الموقع — مع منع القطع
        lbl_x = max(4, min(x1 + 4, W - 220))
        lbl_y = y1 - 20
        if lbl_y < HUD_SAFE_TOP:
            lbl_y = y1 + 4
        if lbl_y > H - 30:
            lbl_y = H - 30

        self._draw_label(frame, lbl_text, (lbl_x, lbl_y), color)

        # السطر الثانوي
        secondary = self._build_secondary(track)
        if secondary:
            sub_y = lbl_y + 22
            if sub_y + 20 < H:
                self._draw_sublabel(frame, secondary, (lbl_x, sub_y))

    def _draw_corners(self, frame, box, color, size=15, th=3):
        x1, y1, x2, y2 = box
        L = size
        for (cx, cy, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, th)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, th)

    def _draw_label(self, frame, text, org, color):
        tw, th = ArabicRenderer.measure(text, 18, self.font_path)
        # خلفية
        cv2.rectangle(frame,
                      (org[0] - 4, org[1] - 4),
                      (org[0] + tw + 4, org[1] + th + 6),
                      color, -1)
        ArabicRenderer.draw(frame, text, (org[0], org[1]),
                            (0, 0, 0), 18, self.font_path, thickness=2)

    def _draw_sublabel(self, frame, text, org):
        tw, th = ArabicRenderer.measure(text, 13, self.font_path)
        cv2.rectangle(frame,
                      (org[0] - 3, org[1] - 3),
                      (org[0] + tw + 3, org[1] + th + 4),
                      (255, 255, 255), -1)
        ArabicRenderer.draw(frame, text, (org[0], org[1]),
                            (20, 20, 20), 13, self.font_path)
    def render_smart(self, frame, track, all_tracks=None):
        """رسم ذكي يمنع تراكب التسميات"""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        
        # اللون حسب الحالة
        status = getattr(track, "status", "normal")
        status_str = str(status.value if hasattr(status, "value") else status)
        
        color_map = {
            "normal": (0, 220, 0),      # أخضر
            "warning": (0, 200, 255),   # أصفر
            "alarm": (0, 0, 255),       # أحمر
        }
        color = color_map.get(status_str, (0, 220, 0))
        
        # رسم الصندوق
        thickness = 3 if status_str == "alarm" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        
        # المسار
        positions = getattr(track, "positions", [])
        if len(positions) > 1:
            pts = [(int(p[0]), int(p[1])) for p in positions][-20:]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 1, cv2.LINE_AA)
        
        # الملصق — تحقق من عدم التراكب
        label = self._tr_label(track.stable_label)
        tid = getattr(track, "id", 0)
        susp = int(getattr(track, "suspicion", 0))
        lbl_text = f"ID{tid} {label} {susp}%"
        
        # حساب أبعاد النص
        tw, th = ArabicRenderer.measure(lbl_text, 16, self.font_path)
        
        # موضع التسمية — حاول فوق الصندوق
        lbl_x = max(4, min(x1, W - tw - 20))
        lbl_y = y1 - 8
        
        # إذا خرجت من الأعلى → ضعها داخل الصندوق من الأعلى
        if lbl_y < HUD_SAFE_TOP + th:
            lbl_y = y1 + th + 6
        
        # كشف التراكب مع تسميات أخرى
        if all_tracks:
            for other in all_tracks:
                if other.id == track.id:
                    continue
                ox1, oy1, ox2, oy2 = [int(v) for v in other.bbox]
                # إذا كان الصندوقان متقاربان، أزح التسمية
                if abs(ox1 - x1) < tw + 20 and abs(oy1 - y1) < th + 8:
                    lbl_y = y2 + th + 6  # انقلها أسفل الصندوق
        
        # خلفية الملصق
        cv2.rectangle(
            frame,
            (lbl_x - 4, lbl_y - th - 4),
            (lbl_x + tw + 8, lbl_y + 6),
            color, -1,
        )
        
        # النص
        ArabicRenderer.draw(
            frame, lbl_text,
            (lbl_x, lbl_y - th + 2),
            (0, 0, 0), 16, self.font_path,
        )
        
        # السطر الثانوي (السبب) — أسفل الصندوق
        reason = getattr(track, "reason", "")
        if reason:
            reason_text = self._tr_reason(reason)[:30]
            rtw, rth = ArabicRenderer.measure(reason_text, 12, self.font_path)
            rx = max(4, min(x1, W - rtw - 20))
            ry = y2 + rth + 4
            if ry < H - 20:
                cv2.rectangle(
                    frame,
                    (rx - 3, ry - rth - 3),
                    (rx + rtw + 6, ry + 4),
                    (255, 255, 255), -1,
                )
                ArabicRenderer.draw(
                    frame, reason_text,
                    (rx, ry - rth + 2),
                    (20, 20, 20), 12, self.font_path,
                )

    def _build_secondary(self, track) -> str:
        parts = []
        if getattr(track, "state", "") != "confirmed":
            parts.append(self._t("مبدئي", "tentative"))
        photo_count = len(getattr(track, "photo_events", []))
        if photo_count > 0:
            parts.append(f"{self._t('تصوير', 'photo')}{photo_count}")
        reason = getattr(track, "reason", "")
        if reason:
            parts.append(self._tr_reason(reason))
        return " | ".join(parts[:2])

    def _tr_label(self, label: str) -> str:
        if self.language != "ar":
            return label
        return {
            "person": "شخص",
            "gun": "مسدس", "pistol": "مسدس", "rifle": "بندقية",
            "knife": "سكين", "machete": "ساطور", "axe": "فأس",
            "grenade": "قنبلة يدوية", "taser": "صاعق",
            "mask": "قناع", "balaclava": "قناع رأس",
            "cell phone": "هاتف", "laptop": "حاسوب",
            "camera": "كاميرا", "dslr": "كاميرا احترافية",
            "drone": "طائرة مسيّرة", "quadcopter": "طائرة رباعية",
            "backpack": "حقيبة ظهر", "handbag": "حقيبة يد",
        }.get(label, label)

    def _tr_reason(self, reason: str) -> str:
        if self.language != "ar":
            return reason
        mapping = {
            "dwell": "مكوث",
            "loitering": "تسكع",
            "running": "جري",
            "crouching": "انخفاض",
            "face covered": "الوجه مغطى",
            "repeated photos": "تصوير متكرر",
            "recording device": "جهاز تسجيل",
            "FIREARM": "سلاح ناري",
            "WEAPON": "سلاح",
            "EXPLOSIVE": "عبوة ناسفة",
        }
        parts = [p.strip() for p in reason.split("+")]
        out = []
        for p in parts:
            first = p.split()[0] if p.split() else p
            out.append(mapping.get(first, p))
        return " + ".join(out)

    def _t(self, ar, en):
        return ar if self.language == "ar" else en


# =====================================================================
# PART 72 — Zone Renderer
# =====================================================================
class ZoneRenderer:
    """رسم المناطق المحظورة"""

    def __init__(self, font_path: str = "arial.ttf",
                 language: str = "ar"):
        self.font_path = font_path
        self.language = language

    def render(self, frame: np.ndarray, zone: Any,
               alarm: bool = False) -> None:
        H, W = frame.shape[:2]
        # الإحداثيات
        inner = zone.inner
        x1 = int(W * inner[0])
        y1 = int(H * inner[1])
        x2 = int(W * inner[2])
        y2 = int(H * inner[3])

        color = COLOR_ALARM if alarm else COLOR_ZONE

        # تعبئة خفيفة
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

        # الحدود
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # زوايا
        Draw.corner_brackets(frame, margin=15, size=25, color=color)

        # الملصق أسفل
        label = "منطقة محظورة" if self.language == "ar" else "RESTRICTED ZONE"
        lbl_y = y2 - 34
        tw, th = ArabicRenderer.measure(label, 22, self.font_path, 2)
        cv2.rectangle(frame, (x1 + 8, lbl_y - 4),
                      (x1 + tw + 16, lbl_y + th + 8), (10, 10, 14), -1)
        ArabicRenderer.draw(frame, label, (x1 + 12, lbl_y),
                            color, 22, self.font_path, thickness=2)


# =====================================================================
# PART 73 — Digital Twin Panel
# =====================================================================
class DigitalTwin:
    """لوحة التوأم الرقمي"""

    def __init__(self, size: int = 400, map_path: str = "building_map.jpg",
                 font_path: str = "arial.ttf", language: str = "ar"):
        self.size = size
        self.font_path = font_path
        self.language = language
        self.bg = self._load_or_generate_map(map_path)
        self._blink_phase = 0

    def _load_or_generate_map(self, path: str) -> np.ndarray:
        import os
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.resize(img, (self.size, self.size),
                                  interpolation=cv2.INTER_AREA)
        return self._generate_blueprint()

    def _generate_blueprint(self) -> np.ndarray:
        size = self.size
        canvas = np.full((size, size, 3), (16, 14, 20), dtype=np.uint8)
        # شبكة
        for g in range(0, size, 20):
            cv2.line(canvas, (g, 0), (g, size), (30, 28, 38), 1)
            cv2.line(canvas, (0, g), (size, g), (30, 28, 38), 1)
        for g in range(0, size, 100):
            cv2.line(canvas, (g, 0), (g, size), (52, 48, 68), 1)
            cv2.line(canvas, (0, g), (size, g), (52, 48, 68), 1)
        # غرف
        rooms = [
            (30, 30, 150, 150),
            (170, 30, 370, 110),
            (30, 170, 130, 370),
            (150, 130, 370, 250),
            (250, 270, 370, 370),
        ]
        for i, (x1, y1, x2, y2) in enumerate(rooms):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 200, 220), 1)
            tag = f"R{i+1}" if self.language != "ar" else f"غ{i+1}"
            cv2.rectangle(canvas, (x1, y1), (x1 + 24, y1 + 16),
                          (70, 200, 220), -1)
            cv2.putText(canvas, tag, (x1 + 4, y1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)
        # زوايا
        for (cx, cy, dx, dy) in [(6, 6, 1, 1), (size - 6, 6, -1, 1),
                                 (6, size - 6, 1, -1),
                                 (size - 6, size - 6, -1, -1)]:
            cv2.line(canvas, (cx, cy), (cx + dx * 22, cy), (140, 240, 255), 2)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * 22), (140, 240, 255), 2)
        return canvas

    def render(self, tracks: List[Any], cam_w: int, cam_h: int,
               freeze: bool = False) -> np.ndarray:
        canvas = self.bg.copy()
        H, W = canvas.shape[:2]
        sx = W / float(cam_w)
        sy = H / float(cam_h)

        # خط المسح
        scan_y = int((time.time() * 60) % H)
        cv2.line(canvas, (0, scan_y), (W, scan_y), (80, 220, 240), 1)

        # وميض
        self._blink_phase = (self._blink_phase + 1) % 20
        blink_on = self._blink_phase < 10

        # رسم التتبعات
        for track in tracks:
            if hasattr(track, "state") and hasattr(track.state, "value"):
                if track.state.value == "lost":
                    continue
            cx = (track.bbox[0] + track.bbox[2]) / 2
            cy = (track.bbox[1] + track.bbox[3]) / 2
            mx = int(cx * sx)
            my = int(cy * sy)
            mx = max(0, min(W - 1, mx))
            my = max(0, min(H - 1, my))

            status = getattr(track, "status", "normal")
            if hasattr(status, "value"):
                status = status.value
            color = {
                "normal": COLOR_OK,
                "warning": COLOR_WARN,
                "alarm": COLOR_ALARM,
            }.get(status, COLOR_OK)

            # توهج
            radius = 22 if status == "alarm" else 14
            glow = canvas.copy()
            cv2.circle(glow, (mx, my), radius, color, -1)
            cv2.addWeighted(glow, 0.22, canvas, 0.78, 0, canvas)

            # نقطة
            dot_color = color
            if status == "alarm" and not blink_on:
                dot_color = tuple(int(c * 0.35) for c in color)
            cv2.circle(canvas, (mx, my), 5, dot_color, -1)
            cv2.circle(canvas, (mx, my), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # ملصق
            tid = getattr(track, "id", 0)
            lbl = f"ID{tid}"
            score = getattr(track, "suspicion", 0)
            if score > 0:
                lbl += f" {int(score)}%"
            cv2.putText(canvas, lbl, (mx + 10, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # شريط علوي
        cv2.rectangle(canvas, (0, 0), (W, 22), (10, 10, 14), -1)
        title = ("فالكون آي :: التوأم الرقمي" if self.language == "ar"
                 else "FALCONEYE :: DIGITAL TWIN")
        if freeze:
            title += "  [غير مستقر]" if self.language == "ar" else " [UNSTABLE]"
        cv2.putText(canvas, title, (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 240, 255), 1)

        # شريط سفلي
        cv2.rectangle(canvas, (0, H - 24), (W, H), (10, 10, 14), -1)
        legend = [
            (COLOR_OK, "طبيعي" if self.language == "ar" else "Normal"),
            (COLOR_WARN, "تحذير" if self.language == "ar" else "Warning"),
            (COLOR_ALARM, "إنذار" if self.language == "ar" else "Alarm"),
        ]
        x = 14
        for col, name in legend:
            cv2.circle(canvas, (x, H - 12), 5, col, -1)
            cv2.putText(canvas, name, (x + 10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            x += 80

        # عدد التتبعات
        trk_lbl = f"{'تتبعات' if self.language == 'ar' else 'Tracks'}:{len(tracks):02d}"
        cv2.putText(canvas, trk_lbl, (W - 100, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return canvas


# =====================================================================
# PART 74 — Statistics Panel
# =====================================================================
class StatsPanel:
    """لوحة إحصائيات جانبية"""

    def __init__(self, font_path: str = "arial.ttf",
                 language: str = "ar"):
        self.font_path = font_path
        self.language = language

    def render(self, frame: np.ndarray, stats: Dict[str, Any],
               position: Tuple[int, int] = None) -> None:
        H, W = frame.shape[:2]
        panel_w = 260
        panel_h = 30 + len(stats) * 22
        if position is None:
            x0 = W - panel_w - 10
            y0 = HUD_SAFE_TOP + 10
        else:
            x0, y0 = position

        # خلفية
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0),
                      (x0 + panel_w, y0 + panel_h),
                      COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # حدود
        cv2.rectangle(frame, (x0, y0),
                      (x0 + panel_w, y0 + panel_h),
                      COLOR_TERRACOTTA, 2)

        # عنوان
        title = "الإحصائيات" if self.language == "ar" else "STATISTICS"
        ArabicRenderer.draw(frame, title, (x0 + 10, y0 + 6),
                            COLOR_TERRACOTTA, 16, self.font_path, thickness=2)

        # القيم
        y = y0 + 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            color = COLOR_TEXT
            ArabicRenderer.draw(frame, text, (x0 + 12, y),
                                color, 14, self.font_path)
            y += 22


# =====================================================================
# PART 75 — Track Trail Visualizer
# =====================================================================
class TrailVisualizer:
    """رسم مسارات التتبعات (Trails)"""

    def __init__(self, max_points: int = 30, fade: bool = True):
        self.max_points = max_points
        self.fade = fade

    def render(self, frame: np.ndarray, track: Any,
               color: Tuple[int, int, int]) -> None:
        positions = getattr(track, "positions", None)
        if not positions or len(positions) < 2:
            return
        points = list(positions)[-self.max_points:]
        for i in range(1, len(points)):
            alpha = i / len(points) if self.fade else 1.0
            c = tuple(int(c * alpha) for c in color)
            p1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            p2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, p1, p2, c, 2, cv2.LINE_AA)


# =====================================================================
# PART 76 — Confidence Bar
# =====================================================================
class ConfidenceBar:
    """شريط الثقة/الاحتمال"""

    @staticmethod
    def draw(frame, org: Tuple[int, int], width: int, height: int,
             value: float, color: Tuple[int, int, int] = COLOR_OK,
             bg_color: Tuple[int, int, int] = (40, 40, 40)) -> None:
        x, y = org
        value = max(0.0, min(1.0, value))
        # خلفية
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
        # قيمة
        fill_w = int(width * value)
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
        # حدود
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                      (100, 100, 100), 1)

    @staticmethod
    def draw_with_label(frame, org, width, height, value, label,
                        font_path="arial.ttf", color=COLOR_OK):
        ConfidenceBar.draw(frame, org, width, height, value, color)
        # النص أعلى الشريط
        text = f"{label}: {int(value * 100)}%"
        ArabicRenderer.draw(frame, text,
                            (org[0], org[1] - 20), (200, 200, 200),
                            12, font_path)


# =====================================================================
# PART 77 — Arrow Indicator
# =====================================================================
class ArrowIndicator:
    """سهم يُشير إلى اتجاه التتبع"""

    @staticmethod
    def render(frame, track, color=COLOR_OK, length: int = 40) -> None:
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2
        vx, vy = getattr(track, "motion_dir", (0.0, 0.0))
        mag = math.hypot(vx, vy)
        if mag < 20:
            return
        ux = vx / mag
        uy = vy / mag
        p1 = (int(cx), int(cy))
        p2 = (int(cx + ux * length), int(cy + uy * length))
        cv2.arrowedLine(frame, p1, p2, color, 2, cv2.LINE_AA,
                        tipLength=0.3)


# =====================================================================
# PART 78 — Heatmap Overlay
# =====================================================================
class HeatmapOverlay:
    """طبقة خريطة حرارية فوق الإطار"""

    def __init__(self, alpha: float = 0.45, blur: int = 31):
        self.alpha = alpha
        self.blur = blur

    def render(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        if heatmap is None or heatmap.max() < 1:
            return frame
        norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        if self.blur > 0:
            norm = cv2.GaussianBlur(norm, (self.blur, self.blur), 0)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - self.alpha, colored, self.alpha, 0)


# =====================================================================
# PART 79 — Gesture Visualization
# =====================================================================
class GestureVisualizer:
    """رسم نقاط المفاصل والإيماءات"""

    # اتصالات الهيكل
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    @staticmethod
    def render(frame: np.ndarray, keypoints: List[List[float]],
               color: Tuple[int, int, int] = COLOR_CYAN,
               min_conf: float = 0.3) -> None:
        if not keypoints:
            return
        # النقاط
        for kp in keypoints:
            x, y, c = kp
            if c < min_conf:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

        # الهيكل
        for i, j in GestureVisualizer.SKELETON:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            kp_i = keypoints[i]
            kp_j = keypoints[j]
            if kp_i[2] < min_conf or kp_j[2] < min_conf:
                continue
            cv2.line(frame, (int(kp_i[0]), int(kp_i[1])),
                     (int(kp_j[0]), int(kp_j[1])),
                     color, 1, cv2.LINE_AA)


# =====================================================================
# PART 80 — Scene Composition
# =====================================================================
class SceneCompositor:
    """تركيب المشهد الكامل من الطبقات"""

    def __init__(
        self,
        hud: HUD,
        track_renderer: TrackLabelRenderer,
        zone_renderer: ZoneRenderer,
        banner: AlarmBanner,
        twin: Optional[DigitalTwin] = None,
        font_path: str = "arial.ttf",
    ):
        self.hud = hud
        self.track_renderer = track_renderer
        self.zone_renderer = zone_renderer
        self.banner = banner
        self.twin = twin
        self.font_path = font_path
        self.trail = TrailVisualizer()
        self.heatmap_layer = HeatmapOverlay()

    def compose(
        self,
        frame: np.ndarray,
        tracks: List[Any],
        zones: List[Any],
        stats: HUDStats,
        any_alarm: bool = False,
        heatmap: Optional[np.ndarray] = None,
        show_heatmap: bool = False,
    ) -> np.ndarray:
        """تركيب كل الطبقات"""
        # 1) خريطة حرارية
        if show_heatmap and heatmap is not None:
            frame = self.heatmap_layer.render(frame, heatmap)

        # 2) المناطق
        for zone in zones:
            self.zone_renderer.render(frame, zone, alarm=any_alarm)

        # 3) التتبعات
        for track in tracks:
            if hasattr(track, "state") and hasattr(track.state, "value"):
                if track.state.value != "confirmed":
                    continue
            # الملصق والصندوق
            self.track_renderer.render(frame, track)
            # سهم الاتجاه
            ArrowIndicator.render(frame, track)

        # 4) زوايا الشاشة
        Draw.corner_brackets(
            frame, margin=8, size=28,
            color=COLOR_ALARM if any_alarm else COLOR_CYAN,
            thickness=2
        )

        # 5) HUD
        self.hud.render(frame, stats)

        # 6) شريط الإنذار
        if any_alarm:
            self.banner.render(frame, True,
                               language=self.hud.language)

        return frame


# =====================================================================
# PART 81 — Twin Viewer
# =====================================================================
class TwinViewer:
    """عارض التوأم الرقمي"""

    def __init__(self, twin: DigitalTwin, window_name: str = "Twin"):
        self.twin = twin
        self.window_name = window_name

    def render(self, tracks: List[Any], cam_w: int, cam_h: int,
               freeze: bool = False) -> np.ndarray:
        return self.twin.render(tracks, cam_w, cam_h, freeze)


# =====================================================================
# PART 82 — UI Factory
# =====================================================================
class UIFactory:
    """مصنع مكونات الواجهة"""

    @staticmethod
    def build_default(
        font_path: str = "arial.ttf",
        language: str = "ar",
        twin_size: int = 400,
        twin_map_path: str = "building_map.jpg",
    ) -> Dict[str, Any]:
        ArabicRenderer.initialize()

        hud = HUD(font_path, language)
        track_renderer = TrackLabelRenderer(font_path, language)
        zone_renderer = ZoneRenderer(font_path, language)
        banner = AlarmBanner(font_path)
        twin = DigitalTwin(twin_size, twin_map_path, font_path, language)
        compositor = SceneCompositor(
            hud, track_renderer, zone_renderer, banner, twin, font_path
        )
        stats_panel = StatsPanel(font_path, language)
        trail = TrailVisualizer()
        confidence = ConfidenceBar()
        arrows = ArrowIndicator()
        heatmap = HeatmapOverlay()
        gesture = GestureVisualizer()

        return {
            "hud": hud,
            "track_renderer": track_renderer,
            "zone_renderer": zone_renderer,
            "banner": banner,
            "twin": twin,
            "compositor": compositor,
            "stats_panel": stats_panel,
            "trail": trail,
            "confidence": confidence,
            "arrows": arrows,
            "heatmap": heatmap,
            "gesture": gesture,
        }


# =====================================================================
# PART 83 — UI Theme Manager
# =====================================================================
class UITheme:
    """مدير الألوان والثيمات"""

    LIGHT = {
        "bg": (245, 245, 245),
        "text": (20, 20, 20),
        "accent": (200, 100, 80),
        "border": (180, 180, 180),
    }

    DARK = {
        "bg": (20, 20, 25),
        "text": (240, 240, 240),
        "accent": COLOR_TERRACOTTA,
        "border": (80, 80, 80),
    }

    TERRACOTTA = {
        "bg": (238, 240, 245),
        "text": (30, 30, 30),
        "accent": COLOR_TERRACOTTA,
        "border": COLOR_TERRACOTTA,
    }

    @classmethod
    def get(cls, name: str = "dark") -> Dict[str, Tuple[int, int, int]]:
        return {
            "light": cls.LIGHT,
            "dark": cls.DARK,
            "terracotta": cls.TERRACOTTA,
        }.get(name, cls.DARK)


# =====================================================================
# PART 84 — UI Exports
# =====================================================================
__all__ = [
    # Constants
    "COLOR_OK", "COLOR_WARN", "COLOR_ALARM", "COLOR_ZONE",
    "COLOR_TEXT", "COLOR_DARK", "COLOR_WEAPON", "COLOR_EXPLOSIVE",
    "COLOR_PHOTO", "COLOR_CYAN", "COLOR_TERRACOTTA",
    "HUD_BAR_HEIGHT", "HUD_SAFE_TOP",
    # Renderers
    "ArabicRenderer", "FrameText", "Draw",
    "HUD", "HUDStats", "AlarmBanner",
    "TrackLabelRenderer", "ZoneRenderer",
    "DigitalTwin", "TwinViewer", "StatsPanel",
    "TrailVisualizer", "ConfidenceBar", "ArrowIndicator",
    "HeatmapOverlay", "GestureVisualizer",
    # Composition
    "SceneCompositor", "UIFactory", "UITheme",
]
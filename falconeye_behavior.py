# =====================================================================
# PART 43 — Behavioral Analysis Constants
# =====================================================================
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum, IntEnum
from collections import deque, defaultdict
import time
import math
import threading
import logging
import numpy as np

logger = logging.getLogger("falconeye")


# فئات الأسلحة
FIREARM_CLASSES = {
    "gun", "pistol", "rifle", "handgun", "revolver",
    "shotgun", "submachine gun", "assault rifle",
}

MELEE_CLASSES = {
    "knife", "machete", "axe", "hammer", "baseball bat",
    "sword", "dagger", "box cutter", "crowbar",
}

EXPLOSIVE_CLASSES = {
    "grenade", "bomb", "explosive", "dynamite",
}

CONCEALMENT_CLASSES = {
    "mask", "balaclava", "ski mask",
}

RECORDER_CLASSES = {
    "cell phone", "smartphone", "camera", "dslr",
    "video camera", "drone", "quadcopter", "tablet",
    "gopro", "webcam", "laptop",
}

RESTRAINT_CLASSES = {"handcuffs", "rope", "zip ties"}
SUBSTANCE_CLASSES = {"syringe"}
BURGLARY_CLASSES = {"crowbar", "screwdriver", "lockpick"}

WEAPON_CLASSES = FIREARM_CLASSES | MELEE_CLASSES | EXPLOSIVE_CLASSES


# =====================================================================
# PART 44 — Behavioral Event Types
# =====================================================================
class BehaviorEvent(str, Enum):
    """أنواع الأحداث السلوكية"""
    ENTER_ZONE = "enter_zone"
    EXIT_ZONE = "exit_zone"
    DWELL_WARN = "dwell_warn"
    DWELL_ALARM = "dwell_alarm"
    LOITERING = "loitering"
    RUNNING = "running"
    CROUCHING = "crouching"
    FACE_COVERED = "face_covered"
    WEAPON_BEARING = "weapon_bearing"
    FIREARM_BEARING = "firearm_bearing"
    EXPLOSIVE_BEARING = "explosive_bearing"
    PHOTO_GESTURE = "photo_gesture"
    RECORDING_DEVICE = "recording_device"
    REPEATED_PHOTOS = "repeated_photos"
    RESTRAINT_DETECTED = "restraint"
    SUBSTANCE_DETECTED = "substance"
    BURGLARY_TOOL = "burglary_tool"
    IDENTIFIED_PERSON = "identified_person"
    UNKNOWN_PERSON = "unknown_person"
    ABANDONED_OBJECT = "abandoned_object"
    CROWD_FORMING = "crowd_forming"
    ABNORMAL_SPEED = "abnormal_speed"


class BehaviorSeverity(IntEnum):
    """شدة الحدث"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    FATAL = 5


# =====================================================================
# PART 45 — Behavior Event Dataclass
# =====================================================================
@dataclass
class BehaviorEventRecord:
    """سجل حدث سلوكي واحد"""
    event_type: BehaviorEvent
    severity: BehaviorSeverity
    track_id: int
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "severity": int(self.severity),
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "metadata": self.metadata,
        }


# =====================================================================
# PART 46 — Temporal Voting
# =====================================================================
class TemporalVoter:
    """تصويت زمني لكل قاعدة سلوك"""

    def __init__(self, window: int = 5, threshold: int = 3):
        self.window = window
        self.threshold = threshold
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.RLock()

    def vote(self, key: str, value: bool) -> bool:
        with self._lock:
            self._buffers[key].append(bool(value))
            return sum(self._buffers[key]) >= self.threshold

    def get_state(self, key: str) -> bool:
        with self._lock:
            return sum(self._buffers[key]) >= self.threshold

    def get_confidence(self, key: str) -> float:
        with self._lock:
            buf = self._buffers[key]
            if not buf:
                return 0.0
            return sum(buf) / len(buf)

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key:
                self._buffers.pop(key, None)
            else:
                self._buffers.clear()


# =====================================================================
# PART 47 — Zone Tracker
# =====================================================================
class ZoneTracker:
    """يتتبع دخول/خروج/مكوث كل تتبع في المناطق"""

    def __init__(self):
        self._inside: Dict[Tuple[int, str], float] = {}
        self._lock = threading.RLock()

    def update(self, track_id: int, zone_name: str,
               inside: bool) -> Optional[Tuple[str, float]]:
        """
        إرجاع (نوع الحدث, مدة المكوث) عند التغيير
        """
        key = (track_id, zone_name)
        now = time.time()
        with self._lock:
            was_inside = key in self._inside
            if inside and not was_inside:
                self._inside[key] = now
                return ("enter", 0.0)
            if not inside and was_inside:
                duration = now - self._inside.pop(key)
                return ("exit", duration)
        return None

    def get_dwell(self, track_id: int, zone_name: str) -> float:
        with self._lock:
            started = self._inside.get((track_id, zone_name))
            if started is None:
                return 0.0
            return time.time() - started

    def cleanup(self, active_ids: Set[int]) -> None:
        with self._lock:
            to_remove = [
                k for k in self._inside if k[0] not in active_ids
            ]
            for k in to_remove:
                del self._inside[k]


# =====================================================================
# PART 48 — Photo Gesture Detector
# =====================================================================
class PhotoGestureDetector:
    """كشف إيماءة التصوير السري"""

    # فهارس نقاط المفاصل
    KP_NOSE = 0
    KP_LWRIST = 9
    KP_RWRIST = 10
    KP_LEYE = 1
    KP_REYE = 2

    def __init__(
        self,
        phone_nose_dist_frac: float = 0.25,
        upper_body_frac: float = 0.45,
        min_keypoint_conf: float = 0.30,
        containment_min: float = 0.25,
        max_speed: float = 110.0,
    ):
        self.phone_nose_dist_frac = phone_nose_dist_frac
        self.upper_body_frac = upper_body_frac
        self.min_kp_conf = min_keypoint_conf
        self.containment_min = containment_min
        self.max_speed = max_speed

    def is_photo_gesture(
        self,
        person_box: Tuple[int, int, int, int],
        keypoints: Optional[List[List[float]]],
        phone_box: Tuple[int, int, int, int],
        person_speed: float = 0.0,
    ) -> bool:
        """فحص شامل لإيماءة التصوير"""
        if keypoints is None or phone_box is None:
            return False

        # 1) فحص السرعة
        if person_speed > self.max_speed:
            return False

        # 2) فحص الاحتواء
        if not self._check_containment(phone_box, person_box):
            return False

        # 3) فحص موضع الهاتف
        if not self._check_phone_position(phone_box, person_box):
            return False

        # 4) فحص قرب المعصم من الوجه
        if not self._check_wrist_proximity(keypoints, person_box):
            return False

        return True

    def _check_containment(
        self, phone_box: Tuple[int, int, int, int],
        person_box: Tuple[int, int, int, int],
    ) -> bool:
        x1 = max(phone_box[0], person_box[0])
        y1 = max(phone_box[1], person_box[1])
        x2 = min(phone_box[2], person_box[2])
        y2 = min(phone_box[3], person_box[3])
        if x2 <= x1 or y2 <= y1:
            return False
        inter = (x2 - x1) * (y2 - y1)
        phone_area = (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1])
        return inter / max(1, phone_area) >= self.containment_min

    def _check_phone_position(
        self, phone_box: Tuple[int, int, int, int],
        person_box: Tuple[int, int, int, int],
    ) -> bool:
        person_h = person_box[3] - person_box[1]
        if person_h <= 0:
            return False
        phone_cy = (phone_box[1] + phone_box[3]) / 2
        upper_limit = person_box[1] + self.upper_body_frac * person_h
        return phone_cy <= upper_limit

    def _check_wrist_proximity(
        self, keypoints: List[List[float]],
        person_box: Tuple[int, int, int, int],
    ) -> bool:
        person_h = person_box[3] - person_box[1]
        if person_h <= 0:
            return False

        # نقطة الأنف
        if len(keypoints) <= self.KP_NOSE:
            return False
        nose = keypoints[self.KP_NOSE]
        if nose[2] < self.min_kp_conf:
            return False
        nose_xy = (nose[0], nose[1])

        # فحص كل معصم
        for kp_idx in (self.KP_LWRIST, self.KP_RWRIST):
            if len(keypoints) <= kp_idx:
                continue
            wrist = keypoints[kp_idx]
            if wrist[2] < self.min_kp_conf:
                continue
            dist = math.hypot(wrist[0] - nose_xy[0], wrist[1] - nose_xy[1])
            if dist <= self.phone_nose_dist_frac * person_h:
                return True
        return False


# =====================================================================
# PART 49 — Weapon Verifier
# =====================================================================
class WeaponVerifier:
    """تحقّق زمني من الأسلحة لمنع الإنذارات الكاذبة"""

    def __init__(
        self,
        window: int = 8,
        needed: int = 4,
        min_conf: float = 0.25,
        static_timeout: float = 8.0,
        close_dist_frac: float = 0.90,
        min_containment: float = 0.30,
    ):
        self.window = window
        self.needed = needed
        self.min_conf = min_conf
        self.static_timeout = static_timeout
        self.close_dist_frac = close_dist_frac
        self.min_containment = min_containment

        self._votes: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.RLock()

    def record(self, track_id: int, conf: float) -> None:
        with self._lock:
            self._votes[track_id].append(conf >= self.min_conf)

    def verify(self, track_id: int, current_conf: float) -> bool:
        with self._lock:
            votes = self._votes.get(track_id)
            if votes is None:
                return False
            return sum(votes) >= self.needed

    def temporal_score(self, track_id: int) -> float:
        with self._lock:
            votes = self._votes.get(track_id)
            if not votes:
                return 0.0
            return sum(votes) / len(votes)

    def check_owner(
        self,
        weapon_box: Tuple[int, int, int, int],
        weapon_speed: float,
        persons: List[Dict[str, Any]],
    ) -> Optional[int]:
        """البحث عن مالك السلاح"""
        # 1) الاحتواء
        best_pid = None
        best_cont = 0.0
        for p in persons:
            p_box = p.get("box")
            if p_box is None:
                continue
            cont = box_containment(weapon_box, p_box)
            if cont > best_cont:
                best_cont = cont
                best_pid = p.get("id")

        if best_pid is not None and best_cont >= self.min_containment:
            return best_pid

        # 2) القرب مع الحركة
        if best_pid is not None:
            p = next((x for x in persons if x.get("id") == best_pid), None)
            if p:
                ph = p["box"][3] - p["box"][1]
                wx = (weapon_box[0] + weapon_box[2]) / 2
                wy = (weapon_box[1] + weapon_box[3]) / 2
                px = (p["box"][0] + p["box"][2]) / 2
                py = (p["box"][1] + p["box"][3]) / 2
                dist = math.hypot(wx - px, wy - py)
                if dist < self.close_dist_frac * ph and weapon_speed > 15:
                    return best_pid

        return None

    def cleanup(self, active_ids: Set[int]) -> None:
        with self._lock:
            for k in list(self._votes.keys()):
                if k not in active_ids:
                    del self._votes[k]


# =====================================================================
# PART 50 — Threat Score Calculator
# =====================================================================
class ThreatScoreCalculator:
    """حساب درجة التهديد لكل تتبع"""

    # أوزان كل سلوك
    WEIGHTS = {
        "firearm": 100.0,
        "explosive": 100.0,
        "melee": 85.0,
        "weapon_bearing": 90.0,
        "concealment": 35.0,
        "photo_repeated": 90.0,
        "recording_in_zone": 60.0,
        "restraint": 45.0,
        "substance": 30.0,
        "burglary_tool": 55.0,
        "loitering": 25.0,
        "running_in_zone": 20.0,
        "crouching_in_zone": 30.0,
        "dwell_short": 15.0,
        "dwell_medium": 25.0,
        "dwell_long": 40.0,
        "identified_known": -30.0,   # يخفض الخطر
        "unknown_person": 10.0,
    }

    def __init__(self, score_warn: float = 35.0, score_alarm: float = 70.0,
                 hard_lock_threshold: float = 100.0):
        self.score_warn = score_warn
        self.score_alarm = score_alarm
        self.hard_lock_threshold = hard_lock_threshold

    def compute(
        self,
        soft_signals: Dict[str, bool],
        hard_signals: Dict[str, bool],
        dwell_secs: float = 0.0,
        identified_name: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """إرجاع (الدرجة, الأسباب)"""
        score = 0.0
        reasons: List[str] = []

        # 1) الإشارات الصلبة
        if hard_signals.get("explosive"):
            score = max(score, 100.0)
            reasons.append("EXPLOSIVE")
        elif hard_signals.get("firearm"):
            score = max(score, 100.0)
            reasons.append("FIREARM")
        elif hard_signals.get("melee"):
            score = max(score, 85.0)
            reasons.append("WEAPON")

        # 2) الإشارات الناعمة
        for key, active in soft_signals.items():
            if not active:
                continue
            weight = self.WEIGHTS.get(key, 0.0)
            if weight == 0:
                continue
            score += weight
            reasons.append(key.upper())

        # 3) المكوث
        if dwell_secs > 0:
            if dwell_secs >= 6.0:
                score += self.WEIGHTS["dwell_long"]
                reasons.append(f"DWELL {dwell_secs:.1f}s")
            elif dwell_secs >= 3.0:
                score += self.WEIGHTS["dwell_medium"]
                reasons.append(f"DWELL {dwell_secs:.1f}s")
            elif dwell_secs >= 1.5:
                score += self.WEIGHTS["dwell_short"]

        # 4) الهوية
        if identified_name:
            score += self.WEIGHTS["identified_known"]
            reasons.append(f"IDENTIFIED:{identified_name}")

        # clamp
        score = max(0.0, min(100.0, score))
        return score, reasons

    def classify(self, score: float) -> str:
        if score >= self.score_alarm:
            return "alarm"
        if score >= self.score_warn:
            return "warning"
        return "normal"


# =====================================================================
# PART 51 — Behavior Analyzer (Core)
# =====================================================================
class BehaviorAnalyzer:
    """
    المُحلّل السلوكي الرئيسي
    يستقبل التتبعات الحالية + الكشوفات ويصدر أحداثاً سلوكية
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.warn_dwell = cfg.get("warn_dwell", 2.5)
        self.alarm_dwell = cfg.get("alarm_dwell", 4.0)
        self.loiter_radius = cfg.get("loiter_radius", 60.0)
        self.loiter_secs = cfg.get("loiter_secs", 6.0)
        self.run_speed = cfg.get("run_speed", 260.0)
        self.crouch_ratio = cfg.get("crouch_ratio", 1.20)
        self.photo_events_to_alarm = cfg.get("photo_events_to_alarm", 3)
        self.photo_event_window = cfg.get("photo_event_window", 30.0)
        self.alarm_hold_secs = cfg.get("alarm_hold_secs", 5.0)

        # المكونات
        self.zone_tracker = ZoneTracker()
        self.photo_detector = PhotoGestureDetector()
        self.weapon_verifier = WeaponVerifier()
        self.score_calc = ThreatScoreCalculator(
            score_warn=cfg.get("score_warn", 35.0),
            score_alarm=cfg.get("score_alarm", 70.0),
        )

        # حالة التتبع
        self._voters: Dict[int, TemporalVoter] = {}
        self._alarm_hold: Dict[int, float] = {}
        self._last_reasons: Dict[int, List[str]] = {}

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def analyze(
        self,
        tracks: List[Any],
        detections: List[Any],
        zones: List[Any],
        frame_shape: Tuple[int, ...],
    ) -> List[BehaviorEventRecord]:
        """تحليل جميع التتبعات وإرجاع الأحداث"""
        events: List[BehaviorEventRecord] = []

        # فهارس الكشوفات
        recorders = [d for d in detections if d.label in RECORDER_CLASSES]
        weapons = [d for d in detections if d.label in WEAPON_CLASSES]
        persons = [d for d in detections if d.label == "person"]
        conceal = [d for d in detections if d.label in CONCEALMENT_CLASSES]
        restraints = [d for d in detections if d.label in RESTRAINT_CLASSES]
        substances = [d for d in detections if d.label in SUBSTANCE_CLASSES]
        burglary = [d for d in detections if d.label in BURGLARY_CLASSES]

        # تحديث قوائم الأسلحة
        weapon_by_owner: Dict[int, List[str]] = defaultdict(list)
        for w in weapons:
            owner_id = self._match_weapon_to_person(w, persons)
            if owner_id is not None:
                weapon_by_owner[owner_id].append(w.label)
                self.weapon_verifier.record(owner_id, w.confidence)

        active_ids = set()

        for track in tracks:
            if not hasattr(track, "state"):
                continue
            if track.state.value != "confirmed":
                continue
            tid = track.id
            active_ids.add(tid)

            # هل هو في حالة hold؟
            if self._is_in_hold(tid):
                continue

            # الحصول على voter
            voter = self._voters.setdefault(tid, TemporalVoter())

            # 1) فحص المناطق
            zone_events = self._check_zones(track, zones, frame_shape)
            events.extend(zone_events)

            # 2) فحص السلوكيات إن كان شخصاً
            if track.is_person:
                events.extend(
                    self._check_person_behaviors(
                        track, voter, frame_shape,
                        persons, recorders, weapons,
                        conceal, restraints, substances, burglary,
                        weapon_by_owner,
                    )
                )

            # 3) فحص الأسلحة المملوكة
            if tid in weapon_by_owner:
                events.extend(
                    self._check_weapon_ownership(track, weapon_by_owner[tid])
                )

        # تنظيف
        self.zone_tracker.cleanup(active_ids)
        self.weapon_verifier.cleanup(active_ids)
        for tid in list(self._voters.keys()):
            if tid not in active_ids:
                del self._voters[tid]

        return events

    # ------------------------------------------------------------------
    # Zone checks
    # ------------------------------------------------------------------
    def _check_zones(
        self,
        track: Any,
        zones: List[Any],
        frame_shape: Tuple[int, ...],
    ) -> List[BehaviorEventRecord]:
        events: List[BehaviorEventRecord] = []
        cx, cy = _track_center(track)
        for zone in zones:
            if not zone.enabled:
                continue
            inside = zone.contains((cx, cy), frame_shape)
            result = self.zone_tracker.update(track.id, zone.name, inside)
            if result is None:
                continue
            kind, duration = result
            if kind == "enter":
                events.append(BehaviorEventRecord(
                    event_type=BehaviorEvent.ENTER_ZONE,
                    severity=BehaviorSeverity.INFO,
                    track_id=track.id,
                    reason=f"دخل المنطقة {zone.name}",
                    metadata={"zone": zone.name},
                ))
            elif kind == "exit":
                events.append(BehaviorEventRecord(
                    event_type=BehaviorEvent.EXIT_ZONE,
                    severity=BehaviorSeverity.INFO,
                    track_id=track.id,
                    reason=f"خرج من {zone.name} بعد {duration:.1f}s",
                    metadata={"zone": zone.name, "dwell": duration},
                ))
        return events

    # ------------------------------------------------------------------
    # Person behaviors
    # ------------------------------------------------------------------
    def _check_person_behaviors(
        self,
        track: Any,
        voter: TemporalVoter,
        frame_shape: Tuple[int, ...],
        persons: List[Any],
        recorders: List[Any],
        weapons: List[Any],
        conceal: List[Any],
        restraints: List[Any],
        substances: List[Any],
        burglary: List[Any],
        weapon_by_owner: Dict[int, List[str]],
    ) -> List[BehaviorEventRecord]:
        events: List[BehaviorEventRecord] = []

        # a) التسكع
        loiter = track.is_loitering(
            radius=self.loiter_radius, window_secs=self.loiter_secs
        )
        if voter.vote("loiter", loiter):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.LOITERING,
                severity=BehaviorSeverity.MEDIUM,
                track_id=track.id,
                reason="تسكع مريب",
            ))

        # b) الجري
        running = track.get_speed() > self.run_speed
        if voter.vote("running", running):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.RUNNING,
                severity=BehaviorSeverity.MEDIUM,
                track_id=track.id,
                reason=f"جري بسرعة {track.get_speed():.0f}px/s",
            ))

        # c) الانخفاض
        crouching = track.is_crouching(ratio_threshold=self.crouch_ratio)
        if voter.vote("crouch", crouching):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.CROUCHING,
                severity=BehaviorSeverity.MEDIUM,
                track_id=track.id,
                reason="انخفاض مريب",
            ))

        # d) تغطية الوجه
        face_covered = self._check_concealment(track, conceal)
        if voter.vote("conceal", face_covered):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.FACE_COVERED,
                severity=BehaviorSeverity.MEDIUM,
                track_id=track.id,
                reason="الوجه مغطى",
            ))

        # e) إيماءة التصوير
        photo = self._check_photo_gesture(track, recorders)
        if voter.vote("photo", photo):
            track.register_photo_event()
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.PHOTO_GESTURE,
                severity=BehaviorSeverity.HIGH,
                track_id=track.id,
                reason=f"إيماءة تصوير ({len(track.photo_events)})",
                metadata={"count": len(track.photo_events)},
            ))

        # f) التصوير المتكرر
        if self._is_repeated_photo(track):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.REPEATED_PHOTOS,
                severity=BehaviorSeverity.CRITICAL,
                track_id=track.id,
                reason=f"تصوير متكرر ({len(track.photo_events)})",
                metadata={"count": len(track.photo_events)},
            ))
            self._set_alarm_hold(track.id)

        # g) قرب الأدوات
        if any(_near_object(track, o) for o in restraints):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.RESTRAINT_DETECTED,
                severity=BehaviorSeverity.HIGH,
                track_id=track.id,
                reason="أداة تقييد قريبة",
            ))

        if any(_near_object(track, o) for o in substances):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.SUBSTANCE_DETECTED,
                severity=BehaviorSeverity.MEDIUM,
                track_id=track.id,
                reason="مادة مشبوهة",
            ))

        if any(_near_object(track, o) for o in burglary):
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.BURGLARY_TOOL,
                severity=BehaviorSeverity.HIGH,
                track_id=track.id,
                reason="أداة سرقة",
            ))

        # h) هوية الشخص
        identified = getattr(track, "metadata", {}).get("identified")
        if identified:
            events.append(BehaviorEventRecord(
                event_type=BehaviorEvent.IDENTIFIED_PERSON,
                severity=BehaviorSeverity.INFO,
                track_id=track.id,
                reason=f"تم التعرف: {identified}",
                metadata={"name": identified},
            ))

        # i) مالك السلاح
        weapon_labels = weapon_by_owner.get(track.id, [])
        if weapon_labels:
            for label in weapon_labels:
                if label in FIREARM_CLASSES:
                    evt = BehaviorEvent.FIREARM_BEARING
                    sev = BehaviorSeverity.FATAL
                elif label in EXPLOSIVE_CLASSES:
                    evt = BehaviorEvent.EXPLOSIVE_BEARING
                    sev = BehaviorSeverity.FATAL
                else:
                    evt = BehaviorEvent.WEAPON_BEARING
                    sev = BehaviorSeverity.CRITICAL
                events.append(BehaviorEventRecord(
                    event_type=evt,
                    severity=sev,
                    track_id=track.id,
                    reason=f"يحمل: {label}",
                    metadata={"weapon": label},
                ))
            self._set_alarm_hold(track.id)

        return events

    # ------------------------------------------------------------------
    # Weapon ownership
    # ------------------------------------------------------------------
    def _check_weapon_ownership(
        self, track: Any, weapons: List[str]
    ) -> List[BehaviorEventRecord]:
        events: List[BehaviorEventRecord] = []
        for label in weapons:
            if label in FIREARM_CLASSES:
                events.append(BehaviorEventRecord(
                    event_type=BehaviorEvent.FIREARM_BEARING,
                    severity=BehaviorSeverity.FATAL,
                    track_id=track.id,
                    reason=f"سلاح ناري: {label}",
                ))
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _match_weapon_to_person(
        self, weapon: Any, persons: List[Any]
    ) -> Optional[int]:
        if not persons:
            return None
        best_id = None
        best_cont = 0.0
        for p in persons:
            cont = box_containment(weapon.bbox, p.bbox)
            if cont > best_cont and cont > 0.3:
                best_cont = cont
                best_id = getattr(p, "id", None)
        return best_id

    def _check_concealment(
        self, track: Any, conceal: List[Any]
    ) -> bool:
        for c in conceal:
            if compute_iou(track.bbox, c.bbox) > 0.05:
                return True
        return False

    def _check_photo_gesture(
        self, track: Any, recorders: List[Any]
    ) -> bool:
        if not recorders:
            return False
        kps = getattr(track, "keypoints", None)
        if kps is None:
            return False
        for rec in recorders:
            if rec.confidence < 0.4:
                continue
            if self.photo_detector.is_photo_gesture(
                track.bbox, kps, rec.bbox, track.get_speed()
            ):
                return True
        return False

    def _is_repeated_photo(self, track: Any) -> bool:
        if len(track.photo_events) < self.photo_events_to_alarm:
            return False
        now = time.time()
        recent = [t for t in track.photo_events
                  if now - t < self.photo_event_window]
        return len(recent) >= self.photo_events_to_alarm

    def _is_in_hold(self, track_id: int) -> bool:
        with self._lock:
            until = self._alarm_hold.get(track_id, 0.0)
            return time.time() < until

    def _set_alarm_hold(self, track_id: int) -> None:
        with self._lock:
            self._alarm_hold[track_id] = time.time() + self.alarm_hold_secs

    def clear_hold(self, track_id: Optional[int] = None) -> None:
        with self._lock:
            if track_id is None:
                self._alarm_hold.clear()
            else:
                self._alarm_hold.pop(track_id, None)

    def reset(self) -> None:
        with self._lock:
            self._voters.clear()
            self._alarm_hold.clear()
            self._last_reasons.clear()
            self.zone_tracker._inside.clear()
            self.weapon_verifier._votes.clear()


# =====================================================================
# PART 52 — Helper functions
# =====================================================================
def _track_center(track: Any) -> Tuple[float, float]:
    box = track.bbox
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _near_object(track: Any, obj: Any,
                 contain_thresh: float = 0.35,
                 dist_frac: float = 0.6) -> bool:
    if box_containment(obj.bbox, track.bbox) > contain_thresh:
        return True
    ph = track.bbox[3] - track.bbox[1]
    if ph <= 0:
        return False
    tc = _track_center(track)
    oc = ((obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2)
    return math.hypot(tc[0] - oc[0], tc[1] - oc[1]) < dist_frac * ph


# =====================================================================
# PART 53 — Alarm Priority Queue
# =====================================================================
class AlarmPriorityQueue:
    """قائمة أولويات للإنذارات"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._items: List[BehaviorEventRecord] = []
        self._lock = threading.RLock()

    def push(self, event: BehaviorEventRecord) -> None:
        with self._lock:
            self._items.append(event)
            self._items.sort(
                key=lambda e: (e.severity, e.timestamp), reverse=True
            )
            if len(self._items) > self.max_size:
                self._items = self._items[:self.max_size]

    def pop(self) -> Optional[BehaviorEventRecord]:
        with self._lock:
            return self._items.pop(0) if self._items else None

    def peek(self, n: int = 10) -> List[BehaviorEventRecord]:
        with self._lock:
            return list(self._items[:n])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# =====================================================================
# PART 54 — Threat State Machine
# =====================================================================
class TrackThreatState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ThreatStateMachine:
    """آلة حالة لكل تتبع"""

    def __init__(
        self,
        warn_threshold: float = 35.0,
        alarm_threshold: float = 70.0,
        hold_secs: float = 5.0,
        decay_rate: float = 12.0,
    ):
        self.warn_threshold = warn_threshold
        self.alarm_threshold = alarm_threshold
        self.hold_secs = hold_secs
        self.decay_rate = decay_rate

        self._states: Dict[int, TrackThreatState] = {}
        self._scores: Dict[int, float] = {}
        self._hold_until: Dict[int, float] = {}
        self._last_update: Dict[int, float] = {}
        self._lock = threading.RLock()

    def update(self, track_id: int, score: float,
               force_alarm: bool = False) -> TrackThreatState:
        with self._lock:
            now = time.time()
            prev = self._states.get(track_id, TrackThreatState.NORMAL)
            prev_score = self._scores.get(track_id, 0.0)

            # hold
            if now < self._hold_until.get(track_id, 0.0):
                return prev

            # decay if decreasing
            if score < prev_score:
                dt = max(0.01, now - self._last_update.get(track_id, now))
                score = max(score, prev_score - self.decay_rate * dt)

            self._scores[track_id] = score
            self._last_update[track_id] = now

            if force_alarm or score >= self.alarm_threshold:
                new_state = TrackThreatState.ALARM
                self._hold_until[track_id] = now + self.hold_secs
            elif score >= self.warn_threshold:
                new_state = TrackThreatState.WARNING
            else:
                new_state = TrackThreatState.NORMAL

            self._states[track_id] = new_state
            return new_state

    def get_state(self, track_id: int) -> TrackThreatState:
        with self._lock:
            return self._states.get(track_id, TrackThreatState.NORMAL)

    def get_score(self, track_id: int) -> float:
        with self._lock:
            return self._scores.get(track_id, 0.0)

    def acknowledge(self, track_id: int) -> None:
        with self._lock:
            self._states[track_id] = TrackThreatState.ACKNOWLEDGED
            self._hold_until.pop(track_id, None)

    def resolve(self, track_id: int) -> None:
        with self._lock:
            self._states[track_id] = TrackThreatState.RESOLVED
            self._scores[track_id] = 0.0
            self._hold_until.pop(track_id, None)

    def cleanup(self, active_ids: Set[int]) -> None:
        with self._lock:
            for tid in list(self._states.keys()):
                if tid not in active_ids:
                    del self._states[tid]
                    self._scores.pop(tid, None)
                    self._hold_until.pop(tid, None)
                    self._last_update.pop(tid, None)


# =====================================================================
# PART 55 — Alarm Manager
# =====================================================================
class AlarmManager:
    """
    مدير الإنذارات المركزي
    يستقبل الأحداث السلوكية ويقرر الإنذار النهائي
    """

    def __init__(
        self,
        score_calc: ThreatScoreCalculator,
        state_machine: ThreatStateMachine,
        priority_queue: AlarmPriorityQueue,
        alarm_hold_secs: float = 5.0,
    ):
        self.score_calc = score_calc
        self.state_machine = state_machine
        self.priority_queue = priority_queue
        self.alarm_hold_secs = alarm_hold_secs

        self._callbacks: List[callable] = []
        self._lock = threading.RLock()

        # عدّادات
        self.total_alarms = 0
        self.total_warnings = 0
        self.total_events = 0

    def register_callback(self, fn: callable) -> None:
        """تسجيل دالة تُستدعى عند الإنذار"""
        with self._lock:
            self._callbacks.append(fn)

    def process_events(
        self,
        track: Any,
        events: List[BehaviorEventRecord],
    ) -> Tuple[float, str, List[str]]:
        """
        معالجة أحداث تتبع واحد
        إرجاع: (score, state, reasons)
        """
        self.total_events += len(events)

        # تحويل الأحداث إلى soft/hard signals
        soft_signals: Dict[str, bool] = {}
        hard_signals: Dict[str, bool] = {}
        reasons: List[str] = []

        for evt in events:
            reasons.append(evt.reason)
            if evt.event_type == BehaviorEvent.FIREARM_BEARING:
                hard_signals["firearm"] = True
            elif evt.event_type == BehaviorEvent.EXPLOSIVE_BEARING:
                hard_signals["explosive"] = True
            elif evt.event_type == BehaviorEvent.WEAPON_BEARING:
                hard_signals["melee"] = True
            elif evt.event_type == BehaviorEvent.LOITERING:
                soft_signals["loitering"] = True
            elif evt.event_type == BehaviorEvent.RUNNING:
                soft_signals["running_in_zone"] = True
            elif evt.event_type == BehaviorEvent.CROUCHING:
                soft_signals["crouching_in_zone"] = True
            elif evt.event_type == BehaviorEvent.FACE_COVERED:
                soft_signals["concealment"] = True
            elif evt.event_type == BehaviorEvent.PHOTO_GESTURE:
                soft_signals["recording_in_zone"] = True
            elif evt.event_type == BehaviorEvent.REPEATED_PHOTOS:
                soft_signals["photo_repeated"] = True
            elif evt.event_type == BehaviorEvent.RESTRAINT_DETECTED:
                soft_signals["restraint"] = True
            elif evt.event_type == BehaviorEvent.SUBSTANCE_DETECTED:
                soft_signals["substance"] = True
            elif evt.event_type == BehaviorEvent.BURGLARY_TOOL:
                soft_signals["burglary_tool"] = True

        dwell_secs = getattr(track, "dwell", 0.0)
        identified = getattr(track, "metadata", {}).get("identified")

        score, score_reasons = self.score_calc.compute(
            soft_signals, hard_signals, dwell_secs, identified
        )

        # دفع المهم إلى قائمة الأولويات
        for evt in events:
            if evt.severity >= BehaviorSeverity.HIGH:
                self.priority_queue.push(evt)

        # تحديث آلة الحالة
        force_alarm = bool(hard_signals)
        state = self.state_machine.update(track.id, score, force_alarm)

        if state == TrackThreatState.ALARM:
            self.total_alarms += 1
            self._fire_callbacks(track, score, reasons)
        elif state == TrackThreatState.WARNING:
            self.total_warnings += 1

        return score, state.value, reasons

    def _fire_callbacks(
        self, track: Any, score: float, reasons: List[str]
    ) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for fn in callbacks:
            try:
                fn(track, score, reasons)
            except Exception as e:
                logger.error("خطأ في callback: %s", e)

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_events": self.total_events,
            "total_warnings": self.total_warnings,
            "total_alarms": self.total_alarms,
            "queue_size": len(self.priority_queue),
        }


# =====================================================================
# PART 56 — Rules Engine
# =====================================================================
class Rule:
    """قاعدة سلوكية واحدة"""

    def __init__(
        self,
        name: str,
        condition: callable,
        event_type: BehaviorEvent,
        severity: BehaviorSeverity,
        cooldown: float = 5.0,
        enabled: bool = True,
    ):
        self.name = name
        self.condition = condition
        self.event_type = event_type
        self.severity = severity
        self.cooldown = cooldown
        self.enabled = enabled
        self._last_fire: Dict[int, float] = {}

    def evaluate(self, track: Any, context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        last = self._last_fire.get(track.id, 0.0)
        if now - last < self.cooldown:
            return False
        try:
            if self.condition(track, context):
                self._last_fire[track.id] = now
                return True
        except Exception:
            return False
        return False


class RulesEngine:
    """محرك قواعد قابل للتوسيع"""

    def __init__(self):
        self.rules: List[Rule] = []
        self._lock = threading.RLock()

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        with self._lock:
            before = len(self.rules)
            self.rules = [r for r in self.rules if r.name != name]
            return len(self.rules) < before

    def evaluate(self, track: Any,
                 context: Dict[str, Any]) -> List[BehaviorEventRecord]:
        events: List[BehaviorEventRecord] = []
        with self._lock:
            rules = list(self.rules)
        for rule in rules:
            if rule.evaluate(track, context):
                events.append(BehaviorEventRecord(
                    event_type=rule.event_type,
                    severity=rule.severity,
                    track_id=track.id,
                    reason=rule.name,
                ))
        return events

    def list_rules(self) -> List[str]:
        return [r.name for r in self.rules]


# =====================================================================
# PART 57 — Default Rules Builder
# =====================================================================
def build_default_rules(analyzer: BehaviorAnalyzer) -> RulesEngine:
    """بناء قواعد سلوكية افتراضية"""
    engine = RulesEngine()

    # قاعدة: شخص مجهول يحمل سلاحاً في المنطقة
    def _armed_intruder(track: Any, ctx: Dict[str, Any]) -> bool:
        return (
            track.is_person
            and "identified" not in track.metadata
            and ctx.get("in_zone", False)
            and any(
                lbl in FIREARM_CLASSES or lbl in EXPLOSIVE_CLASSES
                for lbl in ctx.get("weapons", [])
            )
        )

    engine.add_rule(Rule(
        name="مسلح مجهول في المنطقة",
        condition=_armed_intruder,
        event_type=BehaviorEvent.FIREARM_BEARING,
        severity=BehaviorSeverity.FATAL,
        cooldown=10.0,
    ))

    # قاعدة: تسكع + تغطية وجه
    def _suspicious_loiterer(track: Any, ctx: Dict[str, Any]) -> bool:
        return (
            track.is_person
            and track.is_loitering(radius=80, window_secs=8)
            and any(
                compute_iou(track.bbox, c.bbox) > 0.05
                for c in ctx.get("conceal", [])
            )
        )

    engine.add_rule(Rule(
        name="تسكع مع تغطية وجه",
        condition=_suspicious_loiterer,
        event_type=BehaviorEvent.LOITERING,
        severity=BehaviorSeverity.HIGH,
        cooldown=15.0,
    ))

    # قاعدة: تصوير حساس في منطقة محظورة
    def _photo_in_restricted(track: Any, ctx: Dict[str, Any]) -> bool:
        return (
            track.is_person
            and ctx.get("in_zone", False)
            and len(track.photo_events) >= 2
        )

    engine.add_rule(Rule(
        name="تصوير في منطقة محظورة",
        condition=_photo_in_restricted,
        event_type=BehaviorEvent.REPEATED_PHOTOS,
        severity=BehaviorSeverity.CRITICAL,
        cooldown=20.0,
    ))

    return engine


# =====================================================================
# PART 58 — Behavioral Context Builder
# =====================================================================
class ContextBuilder:
    """بناء سياق التحليل لكل تتبع"""

    def __init__(self, analyzer: BehaviorAnalyzer):
        self.analyzer = analyzer

    def build(
        self,
        track: Any,
        detections: List[Any],
        zones: List[Any],
        frame_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        cx, cy = _track_center(track)
        in_zone = any(
            z.enabled and z.contains((cx, cy), frame_shape)
            for z in zones
        )

        persons = [d for d in detections if d.label == "person"]
        weapons = [d for d in detections if d.label in WEAPON_CLASSES]
        conceal = [d for d in detections if d.label in CONCEALMENT_CLASSES]
        recorders = [d for d in detections if d.label in RECORDER_CLASSES]
        restraints = [d for d in detections if d.label in RESTRAINT_CLASSES]

        return {
            "in_zone": in_zone,
            "persons": persons,
            "weapons": [w.label for w in weapons],
            "conceal": conceal,
            "recorders": recorders,
            "restraints": restraints,
            "frame_shape": frame_shape,
        }


# =====================================================================
# PART 59 — Anomaly Detector (optional ML)
# =====================================================================
class AnomalyDetector:
    """كشف الشذوذ الإحصائي"""

    def __init__(self, history_size: int = 100, z_threshold: float = 3.0):
        self.history_size = history_size
        self.z_threshold = z_threshold
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._lock = threading.RLock()

    def record(self, metric: str, value: float) -> None:
        with self._lock:
            self._history[metric].append(value)

    def is_anomaly(self, metric: str, value: float) -> bool:
        with self._lock:
            hist = list(self._history[metric])
        if len(hist) < 10:
            return False
        mu = sum(hist) / len(hist)
        var = sum((x - mu) ** 2 for x in hist) / len(hist)
        sigma = math.sqrt(var)
        if sigma < 1e-6:
            return False
        z = abs(value - mu) / sigma
        return z > self.z_threshold

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


# =====================================================================
# PART 60 — Track Behavior Profile
# =====================================================================
@dataclass
class TrackBehaviorProfile:
    """ملف سلوكي لكل تتبع"""
    track_id: int
    total_photos: int = 0
    total_alarms: int = 0
    total_warnings: int = 0
    zones_entered: Set[str] = field(default_factory=set)
    max_suspicion: float = 0.0
    first_alert: Optional[float] = None
    last_alert: Optional[float] = None
    weapons_seen: List[str] = field(default_factory=list)
    identified_as: Optional[str] = None

    def update_from_event(self, evt: BehaviorEventRecord) -> None:
        if evt.event_type == BehaviorEvent.PHOTO_GESTURE:
            self.total_photos += 1
        elif evt.severity >= BehaviorSeverity.HIGH:
            self.total_alarms += 1
            self.last_alert = evt.timestamp
            if self.first_alert is None:
                self.first_alert = evt.timestamp
        elif evt.severity >= BehaviorSeverity.MEDIUM:
            self.total_warnings += 1
        if evt.event_type == BehaviorEvent.ENTER_ZONE:
            z = evt.metadata.get("zone")
            if z:
                self.zones_entered.add(z)
        if "weapon" in evt.metadata:
            self.weapons_seen.append(evt.metadata["weapon"])
        if evt.event_type == BehaviorEvent.IDENTIFIED_PERSON:
            self.identified_as = evt.metadata.get("name")


class BehaviorProfileStore:
    """مخزن ملفات السلوك"""

    def __init__(self):
        self._profiles: Dict[int, TrackBehaviorProfile] = {}
        self._lock = threading.RLock()

    def get_or_create(self, track_id: int) -> TrackBehaviorProfile:
        with self._lock:
            if track_id not in self._profiles:
                self._profiles[track_id] = TrackBehaviorProfile(track_id=track_id)
            return self._profiles[track_id]

    def update(self, track_id: int, events: List[BehaviorEventRecord]) -> None:
        profile = self.get_or_create(track_id)
        for e in events:
            profile.update_from_event(e)

    def get(self, track_id: int) -> Optional[TrackBehaviorProfile]:
        with self._lock:
            return self._profiles.get(track_id)

    def cleanup(self, active_ids: Set[int]) -> None:
        with self._lock:
            for tid in list(self._profiles.keys()):
                if tid not in active_ids:
                    del self._profiles[tid]


# =====================================================================
# PART 61 — Full Behavior Pipeline
# =====================================================================
class BehaviorPipeline:
    """
    خط أنابيب كامل: من التتبعات إلى الإنذارات
    يدمج: Analyzer + Rules + AlarmManager + Profiles + State Machine
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.analyzer = BehaviorAnalyzer(cfg)
        self.rules_engine = build_default_rules(self.analyzer)
        self.state_machine = ThreatStateMachine(
            warn_threshold=cfg.get("score_warn", 35.0),
            alarm_threshold=cfg.get("score_alarm", 70.0),
            hold_secs=cfg.get("alarm_hold_secs", 5.0),
        )
        self.priority_queue = AlarmPriorityQueue()
        self.alarm_manager = AlarmManager(
            score_calc=self.analyzer.score_calc,
            state_machine=self.state_machine,
            priority_queue=self.priority_queue,
            alarm_hold_secs=cfg.get("alarm_hold_secs", 5.0),
        )
        self.profiles = BehaviorProfileStore()
        self.context_builder = ContextBuilder(self.analyzer)
        self.anomaly = AnomalyDetector()

        # stat counters
        self.frames_processed = 0
        self.events_generated = 0

    def process(
        self,
        tracks: List[Any],
        detections: List[Any],
        zones: List[Any],
        frame_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """معالجة كاملة لكل الإطارات"""
        self.frames_processed += 1

        # 1) تحليل سلوكي أساسي
        events = self.analyzer.analyze(tracks, detections, zones, frame_shape)

        # 2) تطبيق القواعد
        for track in tracks:
            if track.state.value != "confirmed":
                continue
            ctx = self.context_builder.build(track, detections, zones, frame_shape)
            events.extend(self.rules_engine.evaluate(track, ctx))

        self.events_generated += len(events)

        # 3) تحديث الملفات
        for track in tracks:
            tid_events = [e for e in events if e.track_id == track.id]
            if tid_events:
                self.profiles.update(track.id, tid_events)

        # 4) حساب الدرجات والإنذارات
        results = []
        for track in tracks:
            if track.state.value != "confirmed":
                continue
            tid_events = [e for e in events if e.track_id == track.id]
            if not tid_events:
                # تحديث بالانحدار فقط
                score = self.state_machine.get_score(track.id)
                state = self.state_machine.update(track.id, score * 0.9)
                continue
            score, state_str, reasons = self.alarm_manager.process_events(
                track, tid_events
            )
            # تحديث suspicion على التتبع نفسه
            if hasattr(track, "suspicion"):
                track.suspicion = score
            if hasattr(track, "reason"):
                track.reason = " + ".join(reasons[:3])
            if hasattr(track, "status"):
                track.status = state_str
            results.append({
                "track_id": track.id,
                "score": score,
                "state": state_str,
                "reasons": reasons,
            })

        # 5) تنظيف
        active_ids = {t.id for t in tracks if hasattr(t, "id")}
        self.state_machine.cleanup(active_ids)
        self.profiles.cleanup(active_ids)

        return {
            "events": [e.to_dict() for e in events],
            "tracks_state": results,
            "stats": self.alarm_manager.get_stats(),
            "queue_top": [e.to_dict() for e in self.priority_queue.peek(5)],
        }

    def reset(self) -> None:
        self.analyzer.reset()
        self.state_machine._states.clear()
        self.state_machine._scores.clear()
        self.priority_queue.clear()
        self.profiles._profiles.clear()
        self.frames_processed = 0
        self.events_generated = 0


# =====================================================================
# PART 62 — Alarm Callback Integration
# =====================================================================
class AlarmBridge:
    """
    جسر بين AlarmManager وبقية النظام
    (TTS + إشعارات + حفظ أدلة)
    """

    def __init__(
        self,
        alarm_manager: AlarmManager,
        tts_callback: Optional[callable] = None,
        notification_callback: Optional[callable] = None,
        evidence_callback: Optional[callable] = None,
    ):
        self.alarm_manager = alarm_manager
        self.tts_callback = tts_callback
        self.notification_callback = notification_callback
        self.evidence_callback = evidence_callback

        alarm_manager.register_callback(self._on_alarm)

    def _on_alarm(self, track: Any, score: float, reasons: List[str]) -> None:
        reason_text = " | ".join(reasons[:3]) if reasons else "تهديد"

        # TTS
        if self.tts_callback:
            try:
                self.tts_callback(track, reason_text)
            except Exception as e:
                logger.error("TTS error: %s", e)

        # إشعارات
        if self.notification_callback:
            try:
                subject = f"⚠️ إنذار FalconEye — {reason_text}"
                body = (
                    f"معرّف التتبع: {track.id}\n"
                    f"الفئة: {getattr(track, 'stable_label', '?')}\n"
                    f"درجة الاشتباه: {score:.0f}%\n"
                    f"الأسباب: {reason_text}\n"
                    f"الوقت: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.notification_callback(subject, body)
            except Exception as e:
                logger.error("Notification error: %s", e)

        # حفظ الأدلة
        if self.evidence_callback:
            try:
                self.evidence_callback(track, reason_text)
            except Exception as e:
                logger.error("Evidence error: %s", e)


# =====================================================================
# PART 63 — Behavior Reports
# =====================================================================
class BehaviorReporter:
    """تقارير سلوكية"""

    def __init__(self, profile_store: BehaviorProfileStore):
        self.profile_store = profile_store

    def summary(self) -> Dict[str, Any]:
        profiles = self.profile_store._profiles
        if not profiles:
            return {"total": 0}
        by_identified = defaultdict(int)
        weapons_count = defaultdict(int)
        total_alarms = 0
        total_warnings = 0
        for p in profiles.values():
            total_alarms += p.total_alarms
            total_warnings += p.total_warnings
            if p.identified_as:
                by_identified[p.identified_as] += 1
            for w in p.weapons_seen:
                weapons_count[w] += 1
        return {
            "total_profiles": len(profiles),
            "total_alarms": total_alarms,
            "total_warnings": total_warnings,
            "identified_persons": dict(by_identified),
            "weapons_seen": dict(weapons_count),
        }

    def top_threats(self, n: int = 5) -> List[Dict[str, Any]]:
        profiles = list(self.profile_store._profiles.values())
        profiles.sort(key=lambda p: p.max_suspicion, reverse=True)
        return [
            {
                "track_id": p.track_id,
                "score": round(p.max_suspicion, 1),
                "alarms": p.total_alarms,
                "photos": p.total_photos,
                "weapons": p.weapons_seen,
                "identified": p.identified_as,
            }
            for p in profiles[:n]
        ]


# =====================================================================
# PART 64 — Behavior Exports
# =====================================================================
__all__ = [
    # Enums
    "BehaviorEvent", "BehaviorSeverity",
    # Records
    "BehaviorEventRecord", "TrackBehaviorProfile",
    # Core
    "BehaviorAnalyzer", "BehaviorPipeline", "BehaviorReporter",
    "AlarmManager", "AlarmBridge", "AlarmPriorityQueue",
    "ThreatStateMachine", "TrackThreatState",
    "ThreatScoreCalculator",
    # Detectors
    "PhotoGestureDetector", "WeaponVerifier", "AnomalyDetector",
    # Zones & Voting
    "ZoneTracker", "TemporalVoter",
    # Rules
    "Rule", "RulesEngine", "build_default_rules",
    # Store
    "BehaviorProfileStore", "ContextBuilder",
    # Constants
    "FIREARM_CLASSES", "MELEE_CLASSES", "EXPLOSIVE_CLASSES",
    "CONCEALMENT_CLASSES", "RECORDER_CLASSES", "WEAPON_CLASSES",
]
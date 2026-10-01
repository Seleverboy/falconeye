# -*- coding: utf-8 -*-
"""
FalconEye v13.0 — Enterprise Edition
=====================================
الجزء الأول: البنية الأساسية
  • الإعدادات (Configuration)
  • قاعدة البيانات (Database)
  • الإحصائيات (Statistics)
  • سجل HMAC (Manifest)
  • إدارة الأدلة (Evidence)
  • الإشعارات (Notifications)
  • REST API
  • مراقبة النظام (Monitor)
  • النسخ الاحتياطي (Backup)
  • التصدير (Export)
  • الإضافات (Plugins)
  • CLI

يمكن تشغيل هذا الملف مباشرة للأوامر الإدارية:
    python falconeye.py --stats
    python falconeye.py --create-user admin pass admin
    python falconeye.py --backup
    python falconeye.py --export csv
"""

# =====================================================================
# 1) الاستيرادات
# =====================================================================
import argparse
import csv
import glob
import hashlib
import hmac
import html
import json
import logging
import logging.handlers
import math
import os
import pickle
import platform
import secrets
import shutil
import signal
import smtplib
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, IntEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, RLock, Semaphore
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional,
    Sequence, Set, Tuple, Type, TypeVar, Union,
)

import numpy as np

# مكتبات اختيارية
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from cryptography.fernet import Fernet
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


# =====================================================================
# 2) البيانات الوصفية
# =====================================================================
__version__ = "13.0.0"
__author__ = "Mohammed Amine Nejib Al-Ghazouani"
__project__ = "FalconEye Enterprise"
__build__ = "2026.01"


# =====================================================================
# 3) التصنيفات (Enums)
# =====================================================================
class ThreatLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    FATAL = 4


class AlarmType(str, Enum):
    WEAPON = "weapon"
    FIREARM = "firearm"
    EXPLOSIVE = "explosive"
    RECORDING = "recording"
    PHOTO = "photo"
    CONCEALMENT = "concealment"
    LOITERING = "loitering"
    RUNNING = "running"
    CROUCHING = "crouching"
    DWELL = "dwell"
    INTRUSION = "intrusion"
    FACE_MATCH = "face_match"
    SOUND = "sound"
    SYSTEM = "system"


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    AUDITOR = "auditor"


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    HTML = "html"


# =====================================================================
# 4) السجل (Logging)
# =====================================================================
logger = logging.getLogger("falconeye")


def setup_logging(cfg: "SystemConfig" = None) -> logging.Logger:
    """تهيئة نظام السجل"""
    global logger
    logger = logging.getLogger("falconeye")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # الملف الدوّار
    log_file = cfg.log_file if cfg else "falconeye_events.log"
    max_bytes = cfg.log_max_bytes if cfg else (10 * 1024 * 1024)
    backup_count = cfg.log_backup_count if cfg else 5

    try:
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

    # الطرفية
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# =====================================================================
# 5) الإعدادات (Configuration)
# =====================================================================
@dataclass
class CameraConfig:
    camera_id: str = "cam_01"
    name: str = "الكاميرا الرئيسية"
    source: Union[str, int] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True
    location: str = "غير محدد"
    zone_inner: Tuple[float, float, float, float] = (0.30, 0.25, 0.70, 0.80)
    zone_outer: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.90)


@dataclass
class DetectionConfig:
    world_model: str = "yolov8n-world.pt"
    pose_model: str = "yolov8n-pose.pt"
    face_model: str = "yolov8n-face.pt"
    detect_imgsz: int = 960
    detect_imgsz_min: int = 640
    detect_skip: int = 1
    pose_skip: int = 2
    conf_detect: float = 0.28
    conf_person: float = 0.35
    conf_weapon: float = 0.15
    conf_face: float = 0.60
    enable_face_recognition: bool = False
    enable_pose: bool = True


@dataclass
class TrackerConfig:
    min_hits_confirm: int = 5
    max_coast_frames: int = 3
    reid_threshold: float = 0.80
    reid_max_dist: int = 150
    reid_max_age: float = 5.0
    dup_iou_threshold: float = 0.45
    merge_iou_threshold: float = 0.60
    min_track_age: float = 2.5
    assoc_max_cost: float = 0.55
    min_match_iou: float = 0.20
    max_track_speed: float = 180.0
    max_jump_frac: float = 0.20


@dataclass
class AlarmConfig:
    warn_dwell: float = 2.5
    alarm_dwell: float = 4.0
    loiter_radius: float = 60.0
    loiter_secs: float = 6.0
    run_speed: float = 260.0
    crouch_ratio: float = 1.20
    photo_events_to_alarm: int = 3
    alarm_hold_secs: float = 5.0
    score_warn: float = 35.0
    score_alarm: float = 70.0
    snapshot_cooldown: float = 15.0


@dataclass
class EvidenceConfig:
    evidence_dir: str = "threats"
    daily_subdir: bool = True
    by_threat: bool = True
    save_full_frame: bool = True
    encrypt: bool = False
    encryption_key_path: str = ".falconeye_enc.key"
    hmac_key_path: str = ".falconeye_hmac.key"
    verify_chain_on_start: bool = True
    manifest_name: str = "tamper_proof_manifest.sha256"
    compression_quality: int = 92
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_dir: str = "backups"


@dataclass
class NotificationConfig:
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    webhook_enabled: bool = False
    webhook_url: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    notify_on_levels: List[int] = field(
        default_factory=lambda: [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
    )


@dataclass
class APIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    api_key_required: bool = False
    api_keys: List[str] = field(default_factory=list)
    cors_enabled: bool = True


@dataclass
class DatabaseConfig:
    db_path: str = "falconeye.db"
    auto_vacuum: bool = True
    retention_days: int = 90


@dataclass
class UIConfig:
    ui_language: str = "ar"
    arabic_font_path: str = "arial.ttf"
    arabic_font_size: int = 22
    arabic_numerals: bool = False
    dark_mode: bool = True
    show_hud: bool = True
    show_heatmap: bool = False
    show_zones: bool = True
    hud_bar_height: int = 34
    hud_safe_top: int = 40
    twin_size: int = 400
    twin_blink_hz: float = 4.0


@dataclass
class SystemConfig:
    """الإعدادات الكاملة"""
    version: str = __version__
    cameras: List[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    alarms: AlarmConfig = field(default_factory=AlarmConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log_level: str = "INFO"
    log_file: str = "falconeye_events.log"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    tts_enabled: bool = True
    tts_language: str = "ar"
    tts_rate: int = 175
    tts_volume: float = 1.0
    tts_cooldown: float = 5.0

    # ------------------------------------------------------------------
    def save(self, path: str = "falconeye_config.json") -> None:
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: str = "falconeye_config.json") -> "SystemConfig":
        if not os.path.exists(path):
            cfg = cls()
            try:
                cfg.save(path)
            except Exception:
                pass
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls()

        # استعادة الكاميرات
        cams = [CameraConfig(**c) for c in data.get("cameras", [])]
        data["cameras"] = cams if cams else [CameraConfig()]

        # استعادة الكائنات المتداخلة
        for key, klass in [
            ("detection", DetectionConfig),
            ("tracker", TrackerConfig),
            ("alarms", AlarmConfig),
            ("evidence", EvidenceConfig),
            ("notifications", NotificationConfig),
            ("api", APIConfig),
            ("database", DatabaseConfig),
            ("ui", UIConfig),
        ]:
            if key in data and isinstance(data[key], dict):
                data[key] = klass(**data[key])

        # حذف الحقول غير المعروفة
        known = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)


# =====================================================================
# 6) أدوات مساعدة
# =====================================================================
class TimeUtils:
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def iso(dt: Optional[datetime] = None) -> str:
        return (dt or TimeUtils.now()).isoformat(timespec="seconds")

    @staticmethod
    def human(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f} ث"
        if seconds < 3600:
            return f"{seconds / 60:.1f} د"
        if seconds < 86400:
            return f"{seconds / 3600:.1f} س"
        return f"{seconds / 86400:.1f} يوم"


class HashUtils:
    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def file_hash(path: str, algo: str = "sha256") -> str:
        h = hashlib.new(algo)
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except Exception:
            return ""
        return h.hexdigest()

    @staticmethod
    def short_id(prefix: str = "") -> str:
        return f"{prefix}{uuid.uuid4().hex[:12]}"


class StringUtils:
    @staticmethod
    def safe_tag(s: str) -> str:
        base = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in str(s)
        )[:32] or "item"
        h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:6]
        return f"{base}_{h}"

    @staticmethod
    def has_arabic(s: str) -> bool:
        for ch in str(s):
            if "\u0600" <= ch <= "\u06FF":
                return True
        return False

    @staticmethod
    def tsv_escape(s: str) -> str:
        return str(s).replace("\t", " ").replace("\n", " ").replace("\r", "")


# =====================================================================
# 7) قاعدة البيانات
# =====================================================================
class Database:
    """قاعدة بيانات SQLite لكل الأحداث والبيانات"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT,
        track_id INTEGER,
        threat_type TEXT,
        threat_level INTEGER DEFAULT 0,
        label TEXT,
        confidence REAL DEFAULT 0,
        reason TEXT,
        bbox TEXT,
        evidence_path TEXT,
        evidence_hash TEXT,
        acknowledged INTEGER DEFAULT 0,
        acknowledged_by TEXT,
        notes TEXT,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ev_ts ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_ev_type ON events(threat_type);
    CREATE INDEX IF NOT EXISTS idx_ev_level ON events(threat_level);

    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        camera_id TEXT,
        first_seen TEXT,
        last_seen TEXT,
        label TEXT,
        total_frames INTEGER DEFAULT 0,
        max_suspicion REAL DEFAULT 0,
        status TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL,
        email TEXT,
        created_at TEXT,
        last_login TEXT,
        active INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user TEXT,
        action TEXT,
        resource TEXT,
        details TEXT,
        ip_address TEXT
    );

    CREATE TABLE IF NOT EXISTS system_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        cpu_percent REAL DEFAULT 0,
        memory_mb REAL DEFAULT 0,
        disk_percent REAL DEFAULT 0,
        fps REAL DEFAULT 0,
        tracks INTEGER DEFAULT 0,
        metadata TEXT
    );
    """

    def __init__(self, cfg: DatabaseConfig):
        self.cfg = cfg
        self._lock = RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            self.cfg.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        with suppress(Exception):
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    # -------------------- Events --------------------
    def insert_event(self, **kw) -> int:
        with self._lock:
            cur = self._conn.execute("""
                INSERT INTO events (
                    timestamp, camera_id, track_id, threat_type,
                    threat_level, label, confidence, reason,
                    bbox, evidence_path, evidence_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kw.get("timestamp", TimeUtils.iso()),
                kw.get("camera_id", "default"),
                kw.get("track_id", 0),
                kw.get("threat_type", "unknown"),
                int(kw.get("threat_level", 0)),
                kw.get("label", ""),
                float(kw.get("confidence", 0.0)),
                kw.get("reason", ""),
                json.dumps(kw.get("bbox")) if kw.get("bbox") else None,
                kw.get("evidence_path", ""),
                kw.get("evidence_hash", ""),
                json.dumps(kw.get("metadata", {})),
            ))
            self._conn.commit()
            return cur.lastrowid

    def query_events(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        camera_id: Optional[str] = None,
        threat_type: Optional[str] = None,
        min_level: int = 0,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM events WHERE 1=1"
        p: List[Any] = []
        if since:
            q += " AND timestamp >= ?"
            p.append(TimeUtils.iso(since))
        if until:
            q += " AND timestamp <= ?"
            p.append(TimeUtils.iso(until))
        if camera_id:
            q += " AND camera_id = ?"
            p.append(camera_id)
        if threat_type:
            q += " AND threat_type = ?"
            p.append(threat_type)
        if min_level:
            q += " AND threat_level >= ?"
            p.append(min_level)
        q += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        p.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(q, p).fetchall()
        return [dict(r) for r in rows]

    def count_events(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return r[0] if r else 0

    def acknowledge_event(self, event_id: int, user: str,
                          notes: str = "") -> bool:
        with self._lock:
            cur = self._conn.execute("""
                UPDATE events SET acknowledged=1, acknowledged_by=?, notes=?
                WHERE id=?
            """, (user, notes, event_id))
            self._conn.commit()
            return cur.rowcount > 0

    # -------------------- Users --------------------
    def create_user(self, username: str, password: str,
                    role: UserRole, email: str = "") -> bool:
        salt = secrets.token_hex(16)
        pw_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100_000
        ).hex()
        try:
            with self._lock:
                self._conn.execute("""
                    INSERT INTO users
                    (username, password_hash, salt, role, email, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, pw_hash, salt, role.value, email,
                      TimeUtils.iso()))
                self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username: str, password: str
                    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username=? AND active=1",
                (username,)
            ).fetchone()
        if not row:
            return None
        u = dict(row)
        pw_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), u["salt"].encode(), 100_000
        ).hex()
        if not hmac.compare_digest(pw_hash, u["password_hash"]):
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE users SET last_login=? WHERE id=?",
                (TimeUtils.iso(), u["id"])
            )
            self._conn.commit()
        return u

    # -------------------- Audit --------------------
    def log_audit(self, user: str, action: str, resource: str,
                  details: str = "", ip: str = "") -> None:
        with self._lock:
            self._conn.execute("""
                INSERT INTO audit_log
                (timestamp, user, action, resource, details, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (TimeUtils.iso(), user, action, resource, details, ip))
            self._conn.commit()

    # -------------------- Metrics --------------------
    def insert_metrics(self, **kw) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT INTO system_metrics
                (timestamp, cpu_percent, memory_mb, disk_percent, fps,
                 tracks, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                TimeUtils.iso(),
                float(kw.get("cpu_percent", 0)),
                float(kw.get("memory_mb", 0)),
                float(kw.get("disk_percent", 0)),
                float(kw.get("fps", 0)),
                int(kw.get("tracks", 0)),
                json.dumps(kw.get("metadata", {})),
            ))
            self._conn.commit()

    # -------------------- Cleanup --------------------
    def purge_old(self, days: int) -> int:
        cutoff = TimeUtils.iso(
            TimeUtils.now() - timedelta(days=days)
        )
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE timestamp < ?", (cutoff,)
            )
            deleted = cur.rowcount
            self._conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
        return deleted

    def close(self) -> None:
        with self._lock:
            if self._conn:
                with suppress(Exception):
                    self._conn.close()
                self._conn = None


# =====================================================================
# 8) محرك الإحصائيات
# =====================================================================
class StatisticsEngine:
    """إحصائيات شاملة"""

    def __init__(self, db: Database):
        self.db = db

    def events_per_hour(self, hours: int = 24) -> Dict[str, int]:
        since = TimeUtils.now() - timedelta(hours=hours)
        events = self.db.query_events(since=since, limit=100_000)
        buckets: Dict[str, int] = defaultdict(int)
        for e in events:
            key = (e.get("timestamp") or "")[:13]
            if key:
                buckets[key] += 1
        return dict(buckets)

    def threat_distribution(self, days: int = 7) -> Dict[str, int]:
        since = TimeUtils.now() - timedelta(days=days)
        events = self.db.query_events(since=since, limit=100_000)
        return dict(Counter(e.get("threat_type", "unknown") for e in events))

    def top_cameras(self, days: int = 7, n: int = 5
                    ) -> List[Tuple[str, int]]:
        since = TimeUtils.now() - timedelta(days=days)
        events = self.db.query_events(since=since, limit=100_000)
        c = Counter(e.get("camera_id", "unknown") for e in events)
        return c.most_common(n)

    def generate_report(self) -> Dict[str, Any]:
        return {
            "generated_at": TimeUtils.iso(),
            "total_events": self.db.count_events(),
            "events_per_hour_24h": self.events_per_hour(24),
            "threat_distribution_7d": self.threat_distribution(7),
            "top_cameras_7d": self.top_cameras(7),
        }


# =====================================================================
# 9) سجل HMAC
# =====================================================================
class HMACManifest:
    """سلسلة HMAC غير قابلة للتلاعب"""

    GENESIS = "FALCONEYE_GENESIS"

    def __init__(self, key_path: str, manifest_path: str):
        self.key_path = key_path
        self.manifest_path = manifest_path
        self._key = self._load_or_create_key()
        self._lock = RLock()
        self._state = {"chain": self.GENESIS}

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = secrets.token_bytes(32)
        os.makedirs(os.path.dirname(os.path.abspath(self.key_path)),
                    exist_ok=True)
        with open(self.key_path, "wb") as f:
            f.write(key)
            f.flush()
            with suppress(Exception):
                os.fsync(f.fileno())
        with suppress(Exception):
            os.chmod(self.key_path, 0o600)
        return key

    def _chain(self, prev: str, payload: str) -> str:
        return hmac.new(
            self._key,
            f"{prev}|{payload}".encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def verify_chain(self) -> Tuple[bool, str]:
        if not os.path.exists(self.manifest_path):
            return True, self.GENESIS
        prev = self.GENESIS
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if "  chain=" not in line:
                    continue
                payload, _, chain_val = line.rpartition("  chain=")
                expected = self._chain(prev, payload)
                if not hmac.compare_digest(expected, chain_val):
                    return False, f"كسر السلسلة عند السطر {lineno}"
                prev = chain_val
        return True, prev

    def append(self, payload: str) -> str:
        with self._lock:
            chain = self._chain(self._state["chain"], payload)
            self._state["chain"] = chain
            os.makedirs(
                os.path.dirname(os.path.abspath(self.manifest_path)),
                exist_ok=True
            )
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(payload + f"  chain={chain}\n")
                f.flush()
                with suppress(Exception):
                    os.fsync(f.fileno())
            return chain


# =====================================================================
# 10) إدارة الأدلة
# =====================================================================
class EvidenceManager:
    """حفظ وتشفير الأدلة"""

    def __init__(self, cfg: EvidenceConfig):
        self.cfg = cfg
        self.manifest = HMACManifest(
            cfg.hmac_key_path,
            os.path.join(cfg.evidence_dir, cfg.manifest_name)
        )
        self._fernet = None
        if cfg.encrypt and HAVE_CRYPTO:
            self._fernet = self._get_fernet()

    def _get_fernet(self):
        if os.path.exists(self.cfg.encryption_key_path):
            with open(self.cfg.encryption_key_path, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.cfg.encryption_key_path, "wb") as f:
                f.write(key)
            with suppress(Exception):
                os.chmod(self.cfg.encryption_key_path, 0o600)
        return Fernet(key)

    def _get_dir(self, threat_cat: Optional[str] = None) -> str:
        parts = [self.cfg.evidence_dir]
        if self.cfg.daily_subdir:
            parts.append(datetime.now().strftime("%Y-%m-%d"))
        if self.cfg.by_threat and threat_cat:
            parts.append(threat_cat)
        d = os.path.join(*parts)
        os.makedirs(d, exist_ok=True)
        return d

    def save_image(self, image: np.ndarray, threat_category: str,
                   prefix: str = "evidence") -> Tuple[str, str]:
        if not HAVE_CV2:
            return "", ""
        try:
            ok, buf = cv2.imencode(
                ".jpg", image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.compression_quality]
            )
            if not ok:
                return "", ""
            data = buf.tobytes()
            h = HashUtils.sha256(data)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"{ts}_{prefix}_{h[:8]}"
            directory = self._get_dir(threat_category)

            if self._fernet:
                fp = os.path.join(directory, fname + ".jpg.enc")
                with open(fp, "wb") as f:
                    f.write(self._fernet.encrypt(data))
            else:
                fp = os.path.join(directory, fname + ".jpg")
                with open(fp, "wb") as f:
                    f.write(data)

            payload = (
                f"{h}  {TimeUtils.iso()}  "
                f"file={os.path.basename(fp)}  "
                f"category={threat_category}"
            )
            self.manifest.append(payload)
            return fp, h
        except Exception as e:
            logger.error("فشل حفظ الدليل: %s", e)
            return "", ""

    def count_files(self) -> int:
        total = 0
        try:
            for root, _, files in os.walk(self.cfg.evidence_dir):
                for f in files:
                    if f.endswith((".jpg", ".enc")):
                        total += 1
        except Exception:
            pass
        return total


# =====================================================================
# 11) الإشعارات
# =====================================================================
class NotificationManager:
    """إشعارات متعددة القنوات"""

    def __init__(self, cfg: NotificationConfig):
        self.cfg = cfg
        self._pool = []
        self._lock = RLock()

    def notify(self, subject: str, body: str,
               level: int = int(ThreatLevel.HIGH),
               attachments: Optional[List[str]] = None) -> None:
        if level not in self.cfg.notify_on_levels:
            return
        if self.cfg.email_enabled:
            t = threading.Thread(
                target=self._send_email,
                args=(subject, body, attachments or []),
                daemon=True,
            )
            t.start()
        if self.cfg.webhook_enabled:
            threading.Thread(
                target=self._send_webhook,
                args=(subject, body),
                daemon=True,
            ).start()
        if self.cfg.telegram_enabled:
            threading.Thread(
                target=self._send_telegram,
                args=(subject, body),
                daemon=True,
            ).start()

    def _send_email(self, subject: str, body: str,
                    attachments: List[str]) -> None:
        if not self.cfg.email_recipients:
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = self.cfg.smtp_user
            msg["To"] = ", ".join(self.cfg.email_recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            for path in attachments:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(path)}"
                )
                msg.attach(part)
            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port,
                              timeout=10) as smtp:
                smtp.starttls()
                smtp.login(self.cfg.smtp_user, self.cfg.smtp_password)
                smtp.send_message(msg)
            logger.info("تم إرسال البريد")
        except Exception as e:
            logger.error("فشل البريد: %s", e)

    def _send_webhook(self, subject: str, body: str) -> None:
        try:
            payload = json.dumps({
                "subject": subject, "body": body,
                "timestamp": TimeUtils.iso(),
            }).encode("utf-8")
            req = urllib.request.Request(
                self.cfg.webhook_url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            logger.info("تم إرسال Webhook")
        except Exception as e:
            logger.error("فشل Webhook: %s", e)

    def _send_telegram(self, subject: str, body: str) -> None:
        try:
            text = f"*{subject}*\n\n{body}"
            url = (
                f"https://api.telegram.org/bot"
                f"{self.cfg.telegram_bot_token}/sendMessage"
            )
            data = urllib.parse.urlencode({
                "chat_id": self.cfg.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10):
                pass
            logger.info("تم إرسال Telegram")
        except Exception as e:
            logger.error("فشل Telegram: %s", e)

    def shutdown(self) -> None:
        pass


# =====================================================================
# 12) REST API
# =====================================================================
class APIHandler(BaseHTTPRequestHandler):
    db: Database = None
    api_cfg: APIConfig = None
    stats_engine: StatisticsEngine = None

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        if not self.api_cfg or not self.api_cfg.api_key_required:
            return True
        key = self.headers.get("X-API-Key", "")
        return key in (self.api_cfg.api_keys or [])

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/health":
            return self._send_json({
                "status": "ok", "version": __version__,
                "ts": TimeUtils.iso(),
            })

        if not self._check_auth():
            return self._send_json({"error": "unauthorized"}, 401)

        try:
            if path == "/api/events":
                q = urllib.parse.parse_qs(parsed.query)
                events = self.db.query_events(
                    limit=int(q.get("limit", ["100"])[0]),
                    offset=int(q.get("offset", ["0"])[0]),
                )
                return self._send_json({
                    "count": len(events), "events": events,
                })
            if path == "/api/stats":
                return self._send_json(
                    self.stats_engine.generate_report()
                )
            if path == "/api/version":
                return self._send_json({
                    "version": __version__,
                    "author": __author__,
                })
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if not self._check_auth():
            return self._send_json({"error": "unauthorized"}, 401)

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body)
        except Exception:
            return self._send_json({"error": "invalid json"}, 400)

        try:
            if path == "/api/acknowledge":
                eid = data.get("event_id")
                user = data.get("user", "api")
                if eid is None:
                    return self._send_json({"error": "missing event_id"}, 400)
                ok = self.db.acknowledge_event(eid, user)
                return self._send_json({"success": ok})
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

        self._send_json({"error": "not found"}, 404)


class APIServer:
    """خادم REST API"""

    def __init__(self, cfg: APIConfig, db: Database,
                 stats_engine: StatisticsEngine):
        self.cfg = cfg
        self.db = db
        self.stats = stats_engine
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        APIHandler.db = self.db
        APIHandler.api_cfg = self.cfg
        APIHandler.stats_engine = self.stats
        try:
            self._server = ThreadingHTTPServer(
                (self.cfg.host, self.cfg.port), APIHandler
            )
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="api-server",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "REST API على http://%s:%d",
                self.cfg.host, self.cfg.port,
            )
        except Exception as e:
            logger.error("فشل API: %s", e)

    def stop(self) -> None:
        if self._server:
            with suppress(Exception):
                self._server.shutdown()
                self._server.server_close()


# =====================================================================
# 13) مراقب النظام
# =====================================================================
class SystemMonitor:
    def __init__(self, db: Database, interval: float = 30.0):
        self.db = db
        self.interval = interval
        self._stop = Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="sys-monitor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                m = self._collect()
                self.db.insert_metrics(**m)
            except Exception as e:
                logger.error("Monitor: %s", e)
            self._stop.wait(self.interval)

    def _collect(self) -> Dict[str, Any]:
        m = {
            "cpu_percent": 0.0, "memory_mb": 0.0,
            "disk_percent": 0.0, "fps": 0.0, "tracks": 0,
        }
        if HAVE_PSUTIL:
            try:
                m["cpu_percent"] = psutil.cpu_percent(interval=0.3)
                m["memory_mb"] = (
                    psutil.Process().memory_info().rss / (1024 * 1024)
                )
                m["disk_percent"] = psutil.disk_usage("/").percent
            except Exception:
                pass
        return m

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# =====================================================================
# 14) التصدير
# =====================================================================
class ExportManager:
    """تصدير الأحداث"""

    def __init__(self, db: Database):
        self.db = db

    def to_json(self, events: List[Dict], output: str) -> str:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False, default=str)
        return output

    def to_csv(self, events: List[Dict], output: str) -> str:
        if not events:
            return ""
        with open(output, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)
        return output

    def to_html(self, events: List[Dict], output: str,
                title: str = "تقرير FalconEye") -> str:
        rows = ""
        for e in events[:500]:
            rows += "<tr>"
            for v in e.values():
                rows += f"<td>{html.escape(str(v))}</td>"
            rows += "</tr>"
        headers = ""
        if events:
            headers = "".join(
                f"<th>{html.escape(k)}</th>" for k in events[0].keys()
            )
        content = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:2em;background:#f5f5f5;}}
h1{{color:#C9604F;}}
table{{border-collapse:collapse;width:100%;background:#fff;}}
th{{background:#C9604F;color:#fff;padding:10px;}}
td{{border:1px solid #ddd;padding:8px;}}
tr:nth-child(even){{background:#f9f9f9;}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>تم الإنشاء: {TimeUtils.iso()}</p>
<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        return output


# =====================================================================
# 15) النسخ الاحتياطي
# =====================================================================
class BackupManager:
    """نسخ احتياطي دوري"""

    def __init__(self, ev_cfg: EvidenceConfig, db_cfg: DatabaseConfig):
        self.ev_cfg = ev_cfg
        self.db_cfg = db_cfg
        self._stop = Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.ev_cfg.backup_enabled:
            return
        self._thread = threading.Thread(
            target=self._run, name="backup", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # انتظار أول دورة
        self._stop.wait(60)
        while not self._stop.is_set():
            try:
                self.create_backup()
            except Exception as e:
                logger.error("Backup: %s", e)
            self._stop.wait(self.ev_cfg.backup_interval_hours * 3600)

    def create_backup(self) -> str:
        os.makedirs(self.ev_cfg.backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = os.path.join(
            self.ev_cfg.backup_dir,
            f"falconeye_backup_{ts}.zip",
        )
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            if os.path.exists(self.db_cfg.db_path):
                zf.write(self.db_cfg.db_path, arcname="falconeye.db")
            if os.path.exists(self.ev_cfg.evidence_dir):
                for root, _, files in os.walk(self.ev_cfg.evidence_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        arc = os.path.relpath(
                            fp,
                            start=os.path.dirname(self.ev_cfg.evidence_dir)
                        )
                        zf.write(fp, arcname=arc)
        logger.info("نسخة احتياطية: %s", archive)
        return archive

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# =====================================================================
# 16) نظام الإضافات
# =====================================================================
class PluginBase(ABC):
    name: str = "plugin"
    version: str = "1.0.0"

    @abstractmethod
    def on_start(self, ctx: Dict[str, Any]) -> None:
        ...

    def on_event(self, event: Dict[str, Any]) -> None:
        pass

    def on_alarm(self, track: Any, alarm_type: str) -> None:
        pass

    def on_shutdown(self) -> None:
        pass


class PluginManager:
    """تحميل الإضافات ديناميكياً"""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: List[PluginBase] = []
        os.makedirs(plugin_dir, exist_ok=True)

    def load_all(self) -> int:
        import importlib.util
        count = 0
        for path in glob.glob(os.path.join(self.plugin_dir, "*.py")):
            name = os.path.splitext(os.path.basename(path))[0]
            if name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type)
                            and issubclass(attr, PluginBase)
                            and attr is not PluginBase):
                        inst = attr()
                        self.plugins.append(inst)
                        count += 1
                        logger.info("تحميل إضافة: %s", inst.name)
            except Exception as e:
                logger.error("فشل %s: %s", path, e)
        return count

    def shutdown_all(self) -> None:
        for p in self.plugins:
            with suppress(Exception):
                p.on_shutdown()


# =====================================================================
# 17) CLI
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="falconeye",
        description=f"FalconEye v{__version__}",
    )
    p.add_argument("--config", default="falconeye_config.json")
    p.add_argument("--create-user", nargs=3,
                   metavar=("USER", "PASS", "ROLE"),
                   help="إنشاء مستخدم")
    p.add_argument("--stats", action="store_true", help="إحصائيات")
    p.add_argument("--backup", action="store_true", help="نسخة احتياطية")
    p.add_argument("--export", choices=["json", "csv", "html"],
                   help="تصدير الأحداث")
    p.add_argument("--init", action="store_true",
                   help="إنشاء ملف الإعدادات")
    p.add_argument("--version", action="version", version=__version__)
    return p


def handle_cli(args, cfg: SystemConfig, db: Database,
               stats_engine: StatisticsEngine,
               backup: BackupManager) -> bool:
    """معالجة أوامر CLI — إرجاع True إذا نُفّذ أمر"""
    if args.create_user:
        user, pw, role = args.create_user
        try:
            role_enum = UserRole(role.lower())
        except ValueError:
            print(f"[خطأ] الأدوار: {[r.value for r in UserRole]}")
            return True
        ok = db.create_user(user, pw, role_enum)
        print(f"[{'نجح' if ok else 'فشل'}] إنشاء المستخدم")
        return True

    if args.stats:
        print(json.dumps(stats_engine.generate_report(),
                         indent=2, ensure_ascii=False))
        return True

    if args.backup:
        path = backup.create_backup()
        print(f"[نجح] {path}")
        return True

    if args.export:
        events = db.query_events(limit=100_000)
        exp = ExportManager(db)
        if args.export == "json":
            out = exp.to_json(events, "export.json")
        elif args.export == "csv":
            out = exp.to_csv(events, "export.csv")
        else:
            out = exp.to_html(events, "export.html")
        print(f"[نجح] {out}")
        return True

    if args.init:
        cfg.save(args.config)
        print(f"[نجح] تم إنشاء {args.config}")
        return True

    return False


# =====================================================================
# 18) نقطة البداية
# =====================================================================
def main() -> int:
    args = build_parser().parse_args()

    cfg = SystemConfig.load(args.config)
    setup_logging(cfg)

    db = Database(cfg.database)
    stats_engine = StatisticsEngine(db)
    backup = BackupManager(cfg.evidence, cfg.database)

    # تنفيذ أمر CLI
    if handle_cli(args, cfg, db, stats_engine, backup):
        db.close()
        return 0

    # بدون أمر → رسالة إرشادية
    print(f"""
╔════════════════════════════════════════════════════╗
║  FalconEye v{__version__} — Enterprise             ║
╠════════════════════════════════════════════════════╣
║  هذا الجزء يحتوي على البنية الأساسية فقط.         ║
║  لتشغيل النظام الكامل، استخدم:                    ║
║                                                    ║
║      python run.py                                 ║
║                                                    ║
║  أوامر متاحة:                                      ║
║    --stats             إحصائيات                    ║
║    --backup            نسخة احتياطية              ║
║    --export csv        تصدير الأحداث              ║
║    --create-user ...   إنشاء مستخدم                ║
║    --init              إنشاء الإعدادات             ║
╚════════════════════════════════════════════════════╝
""")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
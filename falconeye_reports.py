# =====================================================================
# PART 97 — Report Constants & Data Classes
# =====================================================================
import base64
import csv
import html
import io
import json
import math
import os
import statistics
import threading
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


REPORTS_VERSION = "1.0.0"


# =====================================================================
# PART 98 — Report Enums
# =====================================================================
class ReportFormat(str, Enum):
    """صيغ التقارير المدعومة"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    XML = "xml"
    MARKDOWN = "markdown"


class ReportPeriod(str, Enum):
    """فترات التقارير الجاهزة"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class ReportType(str, Enum):
    """أنواع التقارير"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    THREAT_ANALYSIS = "threat_analysis"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    EVIDENCE = "evidence"
    CAMERA_HEALTH = "camera_health"


# =====================================================================
# PART 99 — Report Data Structures
# =====================================================================
@dataclass
class ReportMetadata:
    """بيانات وصفية للتقرير"""
    title: str
    report_type: ReportType
    period: ReportPeriod
    generated_at: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    author: str = "FalconEye System"
    version: str = REPORTS_VERSION
    language: str = "ar"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "type": self.report_type.value,
            "period": self.period.value,
            "generated_at": self.generated_at,
            "generated_at_iso": datetime.fromtimestamp(
                self.generated_at, tz=timezone.utc
            ).isoformat(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "author": self.author,
            "version": self.version,
            "language": self.language,
        }


@dataclass
class ReportSection:
    """قسم واحد داخل التقرير"""
    title: str
    content: Any
    section_type: str = "text"  # text | table | chart | image | metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """تقرير كامل"""
    metadata: ReportMetadata
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, title: str, content: Any,
                    section_type: str = "text",
                    **metadata) -> None:
        self.sections.append(ReportSection(
            title=title,
            content=content,
            section_type=section_type,
            metadata=metadata,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "summary": self.summary,
            "sections": [
                {
                    "title": s.title,
                    "type": s.section_type,
                    "content": s.content,
                    "metadata": s.metadata,
                }
                for s in self.sections
            ],
        }


# =====================================================================
# PART 100 — Time Range Helper
# =====================================================================
class TimeRange:
    """مساعد لحساب الفترات الزمنية"""

    @staticmethod
    def from_period(period: ReportPeriod,
                    ref: Optional[datetime] = None
                    ) -> Tuple[datetime, datetime]:
        now = ref or datetime.now(timezone.utc)
        if period == ReportPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == ReportPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == ReportPeriod.WEEKLY:
            start = (now - timedelta(days=now.weekday())
                    ).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        elif period == ReportPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # آخر يوم من الشهر
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif period == ReportPeriod.QUARTERLY:
            quarter = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter, day=1,
                               hour=0, minute=0, second=0, microsecond=0)
            end_month = quarter + 3
            if end_month > 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=end_month)
        elif period == ReportPeriod.YEARLY:
            start = now.replace(month=1, day=1,
                               hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=start.year + 1)
        else:
            end = now
            start = now - timedelta(hours=24)
        return start, end

    @staticmethod
    def format_range(start: datetime, end: datetime,
                     lang: str = "ar") -> str:
        fmt = "%Y-%m-%d %H:%M"
        if lang == "ar":
            return f"من {start.strftime(fmt)} إلى {end.strftime(fmt)}"
        return f"From {start.strftime(fmt)} to {end.strftime(fmt)}"


# =====================================================================
# PART 101 — Data Aggregator
# =====================================================================
class DataAggregator:
    """تجميع البيانات من قاعدة البيانات"""

    def __init__(self, db):
        self.db = db

    def aggregate_events(
        self,
        start: datetime,
        end: datetime,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """تجميع شامل للأحداث"""
        events = self.db.query_events(
            since=start, until=end, camera_id=camera_id, limit=100_000
        )
        if not events:
            return self._empty_aggregate()

        # العدّ
        total = len(events)
        by_type = Counter(e.get("threat_type", "unknown") for e in events)
        by_camera = Counter(e.get("camera_id", "unknown") for e in events)
        by_level = Counter(e.get("threat_level", 0) for e in events)
        by_label = Counter(e.get("label", "unknown") for e in events)

        # الأحداث حسب الساعة
        by_hour = defaultdict(int)
        by_day = defaultdict(int)
        for e in events:
            ts = e.get("timestamp", "")
            if len(ts) >= 13:
                by_hour[ts[:13]] += 1
            if len(ts) >= 10:
                by_day[ts[:10]] += 1

        # درجات
        scores = [e.get("confidence", 0) for e in events
                  if e.get("confidence")]
        avg_score = statistics.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # الأحداث المؤكدة
        acknowledged = sum(1 for e in events if e.get("acknowledged"))

        return {
            "total_events": total,
            "by_type": dict(by_type),
            "by_camera": dict(by_camera),
            "by_level": dict(by_level),
            "by_label": dict(by_label),
            "by_hour": dict(by_hour),
            "by_day": dict(by_day),
            "avg_score": round(avg_score, 2),
            "max_score": round(max_score, 2),
            "acknowledged": acknowledged,
            "pending": total - acknowledged,
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        }

    def aggregate_behavior(
        self,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """تجميع البيانات السلوكية"""
        events = self.db.query_events(
            since=start, until=end, limit=100_000
        )
        reasons = Counter()
        for e in events:
            reason = (e.get("reason") or "").lower()
            for key in ["dwell", "loitering", "running", "crouching",
                        "face covered", "photo", "recording device",
                        "firearm", "weapon", "explosive"]:
                if key in reason:
                    reasons[key] += 1
        return dict(reasons)

    def aggregate_performance(
        self,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """تجميع بيانات الأداء من system_metrics"""
        try:
            with self.db._lock:
                rows = self.db._conn.execute("""
                    SELECT * FROM system_metrics
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """, (start.isoformat(), end.isoformat())).fetchall()
        except Exception:
            return {}
        if not rows:
            return {}
        cpus = [r["cpu_percent"] for r in rows if r["cpu_percent"]]
        mems = [r["memory_mb"] for r in rows if r["memory_mb"]]
        fpss = [r["fps"] for r in rows if r["fps"]]
        tracks = [r["tracks"] for r in rows if r["tracks"]]
        return {
            "samples": len(rows),
            "cpu": {
                "avg": round(statistics.mean(cpus), 1) if cpus else 0,
                "max": round(max(cpus), 1) if cpus else 0,
                "min": round(min(cpus), 1) if cpus else 0,
            },
            "memory_mb": {
                "avg": round(statistics.mean(mems), 1) if mems else 0,
                "max": round(max(mems), 1) if mems else 0,
            },
            "fps": {
                "avg": round(statistics.mean(fpss), 1) if fpss else 0,
                "max": round(max(fpss), 1) if fpss else 0,
                "min": round(min(fpss), 1) if fpss else 0,
            },
            "tracks": {
                "avg": round(statistics.mean(tracks), 1) if tracks else 0,
                "max": max(tracks) if tracks else 0,
            },
        }

    @staticmethod
    def _empty_aggregate() -> Dict[str, Any]:
        return {
            "total_events": 0,
            "by_type": {},
            "by_camera": {},
            "by_level": {},
            "by_label": {},
            "by_hour": {},
            "by_day": {},
            "avg_score": 0.0,
            "max_score": 0.0,
            "acknowledged": 0,
            "pending": 0,
        }


# =====================================================================
# PART 102 — Report Builder
# =====================================================================
class ReportBuilder:
    """بناء التقارير بكل الأنواع"""

    def __init__(self, db, language: str = "ar"):
        self.db = db
        self.aggregator = DataAggregator(db)
        self.language = language

    def _t(self, ar: str, en: str) -> str:
        return ar if self.language == "ar" else en

    def build_summary_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Report:
        """تقرير موجز"""
        if start is None or end is None:
            start, end = TimeRange.from_period(period)

        meta = ReportMetadata(
            title=self._t("تقرير موجز", "Summary Report"),
            report_type=ReportType.SUMMARY,
            period=period,
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            language=self.language,
        )
        report = Report(metadata=meta)

        # الملخص
        agg = self.aggregator.aggregate_events(start, end)
        report.summary = {
            self._t("إجمالي الأحداث", "Total Events"):
                agg["total_events"],
            self._t("الأحداث المؤكدة", "Acknowledged"):
                agg["acknowledged"],
            self._t("قيد المراجعة", "Pending"):
                agg["pending"],
            self._t("متوسط الدرجة", "Avg Score"):
                agg["avg_score"],
            self._t("أعلى درجة", "Max Score"):
                agg["max_score"],
        }

        # قسم الأحداث حسب النوع
        report.add_section(
            title=self._t("الأحداث حسب النوع", "Events by Type"),
            content=agg["by_type"],
            section_type="table",
        )

        # الأحداث حسب الكاميرا
        report.add_section(
            title=self._t("الأحداث حسب الكاميرا", "Events by Camera"),
            content=agg["by_camera"],
            section_type="table",
        )

        # التوزيع الزمني
        report.add_section(
            title=self._t("التوزيع اليومي", "Daily Distribution"),
            content=agg["by_day"],
            section_type="chart",
        )

        # السلوكيات
        behavior = self.aggregator.aggregate_behavior(start, end)
        report.add_section(
            title=self._t("السلوكيات المرصودة", "Observed Behaviors"),
            content=behavior,
            section_type="table",
        )

        return report

    def build_detailed_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY,
    ) -> Report:
        """تقرير مفصل مع القائمة الكاملة"""
        start, end = TimeRange.from_period(period)
        meta = ReportMetadata(
            title=self._t("تقرير مفصل", "Detailed Report"),
            report_type=ReportType.DETAILED,
            period=period,
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            language=self.language,
        )
        report = Report(metadata=meta)
        events = self.db.query_events(
            since=start, until=end, limit=10_000
        )
        report.summary = {
            self._t("عدد الأحداث", "Event Count"): len(events),
            self._t("الفترة", "Period"):
                TimeRange.format_range(start, end, self.language),
        }
        report.add_section(
            title=self._t("جميع الأحداث", "All Events"),
            content=events,
            section_type="table",
        )
        return report

    def build_threat_analysis(
        self,
        period: ReportPeriod = ReportPeriod.WEEKLY,
    ) -> Report:
        """تحليل التهديدات"""
        start, end = TimeRange.from_period(period)
        meta = ReportMetadata(
            title=self._t("تحليل التهديدات", "Threat Analysis"),
            report_type=ReportType.THREAT_ANALYSIS,
            period=period,
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            language=self.language,
        )
        report = Report(metadata=meta)
        agg = self.aggregator.aggregate_events(start, end)

        # أعلى تهديدات
        threat_levels = {
            0: self._t("منخفض", "Low"),
            1: self._t("متوسط", "Medium"),
            2: self._t("عالي", "High"),
            3: self._t("حرج", "Critical"),
        }
        threat_dist = {}
        for lvl, count in agg["by_level"].items():
            try:
                threat_dist[threat_levels.get(int(lvl), str(lvl))] = count
            except (ValueError, TypeError):
                threat_dist[str(lvl)] = count

        report.summary = {
            self._t("إجمالي التهديدات", "Total Threats"):
                agg["total_events"],
            self._t("تهديدات حرجة", "Critical"):
                sum(c for l, c in agg["by_level"].items()
                    if int(l) >= 3),
            self._t("تهديدات عالية", "High"):
                sum(c for l, c in agg["by_level"].items()
                    if int(l) == 2),
        }

        report.add_section(
            title=self._t("التوزيع حسب الخطورة", "Severity Distribution"),
            content=threat_dist,
            section_type="table",
        )
        report.add_section(
            title=self._t("أعلى التهديدات", "Top Threats"),
            content=dict(sorted(
                agg["by_type"].items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
            section_type="table",
        )
        return report

    def build_performance_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY,
    ) -> Report:
        """تقرير الأداء"""
        start, end = TimeRange.from_period(period)
        meta = ReportMetadata(
            title=self._t("تقرير الأداء", "Performance Report"),
            report_type=ReportType.PERFORMANCE,
            period=period,
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            language=self.language,
        )
        report = Report(metadata=meta)
        perf = self.aggregator.aggregate_performance(start, end)
        report.summary = perf
        report.add_section(
            title=self._t("مقاييس النظام", "System Metrics"),
            content=perf,
            section_type="metrics",
        )
        return report

    def build_custom_report(
        self,
        title: str,
        sections: List[Tuple[str, Any, str]],
        summary: Optional[Dict[str, Any]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Report:
        """تقرير مخصص"""
        now = datetime.now(timezone.utc)
        meta = ReportMetadata(
            title=title,
            report_type=ReportType.SUMMARY,
            period=ReportPeriod.CUSTOM,
            start_time=(start or now - timedelta(days=1)).timestamp(),
            end_time=(end or now).timestamp(),
            language=self.language,
        )
        report = Report(metadata=meta, summary=summary or {})
        for sec_title, content, sec_type in sections:
            report.add_section(sec_title, content, sec_type)
        return report


# =====================================================================
# PART 103 — JSON Exporter
# =====================================================================
class JSONExporter:
    """مصدّر JSON"""

    @staticmethod
    def export(report: Report, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2,
                     ensure_ascii=False, default=str)
        return output_path


# =====================================================================
# PART 104 — CSV Exporter
# =====================================================================
class CSVExporter:
    """مصدّر CSV"""

    @staticmethod
    def export(report: Report, output_path: str) -> str:
        rows = []
        # metadata
        meta = report.metadata.to_dict()
        for k, v in meta.items():
            rows.append(["META", k, str(v)])
        rows.append(["", "", ""])
        # summary
        for k, v in report.summary.items():
            rows.append(["SUMMARY", k, str(v)])
        rows.append(["", "", ""])
        # sections
        for section in report.sections:
            rows.append(["SECTION", section.title, section.section_type])
            content = section.content
            if isinstance(content, dict):
                for k, v in content.items():
                    rows.append(["", str(k), str(v)])
            elif isinstance(content, list):
                if content and isinstance(content[0], dict):
                    # رأس الجدول
                    headers = list(content[0].keys())
                    rows.append(["HEADERS", ",".join(headers), ""])
                    for item in content:
                        row = [str(item.get(h, "")) for h in headers]
                        rows.append(["DATA", " | ".join(row), ""])
                else:
                    for item in content:
                        rows.append(["ITEM", str(item), ""])
            else:
                rows.append(["TEXT", str(content), ""])
            rows.append(["", "", ""])

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return output_path


# =====================================================================
# PART 105 — XML Exporter
# =====================================================================
class XMLExporter:
    """مصدّر XML"""

    @staticmethod
    def export(report: Report, output_path: str) -> str:
        def _escape(s: Any) -> str:
            return html.escape(str(s), quote=True)

        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<report>')

        # metadata
        lines.append('  <metadata>')
        for k, v in report.metadata.to_dict().items():
            lines.append(f'    <{k}>{_escape(v)}</{k}>')
        lines.append('  </metadata>')

        # summary
        lines.append('  <summary>')
        for k, v in report.summary.items():
            key = k.replace(" ", "_").replace("/", "_")
            lines.append(f'    <item name="{_escape(key)}">{_escape(v)}</item>')
        lines.append('  </summary>')

        # sections
        lines.append('  <sections>')
        for section in report.sections:
            lines.append(f'    <section title="{_escape(section.title)}" '
                        f'type="{_escape(section.section_type)}">')
            content = section.content
            if isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f'      <entry key="{_escape(k)}">'
                                f'{_escape(v)}</entry>')
            elif isinstance(content, list):
                for item in content[:500]:
                    lines.append(f'      <item>{_escape(item)}</item>')
            else:
                lines.append(f'      <text>{_escape(content)}</text>')
            lines.append('    </section>')
        lines.append('  </sections>')

        lines.append('</report>')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return output_path


# =====================================================================
# PART 106 — HTML Exporter
# =====================================================================
class HTMLExporter:
    """مصدّر HTML أنيق"""

    STYLE = """
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Segoe UI', Tahoma, sans-serif;
        background: #f5f5f5; color: #1a1a1a;
        padding: 40px; direction: rtl;
    }
    .report {
        max-width: 1200px; margin: 0 auto;
        background: white; border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .header {
        background: #C9604F; color: white;
        padding: 30px 40px;
    }
    .header h1 { font-size: 28px; margin-bottom: 8px; }
    .header .meta { opacity: 0.9; font-size: 14px; }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px; padding: 30px 40px;
        background: #fafafa;
        border-bottom: 1px solid #eee;
    }
    .summary-card {
        background: white; padding: 16px;
        border-radius: 8px;
        border-right: 4px solid #C9604F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .summary-card .label {
        color: #888; font-size: 12px;
        margin-bottom: 4px;
    }
    .summary-card .value {
        font-size: 24px; font-weight: bold;
        color: #C9604F;
    }
    .section {
        padding: 30px 40px;
        border-bottom: 1px solid #eee;
    }
    .section:last-child { border-bottom: none; }
    .section h2 {
        font-size: 20px;
        color: #C9604F;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #C9604F;
        display: inline-block;
    }
    table {
        width: 100%; border-collapse: collapse;
        margin-top: 12px;
    }
    th {
        background: #C9604F; color: white;
        padding: 10px 14px; text-align: right;
        font-size: 13px;
    }
    td {
        padding: 9px 14px; border-bottom: 1px solid #eee;
        font-size: 13px;
    }
    tr:hover { background: #fafafa; }
    .chart-bar {
        display: flex; align-items: center;
        margin-bottom: 8px; gap: 12px;
    }
    .chart-bar .label {
        min-width: 140px; font-size: 13px;
        text-align: right;
    }
    .chart-bar .bar {
        flex: 1; height: 22px;
        background: #C9604F;
        border-radius: 4px;
        position: relative;
        transition: width 0.3s;
    }
    .chart-bar .value {
        min-width: 60px; font-size: 13px;
        font-weight: bold; color: #C9604F;
    }
    .footer {
        background: #2a2a2f; color: #ccc;
        padding: 20px 40px;
        text-align: center;
        font-size: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-critical { background: #fee; color: #c00; }
    .badge-high { background: #ffe; color: #c60; }
    .badge-medium { background: #fef; color: #808; }
    .badge-low { background: #efe; color: #060; }
    </style>
    """

    @classmethod
    def export(cls, report: Report, output_path: str) -> str:
        parts = []
        parts.append('<!DOCTYPE html>')
        parts.append('<html lang="ar" dir="rtl">')
        parts.append('<head>')
        parts.append('<meta charset="UTF-8">')
        parts.append(f'<title>{html.escape(report.metadata.title)}</title>')
        parts.append(cls.STYLE)
        parts.append('</head>')
        parts.append('<body>')
        parts.append('<div class="report">')

        # الترويسة
        parts.append('<div class="header">')
        parts.append(f'<h1>{html.escape(report.metadata.title)}</h1>')
        meta = report.metadata
        gen_at = datetime.fromtimestamp(
            meta.generated_at, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        parts.append(f'<div class="meta">تم الإنشاء: {gen_at}</div>')
        if meta.start_time and meta.end_time:
            s = datetime.fromtimestamp(meta.start_time, tz=timezone.utc)
            e = datetime.fromtimestamp(meta.end_time, tz=timezone.utc)
            parts.append(
                f'<div class="meta">'
                f'{TimeRange.format_range(s, e, meta.language)}</div>'
            )
        parts.append('</div>')

        # الملخص
        if report.summary:
            parts.append('<div class="summary-grid">')
            for k, v in report.summary.items():
                parts.append('<div class="summary-card">')
                parts.append(f'<div class="label">{html.escape(str(k))}</div>')
                parts.append(f'<div class="value">{html.escape(str(v))}</div>')
                parts.append('</div>')
            parts.append('</div>')

        # الأقسام
        for section in report.sections:
            parts.append('<div class="section">')
            parts.append(f'<h2>{html.escape(section.title)}</h2>')

            content = section.content
            if section.section_type == "table":
                parts.append(cls._render_table(content))
            elif section.section_type == "chart":
                parts.append(cls._render_chart(content))
            elif section.section_type == "metrics":
                parts.append(cls._render_metrics(content))
            else:
                parts.append(cls._render_text(content))
            parts.append('</div>')

        # التذييل
        parts.append('<div class="footer">')
        parts.append(f'FalconEye v{meta.version} — '
                    f'{html.escape(meta.author)}')
        parts.append('</div>')

        parts.append('</div>')
        parts.append('</body>')
        parts.append('</html>')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return output_path

    @classmethod
    def _render_table(cls, content: Any) -> str:
        if not content:
            return '<p style="color:#888;">لا توجد بيانات</p>'
        if isinstance(content, dict):
            rows = list(content.items())
            if not rows:
                return '<p style="color:#888;">لا توجد بيانات</p>'
            parts = ['<table><thead><tr>',
                     '<th>البند</th><th>القيمة</th></tr></thead><tbody>']
            for k, v in rows:
                parts.append(f'<tr><td>{html.escape(str(k))}</td>'
                            f'<td>{html.escape(str(v))}</td></tr>')
            parts.append('</tbody></table>')
            return "".join(parts)
        if isinstance(content, list) and content and isinstance(content[0], dict):
            headers = list(content[0].keys())
            parts = ['<table><thead><tr>']
            for h in headers:
                parts.append(f'<th>{html.escape(str(h))}</th>')
            parts.append('</tr></thead><tbody>')
            for item in content[:500]:
                parts.append('<tr>')
                for h in headers:
                    val = item.get(h, "")
                    parts.append(f'<td>{html.escape(str(val))}</td>')
                parts.append('</tr>')
            parts.append('</tbody></table>')
            return "".join(parts)
        return f'<p>{html.escape(str(content))}</p>'

    @classmethod
    def _render_chart(cls, content: Dict[str, Any]) -> str:
        if not isinstance(content, dict) or not content:
            return '<p style="color:#888;">لا توجد بيانات</p>'
        items = sorted(content.items(), key=lambda x: x[1], reverse=True)[:15]
        max_val = max(v for _, v in items) if items else 1
        parts = []
        for label, value in items:
            try:
                pct = (float(value) / float(max_val)) * 100
            except (TypeError, ValueError):
                pct = 0
            parts.append('<div class="chart-bar">')
            parts.append(f'<div class="label">{html.escape(str(label))}</div>')
            parts.append(f'<div class="bar" style="width:{pct:.1f}%"></div>')
            parts.append(f'<div class="value">{html.escape(str(value))}</div>')
            parts.append('</div>')
        return "".join(parts)

    @classmethod
    def _render_metrics(cls, content: Any) -> str:
        if not isinstance(content, dict):
            return cls._render_text(content)
        parts = ['<table><thead><tr>',
                 '<th>المقياس</th><th>القيمة</th></tr></thead><tbody>']
        for k, v in content.items():
            if isinstance(v, dict):
                val = " | ".join(f"{kk}: {vv}" for kk, vv in v.items())
            else:
                val = v
            parts.append(f'<tr><td>{html.escape(str(k))}</td>'
                        f'<td>{html.escape(str(val))}</td></tr>')
        parts.append('</tbody></table>')
        return "".join(parts)

    @staticmethod
    def _render_text(content: Any) -> str:
        if isinstance(content, str):
            return f'<p>{html.escape(content)}</p>'
        return f'<pre>{html.escape(str(content))}</pre>'


# =====================================================================
# PART 107 — Markdown Exporter
# =====================================================================
class MarkdownExporter:
    """مصدّر Markdown"""

    @staticmethod
    def export(report: Report, output_path: str) -> str:
        lines = []
        lines.append(f"# {report.metadata.title}")
        lines.append("")
        meta = report.metadata
        lines.append(f"**تم الإنشاء:** "
                    f"{datetime.fromtimestamp(meta.generated_at).strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        if report.summary:
            lines.append("## الملخص")
            lines.append("")
            for k, v in report.summary.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            content = section.content
            if isinstance(content, dict):
                lines.append("| البند | القيمة |")
                lines.append("|-------|--------|")
                for k, v in content.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            elif isinstance(content, list) and content and isinstance(content[0], dict):
                headers = list(content[0].keys())
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for item in content[:200]:
                    lines.append("| " + " | ".join(
                        str(item.get(h, "")) for h in headers
                    ) + " |")
                lines.append("")
            else:
                lines.append(str(content))
                lines.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return output_path


# =====================================================================
# PART 108 — PDF Exporter (via HTML)
# =====================================================================
class PDFExporter:
    """مصدّر PDF — يحوّل HTML إلى PDF عبر wkhtmltopdf أو weasyprint"""

    @staticmethod
    def export(report: Report, output_path: str) -> str:
        # نحاول استخدام weasyprint
        try:
            from weasyprint import HTML
            tmp_html = output_path.replace(".pdf", "_tmp.html")
            HTMLExporter.export(report, tmp_html)
            HTML(filename=tmp_html).write_pdf(output_path)
            os.remove(tmp_html)
            return output_path
        except ImportError:
            pass

        # نحاول wkhtmltopdf
        try:
            import subprocess
            tmp_html = output_path.replace(".pdf", "_tmp.html")
            HTMLExporter.export(report, tmp_html)
            subprocess.run(
                ["wkhtmltopdf", tmp_html, output_path],
                check=True, capture_output=True,
            )
            os.remove(tmp_html)
            return output_path
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # fallback: HTML على الأقل
        html_path = output_path.replace(".pdf", ".html")
        HTMLExporter.export(report, html_path)
        return html_path


# =====================================================================
# PART 109 — Unified Export Manager
# =====================================================================
class ReportExporter:
    """مصدّر موحّد لكل الصيغ"""

    EXPORTERS = {
        ReportFormat.JSON: JSONExporter,
        ReportFormat.CSV: CSVExporter,
        ReportFormat.XML: XMLExporter,
        ReportFormat.HTML: HTMLExporter,
        ReportFormat.MARKDOWN: MarkdownExporter,
        ReportFormat.PDF: PDFExporter,
    }

    @classmethod
    def export(cls, report: Report, output_path: str,
               fmt: ReportFormat) -> str:
        exporter = cls.EXPORTERS.get(fmt)
        if exporter is None:
            raise ValueError(f"صيغة غير مدعومة: {fmt}")
        return exporter.export(report, output_path)

    @classmethod
    def export_multi(cls, report: Report, base_path: str,
                     formats: List[ReportFormat]) -> Dict[str, str]:
        """تصدير نفس التقرير بصيغ متعددة"""
        results = {}
        for fmt in formats:
            ext = fmt.value
            path = f"{base_path}.{ext}"
            try:
                results[fmt.value] = cls.export(report, path, fmt)
            except Exception as e:
                results[fmt.value] = f"error: {e}"
        return results


# =====================================================================
# PART 110 — Heatmap Generator (Advanced)
# =====================================================================
class AdvancedHeatmap:
    """خريطة حرارية متقدمة مع عدة أنواع"""

    def __init__(self, width: int = 1280, height: int = 720,
                 decay: float = 0.995, blur: int = 41):
        self.width = width
        self.height = height
        self.decay = decay
        self.blur = blur
        # خرائط متعددة
        self.map_all = np.zeros((height, width), dtype=np.float32)
        self.map_threat = np.zeros((height, width), dtype=np.float32)
        self.map_motion = np.zeros((height, width), dtype=np.float32)
        self.map_dwell = np.zeros((height, width), dtype=np.float32)
        self._lock = threading.RLock()
        self._frame_count = 0

    def update(self, tracks: List[Any]) -> None:
        with self._lock:
            self._frame_count += 1
            self.map_all *= self.decay
            self.map_threat *= self.decay
            self.map_motion *= self.decay
            self.map_dwell *= self.decay

            for track in tracks:
                cx, cy = self._track_center(track)
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    continue
                ix, iy = int(cx), int(cy)

                # الخريطة العامة
                self.map_all[iy, ix] += 1.0

                # خريطة التهديدات
                score = getattr(track, "suspicion", 0.0)
                if score > 0:
                    self.map_threat[iy, ix] += score / 100.0

                # خريطة الحركة
                speed = self._track_speed(track)
                if speed > 20:
                    self.map_motion[iy, ix] += min(1.0, speed / 500.0)

                # خريطة المكوث
                dwell = getattr(track, "dwell", 0.0)
                if dwell > 0:
                    self.map_dwell[iy, ix] += min(1.0, dwell / 10.0)

    @staticmethod
    def _track_center(track: Any) -> Tuple[float, float]:
        bbox = getattr(track, "bbox", (0, 0, 0, 0))
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _track_speed(track: Any) -> float:
        if hasattr(track, "get_speed"):
            try:
                return track.get_speed()
            except Exception:
                return 0.0
        return 0.0

    def render(self, frame: np.ndarray,
               mode: str = "all", alpha: float = 0.45) -> np.ndarray:
        if not HAVE_CV2:
            return frame
        with self._lock:
            source = {
                "all": self.map_all,
                "threat": self.map_threat,
                "motion": self.map_motion,
                "dwell": self.map_dwell,
            }.get(mode, self.map_all).copy()

        if source.max() < 0.01:
            return frame

        norm = cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.blur > 0:
            norm = cv2.GaussianBlur(norm, (self.blur, self.blur), 0)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)

    def get_peaks(self, mode: str = "all", top_n: int = 10,
                  min_dist: int = 60) -> List[Tuple[int, int, float]]:
        with self._lock:
            source = {
                "all": self.map_all,
                "threat": self.map_threat,
                "motion": self.map_motion,
                "dwell": self.map_dwell,
            }.get(mode, self.map_all).copy()

        peaks = []
        for _ in range(top_n):
            if source.max() < 0.5:
                break
            idx = np.unravel_index(np.argmax(source), source.shape)
            cy, cx = int(idx[0]), int(idx[1])
            val = float(source[cy, cx])
            peaks.append((cx, cy, val))
            y1 = max(0, cy - min_dist)
            y2 = min(self.height, cy + min_dist)
            x1 = max(0, cx - min_dist)
            x2 = min(self.width, cx + min_dist)
            source[y1:y2, x1:x2] = 0
        return peaks

    def to_grayscale(self, mode: str = "all") -> np.ndarray:
        with self._lock:
            source = {
                "all": self.map_all,
                "threat": self.map_threat,
                "motion": self.map_motion,
                "dwell": self.map_dwell,
            }.get(mode, self.map_all).copy()
        if source.max() < 0.01:
            return np.zeros_like(source, dtype=np.uint8)
        return cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def reset(self, mode: Optional[str] = None) -> None:
        with self._lock:
            if mode is None:
                self.map_all.fill(0)
                self.map_threat.fill(0)
                self.map_motion.fill(0)
                self.map_dwell.fill(0)
            else:
                m = {
                    "all": self.map_all,
                    "threat": self.map_threat,
                    "motion": self.map_motion,
                    "dwell": self.map_dwell,
                }.get(mode)
                if m is not None:
                    m.fill(0)

    def export_png(self, output_path: str,
                   mode: str = "all") -> Optional[str]:
        if not HAVE_CV2:
            return None
        gray = self.to_grayscale(mode)
        if gray.max() < 1:
            return None
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, colored)
        return output_path


# =====================================================================
# PART 111 — Trend Analyzer
# =====================================================================
class TrendAnalyzer:
    """تحليل الاتجاهات"""

    def __init__(self, db):
        self.db = db

    def hourly_trend(self, days: int = 7) -> Dict[str, Any]:
        """اتجاه الأحداث على مدار الساعات"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.db.query_events(since=since, limit=100_000)
        buckets = defaultdict(int)
        for e in events:
            ts = e.get("timestamp", "")
            if len(ts) >= 13:
                buckets[ts[:13]] += 1
        if not buckets:
            return {"trend": "stable", "data": {}}
        values = list(buckets.values())
        trend = self._classify_trend(values)
        return {
            "trend": trend,
            "data": dict(sorted(buckets.items())),
            "avg": round(statistics.mean(values), 2),
            "max": max(values),
            "min": min(values),
        }

    def threat_trend(self, days: int = 30) -> Dict[str, Any]:
        """اتجاه التهديدات"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.db.query_events(since=since, limit=100_000)
        daily = defaultdict(int)
        for e in events:
            ts = e.get("timestamp", "")
            if len(ts) >= 10:
                daily[ts[:10]] += 1
        if not daily:
            return {"trend": "stable", "data": {}}
        values = [daily[k] for k in sorted(daily.keys())]
        return {
            "trend": self._classify_trend(values),
            "data": dict(sorted(daily.items())),
            "avg": round(statistics.mean(values), 2),
        }

    def compare_periods(
        self,
        period1: Tuple[datetime, datetime],
        period2: Tuple[datetime, datetime],
    ) -> Dict[str, Any]:
        """مقارنة فترتين"""
        s1, e1 = period1
        s2, e2 = period2
        c1 = self.db.count_events() if not hasattr(self.db, "query_events") else len(
            self.db.query_events(since=s1, until=e1, limit=100_000)
        )
        c2 = len(self.db.query_events(since=s2, until=e2, limit=100_000))
        change = ((c2 - c1) / max(1, c1)) * 100 if c1 > 0 else 0
        return {
            "period1_count": c1,
            "period2_count": c2,
            "change_percent": round(change, 1),
            "direction": "up" if change > 5 else ("down" if change < -5 else "stable"),
        }

    @staticmethod
    def _classify_trend(values: List[float]) -> str:
        if len(values) < 3:
            return "stable"
        n = len(values)
        first = sum(values[:n // 3]) / max(1, n // 3)
        last = sum(values[-n // 3:]) / max(1, n // 3)
        if last > first * 1.15:
            return "increasing"
        if last < first * 0.85:
            return "decreasing"
        return "stable"


# =====================================================================
# PART 112 — Scheduled Report Manager
# =====================================================================
class ScheduledReport:
    """تقرير مجدول"""

    def __init__(
        self,
        name: str,
        report_type: ReportType,
        period: ReportPeriod,
        interval_hours: float = 24.0,
        formats: Optional[List[ReportFormat]] = None,
        recipients: Optional[List[str]] = None,
        output_dir: str = "reports",
    ):
        self.name = name
        self.report_type = report_type
        self.period = period
        self.interval_hours = interval_hours
        self.formats = formats or [ReportFormat.PDF, ReportFormat.JSON]
        self.recipients = recipients or []
        self.output_dir = output_dir
        self.last_run: Optional[float] = None
        self.enabled = True


class ScheduledReportManager:
    """مدير التقارير المجدولة"""

    def __init__(self, builder: ReportBuilder,
                 notifier: Optional[Any] = None):
        self.builder = builder
        self.notifier = notifier
        self.schedules: List[ScheduledReport] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.history: List[Dict[str, Any]] = []

    def add(self, sched: ScheduledReport) -> None:
        with self._lock:
            self.schedules.append(sched)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="report-scheduler", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_and_run()
            except Exception:
                pass
            self._stop.wait(60.0)  # فحص كل دقيقة

    def _check_and_run(self) -> None:
        now = time.time()
        with self._lock:
            schedules = list(self.schedules)
        for sched in schedules:
            if not sched.enabled:
                continue
            if sched.last_run is None:
                should_run = True
            else:
                elapsed_hours = (now - sched.last_run) / 3600
                should_run = elapsed_hours >= sched.interval_hours
            if should_run:
                try:
                    self._run_schedule(sched)
                    sched.last_run = now
                except Exception as e:
                    self.history.append({
                        "schedule": sched.name,
                        "time": now,
                        "status": "error",
                        "error": str(e),
                    })

    def _run_schedule(self, sched: ScheduledReport) -> None:
        os.makedirs(sched.output_dir, exist_ok=True)
        report = self._build_report(sched.report_type, sched.period)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(
            sched.output_dir,
            f"{sched.name}_{timestamp}"
        )
        exported = {}
        for fmt in sched.formats:
            try:
                path = f"{base_path}.{fmt.value}"
                ReportExporter.export(report, path, fmt)
                exported[fmt.value] = path
            except Exception as e:
                exported[fmt.value] = f"error: {e}"

        # إشعار
        if self.notifier and sched.recipients:
            try:
                self.notifier.notify(
                    subject=f"تقرير مجدول: {sched.name}",
                    body=f"تم إنشاء التقرير بنجاح.\n"
                         f"الملفات: {', '.join(exported.values())}",
                )
            except Exception:
                pass

        self.history.append({
            "schedule": sched.name,
            "time": time.time(),
            "status": "success",
            "files": exported,
        })

    def _build_report(self, rtype: ReportType,
                      period: ReportPeriod) -> Report:
        if rtype == ReportType.SUMMARY:
            return self.builder.build_summary_report(period)
        if rtype == ReportType.DETAILED:
            return self.builder.build_detailed_report(period)
        if rtype == ReportType.THREAT_ANALYSIS:
            return self.builder.build_threat_analysis(period)
        if rtype == ReportType.PERFORMANCE:
            return self.builder.build_performance_report(period)
        return self.builder.build_summary_report(period)

    def run_now(self, name: str) -> bool:
        """تشغيل تقرير يدوياً"""
        with self._lock:
            sched = next((s for s in self.schedules if s.name == name), None)
        if sched is None:
            return False
        try:
            self._run_schedule(sched)
            sched.last_run = time.time()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# =====================================================================
# PART 113 — Report Archive
# =====================================================================
class ReportArchive:
    """أرشيف التقارير"""

    def __init__(self, archive_dir: str = "reports"):
        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, report: Report, formats: List[ReportFormat]
             ) -> Dict[str, str]:
        """حفظ التقرير بأرشيف منظم"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(
            self.archive_dir,
            datetime.now().strftime("%Y-%m"),
        )
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, f"{report.metadata.report_type.value}_{timestamp}")
        return ReportExporter.export_multi(report, base, formats)

    def list_reports(self) -> List[Dict[str, Any]]:
        """قائمة كل التقارير المحفوظة"""
        reports = []
        for root, _, files in os.walk(self.archive_dir):
            for f in files:
                path = os.path.join(root, f)
                try:
                    stat = os.stat(path)
                    reports.append({
                        "path": path,
                        "name": f,
                        "size_kb": round(stat.st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                    })
                except Exception:
                    pass
        reports.sort(key=lambda x: x["modified"], reverse=True)
        return reports

    def create_zip(self, output_path: str,
                   reports: Optional[List[Dict[str, Any]]] = None
                   ) -> str:
        """ضغط التقارير في ملف ZIP"""
        if reports is None:
            reports = self.list_reports()
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in reports:
                try:
                    zf.write(r["path"], arcname=os.path.basename(r["path"]))
                except Exception:
                    pass
        return output_path

    def cleanup_old(self, days: int = 90) -> int:
        """حذف التقارير القديمة"""
        cutoff = time.time() - (days * 86400)
        deleted = 0
        for root, _, files in os.walk(self.archive_dir):
            for f in files:
                path = os.path.join(root, f)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        deleted += 1
                except Exception:
                    pass
        return deleted


# =====================================================================
# PART 114 — Analytics Engine
# =====================================================================
class AnalyticsEngine:
    """محرك تحليلات متقدم"""

    def __init__(self, db):
        self.db = db
        self.aggregator = DataAggregator(db)
        self.trend = TrendAnalyzer(db)

    def full_analysis(self, days: int = 7) -> Dict[str, Any]:
        """تحليل شامل"""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        agg = self.aggregator.aggregate_events(start, end)
        behavior = self.aggregator.aggregate_behavior(start, end)
        performance = self.aggregator.aggregate_performance(start, end)
        trend = self.trend.hourly_trend(days)

        return {
            "period_days": days,
            "events": agg,
            "behavior": behavior,
            "performance": performance,
            "trend": trend,
            "insights": self._generate_insights(agg, behavior, trend),
        }

    def _generate_insights(self, agg: Dict[str, Any],
                           behavior: Dict[str, Any],
                           trend: Dict[str, Any]) -> List[str]:
        """توليد رؤى تلقائية"""
        insights = []
        total = agg.get("total_events", 0)
        if total == 0:
            insights.append("لا توجد أحداث في الفترة المحددة")
            return insights

        # أكثر تهديد
        by_type = agg.get("by_type", {})
        if by_type:
            top_type, top_count = max(by_type.items(), key=lambda x: x[1])
            pct = (top_count / total) * 100
            insights.append(
                f"أكثر تهديد: {top_type} ({pct:.1f}% من الأحداث)"
            )

        # الاتجاه
        trend_type = trend.get("trend", "stable")
        if trend_type == "increasing":
            insights.append("⚠️ اتجاه تصاعدي في الأحداث خلال الفترة الأخيرة")
        elif trend_type == "decreasing":
            insights.append("✅ اتجاه تنازلي في الأحداث — تحسّن ملحوظ")

        # أعلى كاميرا
        by_cam = agg.get("by_camera", {})
        if by_cam:
            top_cam, top_count = max(by_cam.items(), key=lambda x: x[1])
            insights.append(f"أكثر كاميرا نشاطاً: {top_cam} ({top_count})")

        # أعلى سلوك
        if behavior:
            top_behavior = max(behavior.items(), key=lambda x: x[1])
            insights.append(
                f"السلوك الأكثر تكراراً: {top_behavior[0]} "
                f"({top_behavior[1]} مرة)"
            )

        return insights

    def top_threats(self, days: int = 7, n: int = 10) -> List[Dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.db.query_events(since=since, limit=10_000)
        events.sort(key=lambda e: e.get("threat_level", 0), reverse=True)
        return events[:n]

    def hourly_histogram(self, days: int = 7) -> Dict[int, int]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.db.query_events(since=since, limit=100_000)
        hist = defaultdict(int)
        for e in events:
            try:
                ts = e.get("timestamp", "")
                if "T" in ts:
                    hour = int(ts.split("T")[1][:2])
                    hist[hour] += 1
            except Exception:
                pass
        return {h: hist.get(h, 0) for h in range(24)}


# =====================================================================
# PART 115 — Report CLI
# =====================================================================
class ReportCLI:
    """واجهة سطرية للتقارير"""

    @staticmethod
    def print_available_reports() -> None:
        print("""
╔══════════════════════════════════════════════════╗
║  التقارير المتاحة:                              ║
╠══════════════════════════════════════════════════╣
║  1. summary         تقرير موجز                  ║
║  2. detailed        تقرير مفصل                  ║
║  3. threat          تحليل التهديدات             ║
║  4. performance     تقرير الأداء                ║
║  5. audit           سجل المراجعة                ║
║  6. evidence        قائمة الأدلة                ║
╚══════════════════════════════════════════════════╝

الصيغ المتاحة: json, csv, html, pdf, xml, markdown
    """)

    @staticmethod
    def print_report_summary(report: Report) -> None:
        meta = report.metadata
        print(f"""
╔══════════════════════════════════════════════════╗
║  {meta.title}
╠══════════════════════════════════════════════════╣
║  النوع:      {meta.report_type.value}
║  الفترة:     {meta.period.value}
║  اللغة:      {meta.language}
║  الأقسام:    {len(report.sections)}
║  الملخص:     {len(report.summary)} عناصر
╚══════════════════════════════════════════════════╝
        """)
        if report.summary:
            print("الملخص:")
            for k, v in report.summary.items():
                print(f"  • {k}: {v}")


# =====================================================================
# PART 116 — Reports Factory
# =====================================================================
class ReportsFactory:
    """مصنع كل مكونات التقارير"""

    @staticmethod
    def build(db, language: str = "ar",
              archive_dir: str = "reports",
              notifier: Optional[Any] = None) -> Dict[str, Any]:
        builder = ReportBuilder(db, language)
        archive = ReportArchive(archive_dir)
        analytics = AnalyticsEngine(db)
        trend = TrendAnalyzer(db)
        heatmap = None
        if HAVE_CV2:
            heatmap = AdvancedHeatmap()
        scheduler = ScheduledReportManager(builder, notifier)

        return {
            "builder": builder,
            "archive": archive,
            "analytics": analytics,
            "trend": trend,
            "heatmap": heatmap,
            "scheduler": scheduler,
            "exporter": ReportExporter,
        }

    @staticmethod
    def default_schedules(builder: ReportBuilder) -> List[ScheduledReport]:
        """جداول افتراضية"""
        return [
            ScheduledReport(
                name="daily_summary",
                report_type=ReportType.SUMMARY,
                period=ReportPeriod.DAILY,
                interval_hours=24.0,
                formats=[ReportFormat.PDF, ReportFormat.JSON],
            ),
            ScheduledReport(
                name="weekly_threats",
                report_type=ReportType.THREAT_ANALYSIS,
                period=ReportPeriod.WEEKLY,
                interval_hours=168.0,
                formats=[ReportFormat.PDF, ReportFormat.HTML],
            ),
            ScheduledReport(
                name="monthly_performance",
                report_type=ReportType.PERFORMANCE,
                period=ReportPeriod.MONTHLY,
                interval_hours=720.0,
                formats=[ReportFormat.PDF],
            ),
        ]


# =====================================================================
# PART 117 — Full Integration Example
# =====================================================================
def integrate_reports(cfg, db, notifier=None) -> Dict[str, Any]:
    """تكامل كامل للتقارير مع النظام"""
    lang = getattr(cfg, "ui", None)
    language = lang.ui_language if lang else "ar"

    components = ReportsFactory.build(
        db=db,
        language=language,
        archive_dir="reports",
        notifier=notifier,
    )

    # إضافة الجداول الافتراضية
    for sched in ReportsFactory.default_schedules(components["builder"]):
        components["scheduler"].add(sched)

    return components


# =====================================================================
# PART 118 — Reports Exports
# =====================================================================
__all__ = [
    # Constants
    "REPORTS_VERSION",
    # Enums
    "ReportFormat", "ReportPeriod", "ReportType",
    # Data
    "ReportMetadata", "ReportSection", "Report",
    # Helpers
    "TimeRange", "DataAggregator",
    # Builder
    "ReportBuilder",
    # Exporters
    "JSONExporter", "CSVExporter", "XMLExporter",
    "HTMLExporter", "MarkdownExporter", "PDFExporter",
    "ReportExporter",
    # Heatmap
    "AdvancedHeatmap",
    # Trends
    "TrendAnalyzer",
    # Schedules
    "ScheduledReport", "ScheduledReportManager",
    # Archive
    "ReportArchive",
    # Analytics
    "AnalyticsEngine",
    # CLI
    "ReportCLI",
    # Factory
    "ReportsFactory", "integrate_reports",
]
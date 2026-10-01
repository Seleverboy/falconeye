# -*- coding: utf-8 -*-
"""FalconEye v13.0 — المشغّل الرئيسي"""
import sys
import time
import logging
import cv2
import numpy as np

from falconeye import (
    SystemConfig, Database, StatisticsEngine,
    EvidenceManager, APIServer, SystemMonitor,
    setup_logging,
)
from falconeye_core import (
    YOLOWorldDetector, PoseDetector, DetectionFusion,
    TrackManager, FrameProcessor, select_device,
    Zone, ZoneManager,
)
from falconeye_behavior import BehaviorPipeline
from falconeye_ui import UIFactory, HUDStats
from falconeye_reports import ReportBuilder, ReportFormat, ReportExporter

logger = logging.getLogger("falconeye")

TOOL_THREATS = {
    "gun": 95, "pistol": 95, "rifle": 95, "handgun": 95, "revolver": 95,
    "knife": 85, "machete": 90, "axe": 85, "hammer": 75,
    "baseball bat": 70, "sword": 90, "dagger": 85, "box cutter": 75,
    "grenade": 100, "bomb": 100, "explosive": 100, "dynamite": 100,
    "camera": 65, "dslr": 65, "video camera": 65,
    "cell phone": 55, "smartphone": 55, "laptop": 50, "tablet": 50,
    "drone": 70, "quadcopter": 70,
    "mask": 60, "balaclava": 60, "ski mask": 60,
}


def apply_tool_threats(tracker):
    for track in tracker.tracks.values():
        score = TOOL_THREATS.get(track.stable_label, 0)
        if score <= 0:
            continue
        track.suspicion = float(score)
        if score >= 85:
            track.status = "alarm"
            track.reason = f"تهديد خطير: {track.stable_label}"
        elif score >= 60:
            track.status = "warning"
            track.reason = f"أداة مشبوهة: {track.stable_label}"
        else:
            track.status = "warning"
            track.reason = track.stable_label


def main():
    cfg = SystemConfig.load("falconeye_config.json")
    setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("FalconEye v13.0 — بدء التشغيل")
    logger.info("=" * 60)

    db = Database(cfg.database)
    stats_engine = StatisticsEngine(db)
    evidence = EvidenceManager(cfg.evidence)
    api_server = APIServer(cfg.api, db, stats_engine)
    api_server.start()
    monitor = SystemMonitor(db, interval=30.0)
    monitor.start()

    USE_STATIC_IMAGE = True

    if USE_STATIC_IMAGE:
        frame_source = cv2.imread("image.png")
        if frame_source is None:
            logger.error("لم أجد image.png")
            return
        logger.info("وضع الصورة الثابتة — %s", frame_source.shape)
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("لا يمكن فتح الكاميرا")
            return
        logger.info("وضع الكاميرا")

    device = select_device()
    logger.info("الجهاز: %s", device)

    world = YOLOWorldDetector(
        model_path="yolov8s-world.pt",
        conf=0.25, imgsz=640, device=device,
    )
    pose = PoseDetector(
        model_path="yolov8n-pose.pt",
        device=device, enabled=False,
    )
    fusion = DetectionFusion(iou_threshold=0.55)
    tracker = TrackManager(
        max_coast=3, min_hits_confirm=2,
        iou_threshold=0.20, assoc_max_cost=0.55,
    )
    processor = FrameProcessor(
        detector_world=world,
        detector_pose=pose,
        detector_face=None,
        track_manager=tracker,
        fusion=fusion,
        enable_pose=False,
        enable_face=False,
    )

    behavior = BehaviorPipeline({
        "warn_dwell": cfg.alarms.warn_dwell,
        "alarm_dwell": cfg.alarms.alarm_dwell,
        "run_speed": cfg.alarms.run_speed,
        "score_warn": cfg.alarms.score_warn,
        "score_alarm": cfg.alarms.score_alarm,
    })

    ui = UIFactory.build_default(
        font_path="arial.ttf",
        language=cfg.ui.ui_language,
    )

    zones = ZoneManager()
    zones.add_zone(Zone(name="main_zone", inner=(0.08, 0.08, 0.92, 0.92)))

    report_builder = ReportBuilder(db, cfg.ui.ui_language)

    logger.info("=" * 60)
    logger.info("النظام جاهز — اضغط q للخروج، s لحفظ لقطة")
    logger.info("=" * 60)

    fps = 0.0
    frame_count = 0
    prev_time = time.time()

    try:
        while True:
            if USE_STATIC_IMAGE:
                frame = frame_source.copy()
                time.sleep(0.05)
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            frame_count += 1
            processor.process(frame)

            try:
                behavior_result = behavior.process(
                    list(tracker.get_active_tracks()),
                    [d for d in world.detect(frame)],
                    list(zones.zones.values()),
                    frame.shape,
                )
            except Exception as e:
                logger.debug("Behavior error: %s", e)
                behavior_result = {"events": [], "stats": {"total_alarms": 0}}

            apply_tool_threats(tracker)
            # الشخص داخل المنطقة → تحذير
            for t in tracker.tracks.values():
                if not t.is_person:
                    continue
                cx = (t.bbox[0] + t.bbox[2]) / 2
                cy = (t.bbox[1] + t.bbox[3]) / 2
                H, W = frame.shape[:2]
                zx1, zy1, zx2, zy2 = (
                    int(W * 0.30), int(H * 0.25),
                    int(W * 0.70), int(H * 0.80),
                )
                if zx1 < cx < zx2 and zy1 < cy < zy2:
                    if t.suspicion < 35:
                        t.suspicion = 35.0
                        t.status = "warning"
                        t.reason = "داخل المنطقة المحظورة"
            now = time.time()
            if now - prev_time >= 0.5:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # عداد المشتبه بهم
            n_alarm_tracks = 0
            for t in tracker.tracks.values():
                status_str = str(getattr(t, "status", "normal"))
                if status_str in ("alarm", "warning") and t.suspicion > 0:
                    n_alarm_tracks += 1

            any_alarm = n_alarm_tracks > 0

            # حفظ الأدلة
            for t in tracker.get_active_tracks():
                if t.suspicion > 50 and t.last_snapshot < now - 15:
                    evidence.save_image(frame, "threat", f"ID{t.id}")
                    t.last_snapshot = now
                    logger.info("ALARM: ID %d — %.0f%% — %s",
                                t.id, t.suspicion, t.reason)

            n_people = sum(1 for t in tracker.tracks.values() if t.is_person)
            hud_stats = HUDStats(
                fps=fps,
                people=n_people,
                tracks=len(tracker.tracks),
                suspicious=n_alarm_tracks,
                evidence=evidence.count_files(),
                mode="World",
                tracker="Hungarian",
                device=device.upper(),
                threat_level="حرج" if any_alarm else "منخفض",
                threat_color=(0, 0, 255) if any_alarm else (0, 220, 0),
            )

            frame = ui["compositor"].compose(
                frame=frame,
                tracks=list(tracker.get_active_tracks()),
                zones=list(zones.zones.values()),
                stats=hud_stats,
                any_alarm=any_alarm,
            )

            cv2.imshow("FalconEye v13", frame)

            twin = ui["twin"].render(
                list(tracker.get_active_tracks()),
                frame.shape[1], frame.shape[0],
            )
            cv2.imshow("Digital Twin", twin)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                logger.info("إعادة تعيين")
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"screenshot_{ts}.jpg"
                cv2.imwrite(path, frame)
                logger.info("لقطة محفوظة: %s", path)
            elif key == ord('e'):
                report = report_builder.build_summary_report()
                ReportExporter.export_multi(
                    report,
                    base_path="falconeye_report",
                    formats=[ReportFormat.HTML, ReportFormat.JSON],
                )

    except KeyboardInterrupt:
        logger.info("إيقاف بواسطة المستخدم")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        api_server.stop()
        monitor.stop()
        db.close()
        logger.info("=== تم إيقاف FalconEye ===")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""FalconEye v13 — اختبار شامل"""
import sys, time

print("=" * 60)
print("FalconEye v13.0 — اختبار شامل")
print("=" * 60)

# 1) الإعدادات
print("\n[1/5] الإعدادات...")
try:
    from falconeye import SystemConfig, Database, StatisticsEngine
    cfg = SystemConfig.load("falconeye_config.json")
    db = Database(cfg.database)
    stats = StatisticsEngine(db)
    print(f"   ✅ جاهز — {len(cfg.cameras)} كاميرا")
except Exception as e:
    print(f"   ❌ {e}")

# 2) السلوك
print("\n[2/5] التحليل السلوكي...")
try:
    from falconeye_behavior import BehaviorPipeline
    bp = BehaviorPipeline()
    print(f"   ✅ جاهز — {len(bp.rules_engine.rules)} قاعدة")
except Exception as e:
    print(f"   ❌ {e}")

# 3) الواجهة
print("\n[3/5] الواجهة...")
try:
    from falconeye_ui import UIFactory
    ui = UIFactory.build_default(language="ar")
    print(f"   ✅ جاهزة — {len(ui)} مكوّن")
except Exception as e:
    print(f"   ❌ {e}")

# 4) الويب
print("\n[4/5] لوحة الويب...")
try:
    from falconeye_web import WebServer
    server = WebServer(host="127.0.0.1", port=8080)
    print(f"   ✅ جاهز — {server.url}")
except Exception as e:
    print(f"   ❌ {e}")

# 5) التقارير
print("\n[5/5] التقارير...")
try:
    from falconeye_reports import ReportBuilder
    builder = ReportBuilder(db, "ar")
    report = builder.build_summary_report()
    print(f"   ✅ جاهز — {len(report.sections)} قسم")
except Exception as e:
    print(f"   ❌ {e}")

print("\n" + "=" * 60)
print("✅ اكتمل الاختبار!")
print("=" * 60)
db.close()
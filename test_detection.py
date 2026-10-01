# test_detection.py
import cv2
import sys
import os

print("=" * 60)
print("FalconEye — اختبار الكشف")
print("=" * 60)

# 1) تحقق من الصورة
if not os.path.exists("image.png"):
    print("[خطأ] لم أجد image.png في المجلد")
    print("انسخ صورة الشخص مع الهاتف إلى المجلد")
    sys.exit(1)

img = cv2.imread("image.png")
print(f"[OK] حجم الصورة: {img.shape}")

# 2) حمّل الموديل
print("\n[1/3] تحميل YOLO-World...")
print("      ملاحظة: سيُحمَّل الملف ~150MB في المرة الأولى")
from falconeye_core import YOLOWorldDetector

det = YOLOWorldDetector(
    model_path="yolov8s-world.pt",
    conf=0.25,
    imgsz=640,
    device="cpu",
)
print("[OK] تم التحميل")

# 3) شغّل الكشف
print("\n[2/3] تشغيل الكشف...")
results = det.detect(img)
print(f"[OK] اكتُشف {len(results)} هدف")

# 4) اعرض النتائج
print("\n[3/3] النتائج:")
for d in results:
    print(f"   • {d.label:20s} ثقة: {d.confidence:.2f}")

# 5) ارسم النتائج
# 5) ارسم النتائج بأسلوب احترافي
print("\n[...] رسم النتائج...")
import numpy as np

# ألوان حسب الفئة (BGR)
COLORS = {
    "person": (0, 220, 0),          # أخضر
    "camera": (0, 0, 255),          # أحمر
    "cell phone": (0, 0, 255),      # أحمر
    "dslr": (0, 0, 255),            # أحمر
    "backpack": (255, 128, 0),      # أزرق
    "handbag": (255, 128, 0),
    "knife": (0, 0, 255),
    "gun": (0, 0, 255),
}

# رسم الصناديق
for d in results:
    x1, y1, x2, y2 = d.bbox
    color = COLORS.get(d.label, (0, 255, 255))  # أصفر للباقي
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

# تجميع التسميات لمنع التراكب
labels = []
for d in results:
    labels.append((d.label, d.confidence, d.bbox, COLORS.get(d.label, (0, 255, 255))))

# ترتيب حسب y (الأعلى أولاً)
labels.sort(key=lambda x: x[2][1])

# رسم كل تسمية في موضع منفصل
y_offset = 0
last_x = -999
for label, conf, bbox, color in labels:
    x1, y1 = bbox[0], bbox[1]
    text = f"{label} {conf:.2f}"
    
    # قياس النص
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # موضع التسمية مع منع التراكب
    # إذا كانت التسمية قريبة من السابقة، أزحها لأسفل
    if abs(x1 - last_x) < 200:
        y_offset += 28
    else:
        y_offset = 0
    last_x = x1
    
    ty = max(30, y1 - 10 - y_offset)    
    # خلفية سوداء للتسمية
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), (0, 0, 0), -1)
    
    # النص
    cv2.putText(img, text, (x1 + 4, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # خط يشير من التسمية إلى الصندوق
    cv2.line(img, (x1 + 4, ty), (x1 + 4, y1), color, 1)
    
    y_offset += 25

# إضافة شعار في الأعلى
cv2.rectangle(img, (0, 0), (img.shape[1], 60), (30, 30, 35), -1)
cv2.putText(img, "FalconEye Detection", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
cv2.putText(img, f"Objects: {len(results)}", 
            (img.shape[1] - 220, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imwrite("test_detection_output.jpg", img)
print("[OK] تم حفظ: test_detection_output.jpg")
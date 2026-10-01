# fix_twin.py — إصلاح النصوص العربية في Digital Twin
with open("falconeye_ui.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1) شريط العنوان
old1 = '''        cv2.putText(canvas, title, (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 240, 255), 1)'''
new1 = '''        ArabicRenderer.draw(canvas, title, (8, 4),
                            (120, 240, 255), 14, self.font_path)'''

# 2) الـ legend (طبيعي/تحذير/إنذار)
old2 = '''            cv2.putText(canvas, name, (x + 10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)'''
new2 = '''            ArabicRenderer.draw(canvas, name, (x + 10, H - 22),
                                (200, 200, 200), 13, self.font_path)'''

# 3) نص التتبعات (أسفل يمين)
old3 = '''        cv2.putText(canvas, trk_lbl, (W - 100, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)'''
new3 = '''        ArabicRenderer.draw(canvas, trk_lbl, (W - 110, H - 22),
                            (180, 180, 180), 13, self.font_path)'''

# 4) ملصقات IDs داخل المسارات
old4 = '''            cv2.putText(canvas, lbl, (mx + 10, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)'''
new4 = '''            ArabicRenderer.draw(canvas, lbl, (mx + 10, my - 16),
                                color, 13, self.font_path)'''

# 5) أسماء الغرف (غ1، غ2...)
old5 = '''            cv2.putText(canvas, tag, (x1 + 4, y1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)'''
new5 = '''            ArabicRenderer.draw(canvas, tag, (x1 + 4, y1 + 2),
                                (10, 10, 10), 13, self.font_path)'''

count = 0
for old, new in [(old1, new1), (old2, new2), (old3, new3), (old4, new4), (old5, new5)]:
    if old in content:
        content = content.replace(old, new)
        count += 1

with open("falconeye_ui.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"[OK] تم إصلاح {count} موضع")
print("[الآن] شغّل: python run.py")
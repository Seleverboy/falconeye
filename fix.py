# fix.py
import re

# اقرأ الملف
with open("falconeye_core.py", "r", encoding="utf-8") as f:
    content = f.read()

# ابحث عن المقطع الخاطئ
# النمط: مسافات خاطئة قبل self.hits + سطران مكرران
pattern = re.compile(
    r"(self\.last_seen = time\.time\(\)\n)"
    r"\s+self\.hits \+= 1\n"
    r"(\s+self\.misses = 0\n)"
    r"(\s+self\.age_frames \+= 1\n)"
    r"\s*\n"
    r"\s*if self\.state == TrackState\.TENTATIVE and self\.hits >= 2:\n"
    r"\s+self\.state = TrackState\.CONFIRMED\n"
    r"\s+self\.misses = 0\n"
    r"\s+self\.age_frames \+= 1\n",
    re.MULTILINE
)

# الإصلاح: 8 مسافات فقط
replacement = (
    "self.last_seen = time.time()\n"
    "        self.hits += 1\n"
    "        self.misses = 0\n"
    "        self.age_frames += 1\n"
    "\n"
    "        if self.state == TrackState.TENTATIVE and self.hits >= 2:\n"
    "            self.state = TrackState.CONFIRMED\n"
)

new_content, count = pattern.subn(replacement, content)

if count == 0:
    print("[!] لم يتم العثور على النمط — ربما أُصلح مسبقاً")
else:
    with open("falconeye_core.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[OK] تم إصلاح {count} موضع")

print("[تم] يمكنك الآن تشغيل: python run.py")
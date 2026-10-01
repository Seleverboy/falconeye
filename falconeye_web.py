# =====================================================================
# PART 85 — Web Constants & Template Strings
# =====================================================================
import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import queue
import secrets
import socket
import struct
import threading
import time
import urllib.parse
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


WEB_VERSION = "1.0.0"


# =====================================================================
# PART 86 — WebSocket Protocol Implementation
# =====================================================================
class WebSocketOpcode:
    """أكواد عمليات WebSocket"""
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA


class WebSocketFrame:
    """إطار WebSocket واحد"""

    def __init__(self, opcode: int, payload: bytes,
                 fin: bool = True, mask: bool = False,
                 mask_key: bytes = b""):
        self.opcode = opcode
        self.payload = payload
        self.fin = fin
        self.mask = mask
        self.mask_key = mask_key

    def to_bytes(self) -> bytes:
        header = bytearray()
        header.append((0x80 if self.fin else 0x00) | self.opcode)
        length = len(self.payload)
        mask_bit = 0x80 if self.mask else 0x00
        if length < 126:
            header.append(mask_bit | length)
        elif length < 65536:
            header.append(mask_bit | 126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(mask_bit | 127)
            header.extend(struct.pack(">Q", length))
        if self.mask:
            header.extend(self.mask_key)
            masked = bytes(b ^ self.mask_key[i % 4]
                          for i, b in enumerate(self.payload))
            return bytes(header) + masked
        return bytes(header) + self.payload

    @classmethod
    def parse(cls, data: bytes) -> Tuple[Optional["WebSocketFrame"], int]:
        """إرجاع (الإطار, عدد البايتات المستهلكة)"""
        if len(data) < 2:
            return None, 0
        b0, b1 = data[0], data[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        offset = 2
        if length == 126:
            if len(data) < 4:
                return None, 0
            length = struct.unpack(">H", data[2:4])[0]
            offset = 4
        elif length == 127:
            if len(data) < 10:
                return None, 0
            length = struct.unpack(">Q", data[2:10])[0]
            offset = 10
        mask_key = b""
        if masked:
            if len(data) < offset + 4:
                return None, 0
            mask_key = data[offset:offset + 4]
            offset += 4
        if len(data) < offset + length:
            return None, 0
        payload = data[offset:offset + length]
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return cls(opcode, payload, fin, masked, mask_key), offset + length


class WebSocketClient:
    """عميل WebSocket متصل"""

    def __init__(self, sock: socket.socket, address: str):
        self.sock = sock
        self.address = address
        self.id = uuid.uuid4().hex[:12]
        self.subscriptions: Set[str] = set()
        self.connected_at = time.time()
        self.last_ping = time.time()
        self._lock = threading.RLock()
        self._closed = False

    def send_text(self, text: str) -> bool:
        return self.send_frame(WebSocketOpcode.TEXT, text.encode("utf-8"))

    def send_json(self, data: Dict[str, Any]) -> bool:
        try:
            return self.send_text(json.dumps(data, ensure_ascii=False,
                                             default=str))
        except Exception:
            return False

    def send_binary(self, data: bytes) -> bool:
        return self.send_frame(WebSocketOpcode.BINARY, data)

    def send_frame(self, opcode: int, payload: bytes) -> bool:
        with self._lock:
            if self._closed:
                return False
            try:
                frame = WebSocketFrame(opcode, payload, fin=True, mask=False)
                self.sock.sendall(frame.to_bytes())
                return True
            except Exception:
                self._closed = True
                return False

    def ping(self) -> bool:
        self.last_ping = time.time()
        return self.send_frame(WebSocketOpcode.PING, b"")

    def close(self, code: int = 1000) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                payload = struct.pack(">H", code)
                self.send_frame(WebSocketOpcode.CLOSE, payload)
                self.sock.close()
            except Exception:
                pass
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class WebSocketHub:
    """مدير جميع عملاء WebSocket"""

    def __init__(self):
        self.clients: Dict[str, WebSocketClient] = {}
        self._lock = threading.RLock()
        self._event_log: deque = deque(maxlen=200)

    def add(self, client: WebSocketClient) -> None:
        with self._lock:
            self.clients[client.id] = client

    def remove(self, client_id: str) -> None:
        with self._lock:
            client = self.clients.pop(client_id, None)
            if client:
                client.close()

    def broadcast(self, message: Dict[str, Any],
                  channel: Optional[str] = None) -> int:
        """بث إلى جميع العملاء (أو قناة معينة)"""
        with self._lock:
            clients = list(self.clients.values())
        sent = 0
        for c in clients:
            if channel and channel not in c.subscriptions:
                continue
            if c.send_json(message):
                sent += 1
            else:
                self.remove(c.id)
        return sent

    def broadcast_event(self, event: Dict[str, Any]) -> None:
        self._event_log.append({
            "ts": time.time(),
            "event": event,
        })
        self.broadcast({"type": "event", "data": event}, "events")

    def get_recent_events(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._event_log)[-n:]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.clients)

    def ping_all(self) -> None:
        with self._lock:
            clients = list(self.clients.values())
        for c in clients:
            if not c.ping():
                self.remove(c.id)


# =====================================================================
# PART 87 — Authentication
# =====================================================================
class WebAuth:
    """مصادقة للوحة الويب"""

    def __init__(self, secret_key: Optional[str] = None,
                 session_timeout: int = 3600):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.session_timeout = session_timeout
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_session(self, username: str, role: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {
                "username": username,
                "role": role,
                "created": time.time(),
                "last_seen": time.time(),
            }
        return token

    def validate(self, token: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            s = self._sessions.get(token)
            if not s:
                return None
            if time.time() - s["last_seen"] > self.session_timeout:
                del self._sessions[token]
                return None
            s["last_seen"] = time.time()
            return dict(s)

    def destroy(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [
                t for t, s in self._sessions.items()
                if now - s["last_seen"] > self.session_timeout
            ]
            for t in expired:
                del self._sessions[t]
        return len(expired)


# =====================================================================
# PART 88 — MJPEG Streamer
# =====================================================================
class MJPEGStreamer:
    """بث فيديو MJPEG للوحة الويب"""

    def __init__(self, fps: int = 15, quality: int = 70,
                 max_width: int = 960):
        self.fps = fps
        self.quality = quality
        self.max_width = max_width
        self._latest: Optional[bytes] = None
        self._lock = threading.RLock()
        self._subscribers: Set[str] = set()

    def publish(self, frame: np.ndarray) -> None:
        if not HAVE_CV2:
            return
        try:
            # تصغير
            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                frame = cv2.resize(
                    frame,
                    (self.max_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not ok:
                return
            with self._lock:
                self._latest = buf.tobytes()
        except Exception:
            pass

    def get_latest(self) -> Optional[bytes]:
        with self._lock:
            return self._latest

    def subscribe(self, client_id: str) -> None:
        with self._lock:
            self._subscribers.add(client_id)

    def unsubscribe(self, client_id: str) -> None:
        with self._lock:
            self._subscribers.discard(client_id)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# =====================================================================
# PART 89 — Web Dashboard HTML Template
# =====================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FalconEye — لوحة التحكم</title>
<style>
:root {
  --terracotta: #C9604F;
  --dark: #1a1a1f;
  --panel: #24242d;
  --border: #3a3a45;
  --text: #e8e8ec;
  --muted: #888899;
  --ok: #22c55e;
  --warn: #f59e0b;
  --alarm: #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Tahoma, sans-serif;
  background: var(--dark);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}
header {
  background: var(--panel);
  border-bottom: 2px solid var(--terracotta);
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky; top: 0; z-index: 100;
}
header h1 {
  font-size: 22px;
  color: var(--terracotta);
}
header .status {
  display: flex; gap: 16px; align-items: center;
  font-size: 13px;
}
.status-dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; background: var(--ok);
  margin-left: 6px;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  padding: 16px 24px;
}
.stat-card {
  background: var(--panel);
  border-radius: 8px;
  padding: 16px;
  border: 1px solid var(--border);
  transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card .label { color: var(--muted); font-size: 12px; }
.stat-card .value {
  font-size: 28px; font-weight: bold;
  color: var(--terracotta);
  margin-top: 4px;
}
.main-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 16px;
  padding: 0 24px 24px;
  min-height: calc(100vh - 200px);
}
@media (max-width: 1000px) {
  .main-grid { grid-template-columns: 1fr; }
}
.panel {
  background: var(--panel);
  border-radius: 8px;
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex; flex-direction: column;
}
.panel-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-weight: bold;
  color: var(--terracotta);
  display: flex; justify-content: space-between;
  align-items: center;
}
.panel-body {
  flex: 1; overflow-y: auto;
  padding: 12px;
}
#video-stream {
  width: 100%; height: auto;
  display: block; background: #000;
}
.event-item {
  padding: 10px;
  border-radius: 6px;
  margin-bottom: 8px;
  background: rgba(255,255,255,0.03);
  border-right: 3px solid var(--muted);
  font-size: 13px;
}
.event-item.sev-critical { border-right-color: var(--alarm); }
.event-item.sev-high { border-right-color: #f97316; }
.event-item.sev-medium { border-right-color: var(--warn); }
.event-item .event-time {
  color: var(--muted); font-size: 11px;
}
.event-item .event-title {
  font-weight: bold; margin: 3px 0;
}
.controls {
  display: flex; gap: 8px; padding: 12px;
  border-top: 1px solid var(--border);
  background: rgba(0,0,0,0.2);
}
button {
  background: var(--terracotta);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  transition: opacity 0.2s;
}
button:hover { opacity: 0.85; }
button.secondary {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
}
.filters {
  display: flex; gap: 8px; padding: 10px;
  border-bottom: 1px solid var(--border);
}
.filters select, .filters input {
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 10px;
  border-radius: 4px;
  font-family: inherit;
  font-size: 12px;
}
#connection-status {
  font-size: 12px; color: var(--muted);
}
#connection-status.connected { color: var(--ok); }
#connection-status.disconnected { color: var(--alarm); }
.logo {
  display: flex; align-items: center; gap: 12px;
}
.logo-icon {
  width: 32px; height: 32px;
  border: 2px solid var(--terracotta);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-weight: bold; color: var(--terracotta);
}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">FE</div>
    <h1>FalconEye — لوحة التحكم</h1>
  </div>
  <div class="status">
    <span id="connection-status" class="disconnected">غير متصل</span>
    <span><span class="status-dot"></span> النظام يعمل</span>
    <span id="clock">--:--:--</span>
  </div>
</header>

<div class="stats-grid">
  <div class="stat-card">
    <div class="label">التتبعات النشطة</div>
    <div class="value" id="stat-tracks">0</div>
  </div>
  <div class="stat-card">
    <div class="label">الأشخاص</div>
    <div class="value" id="stat-people">0</div>
  </div>
  <div class="stat-card">
    <div class="label">الإنذارات</div>
    <div class="value" id="stat-alarms">0</div>
  </div>
  <div class="stat-card">
    <div class="label">الأدلة المحفوظة</div>
    <div class="value" id="stat-evidence">0</div>
  </div>
  <div class="stat-card">
    <div class="label">معدل الإطارات</div>
    <div class="value" id="stat-fps">0.0</div>
  </div>
  <div class="stat-card">
    <div class="label">مستوى التهديد</div>
    <div class="value" id="stat-threat">منخفض</div>
  </div>
</div>

<div class="main-grid">
  <div class="panel">
    <div class="panel-header">
      <span>البث المباشر</span>
      <span id="stream-info" style="font-size:12px;color:var(--muted);">—</span>
    </div>
    <img id="video-stream" src="/api/stream.mjpg" alt="Live stream">
    <div class="controls">
      <button onclick="toggleStream()" id="btn-toggle">إيقاف البث</button>
      <button class="secondary" onclick="resetTracks()">إعادة تعيين</button>
      <button class="secondary" onclick="exportData()">تصدير</button>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <span>سجل الأحداث</span>
      <span id="event-count" style="font-size:12px;color:var(--muted);">0</span>
    </div>
    <div class="filters">
      <select id="filter-severity" onchange="applyFilters()">
        <option value="0">الكل</option>
        <option value="2">متوسط+</option>
        <option value="3">عالي+</option>
        <option value="4">حرج فقط</option>
      </select>
      <select id="filter-type" onchange="applyFilters()">
        <option value="">جميع الأنواع</option>
        <option value="weapon">سلاح</option>
        <option value="photo">تصوير</option>
        <option value="loitering">تسكع</option>
        <option value="dwell">مكوث</option>
      </select>
    </div>
    <div class="panel-body" id="event-list">
      <div style="color:var(--muted);text-align:center;padding:20px;">
        في انتظار الأحداث…
      </div>
    </div>
  </div>
</div>

<script>
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = protocol + '//' + window.location.host + '/ws';
let ws = null;
let reconnectTimer = null;
let allEvents = [];

function connectWS() {
  try {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      setConnectionStatus(true);
      sendSubscription(['events', 'tracks', 'stats']);
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        handleMessage(msg);
      } catch (e) {
        console.error('WS parse error', e);
      }
    };
    ws.onclose = () => {
      setConnectionStatus(false);
      scheduleReconnect();
    };
    ws.onerror = () => {
      setConnectionStatus(false);
    };
  } catch (e) {
    setConnectionStatus(false);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWS();
  }, 3000);
}

function setConnectionStatus(ok) {
  const el = document.getElementById('connection-status');
  el.textContent = ok ? 'متصل' : 'غير متصل';
  el.className = ok ? 'connected' : 'disconnected';
}

function sendSubscription(channels) {
  if (!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({ action: 'subscribe', channels: channels }));
}

function handleMessage(msg) {
  if (msg.type === 'stats') {
    updateStats(msg.data);
  } else if (msg.type === 'event') {
    addEvent(msg.data);
  } else if (msg.type === 'snapshot') {
    // snapshot handling (اختياري)
  }
}

function updateStats(data) {
  setText('stat-tracks', data.tracks ?? 0);
  setText('stat-people', data.people ?? 0);
  setText('stat-alarms', data.alarms ?? 0);
  setText('stat-evidence', data.evidence ?? 0);
  setText('stat-fps', (data.fps ?? 0).toFixed(1));
  const threatEl = document.getElementById('stat-threat');
  threatEl.textContent = data.threat_level ?? 'منخفض';
  const colors = { 'منخفض': '#22c55e', 'متوسط': '#f59e0b',
                   'عالي': '#f97316', 'حرج': '#ef4444' };
  threatEl.style.color = colors[data.threat_level] || '#22c55e';
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function addEvent(evt) {
  allEvents.unshift(evt);
  if (allEvents.length > 500) allEvents = allEvents.slice(0, 500);
  renderEvents();
}

function renderEvents() {
  const minSev = parseInt(document.getElementById('filter-severity').value);
  const typeFilter = document.getElementById('filter-type').value;
  const list = document.getElementById('event-list');

  const filtered = allEvents.filter(e => {
    if ((e.severity ?? 0) < minSev) return false;
    if (typeFilter && !(e.reason || '').toLowerCase().includes(typeFilter)) return false;
    return true;
  });

  if (filtered.length === 0) {
    list.innerHTML = '<div style="color:var(--muted);text-align:center;padding:20px;">لا توجد أحداث</div>';
    setText('event-count', allEvents.length);
    return;
  }

  const sevClass = { 5:'sev-critical', 4:'sev-critical',
                    3:'sev-high', 2:'sev-medium' };
  list.innerHTML = filtered.slice(0, 100).map(e => {
    const sev = e.severity ?? 0;
    const cls = sevClass[sev] || '';
    const t = e.timestamp ? new Date(e.timestamp*1000).toLocaleTimeString('ar-EG') : '--';
    return `<div class="event-item ${cls}">
      <div class="event-time">${t}</div>
      <div class="event-title">ID ${e.track_id ?? '?'} — ${escapeHtml(e.reason ?? '')}</div>
    </div>`;
  }).join('');
  setText('event-count', allEvents.length);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function applyFilters() { renderEvents(); }

let streamActive = true;
function toggleStream() {
  const img = document.getElementById('video-stream');
  const btn = document.getElementById('btn-toggle');
  if (streamActive) {
    img.src = '';
    btn.textContent = 'تشغيل البث';
  } else {
    img.src = '/api/stream.mjpg?' + Date.now();
    btn.textContent = 'إيقاف البث';
  }
  streamActive = !streamActive;
}

async function resetTracks() {
  if (!confirm('إعادة تعيين جميع التتبعات؟')) return;
  try {
    await fetch('/api/tracks/reset', { method: 'POST' });
  } catch (e) { console.error(e); }
}

function exportData() {
  const blob = new Blob([JSON.stringify(allEvents, null, 2)],
                        { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'falconeye_events_' + Date.now() + '.json';
  a.click();
  URL.revokeObjectURL(url);
}

function updateClock() {
  const el = document.getElementById('clock');
  if (el) el.textContent = new Date().toLocaleTimeString('ar-EG');
}
setInterval(updateClock, 1000);
updateClock();

connectWS();
</script>
</body>
</html>
"""


LOGIN_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>تسجيل الدخول — FalconEye</title>
<style>
body {
  font-family: 'Segoe UI', sans-serif;
  background: #1a1a1f; color: #e8e8ec;
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh; margin: 0;
}
.login-box {
  background: #24242d; padding: 40px; border-radius: 12px;
  width: 100%; max-width: 400px;
  border: 1px solid #3a3a45;
  box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
.login-box h1 {
  color: #C9604F; margin: 0 0 24px;
  text-align: center; font-size: 26px;
}
.login-box input {
  width: 100%; padding: 12px;
  margin-bottom: 12px;
  background: rgba(0,0,0,0.3);
  border: 1px solid #3a3a45;
  color: #e8e8ec; border-radius: 6px;
  font-family: inherit; font-size: 14px;
}
.login-box button {
  width: 100%; padding: 12px;
  background: #C9604F; color: white;
  border: none; border-radius: 6px;
  cursor: pointer; font-family: inherit;
  font-size: 15px; font-weight: bold;
}
.login-box button:hover { opacity: 0.9; }
.error { color: #ef4444; font-size: 13px; margin-bottom: 12px; }
</style>
</head>
<body>
<form class="login-box" method="POST" action="/login">
  <h1>FalconEye</h1>
  <div id="error" class="error" style="display:none;"></div>
  <input name="username" placeholder="اسم المستخدم" required autofocus>
  <input name="password" type="password" placeholder="كلمة المرور" required>
  <button type="submit">تسجيل الدخول</button>
</form>
</body>
</html>
"""


# =====================================================================
# PART 90 — Web Request Handler
# =====================================================================
class WebRequestHandler(BaseHTTPRequestHandler):
    """معالج HTTP للوحة الويب"""

    server_version = f"FalconEye-Web/{WEB_VERSION}"

    # سيتم تعيينها من الخارج
    hub: WebSocketHub = None
    streamer: MJPEGStreamer = None
    auth: WebAuth = None
    app_state: Dict[str, Any] = None

    def log_message(self, fmt, *args):
        # تقليل الضوضاء
        pass

    # ==================================================================
    # GET
    # ==================================================================
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._serve_html(DASHBOARD_HTML)
        if path == "/login":
            return self._serve_html(LOGIN_HTML)
        if path == "/logout":
            cookie = self._get_cookie("fe_session")
            if cookie and self.auth:
                self.auth.destroy(cookie)
            self._redirect("/login")
            return

        if path == "/ws":
            return self._handle_websocket_upgrade()

        if path == "/api/stream.mjpg":
            return self._serve_mjpeg()

        if path == "/api/health":
            return self._send_json({"status": "ok", "ts": time.time()})

        if path == "/api/stats":
            return self._send_json(self._get_stats())

        if path == "/api/events":
            q = urllib.parse.parse_qs(parsed.query)
            return self._handle_events_query(q)

        if path == "/api/tracks":
            return self._send_json(self._get_tracks())

        if path.startswith("/api/snapshot/"):
            event_id = path.rsplit("/", 1)[-1]
            return self._serve_snapshot(event_id)

        if path == "/api/config":
            return self._send_json(self._get_config())

        # 404
        self._send_json({"error": "not found", "path": path}, status=404)

    # ==================================================================
    # POST
    # ==================================================================
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/login":
            return self._handle_login()

        if path == "/api/tracks/reset":
            return self._handle_reset_tracks()

        if path == "/api/acknowledge":
            return self._handle_acknowledge()

        if path == "/api/export":
            return self._handle_export()

        self._send_json({"error": "not found"}, status=404)

    # ==================================================================
    # HTML
    # ==================================================================
    def _serve_html(self, html: str, status: int = 200):
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    # ==================================================================
    # JSON
    # ==================================================================
    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ==================================================================
    # Cookies
    # ==================================================================
    def _get_cookie(self, name: str) -> Optional[str]:
        cookie_hdr = self.headers.get("Cookie", "")
        for part in cookie_hdr.split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return None

    def _check_auth(self) -> bool:
        if not self.auth:
            return True
        token = self._get_cookie("fe_session")
        if not token:
            return False
        return self.auth.validate(token) is not None

    # ==================================================================
    # Login
    # ==================================================================
    def _handle_login(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            params = urllib.parse.parse_qs(body)
            username = params.get("username", [""])[0]
            password = params.get("password", [""])[0]
        except Exception:
            return self._serve_html(LOGIN_HTML, 400)

        app = self.app_state or {}
        verify = app.get("verify_user")
        if not verify:
            # بدون مصادقة
            token = self.auth.create_session(username, "admin")
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie",
                f"fe_session={token}; Path=/; HttpOnly; SameSite=Lax"
            )
            self.end_headers()
            return

        user = verify(username, password)
        if user is None:
            return self._serve_html(
                LOGIN_HTML.replace(
                    '<div id="error" class="error" style="display:none;"></div>',
                    '<div id="error" class="error">اسم المستخدم أو كلمة المرور خاطئة</div>'
                ),
                401
            )
        token = self.auth.create_session(user["username"], user["role"])
        self.send_response(302)
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie",
            f"fe_session={token}; Path=/; HttpOnly; SameSite=Lax"
        )
        self.end_headers()

    # ==================================================================
    # WebSocket upgrade
    # ==================================================================
    def _handle_websocket_upgrade(self):
        upgrade = self.headers.get("Upgrade", "").lower()
        if upgrade != "websocket":
            return self._send_json({"error": "expected websocket"}, 400)

        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            return self._send_json({"error": "missing key"}, 400)

        accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        # تسجيل العميل
        client = WebSocketClient(
            self.connection,
            self.client_address[0] if self.client_address else "?"
        )
        # إرسال رأس ترحيب
        self.hub.add(client)
        client.send_json({"type": "welcome", "client_id": client.id})

        try:
            self._ws_loop(client)
        finally:
            self.hub.remove(client.id)

    def _ws_loop(self, client: WebSocketClient):
        """حلقة قراءة رسائل WebSocket"""
        self.connection.settimeout(60.0)
        buffer = bytearray()
        while not client.closed:
            try:
                data = self.connection.recv(4096)
            except socket.timeout:
                # ping keepalive
                if not client.ping():
                    break
                continue
            except Exception:
                break
            if not data:
                break
            buffer.extend(data)
            while True:
                frame, consumed = WebSocketFrame.parse(bytes(buffer))
                if frame is None:
                    break
                buffer = buffer[consumed:]
                if not self._handle_ws_frame(client, frame):
                    return

    def _handle_ws_frame(self, client: WebSocketClient,
                         frame: WebSocketFrame) -> bool:
        """معالجة إطار — إرجاع False للإغلاق"""
        if frame.opcode == WebSocketOpcode.CLOSE:
            return False
        if frame.opcode == WebSocketOpcode.PING:
            client.send_frame(WebSocketOpcode.PONG, frame.payload)
            return True
        if frame.opcode == WebSocketOpcode.PONG:
            return True
        if frame.opcode == WebSocketOpcode.TEXT:
            try:
                msg = json.loads(frame.payload.decode("utf-8"))
                self._handle_ws_message(client, msg)
            except Exception:
                pass
        return True

    def _handle_ws_message(self, client: WebSocketClient,
                           msg: Dict[str, Any]):
        action = msg.get("action")
        if action == "subscribe":
            channels = msg.get("channels", [])
            for ch in channels:
                if isinstance(ch, str):
                    client.subscriptions.add(ch)
            client.send_json({
                "type": "subscription",
                "channels": list(client.subscriptions),
            })
        elif action == "unsubscribe":
            channels = msg.get("channels", [])
            for ch in channels:
                client.subscriptions.discard(ch)
        elif action == "ping":
            client.send_json({"type": "pong", "ts": time.time()})

    # ==================================================================
    # MJPEG streaming
    # ==================================================================
    def _serve_mjpeg(self):
        if not self.streamer:
            return self._send_json({"error": "streamer not ready"}, 503)
        boundary = "frame"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}"
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        client_id = uuid.uuid4().hex[:8]
        self.streamer.subscribe(client_id)
        try:
            while True:
                frame_bytes = self.streamer.get_latest()
                if frame_bytes is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(b"--" + boundary.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(1.0 / max(1, self.streamer.fps))
        finally:
            self.streamer.unsubscribe(client_id)

    # ==================================================================
    # API handlers
    # ==================================================================
    def _get_stats(self) -> Dict[str, Any]:
        app = self.app_state or {}
        get_stats = app.get("get_stats")
        if callable(get_stats):
            return get_stats()
        return {
            "tracks": 0, "people": 0, "alarms": 0,
            "evidence": 0, "fps": 0.0, "threat_level": "منخفض",
        }

    def _get_tracks(self) -> Dict[str, Any]:
        app = self.app_state or {}
        get_tracks = app.get("get_tracks")
        if callable(get_tracks):
            return {"tracks": get_tracks()}
        return {"tracks": []}

    def _get_config(self) -> Dict[str, Any]:
        app = self.app_state or {}
        return {
            "version": WEB_VERSION,
            "language": app.get("language", "ar"),
        }

    def _handle_events_query(self, q: Dict[str, List[str]]) -> None:
        app = self.app_state or {}
        get_events = app.get("get_events")
        if callable(get_events):
            limit = int(q.get("limit", ["100"])[0])
            events = get_events(limit=limit)
            return self._send_json({"events": events})
        # fallback
        if self.hub:
            return self._send_json({
                "events": [e["event"] for e in self.hub.get_recent_events(50)]
            })
        return self._send_json({"events": []})

    def _handle_reset_tracks(self) -> None:
        app = self.app_state or {}
        reset = app.get("reset_tracks")
        if callable(reset):
            try:
                reset()
                return self._send_json({"success": True})
            except Exception as e:
                return self._send_json({"success": False, "error": str(e)}, 500)
        return self._send_json({"success": False, "error": "not available"}, 503)

    def _handle_acknowledge(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            event_id = data.get("event_id")
            app = self.app_state or {}
            ack = app.get("acknowledge")
            if callable(ack) and event_id is not None:
                ack(event_id)
                return self._send_json({"success": True})
            return self._send_json({"success": False}, 400)
        except Exception as e:
            return self._send_json({"success": False, "error": str(e)}, 500)

    def _handle_export(self) -> None:
        app = self.app_state or {}
        get_events = app.get("get_events")
        if not callable(get_events):
            return self._send_json({"error": "not available"}, 503)
        events = get_events(limit=10000)
        data = json.dumps(events, ensure_ascii=False, indent=2,
                         default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="falconeye_events_{int(time.time())}.json"'
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_snapshot(self, event_id: str) -> None:
        app = self.app_state or {}
        get_path = app.get("get_evidence_path")
        if not callable(get_path):
            return self._send_json({"error": "not available"}, 503)
        path = get_path(event_id)
        if not path or not os.path.exists(path):
            return self._send_json({"error": "not found"}, 404)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            return self._send_json({"error": "read failed"}, 500)
        ctype, _ = mimetypes.guess_type(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# =====================================================================
# PART 91 — Web Server
# =====================================================================
class WebServer:
    """خادم الويب الرئيسي"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.hub = WebSocketHub()
        self.streamer = MJPEGStreamer()
        self.auth = WebAuth()
        self.app_state: Dict[str, Any] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stats_getter: Optional[Callable] = None
        self._stats_interval = 1.0

    def register_callbacks(
        self,
        get_stats: Optional[Callable] = None,
        get_tracks: Optional[Callable] = None,
        get_events: Optional[Callable] = None,
        reset_tracks: Optional[Callable] = None,
        acknowledge: Optional[Callable] = None,
        verify_user: Optional[Callable] = None,
        get_evidence_path: Optional[Callable] = None,
        language: str = "ar",
    ) -> None:
        """تسجيل كل الدوال التي تحتاجها اللوحة"""
        self.app_state = {
            "get_stats": get_stats,
            "get_tracks": get_tracks,
            "get_events": get_events,
            "reset_tracks": reset_tracks,
            "acknowledge": acknowledge,
            "verify_user": verify_user,
            "get_evidence_path": get_evidence_path,
            "language": language,
        }
        self._stats_getter = get_stats

    def start(self) -> bool:
        try:
            # تثبيت المراجع في المعالج
            WebRequestHandler.hub = self.hub
            WebRequestHandler.streamer = self.streamer
            WebRequestHandler.auth = self.auth
            WebRequestHandler.app_state = self.app_state

            self._server = ThreadingHTTPServer(
                (self.host, self.port), WebRequestHandler
            )
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="web-server",
                daemon=True,
            )
            self._thread.start()
            # بدء بث الإحصائيات
            self._stats_thread = threading.Thread(
                target=self._stats_broadcast_loop,
                name="web-stats",
                daemon=True,
            )
            self._stats_thread.start()
            return True
        except Exception as e:
            print(f"[WEB] فشل بدء الخادم: {e}")
            return False

    def _stats_broadcast_loop(self) -> None:
        """بث الإحصائيات عبر WebSocket بشكل دوري"""
        while not self._stop.is_set():
            try:
                if self._stats_getter and self.hub.count > 0:
                    stats = self._stats_getter()
                    self.hub.broadcast(
                        {"type": "stats", "data": stats},
                        "stats",
                    )
            except Exception:
                pass
            self._stop.wait(self._stats_interval)

    def push_frame(self, frame: np.ndarray) -> None:
        """دفع إطار إلى بث MJPEG"""
        self.streamer.publish(frame)

    def push_event(self, event: Dict[str, Any]) -> None:
        """دفع حدث إلى جميع عملاء WebSocket"""
        self.hub.broadcast_event(event)

    def stop(self) -> None:
        self._stop.set()
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        # إغلاق جميع العملاء
        with self.hub._lock:
            for client in list(self.hub.clients.values()):
                client.close()

    @property
    def client_count(self) -> int:
        return self.hub.count

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


# =====================================================================
# PART 92 — WebSocket Event Bridge
# =====================================================================
class WebEventBridge:
    """جسر يربط أحداث النظام بلوحة الويب"""

    def __init__(self, server: WebServer):
        self.server = server
        self._last_event_ids: Set[str] = set()
        self._lock = threading.RLock()
        self.total_pushed = 0

    def on_behavior_event(self, event: Any) -> None:
        """يُستدعى من BehaviorPipeline"""
        if isinstance(event, dict):
            payload = event
            uid = event.get("id") or self._make_id(event)
        else:
            payload = getattr(event, "to_dict", lambda: {})()
            uid = self._make_id(payload)

        with self._lock:
            if uid in self._last_event_ids:
                return
            self._last_event_ids.add(uid)
            if len(self._last_event_ids) > 1000:
                self._last_event_ids = set(list(self._last_event_ids)[-500:])

        self.total_pushed += 1
        self.server.push_event(payload)

    def on_alarm(self, track: Any, score: float,
                 reasons: List[str]) -> None:
        payload = {
            "id": f"alarm_{int(time.time()*1000)}",
            "type": "alarm",
            "severity": 4,
            "track_id": getattr(track, "id", 0),
            "label": getattr(track, "stable_label", "?"),
            "score": score,
            "reason": " + ".join(reasons[:3]),
            "timestamp": time.time(),
        }
        self.server.push_event(payload)

    @staticmethod
    def _make_id(event: Dict[str, Any]) -> str:
        key = (
            f"{event.get('type')}_"
            f"{event.get('track_id')}_"
            f"{int(event.get('timestamp', 0) * 1000)}"
        )
        return hashlib.md5(key.encode()).hexdigest()


# =====================================================================
# PART 93 — Dashboard Integration Helper
# =====================================================================
class DashboardIntegration:
    """
    يوفّر واجهة موحدة لدمج لوحة الويب مع FalconEye Core
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        language: str = "ar",
    ):
        self.server = WebServer(host, port)
        self.bridge = WebEventBridge(self.server)
        self.language = language
        self._started = False

    def setup(
        self,
        stats_fn: Callable,
        tracks_fn: Callable,
        events_fn: Callable,
        reset_fn: Optional[Callable] = None,
        ack_fn: Optional[Callable] = None,
        user_fn: Optional[Callable] = None,
        evidence_fn: Optional[Callable] = None,
    ) -> bool:
        """تهيئة جميع الدوال"""
        self.server.register_callbacks(
            get_stats=stats_fn,
            get_tracks=tracks_fn,
            get_events=events_fn,
            reset_tracks=reset_fn,
            acknowledge=ack_fn,
            verify_user=user_fn,
            get_evidence_path=evidence_fn,
            language=self.language,
        )
        self._started = self.server.start()
        return self._started

    def update(self, frame: np.ndarray) -> None:
        """دفع الإطار الحالي إلى البث"""
        self.server.push_frame(frame)

    def push_behavior_event(self, event: Any) -> None:
        self.bridge.on_behavior_event(event)

    def push_alarm(self, track: Any, score: float,
                   reasons: List[str]) -> None:
        self.bridge.on_alarm(track, score, reasons)

    def shutdown(self) -> None:
        self.server.stop()

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def url(self) -> str:
        return self.server.url


# =====================================================================
# PART 94 — System Telemetry for Dashboard
# =====================================================================
class SystemTelemetry:
    """جمع بيانات التليمتري للوحة"""

    def __init__(self, db=None):
        self.db = db
        self._start_time = time.time()
        self._frame_times = deque(maxlen=60)

    def record_frame(self) -> None:
        self._frame_times.append(time.time())

    def current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def uptime(self) -> float:
        return time.time() - self._start_time

    def system_info(self) -> Dict[str, Any]:
        info = {
            "uptime_seconds": round(self.uptime(), 1),
            "uptime_human": self._format_uptime(self.uptime()),
            "fps": round(self.current_fps(), 1),
        }
        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            info["memory_mb"] = round(
                psutil.Process().memory_info().rss / (1024 * 1024), 1
            )
            info["disk_percent"] = psutil.disk_usage("/").percent
        except ImportError:
            pass
        return info

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# =====================================================================
# PART 95 — Dashboard CLI Commands
# =====================================================================
class DashboardCLI:
    """أوامر سطرية لإدارة اللوحة"""

    @staticmethod
    def print_status(integration: DashboardIntegration) -> None:
        server = integration.server
        print(f"""
╔══════════════════════════════════════════════════╗
║  FalconEye Web Dashboard — الحالة                ║
╠══════════════════════════════════════════════════╣
║  الحالة:           {'✓ يعمل' if integration.is_running else '✗ متوقف'}
║  العنوان:          {server.url}
║  العملاء النشطون:  {server.client_count}
║  مشاهدو البث:      {server.streamer.subscriber_count}
║  الأحداث المُرسلة: {integration.bridge.total_pushed}
╚══════════════════════════════════════════════════╝
""")

    @staticmethod
    def print_connection_info(host: str, port: int) -> None:
        # اكتشاف IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        print(f"""
╔══════════════════════════════════════════════════╗
║  للوصول إلى اللوحة:                              ║
╠══════════════════════════════════════════════════╣
║  من نفس الجهاز:   http://localhost:{port}
║  من الشبكة:       http://{local_ip}:{port}
║  المستخدم الافتراضي: admin / admin (غيّره فوراً)
╚══════════════════════════════════════════════════╝
""")


# =====================================================================
# PART 96 — Web Exports
# =====================================================================
__all__ = [
    # Constants
    "WEB_VERSION", "DASHBOARD_HTML", "LOGIN_HTML",
    # WebSocket
    "WebSocketOpcode", "WebSocketFrame", "WebSocketClient", "WebSocketHub",
    # Auth
    "WebAuth",
    # Streaming
    "MJPEGStreamer",
    # Server
    "WebRequestHandler", "WebServer",
    # Integration
    "WebEventBridge", "DashboardIntegration",
    # Telemetry
    "SystemTelemetry",
    # CLI
    "DashboardCLI",
]
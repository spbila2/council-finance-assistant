"""The three limits from architecture.md: per user per hour, per day in money,
and the file size at the door. (The per-minute limit towards Anthropic is
handled by the SDK's retries plus the backoff in agent.py.)
"""
from __future__ import annotations

import base64
import binascii
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import Message, Run, utcnow


def local_day_start_utc(now_utc: datetime | None = None) -> datetime:
    """Midnight today in the business timezone, expressed in naive UTC (how we store)."""
    tz = ZoneInfo(settings.timezone)
    now_utc = now_utc or utcnow()
    local_now = now_utc.replace(tzinfo=timezone.utc).astimezone(tz)
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc).replace(tzinfo=None)


def messages_last_hour(db: Session, user_id: int) -> int:
    since = utcnow() - timedelta(hours=1)
    return int(db.scalar(select(func.count()).select_from(Message)
                         .where(Message.user_id == user_id, Message.role == "user", Message.created_at >= since)) or 0)


def spend_today_usd(db: Session) -> float:
    return float(db.scalar(select(func.coalesce(func.sum(Run.cost_usd), 0.0)).where(Run.created_at >= local_day_start_utc())) or 0.0)


def check_rate_limit(db: Session, user_id: int) -> None:
    used = messages_last_hour(db, user_id)
    if used >= settings.rate_limit_per_hour:
        raise HTTPException(429, f"שלחת {used} הודעות בשעה האחרונה, והמגבלה היא {settings.rate_limit_per_hour}. אפשר לנסות שוב מאוחר יותר.")


def check_daily_cap(db: Session) -> None:
    spent = spend_today_usd(db)
    if spent >= settings.daily_cap_usd:
        raise HTTPException(402, f"התקציב היומי של האפליקציה (${settings.daily_cap_usd:.2f}) נוצל. נחזור מחר.")


def max_b64_chars() -> int:
    """The longest base64 string that still decodes to max_upload_mb (used by the schema, before parsing)."""
    return settings.max_upload_mb * 1024 * 1024 * 4 // 3 + 8


def check_upload_size(data_b64: str | None) -> None:
    if not data_b64:
        return
    approx_bytes = len(data_b64) * 3 // 4
    if approx_bytes > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"התמונה גדולה מ-{settings.max_upload_mb} מ״ב.")


_MAGIC = {"image/png": (b"\x89PNG\r\n\x1a\n",), "image/jpeg": (b"\xff\xd8\xff",),
          "image/gif": (b"GIF87a", b"GIF89a"), "image/webp": (b"RIFF",)}


def check_image(media_type: str, data_b64: str) -> None:
    """The bytes must be valid base64 and start like the image type the client claims (a text file, a script
    or an HTML page labelled image/png is refused before it reaches the model)."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "קובץ התמונה אינו תקין.") from None
    ok = any(raw.startswith(m) for m in _MAGIC.get(media_type, ()))
    if ok and media_type == "image/webp":
        ok = raw[8:12] == b"WEBP"
    if not ok:
        raise HTTPException(400, "הקובץ אינו תמונה מהסוג שהוצהר. אפשר לצרף PNG, JPG, WEBP או GIF.")


# ---- login throttle: failed password attempts per client address, kept in memory --------------------------
# One web replica (.railway/railway.ts), so a dict is enough; a restart clears it, which only helps an attacker
# once per deploy. The shared password makes guessing the one thing worth defending.
LOGIN_MAX_FAILURES = 8
LOGIN_WINDOW_SECONDS = 15 * 60
_login_failures: dict[str, list[float]] = {}


def _recent_failures(client: str, now: float) -> list[float]:
    recent = [t for t in _login_failures.get(client, []) if now - t < LOGIN_WINDOW_SECONDS]
    if recent:
        _login_failures[client] = recent
    else:
        _login_failures.pop(client, None)
    return recent


def check_login_allowed(client: str) -> None:
    if len(_recent_failures(client, time.monotonic())) >= LOGIN_MAX_FAILURES:
        raise HTTPException(429, "יותר מדי ניסיונות כניסה שגויים. אפשר לנסות שוב בעוד כרבע שעה.")


def record_login_failure(client: str) -> None:
    now = time.monotonic()
    _recent_failures(client, now)
    _login_failures.setdefault(client, []).append(now)
    if len(_login_failures) > 5000:  # a flood of addresses must not grow memory without bound
        _login_failures.pop(next(iter(_login_failures)))


def clear_login_failures(client: str | None = None) -> None:
    if client is None:
        _login_failures.clear()
    else:
        _login_failures.pop(client, None)

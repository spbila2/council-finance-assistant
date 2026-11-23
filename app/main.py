"""Council finance assistant backend: the API, the static UI, the health check, the digest hook.

One service does everything (architecture.md, "Where things run"). The agent
loop itself lives in agent.py; this file is doors, locks and plumbing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from . import auth, limits
from .agent import get_model_client, run_turn
from .config import ROOT, settings
from .db import Conversation, Message, SessionLocal, engine, get_db, init_db
from .digest import build_digest, send_digest
from .tools.invoices import recent_invoices


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.init_report = init_db()
    # One line in the deploy log, so a first deploy can be checked (docs/deploy-railway.md, step 7).
    print("seed report: " + ", ".join(f"{k}: {v}" for k, v in app.state.init_report.items()), flush=True)
    app.state.model_client = get_model_client()
    yield


log = logging.getLogger("app.security")

HEARTBEAT_SECONDS = 8  # how long a chat stream may stay silent before a keep-alive line goes out


app = FastAPI(title="Council Finance Assistant", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
class RevalidatedStatic(StaticFiles):
    """Static files with `Cache-Control: no-cache`: the browser keeps a copy but asks, with the ETag, before using it.
    Without this a phone keeps an old app.js or styles.css for hours after a deploy (the file's Last-Modified is
    all it has to go on)."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Content-Security-Policy: our own files, plus the two pinned scripts from cdnjs (each also carries an integrity
# hash in index.html). Nothing inline, no framing, no form posts elsewhere.
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self'; "
       "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
       "form-action 'self'; frame-ancestors 'none'")
MAX_BODY_BYTES = (settings.max_upload_mb + 2) * 1024 * 1024  # an image plus its base64 overhead and some text


class SecurityMiddleware:
    """Pure ASGI (so streaming replies are untouched): security headers on every response, and a request whose
    declared body is larger than any legitimate one is refused before it is read."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"0")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return await Response(status_code=413, content="Request too large")(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                extra = {"content-security-policy": CSP, "x-content-type-options": "nosniff",
                         "x-frame-options": "DENY", "referrer-policy": "no-referrer",
                         "permissions-policy": "camera=(self), microphone=(), geolocation=()"}
                if not settings.is_local:
                    extra["strict-transport-security"] = "max-age=31536000; includeSubDomains"
                headers.extend((k.encode(), v.encode()) for k, v in extra.items())
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityMiddleware)


_PROXY_HEADERS = ("x-real-ip", "x-forwarded-for", "forwarded", "x-envoy-external-address", "true-client-ip",
                  "cf-connecting-ip", "x-forwarded-proto")


def _proxy_headers(request: Request) -> dict[str, str]:
    """The headers that can carry the caller's address, for the log only (never cookies or other headers).
    Used to find out which one Railway fills with the real client address."""
    return {h: request.headers[h][:120] for h in _PROXY_HEADERS if h in request.headers}


def client_address(request: Request) -> str:
    """The caller's address, for the login throttle. Behind Railway's proxy it is X-Real-IP: the edge replaces
    that header (and X-Forwarded-For) with the address it saw, so a caller cannot forge it, and it is the same on
    every request. The last X-Forwarded-For entry is not usable: it is one of several edge proxies. Outside a
    deploy (local, tests) nothing in front sets the header, so it is ignored and the socket address is used."""
    real = request.headers.get("x-real-ip", "").strip()
    if real and not settings.is_local:
        return real[:64]
    return request.client.host if request.client else "unknown"


app.mount("/static", RevalidatedStatic(directory=str(ROOT / "static")), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginIn(BaseModel):
    name: str = Field(min_length=2, max_length=40)
    password: str = Field(min_length=1, max_length=200)


class ImageIn(BaseModel):
    media_type: str = Field(pattern=r"^image/(png|jpeg|webp|gif)$")
    data: str = Field(max_length=limits.max_b64_chars())  # base64, no data: prefix; bounded before it is parsed


class ChatIn(BaseModel):
    conversation_id: int | None = None
    message: str = Field(default="", max_length=4000)
    image: ImageIn | None = None


# ---------------------------------------------------------------------------
# Pages and health
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    """The page, with the version of its own files in every asset URL, so a changed file is always fetched anew
    (a phone cannot keep serving an old app.js), and a small version label the user can read."""
    static = ROOT / "static"
    digest = hashlib.sha1()
    for name in ("app.js", "styles.css", "logo.svg", "index.html"):
        digest.update((static / name).read_bytes())
    files_version = digest.hexdigest()[:8]
    shown = (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "")[:7] or files_version
    html = (static / "index.html").read_text(encoding="utf-8")
    html = (html.replace("/static/app.js", f"/static/app.js?v={files_version}")
                .replace("/static/styles.css", f"/static/styles.css?v={files_version}")
                .replace("{{VERSION}}", shown))
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Railway's health check. 200 only when the database answers and a model is configured. Public, so it says
    nothing about the model, the database or the environment."""
    problems = []
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - exercised on a broken deploy, not in tests
        problems.append(f"db: {type(exc).__name__}")
    if settings.model_provider == "anthropic" and not settings.anthropic_api_key:
        problems.append("ANTHROPIC_API_KEY missing")
    if problems:
        log.error("health check failed: %s", problems)  # the detail stays in the logs, not on the public page
        raise HTTPException(503, {"status": "degraded"})
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/login")
def login(payload: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    address = client_address(request)
    limits.check_login_allowed(address, payload.name)
    if not auth.password_ok(payload.password):
        limits.record_login_failure(address, payload.name)
        log.warning("failed login from %s as %r; proxy headers %s", address, payload.name[:40], _proxy_headers(request))
        raise HTTPException(401, "הסיסמה שגויה.")
    limits.clear_login_failures(address, payload.name)
    user = auth.get_or_create_user(db, auth.normalise_name(payload.name))
    response.set_cookie(auth.COOKIE_NAME, auth.make_token(user.id, user.name), httponly=True, samesite="lax",
                        secure=not settings.is_local, max_age=settings.session_hours * 3600, path="/")
    return {"id": user.id, "name": user.name}


@app.post("/api/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return Response(status_code=204)


@app.get("/api/me")
def me(request: Request, db: Session = Depends(get_db)):
    user = auth.current_user(request, db)
    return {"id": user.id, "name": user.name}


# ---------------------------------------------------------------------------
# Conversations and messages
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
def list_conversations(request: Request, db: Session = Depends(get_db)):
    user = auth.current_user(request, db)
    rows = db.scalars(select(Conversation).where(Conversation.user_id == user.id)
                      .order_by(Conversation.updated_at.desc()).limit(100)).all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat(timespec="seconds")} for c in rows]


@app.get("/api/conversations/{conversation_id}/messages")
def list_messages(conversation_id: int, request: Request, db: Session = Depends(get_db)):
    user = auth.current_user(request, db)
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "השיחה לא נמצאה.")
    rows = db.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)).all()
    return [{"id": m.id, "role": m.role, "content": m.content, "has_image": m.has_image,
             "created_at": m.created_at.isoformat(timespec="seconds")} for m in rows]


@app.post("/api/chat")
def chat(payload: ChatIn, request: Request, db: Session = Depends(get_db)):
    """One turn. Checks the three limits, then streams the agent's events as SSE."""
    user = auth.current_user(request, db)
    if not payload.message.strip() and payload.image is None:
        raise HTTPException(400, "אפשר לכתוב הודעה או לצרף חשבונית.")
    limits.check_rate_limit(db, user.id)
    limits.check_daily_cap(db)
    limits.check_upload_size(payload.image.data if payload.image else None)
    if payload.image:
        limits.check_image(payload.image.media_type, payload.image.data)

    conv = None
    if payload.conversation_id is not None:
        conv = db.get(Conversation, payload.conversation_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(404, "השיחה לא נמצאה.")
    conv_id = conv.id if conv else None
    user_id = user.id
    image = payload.image.model_dump() if payload.image else None
    client = request.app.state.model_client

    def sse():
        """The agent's events as SSE. The turn runs in a worker thread, and while it is silent (the model is
        thinking, a tool is running) a comment line goes out every HEARTBEAT_SECONDS. A phone on mobile data, or a
        proxy in between, drops a connection that stays quiet for too long, and the browser then reports
        "network error". Comment lines are ignored by the page."""
        events: queue.Queue = queue.Queue()
        stop = threading.Event()
        finished = object()

        def worker():
            try:
                # A dedicated session: the request-scoped one may close before the stream ends.
                with SessionLocal() as sdb:
                    suser = sdb.get(type(user), user_id)
                    sconv = sdb.get(Conversation, conv_id) if conv_id else None
                    turn = run_turn(sdb, user=suser, conversation=sconv, text=payload.message, image=image, client=client)
                    try:
                        for event in turn:
                            if stop.is_set():  # the browser went away: closing the turn records it as aborted
                                break
                            events.put(f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
                    finally:
                        turn.close()
            except BaseException as exc:  # handed to the response below, which re-raises it
                events.put(exc)
            finally:
                events.put(finished)

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                try:
                    item = events.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is finished:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


# ---------------------------------------------------------------------------
# Invoices, stats, digest
# ---------------------------------------------------------------------------

@app.get("/api/invoices")
def invoices(request: Request, db: Session = Depends(get_db)):
    auth.current_user(request, db)
    return recent_invoices(db, limit=50)


@app.get("/api/stats")
def stats(request: Request, db: Session = Depends(get_db)):
    user = auth.current_user(request, db)
    return {"spend_today_usd": round(limits.spend_today_usd(db), 4), "daily_cap_usd": settings.daily_cap_usd,
            "messages_last_hour": limits.messages_last_hour(db, user.id), "rate_limit_per_hour": settings.rate_limit_per_hour}


@app.post("/internal/digest")
def digest(db: Session = Depends(get_db), x_cron_secret: str | None = Header(default=None)):
    """Manual or cron trigger for the morning digest. Protected by CRON_SECRET."""
    if not settings.cron_secret or not hmac.compare_digest((x_cron_secret or "").encode(), settings.cron_secret.encode()):
        log.warning("digest trigger refused: bad or missing X-Cron-Secret")
        raise HTTPException(401, "Bad or missing X-Cron-Secret.")
    body = build_digest(db)
    sent = send_digest(body)
    return {"text": body, "sent": sent}

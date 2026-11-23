"""OWASP Top 10 (2025) hardening: one test per finding of the security review (docs/security-review.md)."""
import base64
import re
from dataclasses import replace

import pytest

from app import config, limits
from app.db import Invoice
from app.tools.invoices import log_invoice

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()


@pytest.fixture(autouse=True)
def _fresh_login_counter():
    limits.clear_login_failures()
    yield
    limits.clear_login_failures()


# A07 Authentication failures --------------------------------------------------------------------------
def test_repeated_wrong_passwords_are_throttled_and_a_good_one_is_blocked_too(client):
    client.cookies.clear()
    codes = [client.post("/api/login", json={"name": "Eve", "password": f"guess{i}"}).status_code for i in range(10)]
    assert codes[:limits.LOGIN_MAX_FAILURES] == [401] * limits.LOGIN_MAX_FAILURES
    assert set(codes[limits.LOGIN_MAX_FAILURES:]) == {429}
    assert client.post("/api/login", json={"name": "Eve", "password": "test-pass"}).status_code == 429


def test_a_successful_login_resets_the_counter(client):
    for _ in range(3):
        client.post("/api/login", json={"name": "Dana", "password": "bad"})
    assert client.post("/api/login", json={"name": "Dana", "password": "test-pass"}).status_code == 200
    assert limits._login_failures == {}


def test_a_forged_forwarded_for_header_does_not_dodge_the_throttle(client):
    # The caller controls everything but the last entry (the one Railway's proxy appends).
    for i in range(limits.LOGIN_MAX_FAILURES):
        client.post("/api/login", json={"name": "Eve", "password": "x"}, headers={"X-Forwarded-For": f"1.1.1.{i}, 9.9.9.9"})
    r = client.post("/api/login", json={"name": "Eve", "password": "x"}, headers={"X-Forwarded-For": "2.2.2.2, 9.9.9.9"})
    assert r.status_code == 429


# A02 Security misconfiguration ------------------------------------------------------------------------
def _env(monkeypatch, **values):
    for name in ("APP_ENV", "APP_PASSWORD", "APP_SECRET", "ANTHROPIC_API_KEY", "MODEL_PROVIDER", "RAILWAY_PROJECT_ID",
                 "RAILWAY_ENVIRONMENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    for k, v in values.items():
        monkeypatch.setenv(k, v)


def test_production_refuses_to_start_without_real_secrets(monkeypatch):
    _env(monkeypatch, APP_ENV="production", MODEL_PROVIDER="fake")
    with pytest.raises(RuntimeError, match="Missing"):
        config.load_settings()
    _env(monkeypatch, APP_ENV="production", MODEL_PROVIDER="fake", APP_PASSWORD="council", APP_SECRET="s" * 32)
    with pytest.raises(RuntimeError, match="known default"):
        config.load_settings()
    _env(monkeypatch, APP_ENV="production", MODEL_PROVIDER="fake", APP_PASSWORD="a-real-passphrase", APP_SECRET="short")
    with pytest.raises(RuntimeError, match="32"):
        config.load_settings()


def test_a_railway_deploy_without_app_env_is_production_not_local(monkeypatch):
    _env(monkeypatch, RAILWAY_PROJECT_ID="p1", MODEL_PROVIDER="fake")
    with pytest.raises(RuntimeError):
        config.load_settings()
    _env(monkeypatch, RAILWAY_PROJECT_ID="p1", MODEL_PROVIDER="fake", APP_PASSWORD="a-real-passphrase", APP_SECRET="s" * 32)
    assert config.load_settings().environment == "production"


def test_security_headers_on_pages_static_files_and_the_api(client):
    for path in ("/", "/static/app.js", "/health"):
        h = client.get(path).headers
        assert "script-src 'self' https://cdnjs.cloudflare.com" in h["content-security-policy"], path
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"


def test_health_is_public_and_says_nothing_about_the_setup(client):
    assert client.get("/health").json() == {"status": "ok"}


# A05 Injection (XSS) and A03 supply chain -------------------------------------------------------------
def test_scripts_from_the_cdn_are_pinned_with_integrity_hashes(client):
    tags = re.findall(r"<script[^>]+cdnjs[^>]*>", client.get("/").text, re.S)
    assert len(tags) == 2  # marked and DOMPurify
    for tag in tags:
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]{64}"', tag) and 'crossorigin="anonymous"' in tag
        assert re.search(r"/\d+\.\d+\.\d+/", tag)  # an exact version, never "latest"


def test_the_ui_cleans_the_model_reply_before_it_becomes_html(client):
    js = client.get("/static/app.js").text
    assert "DOMPurify.sanitize(marked.parse(" in js
    assert "return marked.parse" not in js  # the unsanitised path is gone


# A06 Insecure design: uploads -------------------------------------------------------------------------
def test_a_body_larger_than_any_valid_request_is_refused_before_it_is_read(client, logged_in):
    r = client.post("/api/chat", content=b"x" * (20 * 1024 * 1024), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_image_field_is_length_bounded_by_the_schema(client, logged_in):
    huge = "A" * (limits.max_b64_chars() + 1)
    r = client.post("/api/chat", json={"message": "hi", "image": {"media_type": "image/png", "data": huge}})
    assert r.status_code in (413, 422)


@pytest.mark.parametrize("payload", [base64.b64encode(b"<script>alert(1)</script>").decode(),
                                     base64.b64encode(b"GIF89a" + b"0" * 9).decode(), "not base64!!"])
def test_bytes_that_are_not_the_claimed_image_type_are_refused(client, logged_in, fake, payload):
    r = client.post("/api/chat", json={"message": "invoice", "image": {"media_type": "image/png", "data": payload}})
    assert r.status_code == 400


def test_a_real_png_still_goes_through(client, logged_in, fake):
    assert client.post("/api/chat", json={"message": "invoice", "image": {"media_type": "image/png", "data": PNG}}).status_code == 200


# A04 / A09 secrets and logging ------------------------------------------------------------------------
def test_cron_secret_is_checked_and_a_refusal_is_logged(client, caplog):
    with caplog.at_level("WARNING", logger="app.security"):
        assert client.post("/internal/digest").status_code == 401
        assert client.post("/internal/digest", headers={"X-Cron-Secret": "wrong"}).status_code == 401
    assert "X-Cron-Secret" in caplog.text
    assert client.post("/internal/digest", headers={"X-Cron-Secret": "cron-test"}).status_code == 200


def test_a_failed_login_is_logged_with_the_address_but_not_the_password(client, caplog):
    client.cookies.clear()
    with caplog.at_level("WARNING", logger="app.security"):
        client.post("/api/login", json={"name": "Mallory", "password": "hunter2-secret"})
    assert "failed login" in caplog.text and "Mallory" in caplog.text and "hunter2-secret" not in caplog.text


def test_session_cookie_is_secure_outside_local(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", replace(main.settings, environment="staging"))
    r = client.post("/api/login", json={"name": "Dana", "password": "test-pass"})
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie


# A10 Exceptional conditions ---------------------------------------------------------------------------
def test_a_failing_tool_does_not_leak_exception_details(db, monkeypatch):
    from app.tools import budget, registry

    def boom(*a, **k):
        raise ValueError("SECRET-INTERNAL-DETAIL postgres://user:pw@host")

    monkeypatch.setattr(budget, "budget_query", boom)
    out = registry.execute("budget_query", {"metric": "spent", "group_by": "none"}, db)
    assert "SECRET" not in out and "ValueError" not in out and "error" in out


# A08 Integrity ----------------------------------------------------------------------------------------
def test_the_model_cannot_file_an_invoice_under_a_history_reference(db):
    row = log_invoice(db, supplier="החשמל", invoice_number="SEC-1", amount=20000, invoice_date="2026-03-01",
                      source_ref="history-evil")
    assert row["source_ref"] is None
    db.query(Invoice).filter_by(invoice_number="SEC-1").delete()
    db.commit()

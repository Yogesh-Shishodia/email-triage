"""Gmail OAuth, fetching, and sending engine.

Uses the `gmail.modify` scope so we can both read INBOX messages and send
replies. Cached tokens with insufficient scope are auto-invalidated.
"""
from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime
from typing import List, Optional

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    DAYS_BACK,
    GMAIL_CREDENTIALS_FILE,
    GMAIL_SCOPES,
    GMAIL_TOKEN_FILE,
    MAX_BODY_CHARS,
    MAX_EMAILS,
)

logger = logging.getLogger(__name__)


# ---------- Auth ------------------------------------------------------------

def _get_credentials() -> Credentials:
    """Load cached creds or run the OAuth flow.

    If the cached token's scopes don't cover what we now need (e.g. user
    upgraded from readonly → modify), the token is wiped and re-issued.
    """
    creds: Optional[Credentials] = None

    if GMAIL_TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(GMAIL_TOKEN_FILE), GMAIL_SCOPES
            )
        except Exception:
            logger.warning("Cached token unreadable; will re-authenticate")
            creds = None

        # Scope upgrade detection: required scopes must be a subset of granted
        if creds and not set(GMAIL_SCOPES).issubset(set(creds.scopes or [])):
            logger.info(
                "Cached token missing required scopes (have=%s, need=%s) — re-auth",
                creds.scopes, GMAIL_SCOPES,
            )
            try:
                GMAIL_TOKEN_FILE.unlink()
            except FileNotFoundError:
                pass
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Gmail credentials")
            creds.refresh(Request())
        else:
            if not GMAIL_CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"OAuth client secrets missing at {GMAIL_CREDENTIALS_FILE}. "
                    "Create an OAuth Client ID (Desktop app) in Google Cloud "
                    "Console and download it as credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(GMAIL_CREDENTIALS_FILE), GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)

        GMAIL_TOKEN_FILE.write_text(creds.to_json())
        logger.info("Saved Gmail token to %s", GMAIL_TOKEN_FILE)

    return creds


def _build_service():
    """Build a Gmail API client. Each thread can call this for thread safety."""
    return build("gmail", "v1", credentials=_get_credentials(), cache_discovery=False)


# ---------- MIME helpers ----------------------------------------------------

def _decode_body(payload: dict) -> str:
    """Walk the MIME tree and return the best plain-text body we can find."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data:
            try:
                decoded = base64.urlsafe_b64decode(data.encode("ASCII")).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                decoded = ""
            if mime == "text/plain":
                plain_parts.append(decoded)
            elif mime == "text/html":
                html_parts.append(decoded)
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)

    if plain_parts:
        text_out = "\n".join(plain_parts)
    elif html_parts:
        text_out = BeautifulSoup("\n".join(html_parts), "html.parser").get_text("\n")
    else:
        text_out = ""

    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip()
    return text_out[:MAX_BODY_CHARS]


def _header(headers: List[dict], name: str) -> str:
    target = name.lower()
    for h in headers:
        if h.get("name", "").lower() == target:
            return h.get("value", "")
    return ""


# ---------- Public: fetch ---------------------------------------------------

def fetch_recent_emails(days: int = DAYS_BACK) -> List[dict]:
    """Fetch all INBOX messages received in the last `days` days."""
    service = _build_service()

    after_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    query = f"in:inbox after:{after_ts}"
    logger.info("Searching Gmail with query: %s", query)

    message_refs: list[dict] = []
    page_token: Optional[str] = None
    page_size = min(100, MAX_EMAILS)
    while True:
        resp = (
            service.users().messages()
            .list(userId="me", q=query, maxResults=page_size, pageToken=page_token)
            .execute()
        )
        message_refs.extend(resp.get("messages", []) or [])
        if len(message_refs) >= MAX_EMAILS:
            message_refs = message_refs[:MAX_EMAILS]
            break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    logger.info("Found %d message IDs in last %d days", len(message_refs), days)

    emails: List[dict] = []
    for ref in message_refs:
        try:
            msg = (
                service.users().messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            payload = msg.get("payload", {})
            headers = payload.get("headers", [])

            from_raw = _header(headers, "From")
            sender_name, sender_email = parseaddr(from_raw)
            sender_name = sender_name or sender_email or from_raw or "(unknown)"

            date_raw = _header(headers, "Date")
            try:
                received_at = (
                    parsedate_to_datetime(date_raw)
                    if date_raw
                    else datetime.now(timezone.utc)
                )
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=timezone.utc)
            except Exception:
                received_at = datetime.now(timezone.utc)

            emails.append(
                {
                    "email_id": msg["id"],
                    "thread_id": msg.get("threadId"),
                    "message_id_header": _header(headers, "Message-ID") or None,
                    "subject": _header(headers, "Subject") or "(no subject)",
                    "sender": sender_name,
                    "sender_email": sender_email or None,
                    "received_at": received_at.replace(tzinfo=None),  # naive UTC
                    "body": _decode_body(payload),
                    "snippet": msg.get("snippet", ""),
                    "list_unsubscribe": _header(headers, "List-Unsubscribe") or None,
                }
            )
        except Exception:
            logger.exception("Failed to fetch message %s", ref.get("id"))

    logger.info("Successfully hydrated %d emails", len(emails))
    return emails


# ---------- Public: send ----------------------------------------------------

def _re_subject(original: str) -> str:
    """Return `Re: <subject>`, but don't double-prefix."""
    s = (original or "").strip()
    return s if s.lower().startswith("re:") else f"Re: {s}"


def _build_raw_reply(
    to: str,
    subject: str,
    body: str,
    in_reply_to: Optional[str] = None,
) -> str:
    msg = MIMEText(body, "plain", "utf-8")
    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")


def send_reply(
    to: str,
    original_subject: str,
    body: str,
    thread_id: Optional[str] = None,
    in_reply_to_message_id: Optional[str] = None,
) -> dict:
    """Send a reply via Gmail. Threads correctly when thread_id is supplied.

    Raises HttpError on API failure. Returns the API response dict.
    """
    if not to:
        raise ValueError("Recipient address is empty")
    if not body or not body.strip():
        raise ValueError("Reply body is empty")

    service = _build_service()
    raw = _build_raw_reply(
        to=to,
        subject=_re_subject(original_subject),
        body=body,
        in_reply_to=in_reply_to_message_id,
    )

    request_body: dict = {"raw": raw}
    if thread_id:
        request_body["threadId"] = thread_id

    try:
        resp = (
            service.users().messages()
            .send(userId="me", body=request_body)
            .execute()
        )
        logger.info("Sent reply to %s (gmail msg id=%s)", to, resp.get("id"))
        return resp
    except HttpError as e:
        logger.exception("Gmail send failed for %s", to)
        raise

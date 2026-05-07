"""SQLite persistence layer for the triage app.

Single source of truth for processed email state. Idempotent — re-running a
sync will not duplicate rows because `email_id` is the primary key.

Includes a tiny PRAGMA-based migration so users coming from v1 don't have to
delete their database when new columns are added.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from config import DB_URL

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class EmailRecord(Base):
    """One row per Gmail message we have ever seen."""

    __tablename__ = "emails"

    email_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(255))
    message_id_header: Mapped[Optional[str]] = mapped_column(String(512))  # RFC822 Message-ID
    subject: Mapped[str] = mapped_column(Text)
    sender: Mapped[str] = mapped_column(String(255))
    sender_email: Mapped[Optional[str]] = mapped_column(String(255))
    received_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    body: Mapped[str] = mapped_column(Text)
    snippet: Mapped[Optional[str]] = mapped_column(Text)
    list_unsubscribe: Mapped[Optional[str]] = mapped_column(Text)  # raw header value

    # AI outputs
    category: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    urgency: Mapped[Optional[int]] = mapped_column(Integer, default=0)  # 1 (low) – 5 (high)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    entities: Mapped[Optional[dict]] = mapped_column(JSON)
    draft_reply: Mapped[Optional[str]] = mapped_column(Text)

    # Bookkeeping
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    error: Mapped[Optional[str]] = mapped_column(Text)


_engine = create_engine(DB_URL, echo=False, future=True)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


# ---------- Initialization & migration --------------------------------------

# Columns that were added after v1. Mapped to their SQL DDL fragment.
_V2_COLUMNS: dict[str, str] = {
    "message_id_header": "VARCHAR(512)",
    "list_unsubscribe": "TEXT",
    "sent_at": "DATETIME",
}


def _migrate_schema() -> None:
    """Add v2 columns to an existing v1 `emails` table if they're missing."""
    with _engine.begin() as conn:
        try:
            existing = {
                row[1] for row in conn.exec_driver_sql("PRAGMA table_info(emails)").all()
            }
        except Exception:
            return  # table doesn't exist yet — create_all will handle it
        for col, ddl in _V2_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating: ALTER TABLE emails ADD COLUMN %s", col)
                conn.exec_driver_sql(f"ALTER TABLE emails ADD COLUMN {col} {ddl}")


def init_db() -> None:
    """Create or migrate the schema. Safe to call repeatedly."""
    Base.metadata.create_all(_engine)
    _migrate_schema()
    logger.info("Database ready at %s", DB_URL)


@contextmanager
def get_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------- CRUD helpers ----------------------------------------------------

def upsert_email(record: dict) -> None:
    """Insert or update by primary key. Only writes columns the model knows about."""
    valid_keys = {c.name for c in EmailRecord.__table__.columns}
    clean = {k: v for k, v in record.items() if k in valid_keys}
    with get_session() as session:
        existing = session.get(EmailRecord, clean["email_id"])
        if existing:
            for k, v in clean.items():
                setattr(existing, k, v)
        else:
            session.add(EmailRecord(**clean))


def get_unprocessed_email_ids(known_ids: List[str]) -> List[str]:
    """Return the subset of `known_ids` that have NOT yet been AI-processed."""
    if not known_ids:
        return []
    with get_session() as session:
        stmt = select(EmailRecord.email_id).where(
            EmailRecord.email_id.in_(known_ids),
            EmailRecord.processed_at.is_not(None),
        )
        already = {row[0] for row in session.execute(stmt).all()}
    return [eid for eid in known_ids if eid not in already]


def fetch_all_emails() -> List[dict]:
    """Return every stored email as a list of dicts, newest first."""
    with get_session() as session:
        rows = session.execute(
            select(EmailRecord).order_by(EmailRecord.received_at.desc())
        ).scalars().all()
        return [
            {
                "email_id": r.email_id,
                "thread_id": r.thread_id,
                "message_id_header": r.message_id_header,
                "subject": r.subject,
                "sender": r.sender,
                "sender_email": r.sender_email,
                "received_at": r.received_at,
                "body": r.body,
                "snippet": r.snippet,
                "list_unsubscribe": r.list_unsubscribe,
                "category": r.category,
                "urgency": r.urgency,
                "summary": r.summary,
                "entities": r.entities,
                "draft_reply": r.draft_reply,
                "approved": r.approved,
                "processed_at": r.processed_at,
                "sent_at": r.sent_at,
                "error": r.error,
            }
            for r in rows
        ]


def get_email(email_id: str) -> Optional[dict]:
    with get_session() as session:
        r = session.get(EmailRecord, email_id)
        if not r:
            return None
        return {c.name: getattr(r, c.name) for c in EmailRecord.__table__.columns}


def set_approval(email_id: str, approved: bool) -> None:
    with get_session() as session:
        rec = session.get(EmailRecord, email_id)
        if rec:
            rec.approved = approved


def update_draft(email_id: str, new_draft: str) -> None:
    with get_session() as session:
        rec = session.get(EmailRecord, email_id)
        if rec:
            rec.draft_reply = new_draft


def mark_sent(email_id: str, when: Optional[datetime] = None) -> None:
    """Stamp an email as sent so the UI can disable the button."""
    with get_session() as session:
        rec = session.get(EmailRecord, email_id)
        if rec:
            rec.sent_at = when or datetime.utcnow()

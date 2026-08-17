"""InboxIQ — AI Email Triage Dashboard.

Fix from previous version: pd.NaT was crashing the detail view when an email
hadn't been sent yet. All datetime checks now use pd.notna() explicitly.
"""
from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

from agents import CATEGORIES, process_email
from config import DAYS_BACK, LOG_LEVEL, MAX_WORKERS
from database import (
    fetch_all_emails,
    get_unprocessed_email_ids,
    init_db,
    mark_sent,
    set_approval,
    update_draft,
    upsert_email,
)
from gmail_engine import fetch_recent_emails, send_reply

# ---- Branding (rename freely) --------------------------------------------
APP_NAME = "InboxIQ"
APP_TAGLINE = "Your inbox, intelligently sorted — 100% local, zero cloud"

# ---- Visual encoding -----------------------------------------------------
CATEGORY_COLORS = {
    "Job Related": "#3B82F6",
    "Marketing/Spam": "#EF4444",
    "Important/Action Required": "#F59E0B",
    "Newsletters": "#10B981",
    "Personal": "#8B5CF6",
    "Unprocessed": "#9CA3AF",
}
CATEGORY_ICONS = {
    "Job Related": "💼",
    "Marketing/Spam": "🎯",
    "Important/Action Required": "⚡",
    "Newsletters": "📰",
    "Personal": "💬",
    "Unprocessed": "❓",
}

# ---- Logging -------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# ---- Page setup ----------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — AI Email Triage",
    layout="wide",
    page_icon="📬",
    initial_sidebar_state="expanded",
)
init_db()

# ---- Custom CSS ----------------------------------------------------------
# NOTE: this whole block must be ONE unbroken HTML element with no blank
# lines inside <style>...</style> — Streamlit's markdown renderer treats a
# blank line as the end of the raw-HTML block and dumps everything after it
# onto the page as literal text instead of applying it as CSS. Font loading
# uses @import (not a separate <link>) for the same reason: a <link> tag
# ahead of <style> makes the parser treat them as one block that still ends
# at the first blank line.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --accent-1: #667eea;
    --accent-2: #764ba2;
    --ink: #1f2937;
    --ink-soft: #6b7280;
    --line: #e5e7eb;
    --surface: #ffffff;
    --surface-soft: #fafafa;
}
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
/* Hero */
.hero {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
    color: white;
    padding: 1.75rem 2.25rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 28px rgba(102, 126, 234, 0.28);
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 0.45rem 0 0 0; opacity: 0.94; font-size: 1.02rem; }
.hero .badges { margin-top: 1rem; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.4rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.22);
}
/* KPI cards */
[data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 800; color: var(--ink); }
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.07); transform: translateY(-1px); }
/* Table */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface-soft); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] hr { margin: 1rem 0; }
/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border: none;
    box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
}
/* Filter bar */
h5, .stMarkdown h5 { color: var(--ink-soft); font-weight: 700; letter-spacing: 0.03em; }
/* Detail panel */
.detail-header {
    background: linear-gradient(180deg, var(--surface-soft) 0%, var(--surface) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.4rem 1.7rem;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
/* Status pills (used via st.markdown in the detail view) */
.pill {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
/* Empty state */
.empty-state {
    text-align: center;
    padding: 4.5rem 2rem;
    background: var(--surface);
    border-radius: 16px;
    border: 2px dashed var(--line);
}
.empty-state h2 { color: var(--ink); }
.empty-state p { color: var(--ink-soft); font-size: 1.05rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---- Hero ----------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
    <h1>📬 {APP_NAME}</h1>
    <p>{APP_TAGLINE}</p>
    <div class="badges">
        <span class="badge">🔒 Private — runs on your Mac</span>
        <span class="badge">🤖 Qwen 2.5 7B</span>
        <span class="badge">⚡ {MAX_WORKERS} parallel workers</span>
        <span class="badge">📨 Last {DAYS_BACK} days</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---- Helpers -------------------------------------------------------------

def _process_one(email: dict) -> dict:
    try:
        return process_email(email)
    except Exception as e:
        logger.exception("Worker failed on %s", email.get("email_id"))
        return {**email, "error": str(e), "processed_at": datetime.utcnow()}


def run_sync(reprocess_all: bool = False) -> None:
    progress = st.progress(0.0, text="Connecting to Gmail…")
    try:
        emails = fetch_recent_emails(days=DAYS_BACK)
    except Exception as e:
        logger.exception("Gmail fetch failed")
        st.error(f"Gmail fetch failed: {e}")
        return

    if not emails:
        st.warning("No emails found in the requested window.")
        return

    ids = [e["email_id"] for e in emails]
    if reprocess_all:
        targets = emails
    else:
        unprocessed = set(get_unprocessed_email_ids(ids))
        targets = [e for e in emails if e["email_id"] in unprocessed]
        for e in emails:
            if e["email_id"] not in unprocessed:
                upsert_email(e)

    st.info(
        f"Found **{len(emails)}** emails · processing **{len(targets)}** with "
        f"**{MAX_WORKERS}** workers"
    )

    if not targets:
        progress.progress(1.0, text="All emails already processed.")
        return

    total = len(targets)
    started = perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_one, em): em for em in targets}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            try:
                upsert_email(result)
            except Exception:
                logger.exception("DB upsert failed for %s", result.get("email_id"))
            subj = (result.get("subject") or "")[:55]
            progress.progress(completed / total, text=f"[{completed}/{total}] {subj}")

    elapsed = perf_counter() - started
    rate = total / elapsed if elapsed > 0 else 0
    progress.progress(1.0, text="Sync complete!")
    st.success(f"✓ Processed {total} emails in {elapsed:.1f}s ({rate:.1f}/s)")


def do_send(row: dict) -> None:
    if not row.get("draft_reply"):
        st.error("No draft to send.")
        return
    if not row.get("sender_email"):
        st.error("No recipient address known for this email.")
        return
    try:
        send_reply(
            to=row["sender_email"],
            original_subject=row["subject"],
            body=row["draft_reply"],
            thread_id=row.get("thread_id"),
            in_reply_to_message_id=row.get("message_id_header"),
        )
        mark_sent(row["email_id"])
        st.toast(f"✉️ Sent reply to {row['sender_email']}")
    except Exception as e:
        logger.exception("Send failed for %s", row.get("email_id"))
        st.error(f"Send failed: {e}")


def _extract_unsub_url(header_value: str) -> str | None:
    if not header_value:
        return None
    for part in header_value.split(","):
        part = part.strip().strip("<>").strip()
        try:
            parsed = urlparse(part)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return part
        except Exception:
            continue
    return None


def _is_set(value) -> bool:
    """Robust truth check that handles pd.NaT, None, and NaN."""
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except (ValueError, TypeError):
        return bool(value)


# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 📬 {APP_NAME}")
    st.caption(f"Triaging the last **{DAYS_BACK}** days")

    if st.button("🔄 Sync Gmail", type="primary", use_container_width=True):
        with st.spinner("Syncing…"):
            run_sync(reprocess_all=False)
        st.rerun()

    if st.button("♻️ Re-process all", use_container_width=True):
        with st.spinner("Re-processing…"):
            run_sync(reprocess_all=True)
        st.rerun()

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
1. **Sync** pulls last {n} days from Gmail
2. **Pre-filter** removes obvious newsletters (no LLM)
3. **Triage** — LLM classifies into 5 categories
4. **Analyze** — extracts deadlines, names, companies
5. **Draft** — generates a reply for actionable mail
6. **Approve & Send** — review, edit, fire via Gmail
            """.format(n=DAYS_BACK)
        )

    st.divider()

    all_rows_for_chart = fetch_all_emails()
    if all_rows_for_chart:
        st.markdown("##### Category Mix")
        cat_counts = Counter(r["category"] or "Unprocessed" for r in all_rows_for_chart)
        df_cat = pd.DataFrame(
            {"Category": list(cat_counts.keys()), "Count": list(cat_counts.values())}
        )
        fig = px.pie(
            df_cat,
            names="Category",
            values="Count",
            hole=0.65,
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
        )
        fig.update_traces(textposition="inside", textinfo="value", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👉 Click **Sync Gmail** to begin")


# ---- Main content --------------------------------------------------------
rows = fetch_all_emails()

if not rows:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
<div class="empty-state">
    <h2>👋 Welcome to {APP_NAME}</h2>
    <p>Click <strong>🔄 Sync Gmail</strong> in the sidebar to triage your inbox.</p>
    <p style="margin-top:1.5rem; color:#9ca3af; font-size:0.9rem;">
        Your data stays on your Mac. No cloud, no tracking, no ads.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )
    st.stop()

df = pd.DataFrame(rows)
df["received_at"] = pd.to_datetime(df["received_at"])
df["sent_at"] = pd.to_datetime(df["sent_at"])  # ensure datetime dtype
df = df.sort_values("received_at", ascending=False).reset_index(drop=True)

# ---- KPI strip ------------------------------------------------------------
total = len(df)
action_required = int((df["category"] == "Important/Action Required").sum())
job_related = int((df["category"] == "Job Related").sum())
drafts_ready = int(df["draft_reply"].notna().sum())
sent_count = int(df["sent_at"].notna().sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📥 Total", total)
k2.metric("⚡ Action req.", action_required)
k3.metric("💼 Job Related", job_related)
k4.metric("✏️ Drafts", drafts_ready)
k5.metric("✉️ Sent", sent_count)

# ---- Filters --------------------------------------------------------------
st.markdown("##### Filter")
fc1, fc2, fc3, fc4 = st.columns([2, 1.6, 1.2, 1])
with fc1:
    search_q = st.text_input(
        "Search subject or sender",
        placeholder="🔎 Type to filter by subject or sender…",
        label_visibility="collapsed",
    )
with fc2:
    sel_cats = st.multiselect(
        "Category",
        options=CATEGORIES + ["Unprocessed"],
        default=CATEGORIES + ["Unprocessed"],
        label_visibility="collapsed",
        placeholder="All categories",
    )
with fc3:
    urg_min, urg_max = st.slider("Urgency", 1, 5, (1, 5))
with fc4:
    only_drafts = st.checkbox("Has draft", value=False)

mask = df["category"].fillna("Unprocessed").isin(
    sel_cats if sel_cats else (CATEGORIES + ["Unprocessed"])
)
mask &= df["urgency"].fillna(0).between(urg_min, urg_max)
if only_drafts:
    mask &= df["draft_reply"].notna()
if search_q:
    q = search_q.lower()
    mask &= (
        df["subject"].fillna("").str.lower().str.contains(q, regex=False)
        | df["sender"].fillna("").str.lower().str.contains(q, regex=False)
    )

filtered = df[mask].reset_index(drop=True)
st.caption(f"Showing **{len(filtered)}** of {len(df)} emails")


# ---- Email table ----------------------------------------------------------

def _cat_display(cat: str | None) -> str:
    cat = cat or "Unprocessed"
    return f"{CATEGORY_ICONS.get(cat, '·')} {cat}"


def _status_display(r: pd.Series) -> str:
    if pd.notna(r["sent_at"]):
        return "✓ Sent"
    if r.get("approved"):
        return "✓ Approved"
    if r.get("draft_reply"):
        return "• Draft"
    return "—"


table = filtered.copy()
table["received_at"] = table["received_at"].dt.strftime("%b %d %H:%M")
table["category"] = table["category"].apply(_cat_display)
table["status"] = table.apply(_status_display, axis=1)

view_cols = ["received_at", "category", "urgency", "sender", "subject", "summary", "status"]
event = st.dataframe(
    table[view_cols],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "received_at": st.column_config.TextColumn("Received", width="medium"),
        "category": st.column_config.TextColumn("Category", width="medium"),
        "urgency": st.column_config.ProgressColumn(
            "Urgency", format="%d", width="small", min_value=0, max_value=5
        ),
        "sender": st.column_config.TextColumn("Sender", width="medium"),
        "subject": st.column_config.TextColumn("Subject", width="large"),
        "summary": st.column_config.TextColumn("AI Summary", width="large"),
        "status": st.column_config.TextColumn("Status", width="medium"),
    },
    height=480,
)

# ---- Detail view ---------------------------------------------------------
sel_rows = event.selection.rows if event and event.selection else []
if sel_rows:
    row = filtered.iloc[sel_rows[0]].to_dict()

    cat = row.get("category") or "Unprocessed"
    cat_color = CATEGORY_COLORS.get(cat, "#9ca3af")
    cat_icon = CATEGORY_ICONS.get(cat, "·")
    is_sent = _is_set(row.get("sent_at"))
    received_str = pd.to_datetime(row["received_at"]).strftime("%b %d, %Y at %H:%M")

    status_text = _status_display(row)
    status_style = {
        "✓ Sent": ("#DCFCE7", "#15803D"),
        "✓ Approved": ("#DBEAFE", "#1D4ED8"),
        "• Draft": ("#FEF3C7", "#B45309"),
        "—": ("#F3F4F6", "#6B7280"),
    }[status_text]

    st.divider()
    st.markdown(
        f"""
<div class="detail-header">
    <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;'>
        <div style='flex:1; min-width:0;'>
            <span style='color:{cat_color}; font-size:0.78rem; font-weight:700; letter-spacing:0.05em;'>
                {cat_icon} {cat.upper()}
            </span>
            <span class="pill" style='background:{status_style[0]}; color:{status_style[1]}; margin-left:0.5rem;'>
                {status_text}
            </span>
            <h3 style='margin:0.4rem 0 0.3rem 0; word-break:break-word;'>{row.get('subject') or '(no subject)'}</h3>
            <p style='margin:0; color:#6b7280;'>
                From <strong>{row.get('sender')}</strong> · {received_str}
            </p>
        </div>
        <div style='text-align:right; min-width:80px;'>
            <div style='font-size:0.72rem; color:#6b7280; letter-spacing:0.05em;'>URGENCY</div>
            <div style='font-size:1.8rem; font-weight:700; color:{cat_color}; line-height:1;'>
                {int(row.get('urgency') or 0)}<span style='font-size:1rem; opacity:0.5;'>/5</span>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if is_sent:
        st.success(
            f"✓ Reply sent on {pd.to_datetime(row['sent_at']).strftime('%b %d at %H:%M')}"
        )

    if cat == "Newsletters":
        unsub_url = _extract_unsub_url(row.get("list_unsubscribe") or "")
        if unsub_url:
            st.info(f"📭 [Unsubscribe from this newsletter]({unsub_url})")

    # Entities
    ents = row.get("entities") or {}
    if any(ents.get(k) for k in ("deadlines", "names", "companies")):
        with st.expander("🏷️ Extracted information", expanded=True):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.markdown("**📅 Deadlines**")
                if ents.get("deadlines"):
                    for d in ents["deadlines"]:
                        st.markdown(f"- {d}")
                else:
                    st.caption("None detected")
            with ec2:
                st.markdown("**👥 People**")
                if ents.get("names"):
                    for n in ents["names"]:
                        st.markdown(f"- {n}")
                else:
                    st.caption("None detected")
            with ec3:
                st.markdown("**🏢 Companies**")
                if ents.get("companies"):
                    for c in ents["companies"]:
                        st.markdown(f"- {c}")
                else:
                    st.caption("None detected")

    # Body and draft
    body_col, draft_col = st.columns(2)
    with body_col:
        st.markdown("##### 📧 Original message")
        st.text_area(
            "body",
            row["body"] or "(empty)",
            height=380,
            label_visibility="collapsed",
            disabled=True,
        )
    with draft_col:
        st.markdown("##### ✏️ AI draft reply")
        if row["draft_reply"]:
            edited = st.text_area(
                "draft",
                row["draft_reply"],
                height=300,
                label_visibility="collapsed",
                key=f"draft_{row['email_id']}",
                disabled=is_sent,
            )

            approved = bool(row.get("approved"))

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "💾 Save",
                    key=f"save_{row['email_id']}",
                    use_container_width=True,
                    disabled=is_sent,
                ):
                    update_draft(row["email_id"], edited)
                    st.toast("Draft saved")
                    st.rerun()
            with b2:
                approve_now = not approved
                label = "✅ Approve" if approve_now else "↩️ Unapprove"
                if st.button(
                    label,
                    key=f"approve_{row['email_id']}",
                    use_container_width=True,
                    type="primary",
                    disabled=is_sent,
                ):
                    set_approval(row["email_id"], approve_now)
                    st.rerun()
            with b3:
                if is_sent:
                    st.button(
                        "✓ Sent",
                        key=f"sent_{row['email_id']}",
                        use_container_width=True,
                        disabled=True,
                    )
                else:
                    if st.button(
                        "📤 Send",
                        key=f"send_{row['email_id']}",
                        use_container_width=True,
                        disabled=not approved,
                        help="Sends to original sender, threaded correctly"
                        if approved
                        else "Approve the draft first",
                    ):
                        if edited and edited != row["draft_reply"]:
                            update_draft(row["email_id"], edited)
                            row["draft_reply"] = edited
                        do_send(row)
                        st.rerun()

            if approved and not is_sent:
                st.caption(
                    f"📨 Will reply to **{row.get('sender_email') or '(unknown)'}**"
                )
        else:
            st.info(
                "No draft available — generation may have failed for this "
                "email. Try **♻️ Re-process all** in the sidebar."
            )
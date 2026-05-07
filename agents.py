"""LangGraph workflow for triage, analysis, and drafting.

Architecture:

    START
      │
      ▼
   triage  ────► analyze ──┬─► draft ─► END
                           │
                           └─► END   (when not Job/Important)

Plus a `quick_classify` rule-based pre-filter that short-circuits the LLM
for obvious newsletter / cold-marketing senders before the graph runs.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from ollama import Client

from config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

CATEGORIES: list[str] = [
    "Job Related",
    "Marketing/Spam",
    "Important/Action Required",
    "Newsletters",
    "Personal",
]
DRAFT_CATEGORIES: set[str] = set(CATEGORIES)  # draft for every category

_client = Client(host=OLLAMA_HOST)


# ---------- State -----------------------------------------------------------

class EmailState(TypedDict, total=False):
    """LangGraph state. `total=False` lets nodes return partial updates."""
    email_id: str
    subject: str
    sender: str
    body: str
    category: str
    urgency: int
    analysis: dict          # {summary, deadlines, names, companies}
    draft_reply: Optional[str]
    error: Optional[str]


# ---------- Ollama helpers --------------------------------------------------

def _ollama_json(system: str, user: str) -> dict:
    """Call Ollama in JSON mode and parse. Handles occasional preamble."""
    resp = _client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": 0.1},
    )
    content = resp["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return valid JSON: {content[:200]}")


def _truncate(s: str, n: int = 2000) -> str:
    return s[:n] + ("…[truncated]" if len(s) > n else "")


# ---------- Rule-based pre-filter -------------------------------------------

# Sender local-parts that almost always indicate machine-generated mail
_NOREPLY_PATTERNS: tuple[str, ...] = (
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "newsletter", "notifications", "updates", "digest",
    "marketing", "campaigns", "info@",
)

# Subject patterns indicating heavy promotional content
_MARKETING_SUBJECT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b\d{1,3}%\s*off\b", re.I),
    re.compile(r"\blimited[- ]time\b", re.I),
    re.compile(r"\bsale\s+ends\b", re.I),
    re.compile(r"\bflash\s+sale\b", re.I),
    re.compile(r"\bsave\s+\$?\d", re.I),
)

# Bulk-sender domains commonly used for cold marketing infrastructure
_BULK_DOMAINS: set[str] = {
    "mailgun.net", "sendgrid.net", "mandrillapp.com", "amazonses.com",
    "mailchimp.com", "constantcontact.com",
}


def quick_classify(email: dict) -> Optional[dict]:
    """Cheap, conservative pre-filter. Returns a result dict only when confident.

    The bar is high: false positives mean missing a real email. We only short
    circuit when at least two strong signals agree. Otherwise we return None
    and let the LLM decide.
    """
    sender_email = (email.get("sender_email") or "").lower()
    sender_local = sender_email.split("@", 1)[0] if "@" in sender_email else ""
    sender_domain = sender_email.split("@", 1)[1] if "@" in sender_email else ""
    subject = email.get("subject", "") or ""
    has_unsub = bool(email.get("list_unsubscribe"))

    is_noreply = any(p in sender_email for p in _NOREPLY_PATTERNS)
    is_promo_subject = any(p.search(subject) for p in _MARKETING_SUBJECT_PATTERNS)
    is_bulk_domain = sender_domain in _BULK_DOMAINS

    # Strong newsletter signal: List-Unsubscribe header AND machine sender
    if has_unsub and is_noreply and not is_promo_subject:
        return {
            "category": "Newsletters",
            "urgency": 1,
            "analysis": {
                "summary": "(pre-filtered as newsletter — no LLM call)",
                "deadlines": [], "names": [], "companies": [],
            },
            "draft_reply": None,
            "_prefiltered": True,
        }

    # Strong marketing signal: promo subject + (unsubscribe OR bulk infra)
    if is_promo_subject and (has_unsub or is_bulk_domain):
        return {
            "category": "Marketing/Spam",
            "urgency": 1,
            "analysis": {
                "summary": "(pre-filtered as marketing — no LLM call)",
                "deadlines": [], "names": [], "companies": [],
            },
            "draft_reply": None,
            "_prefiltered": True,
        }

    return None


# ---------- Triage node -----------------------------------------------------

TRIAGE_SYSTEM = f"""You are an email triage assistant. Classify each email into ONE category:
{', '.join(CATEGORIES)}

Definitions:
- "Job Related": recruiter outreach, interview invitations, offer letters, application status, hiring manager replies.
- "Marketing/Spam": product promotions, sales pitches, COLD SALES OUTREACH (someone selling YOU something), discounts, "limited time offers". Cold sales emails go here even if they mention your job title.
- "Important/Action Required": time-sensitive, requires reply or action — financial/legal/account alerts, deadlines, bills, security warnings.
- "Newsletters": digest-style content, blog roundups, scheduled platform updates with no required action.
- "Personal": friends, family, casual one-to-one conversation.

Also assign urgency 1 (junk) – 5 (drop everything).

Return ONLY JSON: {{"category": "<exact category name>", "urgency": <int>, "reason": "<short>"}}"""

TRIAGE_FEWSHOT = """Examples:

Email:
Subject: Software Engineer role at Stripe — quick chat?
From: Jane Recruiter <jane@stripe.com>
Body: Hi! I came across your profile and would love to discuss our backend role. Are you free Thursday?
JSON: {"category": "Job Related", "urgency": 4, "reason": "recruiter outreach with meeting request"}

Email:
Subject: 🔥 50% OFF EVERYTHING — Today Only!
From: DealHub <deals@dealhub.io>
Body: Massive savings on hundreds of products. Shop now before midnight.
JSON: {"category": "Marketing/Spam", "urgency": 1, "reason": "promotional discount blast"}

Email:
Subject: Boost your engineering team's velocity with our AI code review
From: Alex from PipelineAI <alex@pipelineai.com>
Body: Hey, saw you're an engineer at Acme. Want to 3x your team's PR throughput? We help teams like yours...
JSON: {"category": "Marketing/Spam", "urgency": 1, "reason": "cold sales outreach selling a product, not offering a job"}

Email:
Subject: Action required: verify your bank login
From: Chase Alerts <noreply@chase.com>
Body: We detected an unusual sign-in. Please review within 24 hours.
JSON: {"category": "Important/Action Required", "urgency": 5, "reason": "security alert with deadline"}

Email:
Subject: Your weekly TLDR — top stories in AI
From: TLDR <newsletter@tldrnewsletter.com>
Body: Here are this week's top stories...
JSON: {"category": "Newsletters", "urgency": 1, "reason": "scheduled digest"}

Email:
Subject: dinner saturday?
From: Mom <mom@gmail.com>
Body: hey honey, are you free saturday for dinner?
JSON: {"category": "Personal", "urgency": 3, "reason": "family social plans"}
"""


def triage_node(state: EmailState) -> EmailState:
    """Classify and score urgency."""
    user_msg = (
        TRIAGE_FEWSHOT
        + "\nNow classify this email:\n"
        + f"Subject: {state['subject']}\n"
        + f"From: {state['sender']}\n"
        + f"Body: {_truncate(state.get('body', ''))}\n"
        + "JSON:"
    )
    try:
        result = _ollama_json(TRIAGE_SYSTEM, user_msg)
        category = result.get("category", "Personal")
        if category not in CATEGORIES:
            for known in CATEGORIES:
                if known.lower() in category.lower() or category.lower() in known.lower():
                    category = known
                    break
            else:
                category = "Personal"

        urgency = max(1, min(5, int(result.get("urgency", 2))))
        logger.info(
            "Triage %s → %s (urgency=%d, reason=%s)",
            state["email_id"][:10], category, urgency, result.get("reason", ""),
        )
        return {"category": category, "urgency": urgency}
    except Exception as e:
        logger.exception("Triage failed for %s", state.get("email_id"))
        return {"category": "Personal", "urgency": 1, "error": f"triage: {e}"}


# ---------- Analysis node ---------------------------------------------------

ANALYSIS_SYSTEM = """You are an email analyst. Extract structured information.
Return ONLY JSON with this exact shape:
{
  "summary": "<one concise sentence: what is this about and what is asked>",
  "deadlines": ["<explicit dates/deadlines, or empty list>"],
  "names": ["<people names mentioned, or empty>"],
  "companies": ["<company/org names mentioned, or empty>"]
}
Do not invent information. If a field is not present, return an empty list."""


def analysis_node(state: EmailState) -> EmailState:
    user_msg = (
        f"Subject: {state['subject']}\n"
        f"From: {state['sender']}\n"
        f"Body: {_truncate(state.get('body', ''))}\n"
        "JSON:"
    )
    try:
        result = _ollama_json(ANALYSIS_SYSTEM, user_msg)
        analysis = {
            "summary": str(result.get("summary", ""))[:500],
            "deadlines": list(result.get("deadlines") or []),
            "names": list(result.get("names") or []),
            "companies": list(result.get("companies") or []),
        }
        logger.debug("Analysis %s → %s", state["email_id"][:10], analysis["summary"])
        return {"analysis": analysis}
    except Exception as e:
        logger.exception("Analysis failed for %s", state.get("email_id"))
        return {
            "analysis": {
                "summary": "(analysis failed)",
                "deadlines": [], "names": [], "companies": [],
            },
            "error": f"analysis: {e}",
        }


# ---------- Drafting node ---------------------------------------------------

DRAFT_SYSTEM = """You write email replies on behalf of the user. Tone and length depend on the email category.

Per-category guidance:
- "Job Related": professional and warm. 3–6 sentences. Confirm interest, propose times if a meeting is requested, ask clarifying questions if details are missing.
- "Important/Action Required": brief and direct. 2–4 sentences. Acknowledge the action, confirm next steps or ask for clarification.
- "Personal": match the sender's warmth and informality. Length matches theirs (short for short).
- "Newsletters": a 1–2 sentence draft that the user will likely NOT send — could be a polite "thanks, will read later" or empty pleasantry. Keep it minimal.
- "Marketing/Spam": a polite, firm decline in 1–2 sentences. Example: "Thanks, but I'm not interested at this time. Please remove me from this list." Do not engage with the offer.

Universal rules:
- Use placeholders like [YOUR NAME] or [DATE] when info is missing — never invent facts.
- Do NOT include a subject line, "Sent from my iPhone" footer, or quoted thread.

Return ONLY JSON: {"draft": "<reply body, plain text with line breaks>"}"""


def drafting_node(state: EmailState) -> EmailState:
    summary = state.get("analysis", {}).get("summary", "")
    user_msg = (
        f"Original email:\n"
        f"Subject: {state['subject']}\n"
        f"From: {state['sender']}\n"
        f"Body: {_truncate(state.get('body', ''))}\n\n"
        f"Context: {summary}\n"
        f"Category: {state.get('category')}\n\n"
        "Write the reply.\nJSON:"
    )
    try:
        result = _ollama_json(DRAFT_SYSTEM, user_msg)
        draft = str(result.get("draft", "")).strip() or None
        return {"draft_reply": draft}
    except Exception as e:
        logger.exception("Draft failed for %s", state.get("email_id"))
        return {"draft_reply": None, "error": f"draft: {e}"}


# ---------- Graph wiring ----------------------------------------------------

def _route_after_analysis(state: EmailState) -> str:
    return "draft" if state.get("category") in DRAFT_CATEGORIES else "skip"


def build_graph():
    g = StateGraph(EmailState)
    g.add_node("triage", triage_node)
    g.add_node("analyze", analysis_node)
    g.add_node("draft", drafting_node)

    g.add_edge(START, "triage")
    g.add_edge("triage", "analyze")
    g.add_edge("analyze", "draft")
    g.add_edge("draft", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph compiled")
    return _graph


# ---------- Top-level helper ------------------------------------------------

def process_email(email: dict) -> dict:
    """Run pre-filter then the full graph. Returns DB-ready dict."""
    quick = quick_classify(email)
    if quick is not None:
        logger.info(
            "Pre-filtered %s → %s (still generating a draft)",
            email["email_id"][:10], quick["category"],
        )

    initial: EmailState = {
        "email_id": email["email_id"],
        "subject": email["subject"],
        "sender": email["sender"],
        "body": email.get("body", ""),
    }
    final = get_graph().invoke(initial)
    analysis = final.get("analysis") or {}
    return {
        **email,
        "category": final.get("category"),
        "urgency": final.get("urgency", 1),
        "summary": analysis.get("summary"),
        "entities": {
            "deadlines": analysis.get("deadlines", []),
            "names": analysis.get("names", []),
            "companies": analysis.get("companies", []),
        },
        "draft_reply": final.get("draft_reply"),
        "processed_at": datetime.utcnow(),
        "error": final.get("error"),
    }

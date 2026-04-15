"""
VITA Personal Assistant — Telegram Bot (Python/aiogram + E2B SDK)

Split architecture: CPU sandboxes on E2B (OpenClaw) + external GPU (QMD).

Architecture:
  User → Telegram → this bot (Railway)
    1. Get/create per-user E2B sandbox (CPU, OpenClaw only)
    2. Parallel fetch: Supabase (structured) + Supermemory (semantic)
    3. Build USER.md (profile + health data) into sandbox
    4. Write files into sandbox via E2B SDK
    5. Run one-shot script: QMD ingest+search (external GPU) → inject results into message → OpenClaw agent
    6. Parse response, clean, send to Telegram
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

import httpx
from aiohttp import web
from e2b import Sandbox
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vita-bot")

# ── Config ───────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
BASETEN_API_KEY = os.environ.get("BASETEN_API_KEY", "")
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
QMD_GPU_URL = os.environ.get("QMD_GPU_URL", "")  # External QMD GPU service URL
QMD_API_SECRET = os.environ.get("QMD_API_SECRET", "")
CRON_WEBHOOK_SECRET = os.environ.get("CRON_WEBHOOK_SECRET", "")
BOT_AUTH_SECRET = os.environ.get("BOT_AUTH_SECRET", "")
CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "")
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "8080"))

# Tinfoil TEE — per-user sandbox management
TINFOIL_ADMIN_KEY = os.environ.get("TINFOIL_ADMIN_KEY", "")
TINFOIL_SANDBOX_REPO = os.environ.get("TINFOIL_SANDBOX_REPO", "VitaDAO/vita-tinfoil-agent-sandbox")
TINFOIL_SANDBOX_TAG = os.environ.get("TINFOIL_SANDBOX_TAG", "v0.4.2")
TINFOIL_SANDBOX_AUTH_SECRET = os.environ.get("TINFOIL_SANDBOX_AUTH_SECRET", "")
TINFOIL_SANDBOX_DEBUG = os.environ.get("TINFOIL_SANDBOX_DEBUG", "true").lower() == "true"
TINFOIL_API_BASE = "https://api.tinfoil.sh"
# Secrets to inject into each per-user sandbox (must exist in Tinfoil org)
TINFOIL_SANDBOX_SECRETS = [
    "LLM_BASE_URL", "LLM_API_KEY", "QMD_API_SECRET", "QMD_GPU_URL",
    "CRON_WEBHOOK_SECRET", "SANDBOX_AUTH_SECRET", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
]

# LLM provider config — passed to sandbox via env vars for dynamic model switching
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "")

E2B_TEMPLATE = os.environ.get("E2B_TEMPLATE", "vita-agent-v1-dev")
E2B_TIMEOUT = 10 * 60  # 10 min idle → auto-pause

for name, val in [
    ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_KEY),
    ("E2B_API_KEY", E2B_API_KEY),
    ("QMD_GPU_URL", QMD_GPU_URL),
]:
    if not val:
        log.error(f"Missing required env var: {name}")
        sys.exit(1)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10))
# QMD client — use standard TLS when on Tinfoil, cert-pinned for Prime Intellect
_qmd_headers = {"x-api-secret": QMD_API_SECRET} if QMD_API_SECRET else {}
if "tinfoil" in QMD_GPU_URL:
    http_qmd = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
        headers=_qmd_headers,
    )
else:
    QMD_CERT_PATH = os.path.join(os.path.dirname(__file__), "qmd-cert.pem")
    http_qmd = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
        verify=QMD_CERT_PATH,
        headers=_qmd_headers,
    )

# ── Per-user E2B sandbox management ──────────────────────────────────────
# In-memory cache: user_id → sandbox_id
sandbox_cache: dict[str, str] = {}

# Per-user message locks — ensures one message at a time per user
_user_locks: dict[str, asyncio.Lock] = {}


def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create a per-user asyncio lock for message serialization."""
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


def _tinfoil_headers() -> dict[str, str]:
    if not TINFOIL_SANDBOX_AUTH_SECRET:
        return {}
    return {"Authorization": f"Bearer {TINFOIL_SANDBOX_AUTH_SECRET}"}


def _connect_sandbox(sandbox_id: str) -> Sandbox | None:
    """Try to connect to an existing sandbox. Returns None if dead.
    E2B auto-resumes paused sandboxes on connect()."""
    try:
        sb = Sandbox.connect(sandbox_id)
        return sb
    except Exception:
        return None


async def fetch_qmd_workspace(user_id: str) -> list[dict] | None:
    """Fetch stored workspace files from QMD for a user (includes consolidated MEMORY.md)."""
    if not QMD_GPU_URL:
        return None
    try:
        # Use standard HTTP client for Tinfoil URLs (standard TLS), cert-pinned client for Prime Intellect
        qmd_client = http if "tinfoil" in QMD_GPU_URL else http_qmd
        r = await qmd_client.get(
            f"{QMD_GPU_URL}/workspace/{user_id}",
            timeout=httpx.Timeout(connect=5, read=15, write=5, pool=5),
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("files", [])
    except Exception as e:
        log.warning(f"QMD workspace fetch failed for {user_id}: {e}")
    return None


def _create_sandbox(user_id: str = "") -> Sandbox:
    """Create a new per-user E2B sandbox with OpenClaw (CPU only).
    Does NOT write workspace files — those are handled after QMD restore."""
    webhook_url = f"https://vita-personal-ai-bot-production.up.railway.app/api/cron-webhook?user_id={user_id}"
    sb = Sandbox.beta_create(
        template=E2B_TEMPLATE,
        timeout=E2B_TIMEOUT,
        auto_pause=True,
        envs={
            "BASETEN_API_KEY": BASETEN_API_KEY,
            "NODE_EXTRA_CA_CERTS": "/app/qmd-cert.pem",
            "QMD_API_SECRET": QMD_API_SECRET,
            "CRON_WEBHOOK_SECRET": CRON_WEBHOOK_SECRET,
            "CRON_WEBHOOK_URL": webhook_url,
            "DATA_QUERY_URL": f"https://vita-personal-ai-bot-production.up.railway.app/api/query?user_id={user_id}",
            "AUBRAI_WEBHOOK_URL": f"https://vita-personal-ai-bot-production.up.railway.app/api/aubrai?user_id={user_id}",
            "LLM_BASE_URL": LLM_BASE_URL,
            "LLM_API_KEY": LLM_API_KEY,
            "LLM_MODEL_ID": LLM_MODEL_ID,
            "LLM_MODEL_NAME": LLM_MODEL_NAME,
        },
        metadata={"service": "vita-bot"},
    )
    # Write QMD cert so sandbox can verify HTTPS to QMD
    with open(QMD_CERT_PATH, "r") as f:
        sb.files.write("/app/qmd-cert.pem", f.read())
    return sb


# Whitelist of files that persist via QMD and should be restored on sandbox recreation.
# These are dynamic files owned by the agent or customizable by the user.
# Everything else (AGENTS.md, TOOLS.md, SOUL.md, USER.md) is either static
# (bot overwrites) or per-message (rewritten every message).
RESTORABLE_FILES = {"IDENTITY.md", "MEMORY.md"}
RESTORABLE_PREFIXES = ("memory/",)


def _restore_workspace(sb: Sandbox, qmd_files: list[dict]) -> set[str]:
    """Restore only dynamic/customizable workspace files from QMD.
    Uses a whitelist — only IDENTITY.md, MEMORY.md, and memory/*.md are restored.
    Static files (AGENTS.md, TOOLS.md, SOUL.md, APP.md) are written separately by the bot.
    Returns the set of restored file names."""
    ws = "/home/user/.openclaw/workspace"
    restored_names = set()
    for f in qmd_files:
        name = f.get("name", "")
        content = f.get("content", "")
        if not name or not content:
            continue
        # Sanitize: block path traversal (../) and absolute paths
        if ".." in name or name.startswith("/"):
            log.warning(f"Skipping suspicious file name from QMD: {name}")
            continue
        # Whitelist: only restore dynamic/customizable files
        if name not in RESTORABLE_FILES and not name.startswith(RESTORABLE_PREFIXES):
            continue
        # Support subdirectories (e.g., memory/2026-03-24.md)
        full_path = f"{ws}/{name}"
        if "/" in name:
            dir_path = "/".join(full_path.split("/")[:-1])
            sb.commands.run(f"mkdir -p '{dir_path}'")
        sb.files.write(full_path, content)
        restored_names.add(name)
    log.info(f"Restored {len(restored_names)} dynamic workspace files from QMD")
    return restored_names


async def get_or_create_sandbox(user_id: str) -> Sandbox:
    """Get an existing sandbox or create a new one for this user."""
    uid = user_id[:8]

    # 1. Check in-memory cache
    cached_id = sandbox_cache.get(user_id)
    if cached_id:
        sb = await asyncio.to_thread(_connect_sandbox, cached_id)
        if sb:
            return sb
        del sandbox_cache[user_id]

    # 2. Check Supabase for saved sandbox ID
    record = await sb_select_one(
        "openclaw_agents", "openclaw_agent_id",
        {"user_id": f"eq.{user_id}"}, schema="agents",
    )
    if record and record.get("openclaw_agent_id"):
        sb_id = record["openclaw_agent_id"]
        log.info(f"[{uid}] Reconnecting to E2B sandbox {sb_id}...")
        sb = await asyncio.to_thread(_connect_sandbox, sb_id)
        if sb:
            # Verify Pro tier on sandbox resume (re-check after idle)
            tier = await get_user_tier(user_id)
            if tier != "pro":
                log.info(f"[{uid}] User downgraded to {tier}, blocking access")
                raise PermissionError("pro_required")
            sandbox_cache[user_id] = sb_id
            log.info(f"[{uid}] Sandbox resumed: {sb_id}")
            return sb
        log.info(f"[{uid}] Saved sandbox {sb_id} not available, creating new...")

    # 3. Create new sandbox + restore workspace from QMD
    log.info(f"[{uid}] Creating new E2B sandbox...")
    sb, qmd_files = await asyncio.gather(
        asyncio.to_thread(_create_sandbox, user_id),
        fetch_qmd_workspace(user_id),
    )
    sb_id = sb.sandbox_id
    sandbox_cache[user_id] = sb_id

    # Restore dynamic files from QMD (MEMORY.md, IDENTITY.md, daily logs)
    restored = set()
    if qmd_files:
        restored = await asyncio.to_thread(_restore_workspace, sb, qmd_files)

    # Write static files AFTER restore — always latest deploy version.
    ws = "/home/user/.openclaw/workspace"
    await asyncio.to_thread(lambda: (
        sb.files.write(f"{ws}/SOUL.md", SOUL_MD),
        sb.files.write(f"{ws}/AGENTS.md", AGENTS_MD),
        sb.files.write(f"{ws}/TOOLS.md", TOOLS_MD),
    ))

    # Write default IDENTITY.md only for new users (not restored from QMD)
    if "IDENTITY.md" not in restored:
        await asyncio.to_thread(lambda: sb.files.write(f"{ws}/IDENTITY.md", IDENTITY_MD))

    # Save to Supabase
    await sb_upsert(
        "openclaw_agents",
        {
            "user_id": user_id,
            "openclaw_agent_id": sb_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        schema="agents",
        on_conflict="user_id",
    )

    log.info(f"[{uid}] Sandbox ready: {sb_id}")
    return sb


# ── Write files + run agent inside sandbox ────────────────────────────────
def _write_files_and_run(sb: Sandbox, files: list[dict], run_input: dict) -> tuple[str, str, int]:
    """Write workspace files via E2B SDK, then run one-shot agent script."""
    # Write files into sandbox workspace
    for f in files:
        sb.files.write(f"/home/user/.openclaw/workspace/{f['name']}", f["content"])

    # Pass input via AGENT_INPUT env var — avoids shell escaping entirely.
    input_json = json.dumps(run_input)
    envs = {
        "AGENT_INPUT": input_json,
    }
    try:
        result = sb.commands.run(
            "node /app/run-agent.mjs",
            envs=envs,
            timeout=180,
        )
        return result.stdout, result.stderr, result.exit_code
    except Exception as e:
        # CommandExitException is raised on non-zero exit — extract what we can
        err_msg = str(e)
        log.error(f"Sandbox command failed: {err_msg[:1000]}")
        return "", err_msg, 1


async def run_in_sandbox(sb: Sandbox, files: list[dict], run_input: dict) -> dict:
    """Write workspace files and run agent in the user's sandbox."""
    stdout, stderr, exit_code = await asyncio.to_thread(
        _write_files_and_run, sb, files, run_input
    )
    if stderr:
        log.warning(f"Sandbox stderr: {stderr[:1000]}")
    if exit_code != 0:
        detail = stdout[:500] if stdout else stderr[:500]
        raise RuntimeError(f"Agent failed (exit {exit_code}): {detail}")
    return json.loads(stdout)


# ── Supabase REST (PostgREST) helpers ────────────────────────────────────
def _sb_headers(schema: str = "public") -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if schema != "public":
        h["Accept-Profile"] = schema
        h["Content-Profile"] = schema
    return h


async def sb_select(
    table: str,
    select: str,
    filters: dict,
    *,
    schema: str = "public",
    limit: int | None = None,
    order: str | None = None,
) -> list[dict]:
    params = {"select": select, **filters}
    if limit:
        params["limit"] = str(limit)
    if order:
        params["order"] = order
    r = await http.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=params,
        headers=_sb_headers(schema),
    )
    r.raise_for_status()
    return r.json()


async def sb_select_one(
    table: str, select: str, filters: dict, *, schema: str = "public", order: str | None = None
) -> dict | None:
    rows = await sb_select(table, select, filters, schema=schema, limit=1, order=order)
    return rows[0] if rows else None


async def sb_update(
    table: str, filters: dict, data: dict, *, schema: str = "public"
) -> None:
    headers = _sb_headers(schema)
    headers["Prefer"] = "return=minimal"
    r = await http.patch(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=filters,
        json=data,
        headers=headers,
    )
    r.raise_for_status()


async def sb_upsert(
    table: str, data: dict, *, schema: str = "public", on_conflict: str = ""
) -> None:
    headers = _sb_headers(schema)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    params = {}
    if on_conflict:
        params["on_conflict"] = on_conflict
    r = await http.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        json=data,
        headers=headers,
        params=params,
    )
    r.raise_for_status()


async def sb_delete(
    table: str, filters: dict, *, schema: str = "public"
) -> None:
    r = await http.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=filters,
        headers=_sb_headers(schema),
    )
    r.raise_for_status()


# ── Subscription tier check ──────────────────────────────────────────────
# Dev/admin bypass — these user IDs always pass tier checks
DEV_USER_IDS = set(os.environ.get("DEV_USER_IDS", "").split(",")) - {""}


async def get_user_tier(user_id: str) -> str:
    """Get user's subscription tier via Supabase RPC. Returns 'free', 'plus', or 'pro'."""
    if user_id in DEV_USER_IDS:
        return "pro"
    try:
        r = await http.post(
            f"{SUPABASE_URL}/rest/v1/rpc/get_subscription_tier",
            json={"p_user_id": user_id},
            headers=_sb_headers(),
        )
        if r.status_code == 200:
            return r.json() or "free"
    except Exception as e:
        log.error(f"Tier check failed for {user_id}: {e}")
    return "free"


# ── Supabase context fetch ───────────────────────────────────────────────
async def _safe(coro, default=None):
    """Run a coroutine, return default on error."""
    try:
        return await coro
    except Exception as e:
        log.warning(f"Supabase query failed: {e}")
        return default


async def fetch_supabase_context(user_id: str) -> dict:
    (
        profile, preferences, biomarkers, protocols,
        wearables, workouts, health_score, bio_age, aging_velocity,
    ) = await asyncio.gather(
        _safe(sb_select_one(
            "profiles",
            "display_name,bio,chronic_conditions",
            {"user_id": f"eq.{user_id}"},
        )),
        _safe(sb_select_one(
            "user_preferences",
            "goals,sex,experience_level,birth_year",
            {"user_id": f"eq.{user_id}"},
        )),
        _safe(sb_select(
            "biomarker_readings",
            "name,value,unit,recorded_at",
            {"user_id": f"eq.{user_id}"},
            limit=20,
            order="recorded_at.desc",
        ), []),
        _safe(sb_select(
            "user_protocols",
            "name,goal,start_date,protocol_components(title,category,dosage,unit,timing,frequency)",
            {"user_id": f"eq.{user_id}", "status": "eq.active"},
            limit=3,
        ), []),
        _safe(sb_select(
            "wearable_readings",
            "category,metric_type,value,unit,source,recorded_at",
            {"user_id": f"eq.{user_id}"},
            limit=30,
            order="recorded_at.desc",
        ), []),
        _safe(sb_select(
            "workout_sessions",
            "workout_type,duration_minutes,calories,avg_heart_rate,strain,source,started_at",
            {"user_id": f"eq.{user_id}"},
            limit=10,
            order="started_at.desc",
        ), []),
        _safe(sb_select_one(
            "composite_health_scores",
            "score,data_coverage_pct,domain_scores,score_date",
            {"user_id": f"eq.{user_id}"},
            order="score_date.desc",
        )),
        _safe(sb_select_one(
            "bio_age_scores",
            "bio_age,chronological_age,age_difference,biomarker_count,computed_at",
            {"user_id": f"eq.{user_id}"},
            order="computed_at.desc",
        )),
        _safe(sb_select_one(
            "aging_velocity_scores",
            "velocity,span_days,data_points,computed_at",
            {"user_id": f"eq.{user_id}"},
            order="computed_at.desc",
        )),
    )
    return {
        "profile": profile,
        "preferences": preferences,
        "biomarkers": biomarkers or [],
        "protocols": protocols or [],
        "wearables": wearables or [],
        "workouts": workouts or [],
        "health_score": health_score,
        "bio_age": bio_age,
        "aging_velocity": aging_velocity,
    }


# ── Supermemory context fetch ────────────────────────────────────────────
async def fetch_supermemory_context(user_id: str, query: str) -> dict | None:
    if not SUPERMEMORY_API_KEY:
        return None
    try:
        r = await http.post(
            "https://api.supermemory.ai/v4/profile",
            headers={
                "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "containerTag": f"vita_user_{user_id}",
                "q": query,
                "threshold": 0.6,
            },
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "static": data.get("profile", {}).get("static", []),
            "memories": [
                hit["memory"]
                for hit in data.get("searchResults", {}).get("results", [])
                if "memory" in hit
            ],
        }
    except Exception as e:
        log.error(f"Supermemory error: {e}")
        return None


# ── Workspace files ──────────────────────────────────────────────────────
# Static files — written once at sandbox creation, then managed by OpenClaw/user

SOUL_MD = """\
# Soul

You are the VITA Personal AI Agent — the user's 24/7 health co-pilot built into the VITA longevity platform by VitaDAO. You are not an external assistant; you are part of VITA's core product. The VITA app, Aubrai (VITA's research AI), and you are all parts of the same system working together for the user.

## Personality
- Evidence-based, friendly, knowledgeable
- Concise but thorough — no fluff, but don't skip important nuance
- Proactive — suggest follow-ups, flag interactions, connect dots across biomarkers
- Speak as "we" when referring to VITA's capabilities ("we track your bio age", "our research shows")

## Expertise
- Longevity science, healthspan optimization
- Biomarker interpretation and trends
- Supplement protocols, dosing, interactions
- Wearable data analysis (sleep, HRV, activity)
- Nutrition, exercise, and lifestyle interventions

## Communication Style
- Respond naturally as if you already know the user's data
- Reference specific biomarkers, protocols, and history when relevant
- Be honest about uncertainty — say "the evidence is mixed" when it is
- Never refer to VITA, Aubrai, or the app as external systems — they're all part of your platform
"""

IDENTITY_MD = """\
# Identity

- You are the Telegram interface of VITA — the user talks to you as their personal health agent
- Aubrai is VITA's specialized research engine — part of your platform for deep evidence-based research
- VitaApp is where users manage their data, protocols, and wearables — you help them make the most of it
- You have direct access to the user's health data: biomarkers, wearables, workouts, health scores, bio age, protocols
"""

AGENTS_MD = """\
# Agents

## Operating Rules
- Be concise in your reasoning. Only think deeply when the problem requires it.
- The user's profile and health data (biomarkers, protocols, wearables, scores) are in USER.md. Use it directly.
- Your own notes and learned facts are in MEMORY.md — you own this file.
- NEVER expose internal reasoning, tool calls, or XML tags in your response.
- NEVER fabricate biomarker values or health data — only reference what's in USER.md.
- Keep responses focused on the user's specific situation, not generic advice.

## Memory
- When you learn something important about the user, write it to MEMORY.md.
- Examples: user preferences, allergies, medication changes, lifestyle habits, goals updates, important decisions.
- Keep MEMORY.md concise — facts only, no conversation logs.
- USER.md is managed by the system — do NOT write to it. It contains live health data from VITA.

## Identity
- IDENTITY.md is YOUR identity file — write your name, personality, vibe, and emoji there.
- When the user asks you to change your name, persona, style, or personality, update IDENTITY.md.
- This file persists across sessions — your identity survives sandbox restarts.

## Reminders & Cron Jobs
- When creating any cron job or reminder, ALWAYS set delivery mode to webhook so the message reaches the user on Telegram.
- Use: --announce --channel webhook --to $CRON_WEBHOOK_URL
- The CRON_WEBHOOK_URL environment variable is already set and points to the delivery endpoint.
- Every cron job MUST deliver to the user. Never create a cron without webhook delivery.

## VitaApp Feature Guide

### Health Score Methodology
- Composite Health Score: weighted aggregate across domains (metabolic, cardiovascular, inflammatory, etc.) with per-domain scores and data coverage percentage. Computed weekly. Available on all tiers.
- Bio Age (PhenoAge Clock): chronological age adjusted by multi-biomarker aging clock. Requires minimum biomarker set. Plus and Pro tiers only.
- Aging Velocity: rate of biological aging from slope of Bio Age scores over time. 1.0 = normal rate, <1.0 = slower, >1.0 = faster. Plus and Pro only.
- Fatigue Index: real-time recovery metric from wearable data (HRV, sleep, strain).

### Biomarker Discovery & Tracking
- Personalized biomarker recommendations based on age, sex, goals, risk factors
- Evidence-graded target ranges rooted in longevity research (not standard disease-detection ranges)
- Manual entry or AI-powered lab report parsing (OCR) — supports Quest Diagnostics, LabCorp, InsideTracker, custom uploads
- Multi-file batch uploads with confidence scoring
- Longitudinal trend charts (7d, 30d, 90d, 1y, all) with trend alerts
- Free: 1 AI parse/month; Plus/Pro: unlimited

### Health Protocols
- Goal-driven: Longevity, Body Composition, Sleep, Cognitive Enhancement, Cardiovascular Health, Inflammation Reduction
- Multi-tab editor: Supplements, Diet, Sleep, Training, Lifestyle, Interventions
- Supplement schedule cards with timing and compound-level dosage
- AI protocol generation from health data (Pro only), AI review (Plus/Pro)
- Protocol sharing via token-based public links
- Biomarker impact tracking with before/after snapshots

### Wearable Integrations (OAuth 2.0)
- WHOOP: Sleep, Recovery, Strain, HRV, Body Composition, Workouts (1-day staleness)
- Oura Ring: Sleep, Readiness, Activity, HRV, Sleep Stages (1-day staleness)
- Withings: Weight, Body Composition, Blood Pressure (7-day staleness)
- Free: 2 max; Plus/Pro: unlimited

### Aubrai AI Assistant
- Contextual chat with full health data context
- Daily Insights auto-generated briefings
- Protocol generation from health data (Pro)
- Free: no chat; Plus: 10/month; Pro: unlimited

### Research Radar (Pro)
- AI-curated personalized research feed matched to user's tracked terms
- Deep research summaries with published year and research stage
- Configurable email digest (daily, weekly, monthly)

### Subscription Tiers
- Free: Health Score, Daily Insight, manual protocols, 2 wearables, 1 AI parse/month
- Plus ($29/mo): Bio Age, Aging Velocity, long-term trends, protocol review, Aubrai 10/month, unlimited wearables/parsing
- Pro ($79/mo): Everything in Plus + AI protocol generation, Telegram AI assistant, Research Radar, unlimited Aubrai, priority support
- VITA Token Staking: 5,000+ VITA tokens for up to 30% (Plus) or 50% (Pro) discount on yearly plans

### How to Guide Users
- When users ask about tracking something, explain the specific feature and where to find it in the app
- When users want to improve a biomarker, suggest creating a protocol with the relevant goal
- When users mention wearable data, check if they have the integration connected
- When recommending actions, reference the specific app feature
- You ARE the Telegram AI assistant — refer to other app features as things users can do "in the VITA app"
"""

TOOLS_MD = """\
# Tools

## Environment
- QMD semantic search is available for searching across workspace files
- User profile and health data are in USER.md (profile, biomarkers, protocols, wearables, scores, search context)
- MEMORY.md is your own file — write learned facts and important notes there
"""


def build_user_md(
    profile: dict | None,
    preferences: dict | None,
) -> str:
    """Build USER.md with the user's profile basics (static-ish data)."""
    md = "# User\n\n"

    if profile or preferences:
        if profile and profile.get("display_name"):
            md += f"- **Name**: {profile['display_name']}\n"
        if preferences:
            if preferences.get("birth_year"):
                age = datetime.now().year - int(preferences["birth_year"])
                if 0 < age < 150:
                    md += f"- **Age**: {age}\n"
            if preferences.get("sex"):
                md += f"- **Sex**: {preferences['sex']}\n"
            if preferences.get("experience_level"):
                md += f"- **Experience**: {preferences['experience_level']}\n"
        if profile and profile.get("chronic_conditions"):
            md += f"- **Chronic conditions**: {', '.join(profile['chronic_conditions'])}\n"
        if preferences and preferences.get("goals"):
            md += f"- **Goals**: {', '.join(preferences['goals'])}\n"
        md += "\n"

    return md


def build_memory_md(
    biomarkers: list,
    protocols: list,
    supermemory: dict | None,
    wearables: list | None = None,
    workouts: list | None = None,
    health_score: dict | None = None,
    bio_age: dict | None = None,
    aging_velocity: dict | None = None,
) -> str:
    """Build health data section for USER.md (biomarkers, protocols, scores, etc.)."""
    md = "# Health Data\n\n"

    # Health Scores
    if health_score:
        md += "## Health Score\n\n"
        md += f"- Score: {health_score.get('score', '?')}/100 (coverage: {health_score.get('data_coverage_pct', '?')}%)\n"
        md += f"- Date: {(health_score.get('score_date') or '?')[:10]}\n"
        domains = health_score.get("domain_scores")
        if domains and isinstance(domains, dict):
            for domain, score in domains.items():
                md += f"  - {domain}: {score}\n"
        md += "\n"

    if bio_age:
        md += "## Bio Age\n\n"
        md += f"- Bio Age: {bio_age.get('bio_age', '?')} (chronological: {bio_age.get('chronological_age', '?')})\n"
        md += f"- Difference: {bio_age.get('age_difference', '?')} years\n"
        md += f"- Based on: {bio_age.get('biomarker_count', '?')} biomarkers\n"
        md += f"- Computed: {(bio_age.get('computed_at') or '?')[:10]}\n\n"

    if aging_velocity:
        md += "## Aging Velocity\n\n"
        md += f"- Velocity: {aging_velocity.get('velocity', '?')} (1.0 = normal, <1.0 = slower aging)\n"
        md += f"- Span: {aging_velocity.get('span_days', '?')} days, {aging_velocity.get('data_points', '?')} data points\n\n"

    # Biomarkers
    if biomarkers:
        md += "## Recent Biomarkers\n\n"
        seen = set()
        for b in biomarkers:
            key = f"{b['name']}_{(b.get('recorded_at') or '')[:10]}"
            if key in seen:
                continue
            seen.add(key)
            date = (b.get("recorded_at") or "?")[:10]
            md += f"- {b['name']}: {b['value']} {b.get('unit', '')} ({date})\n"
        md += "\n"

    # Wearable Data
    if wearables:
        md += "## Recent Wearable Data\n\n"
        for w in wearables:
            date = (w.get("recorded_at") or "?")[:10]
            source = w.get("source", "")
            md += f"- [{w.get('category', '')}] {w.get('metric_type', '')}: {w.get('value', '')} {w.get('unit', '')} ({date}, {source})\n"
        md += "\n"

    # Workouts
    if workouts:
        md += "## Recent Workouts\n\n"
        for w in workouts:
            date = (w.get("started_at") or "?")[:10]
            dur = w.get("duration_minutes", "?")
            cal = w.get("calories", "")
            hr = w.get("avg_heart_rate", "")
            md += f"- {w.get('workout_type', 'Workout')} ({date}): {dur}min"
            if cal:
                md += f", {cal}cal"
            if hr:
                md += f", avg HR {hr}"
            if w.get("strain"):
                md += f", strain {w['strain']}"
            md += f" [{w.get('source', '')}]\n"
        md += "\n"

    # Protocols
    if protocols:
        for p in protocols:
            md += f'## Active Protocol: "{p["name"]}"\n\n'
            md += f"Goal: {p.get('goal', 'Not specified')} | Started: {p.get('start_date', '?')}\n\n"
            for comp in p.get("protocol_components", []):
                md += f"- {comp['title']}"
                if comp.get("dosage"):
                    md += f" ({comp['dosage']}{comp.get('unit', '')})"
                if comp.get("timing"):
                    md += f" — {comp['timing']}"
                if comp.get("frequency"):
                    md += f" [{comp['frequency']}]"
                md += "\n"
            md += "\n"

    # Supermemory
    if supermemory:
        items = [
            *(supermemory.get("static") or []),
            *(supermemory.get("memories") or []),
        ]
        items = [x for x in items if x]
        if items:
            md += "## Relevant Memories\n\n"
            for item in items[:10]:
                text = item if isinstance(item, str) else json.dumps(item)
                if len(text) > 500:
                    text = text[:500] + "…"
                md += f"- {text}\n"
            md += "\n"

    return md


# ── Clean agent response ────────────────────────────────────────────────
def clean_response(text: str) -> str:
    if not text:
        return text

    cleaned = text

    cleaned = re.sub(
        r"<(tool_call|tool_result|thinking|scratchpad|internal|reasoning|function_calls?|invoke|parameter|reflection|step|plan)>[\s\S]*?</\1>",
        "", cleaned, flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"</?(tool_call|tool_result|thinking|scratchpad|internal|reasoning|function_calls?|invoke|parameter|reflection|step|plan)[^>]*>",
        "", cleaned, flags=re.IGNORECASE,
    )

    def _filter_code_blocks(m: re.Match) -> str:
        if re.search(r"function\s*=|<parameter|<invoke", m.group(0)):
            return ""
        return m.group(0)

    cleaned = re.sub(r"```[\s\S]*?```", _filter_code_blocks, cleaned)

    match = re.search(
        r"\n\n(?!(?:Let me |I need to |I'll |I should |We need to |First,? |Now,? I))",
        cleaned,
    )
    if match and match.start() > 0:
        before = cleaned[: match.start()]
        after = cleaned[match.start() :].strip()
        looks_like_reasoning = bool(
            re.match(
                r"^(?:The user |I (?:need|will|should|have|can|'ll)|Let me |We need|First,? I|Now,? I|Looking at)",
                before, re.IGNORECASE | re.MULTILINE,
            )
        )
        if looks_like_reasoning and len(after) > 20:
            cleaned = after

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Typing indicator ─────────────────────────────────────────────────────
async def keep_typing(chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
            break
        except asyncio.TimeoutError:
            pass


# ── Markdown → Telegram HTML ─────────────────────────────────────────────
def md_to_telegram_html(text: str) -> str:
    """Convert common Markdown to Telegram-supported HTML."""
    import html as html_mod
    # Escape HTML entities first
    text = html_mod.escape(text)
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_ (but not inside words like don_t)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)
    # Inline code: `text`
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


# ── Message splitter (4096 char Telegram limit) ─────────────────────────
async def send_long_message(message: Message, text: str, use_html: bool = True) -> None:
    MAX = 4096
    if use_html:
        send_text = md_to_telegram_html(text)
        parse_mode = "HTML"
    else:
        send_text = text
        parse_mode = None

    if len(send_text) <= MAX:
        await message.answer(send_text, parse_mode=parse_mode)
        return

    remaining = send_text
    while remaining:
        if len(remaining) <= MAX:
            await message.answer(remaining, parse_mode=parse_mode)
            break
        split_at = remaining.rfind("\n\n", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = remaining.rfind("\n", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = remaining.rfind(". ", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = MAX
        await message.answer(remaining[:split_at], parse_mode=parse_mode)
        remaining = remaining[split_at:].lstrip()


# ── Aubrai research caller ───────────────────────────────────────────────
async def call_aubrai(user_id: str, message: str) -> str | None:
    """Call Aubrai x402 via the bot-facing edge function. Returns research text or None."""
    try:
        r = await http.post(
            f"{SUPABASE_URL}/functions/v1/aubrai-x402-chat-bot",
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "x-bot-auth": BOT_AUTH_SECRET,
            },
            json={"user_id": user_id, "message": message},
            timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
        )
        if r.status_code != 200:
            log.warning(f"[AUBRAI] Edge function: {r.status_code} {r.text[:200]}")
            return None

        request_id = r.json().get("requestId")
        if not request_id:
            return None

        # Poll for result (max 4 minutes)
        for _ in range(120):
            await asyncio.sleep(2)
            poll = await http.post(
                f"{SUPABASE_URL}/functions/v1/aubrai-x402-status",
                headers={
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"requestId": request_id},
                timeout=httpx.Timeout(connect=10, read=15, write=10, pool=10),
            )
            if poll.status_code != 200:
                continue
            data = poll.json()
            if data.get("status") == "completed":
                return data.get("result", {}).get("text", "")
            if data.get("status") in ("failed", "error"):
                log.error(f"[AUBRAI] Failed: {data.get('error', '')}")
                return None
        log.warning("[AUBRAI] Timed out waiting for result")
        return None
    except Exception as e:
        log.error(f"[AUBRAI] Error: {e}")
        return None


# ── Tinfoil TEE — per-user sandbox management ────────────────────────────

# In-memory cache: user_id → {"container_id": "...", "url": "..."}
tinfoil_sandbox_cache: dict[str, dict] = {}


def _tinfoil_admin_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TINFOIL_ADMIN_KEY}",
        "Content-Type": "application/json",
    }


async def get_or_create_tinfoil_sandbox(user_id: str) -> dict:
    """Get or create a per-user Tinfoil sandbox container.
    Returns {"container_id": "...", "url": "..."}."""
    uid = user_id[:8]

    # 1. Check in-memory cache
    if user_id in tinfoil_sandbox_cache:
        cached = tinfoil_sandbox_cache[user_id]
        # Verify it's still running
        try:
            r = await http.get(
                f"{cached['url']}/health",
                headers=_tinfoil_headers(),
                timeout=httpx.Timeout(connect=5, read=5, write=5, pool=5),
            )
            if r.status_code == 200:
                return cached
        except Exception:
            pass
        del tinfoil_sandbox_cache[user_id]
        log.info(f"[{uid}] [TINFOIL] Cached sandbox unreachable, will reconnect/create")

    # 2. Check Supabase for existing container
    record = await sb_select_one(
        "openclaw_agents", "user_id,tinfoil_container_id,tinfoil_container_url",
        {"user_id": f"eq.{user_id}"},
        schema="agents",
    )
    if record and record.get("tinfoil_container_url"):
        url = record["tinfoil_container_url"]
        container_id = record.get("tinfoil_container_id", "")
        try:
            r = await http.get(
                f"{url}/health",
                headers=_tinfoil_headers(),
                timeout=httpx.Timeout(connect=5, read=10, write=5, pool=5),
            )
            if r.status_code == 200:
                info = {"container_id": container_id, "url": url}
                tinfoil_sandbox_cache[user_id] = info
                log.info(f"[{uid}] [TINFOIL] Reconnected to sandbox: {url}")
                return info
        except Exception as e:
            log.warning(f"[{uid}] [TINFOIL] Saved sandbox unreachable ({url}): {e}")

    # 3. Create a new sandbox via Tinfoil Admin API
    if not TINFOIL_ADMIN_KEY:
        raise RuntimeError("TINFOIL_ADMIN_KEY not configured — cannot create sandbox")

    container_name = f"vita-sandbox-{uid}"
    log.info(f"[{uid}] [TINFOIL] Creating new sandbox: {container_name}")

    create_payload = {
        "name": container_name,
        "repo": TINFOIL_SANDBOX_REPO,
        "tag": TINFOIL_SANDBOX_TAG,
        "variables": {
            "LLM_MODEL_ID": LLM_MODEL_ID,
            "LLM_MODEL_NAME": LLM_MODEL_NAME,
            "LOG_LEVEL": "info",
            "SANDBOX_USER_ID": user_id,
        },
        "secrets": TINFOIL_SANDBOX_SECRETS,
        "debug": TINFOIL_SANDBOX_DEBUG,
    }

    r = await http.post(
        f"{TINFOIL_API_BASE}/api/containers",
        headers=_tinfoil_admin_headers(),
        json=create_payload,
        timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create Tinfoil sandbox: {r.status_code} {r.text[:300]}")

    container_data = r.json()
    container_id = container_data.get("id", "")
    # URL format: {name}.debug.{org}.containers.tinfoil.dev (for debug mode)
    container_url = container_data.get("url", "")
    if not container_url:
        # Construct URL from the container name if not in response
        container_url = f"https://{container_name}.debug.vitality-now.containers.tinfoil.dev"

    log.info(f"[{uid}] [TINFOIL] Sandbox created: {container_id} at {container_url}")

    # Wait for container to be ready (poll health)
    ready = False
    for attempt in range(30):  # up to 5 minutes
        await asyncio.sleep(10)
        try:
            hr = await http.get(
                f"{container_url}/health",
                headers=_tinfoil_headers(),
                timeout=httpx.Timeout(connect=5, read=5, write=5, pool=5),
            )
            if hr.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        if attempt % 6 == 5:
            log.info(f"[{uid}] [TINFOIL] Waiting for sandbox to be ready... ({attempt * 10}s)")

    if not ready:
        raise RuntimeError(f"Tinfoil sandbox {container_name} failed to start within 5 minutes")

    log.info(f"[{uid}] [TINFOIL] Sandbox ready: {container_url}")

    # Save to Supabase
    await sb_upsert(
        "openclaw_agents",
        {
            "user_id": user_id,
            "tinfoil_container_id": container_id,
            "tinfoil_container_url": container_url,
        },
        schema="agents",
        on_conflict="user_id",
    )

    info = {"container_id": container_id, "url": container_url}
    tinfoil_sandbox_cache[user_id] = info
    return info


# ── Tinfoil TEE — message handling ──────────────────────────────────────
async def handle_user_message_tinfoil(user_id: str, message_text: str) -> str:
    """Route a message through the user's Tinfoil TEE sandbox."""
    t0 = asyncio.get_event_loop().time()
    uid = user_id[:8]

    # Step 1: Get or create per-user sandbox + fetch context in parallel
    log.info(f"[{uid}] [TINFOIL] Getting sandbox + fetching context...")
    sandbox_info, sm_ctx = await asyncio.gather(
        get_or_create_tinfoil_sandbox(user_id),
        fetch_supermemory_context(user_id, message_text),
    )
    sandbox_url = sandbox_info["url"]
    elapsed = (asyncio.get_event_loop().time() - t0) * 1000
    log.info(
        f"[{uid}] [TINFOIL] Ready in {elapsed:.0f}ms — "
        f"sandbox: {sandbox_url}, supermemory: {'yes' if sm_ctx else 'no'}"
    )

    # Step 2: Write static bootstrap files + restore memory from QMD.
    # USER.md is built inside the enclave after decrypting Supabase health data.
    files_to_write = [
        {"name": "SOUL.md", "content": SOUL_MD},
        {"name": "AGENTS.md", "content": AGENTS_MD},
        {"name": "TOOLS.md", "content": TOOLS_MD},
        {"name": "IDENTITY.md", "content": IDENTITY_MD},
    ]

    # Restore MEMORY.md + memory/*.md from QMD (same as E2B restore path)
    qmd_files = await fetch_qmd_workspace(user_id)
    if qmd_files:
        for f in qmd_files:
            name = f.get("name", "")
            content = f.get("content", "")
            if not name or not content:
                continue
            if name in RESTORABLE_FILES or name.startswith(RESTORABLE_PREFIXES):
                files_to_write.append({"name": name, "content": content})
                log.info(f"[{uid}] [TINFOIL] Restored {name} from QMD")

    try:
        ingest_r = await http.post(
            f"{sandbox_url}/ingest",
            headers=_tinfoil_headers(),
            json={"files": files_to_write},
            timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
        )
        if ingest_r.status_code != 200:
            log.warning(f"[{uid}] [TINFOIL] Ingest failed: {ingest_r.status_code}")
        else:
            log.info(f"[{uid}] [TINFOIL] Ingested {len(files_to_write)} files")
    except Exception as e:
        log.warning(f"[{uid}] [TINFOIL] Ingest error: {e}")

    # Step 3: Invoke agent via /invoke. Sandbox fetches + decrypts health data and writes USER.md.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_id = f"vita-{uid}-{today}"
    log.info(f"[{uid}] [TINFOIL] Invoking agent...")

    invoke_r = await http.post(
        f"{sandbox_url}/invoke",
        headers=_tinfoil_headers(),
        json={
            "message": message_text,
            "session_id": session_id,
            "user_id": user_id,
            "supermemory": sm_ctx,
        },
        timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
    )
    if invoke_r.status_code == 423:
        return (
            "Your encrypted health vault is locked in the TEE sandbox. "
            "Unlock it first, then try again."
        )
    if invoke_r.status_code != 200:
        raise RuntimeError(f"Tinfoil invoke failed: {invoke_r.status_code} {invoke_r.text[:200]}")

    output = invoke_r.json()
    elapsed = (asyncio.get_event_loop().time() - t0) * 1000

    # Log timing breakdown (same format as E2B path)
    timings = output.get("timings", {})
    if timings:
        qmd_status = "ok" if timings.get("qmdSearchOk") or output.get("qmdResults", 0) > 0 else "FAILED"
        log.info(
            f"[{uid}] [TINFOIL] Timings: "
            f"qmdIngest={timings.get('qmdIngest', '?')}ms, "
            f"qmdSearch={timings.get('qmdSearch', '?')}ms, "
            f"qmdStatus={qmd_status}, "
            f"agent={timings.get('agent', '?')}ms, "
            f"total={timings.get('total', '?')}ms | "
            f"qmdResults={output.get('qmdResults', '?')}"
        )

    if output.get("error"):
        raise RuntimeError(output.get("message", "Agent failed"))

    # Parse response (same as E2B path)
    raw_text = None
    if output.get("payloads"):
        texts = [p.get("text", "") for p in output["payloads"] if p.get("text")]
        raw_text = "\n\n".join(texts) if texts else None
    if not raw_text:
        raw_text = (
            output.get("response")
            or output.get("text")
            or output.get("content")
            or output.get("raw", "")
        )

    if not raw_text:
        return "I'm sorry, I couldn't generate a response. Please try again."

    text = clean_response(raw_text)
    log.info(f"[{uid}] [TINFOIL] Response: {len(raw_text)} → {len(text)} chars, total: {elapsed:.0f}ms")
    return text or "I'm sorry, I couldn't generate a response. Please try again."


# ── Core orchestrator ────────────────────────────────────────────────────
async def handle_user_message(user_id: str, message_text: str) -> str:
    # Feature flag: route Tinfoil test users to TEE sandbox
    if TINFOIL_ADMIN_KEY:
        return await handle_user_message_tinfoil(user_id, message_text)

    t0 = asyncio.get_event_loop().time()
    uid = user_id[:8]

    # Step 1: Parallel — fetch context + get/create sandbox
    log.info(f"[{uid}] Fetching context + sandbox...")
    (sb_ctx, sm_ctx), sandbox = await asyncio.gather(
        _fetch_all_context(user_id, message_text),
        get_or_create_sandbox(user_id),
    )
    elapsed = (asyncio.get_event_loop().time() - t0) * 1000
    log.info(
        f"[{uid}] Ready in {elapsed:.0f}ms — "
        f"biomarkers: {len(sb_ctx['biomarkers'])}, "
        f"supermemory: {'yes' if sm_ctx else 'no'}, "
        f"sandbox: {sandbox.sandbox_id}"
    )

    # Step 2: Build per-message USER.md (profile + health data combined)
    # USER.md is a bootstrap file (auto-injected into agent system prompt).
    # HEALTH.md is NOT a bootstrap file, so we merge health data into USER.md.
    user_md = build_user_md(sb_ctx["profile"], sb_ctx["preferences"])
    health_md = build_memory_md(
        sb_ctx["biomarkers"], sb_ctx["protocols"], sm_ctx,
        wearables=sb_ctx.get("wearables"),
        workouts=sb_ctx.get("workouts"),
        health_score=sb_ctx.get("health_score"),
        bio_age=sb_ctx.get("bio_age"),
        aging_velocity=sb_ctx.get("aging_velocity"),
    )
    combined_user_md = user_md + health_md

    # Step 3: Write files into sandbox + run agent
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_id = f"vita-{uid}-{today}"
    log.info(f"[{uid}] Writing files + running agent in sandbox...")

    # Per-message files (refreshed every message with latest data)
    files = [
        {"name": "USER.md", "content": combined_user_md},
    ]
    run_input = {
        "message": message_text,
        "session_id": session_id,
        "user_id": user_id,
        "qmd_url": QMD_GPU_URL,
    }
    output = await run_in_sandbox(sandbox, files, run_input)
    elapsed = (asyncio.get_event_loop().time() - t0) * 1000

    # Log timing breakdown
    timings = output.get("timings", {})
    if timings:
        qmd_status = "ok" if timings.get("qmdSearchOk") or output.get("qmdResults", 0) > 0 else "FAILED"
        log.info(
            f"[{uid}] Timings: "
            f"qmdIngest={timings.get('qmdIngest', '?')}ms, "
            f"qmdSearch={timings.get('qmdSearch', '?')}ms, "
            f"qmdStatus={qmd_status}, "
            f"agent={timings.get('agent', '?')}ms, "
            f"total={timings.get('total', '?')}ms | "
            f"qmdResults={output.get('qmdResults', '?')}"
        )
        if qmd_status == "FAILED":
            log.warning(f"[{uid}] QMD service unavailable — agent ran without semantic search context")

    if output.get("error"):
        raise RuntimeError(output.get("message", "Agent failed"))

    # Parse OpenClaw JSON output — combine all payloads
    raw_text = None
    if output.get("payloads"):
        texts = [p.get("text", "") for p in output["payloads"] if p.get("text")]
        raw_text = "\n\n".join(texts) if texts else None
    if not raw_text:
        raw_text = (
            output.get("response")
            or output.get("text")
            or output.get("content")
            or output.get("raw", "")
        )

    if not raw_text:
        return "I'm sorry, I couldn't generate a response. Please try again."

    text = clean_response(raw_text)
    log.info(f"[{uid}] Response: {len(raw_text)} → {len(text)} chars, total: {elapsed:.0f}ms")
    return text or "I'm sorry, I couldn't generate a response. Please try again."


async def _fetch_all_context(user_id: str, query: str):
    """Fetch Supabase + Supermemory context in parallel."""
    return await asyncio.gather(
        fetch_supabase_context(user_id),
        fetch_supermemory_context(user_id, query),
    )


# ── /start <token> — Link Telegram to VITA account ──────────────────────
@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    telegram_id = message.from_user.id
    args = message.text.split(maxsplit=1)
    token = args[1].strip() if len(args) > 1 else ""

    existing = await sb_select(
        "telegram_links", "user_id,linked_at",
        {"telegram_id": f"eq.{telegram_id}"},
        schema="agents",
    )
    if existing and existing[0].get("linked_at"):
        await message.answer(
            "You're already connected to VITA! Just send me a message and I'll help with your health questions."
        )
        return

    if not token:
        await message.answer(
            'Welcome! To connect your VITA account, tap the "Connect Telegram" button in the VITA app.\n\n'
            "Once connected, you can message me anytime about your health, supplements, biomarkers, and longevity goals."
        )
        return

    link_records = await sb_select(
        "telegram_links", "id,user_id,linked_at",
        {"link_token": f"eq.{token}"},
        schema="agents",
    )
    if not link_records:
        await message.answer(
            'That link has expired or is invalid. Please tap "Connect Telegram" again in the VITA app.'
        )
        return

    record = link_records[0]
    if record.get("linked_at"):
        await message.answer("This link was already used. You're all set!")
        return

    # Verify Pro subscription before linking
    tier = await get_user_tier(record["user_id"])
    if tier != "pro":
        await message.answer(
            "The Personal AI Agent is available on the Pro plan. "
            "Upgrade in the VITA app to connect your Telegram."
        )
        return

    try:
        await sb_update(
            "telegram_links",
            {"id": f"eq.{record['id']}"},
            {
                "telegram_id": telegram_id,
                "linked_at": datetime.now(timezone.utc).isoformat(),
            },
            schema="agents",
        )
    except Exception as e:
        log.error(f"Link update error: {e}")
        await message.answer(
            "Something went wrong connecting your account. Please try again from the VITA app."
        )
        return

    profile = await sb_select_one(
        "profiles", "display_name",
        {"user_id": f"eq.{record['user_id']}"},
    )
    name = profile.get("display_name", "there") if profile else "there"

    await message.answer(
        f"Connected! Hey {name}, I'm your VITA personal health assistant.\n\n"
        "You can ask me about:\n"
        "- Your biomarkers and health scores\n"
        "- Supplement protocols and interactions\n"
        "- Longevity research and recommendations\n"
        "- Your wearable data trends\n\n"
        "Just send me a message anytime!"
    )


# ── /unlock — Unlock encrypted health data in the TEE sandbox ─────────
@dp.message(Command("unlock"))
async def cmd_unlock(message: Message) -> None:
    """User sends /unlock <passphrase> to decrypt their health data in the TEE."""
    telegram_id = message.from_user.id

    links = await sb_select(
        "telegram_links", "user_id",
        {"telegram_id": f"eq.{telegram_id}", "linked_at": "not.is.null"},
        schema="agents",
    )
    if not links:
        await message.answer("You haven't connected your VITA account yet.")
        return

    user_id = links[0]["user_id"]

    if not TINFOIL_SANDBOX_URL or user_id not in TINFOIL_TEST_USERS:
        await message.answer("Encryption unlock is not available for your account yet.")
        return

    args = message.text.split(maxsplit=1)
    passphrase = args[1].strip() if len(args) > 1 else ""
    if not passphrase:
        await message.answer("Usage: /unlock <your encryption passphrase>")
        return

    try:
        r = await http.post(
            f"{TINFOIL_SANDBOX_URL}/unlock",
            headers=_tinfoil_headers(),
            json={"user_id": user_id, "passphrase": passphrase},
            timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
        )
        if r.status_code == 200:
            data = r.json()
            categories = data.get("categories", [])
            await message.answer(
                f"Health data unlocked inside TEE. Categories: {', '.join(categories)}. "
                "You can now chat normally — your health data will be decrypted securely."
            )
            log.info(f"[{user_id[:8]}] [TINFOIL] Unlock success: {categories}")
        else:
            error = r.json().get("error", "unknown error")
            await message.answer(f"Unlock failed: {error}")
            log.warning(f"[{user_id[:8]}] [TINFOIL] Unlock failed: {r.status_code} {error}")
    except Exception as e:
        log.error(f"[{user_id[:8]}] [TINFOIL] Unlock error: {e}")
        await message.answer("Something went wrong unlocking your health data. Please try again.")

    # Delete the message containing the passphrase for safety
    try:
        await message.delete()
    except Exception:
        pass  # May fail if bot doesn't have delete permission


# ── /unlink — Disconnect Telegram from VITA ──────────────────────────────
@dp.message(Command("unlink"))
async def cmd_unlink(message: Message) -> None:
    telegram_id = message.from_user.id

    # Find user before unlinking (for sandbox cleanup)
    links = await sb_select(
        "telegram_links", "user_id",
        {"telegram_id": f"eq.{telegram_id}"},
        schema="agents",
    )

    try:
        await sb_update(
            "telegram_links",
            {"telegram_id": f"eq.{telegram_id}"},
            {"telegram_id": None, "linked_at": None},
            schema="agents",
        )
    except Exception as e:
        log.error(f"Unlink error: {e}")
        await message.answer("Could not unlink. Please try again.")
        return

    # Clean up sandbox cache
    if links:
        uid = links[0]["user_id"]
        sandbox_cache.pop(uid, None)

    await message.answer(
        "Disconnected from VITA. You can reconnect anytime from the VITA app."
    )


# ── Handle messages — Route through orchestrator ─────────────────────────
@dp.message(F.text)
async def on_message(message: Message) -> None:
    telegram_id = message.from_user.id

    links = await sb_select(
        "telegram_links", "user_id",
        {"telegram_id": f"eq.{telegram_id}", "linked_at": "not.is.null"},
        schema="agents",
    )
    if not links:
        await message.answer(
            "You haven't connected your VITA account yet.\n"
            "Open the VITA app → Settings → Connect Telegram"
        )
        return

    user_id = links[0]["user_id"]

    # Classify message — block off-topic requests
    classified_text = None
    if CLASSIFIER_URL and message.text:
        try:
            cr = await http_qmd.post(
                f"{CLASSIFIER_URL}/classify",
                json={"message": message.text[:1000]},
                timeout=httpx.Timeout(connect=5, read=5, write=5, pool=5),
            )
            if cr.status_code == 200:
                cls = cr.json()
                if not cls.get("allowed", True):
                    deflection = cls.get("deflection", "")
                    if deflection:
                        await message.answer(deflection)
                    else:
                        await message.answer(
                            "I'm your VITA health agent — I can help with health, longevity, "
                            "supplements, biomarkers, and anything related to the VITA app. "
                            "What health topic can I help you with?"
                        )
                    return
                # Mixed intent: tell the agent what's health and what's off-topic
                health_part = cls.get("health_part", "")
                deflection = cls.get("deflection", "")
                if health_part and health_part != message.text and deflection:
                    # Extract the off-topic part by removing health_part from original
                    off_topic = message.text.replace(health_part, "").strip().strip(",").strip("and").strip()
                    classified_text = (
                        f"{health_part}\n\n"
                        f"(The user also asked: \"{off_topic}\" — this is outside our health focus. "
                        f"In your response, warmly acknowledge it and let them know you focus on health and VITA, "
                        f"then answer their health question. Keep it as one natural response.)"
                    )
                else:
                    classified_text = None
        except Exception as e:
            log.warning(f"Classifier error: {e}")
            classified_text = None

    # Use classified health_part if available, otherwise original message
    user_message = classified_text if classified_text else message.text

    # Per-user lock: serialize messages so concurrent sends don't corrupt state
    lock = _get_user_lock(user_id)
    async with lock:
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(keep_typing(message.chat.id, stop_typing))

        try:
            response = await handle_user_message(user_id, user_message)
            stop_typing.set()
            await typing_task
            await send_long_message(message, response)
        except PermissionError:
            stop_typing.set()
            await typing_task
            await message.answer(
                "Your VITA subscription no longer includes the Personal AI Agent. "
                "Upgrade to Pro in the VITA app to continue."
            )
        except Exception as e:
            stop_typing.set()
            await typing_task
            log.error(f"Chat error for user {user_id}: {e}", exc_info=True)

            # Clear in-memory cache so next message retries connection.
            # NEVER delete the Supabase sandbox record — the sandbox is the
            # user's persistent agent with memory, daily logs, and customization.
            sandbox_cache.pop(user_id, None)

            await message.answer(
                "Sorry, something went wrong. Please try again in a moment."
            )


# ── Nightly consolidation + Supabase Storage backup ──────────────────────
CONSOLIDATION_HOUR_UTC = 4  # Run at 4 AM UTC
STORAGE_BUCKET = "workspace-backups"
CONSOLIDATION_MAX_RETRIES = 3
CONSOLIDATION_RETRY_PAUSE = 10 * 60  # 10 min — matches sandbox auto-pause timeout


async def _ensure_storage_bucket() -> None:
    """Create the Supabase Storage bucket if it doesn't exist."""
    r = await http.post(
        f"{SUPABASE_URL}/storage/v1/bucket",
        headers=_sb_headers(),
        json={"id": STORAGE_BUCKET, "name": STORAGE_BUCKET, "public": False},
    )
    if r.status_code == 200:
        log.info(f"[CONSOLIDATE] Created storage bucket: {STORAGE_BUCKET}")
    elif r.status_code != 409:  # 409 = already exists
        log.warning(f"[CONSOLIDATE] Bucket creation: {r.status_code} {r.text[:200]}")


async def _upload_to_storage(user_id: str, files: list[dict]) -> int:
    """Upload workspace files to Supabase Storage. Returns count uploaded."""
    uploaded = 0
    for f in files:
        name = f.get("name", "")
        content = f.get("content", "")
        if not name or not content:
            continue
        path = f"{user_id}/{name}"
        r = await http.post(
            f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "text/markdown",
                "x-upsert": "true",
            },
            content=content.encode("utf-8"),
        )
        if r.status_code in (200, 201):
            uploaded += 1
        else:
            log.warning(f"[CONSOLIDATE] Upload failed {path}: {r.status_code}")
    return uploaded


async def _process_user_consolidation(user_id: str) -> str:
    """Consolidate + backup one user. Returns: ok, skipped, empty, error."""
    uid = user_id[:8]

    # Step 1: Trigger consolidation on QMD
    try:
        r = await http_qmd.post(
            f"{QMD_GPU_URL}/consolidate/{user_id}",
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        log.error(f"[CONSOLIDATE] [{uid}] Request failed: {e}")
        return "error"

    if r.status_code == 409:
        log.info(f"[CONSOLIDATE] [{uid}] User active — skipping")
        return "skipped"
    if r.status_code != 200:
        log.error(f"[CONSOLIDATE] [{uid}] Failed: {r.status_code} {r.text[:200]}")
        return "error"

    body = r.json()
    if body.get("skipped"):
        log.info(f"[CONSOLIDATE] [{uid}] {body.get('reason', 'nothing to consolidate')}")
    else:
        log.info(
            f"[CONSOLIDATE] [{uid}] Consolidated: v{body.get('version')}, "
            f"{body.get('logsConsumed')} logs, {body.get('elapsed')}ms"
        )

    # Step 2: Fetch workspace from QMD
    try:
        r = await http_qmd.get(f"{QMD_GPU_URL}/workspace/{user_id}")
        files = r.json().get("files", []) if r.status_code == 200 else []
    except Exception as e:
        log.error(f"[CONSOLIDATE] [{uid}] Workspace fetch failed: {e}")
        return "error"

    if not files:
        log.info(f"[CONSOLIDATE] [{uid}] No workspace files to backup")
        return "empty"

    # Step 3: Upload to Supabase Storage
    uploaded = await _upload_to_storage(user_id, files)
    log.info(f"[CONSOLIDATE] [{uid}] Backed up {uploaded}/{len(files)} files")
    return "ok"


async def run_nightly_consolidation() -> None:
    """Run consolidation + backup for all users, with retries for active users."""
    start = datetime.now(timezone.utc)
    log.info(f"[CONSOLIDATE] Starting run at {start.isoformat()}")

    await _ensure_storage_bucket()

    # Get all users with sandbox records
    try:
        rows = await sb_select(
            "openclaw_agents", "user_id", {}, schema="agents",
        )
        users = [row["user_id"] for row in rows]
    except Exception as e:
        log.error(f"[CONSOLIDATE] Failed to fetch users: {e}")
        return

    if not users:
        log.info("[CONSOLIDATE] No users found. Done.")
        return

    log.info(f"[CONSOLIDATE] Found {len(users)} users")

    results: dict[str, str] = {}
    pending = list(users)

    for round_num in range(1, CONSOLIDATION_MAX_RETRIES + 1):
        log.info(f"[CONSOLIDATE] Round {round_num}/{CONSOLIDATION_MAX_RETRIES} — {len(pending)} users")

        still_skipped = []
        for user_id in pending:
            result = await _process_user_consolidation(user_id)
            if result == "skipped":
                still_skipped.append(user_id)
            else:
                results[user_id] = result

        if not still_skipped:
            break

        if round_num < CONSOLIDATION_MAX_RETRIES:
            log.info(
                f"[CONSOLIDATE] {len(still_skipped)} users still active, "
                f"retrying in {CONSOLIDATION_RETRY_PAUSE // 60} min..."
            )
            await asyncio.sleep(CONSOLIDATION_RETRY_PAUSE)
            pending = still_skipped
        else:
            for user_id in still_skipped:
                results[user_id] = "skipped"

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    ok = sum(1 for v in results.values() if v == "ok")
    empty = sum(1 for v in results.values() if v == "empty")
    skipped = sum(1 for v in results.values() if v == "skipped")
    errors = sum(1 for v in results.values() if v == "error")
    log.info(
        f"[CONSOLIDATE] Done in {elapsed:.0f}s — "
        f"ok: {ok}, empty: {empty}, skipped: {skipped}, errors: {errors}"
    )
    if skipped:
        ids = [uid[:8] for uid, v in results.items() if v == "skipped"]
        log.warning(f"[CONSOLIDATE] Still active after retries: {ids}")
    if errors:
        ids = [uid[:8] for uid, v in results.items() if v == "error"]
        log.error(f"[CONSOLIDATE] Errors: {ids}")


async def flush_and_sync_sandbox(user_id: str, sandbox_id: str) -> None:
    """Memory flush + sync for a single user sandbox. Used before rebuilds and nightly."""
    uid = user_id[:8]
    try:
        sb = await asyncio.to_thread(Sandbox.connect, sandbox_id)
    except Exception as e:
        log.warning(f"[FLUSH] [{uid}] Cannot connect to sandbox {sandbox_id}: {e}")
        return

    # Step 1: Memory flush — ask the agent to write important facts to MEMORY.md
    log.info(f"[FLUSH] [{uid}] Running memory flush...")
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        flush_sid = f"vita-{uid}-{today}"
        flush_msg = "Review this conversation. Write any important facts or decisions to MEMORY.md. Facts only. If nothing notable, respond NO_REPLY."
        result = sb.commands.run(
            f'openclaw agent --session-id {flush_sid} --message "{flush_msg}" --json',
            timeout=60,
            envs={"HOME": "/home/user"},
        )
        log.info(f"[FLUSH] [{uid}] Memory flush complete")
    except Exception as e:
        log.warning(f"[FLUSH] [{uid}] Memory flush failed: {e}")

    # Step 2: Sync dynamic files to QMD (matching the restore whitelist)
    # Only files that get restored on sandbox recreation need to be synced.
    # Static files (SOUL.md, AGENTS.md, TOOLS.md) and per-message files (USER.md)
    # are NOT synced — they're regenerated by the bot. Syncing them would risk
    # restoring stale versions over fresh ones.
    try:
        ws = "/home/user/.openclaw/workspace"
        files_to_sync = []
        # MEMORY.md + IDENTITY.md
        for name in ("MEMORY.md", "IDENTITY.md"):
            try:
                content = sb.files.read(f"{ws}/{name}")
                if content and len(content) > 5:
                    files_to_sync.append({"name": name, "content": content})
            except Exception:
                pass
        # memory/*.md daily logs
        try:
            for f in sb.files.list(f"{ws}/memory"):
                if f.name.endswith(".md"):
                    try:
                        content = sb.files.read(f"{ws}/memory/{f.name}")
                        if content and len(content) > 5:
                            files_to_sync.append({"name": f"memory/{f.name}", "content": content})
                    except Exception:
                        pass
        except Exception:
            pass  # memory/ dir may not exist

        if files_to_sync:
            r = await http_qmd.post(
                f"{QMD_GPU_URL}/ingest",
                json={"user_id": user_id, "files": files_to_sync},
            )
            names = [f["name"] for f in files_to_sync]
            if r.status_code == 200:
                log.info(f"[FLUSH] [{uid}] Synced to QMD: {names}")
            else:
                log.warning(f"[FLUSH] [{uid}] QMD sync failed: {r.status_code}")
        else:
            log.info(f"[FLUSH] [{uid}] No dynamic files to sync")
    except Exception as e:
        log.warning(f"[FLUSH] [{uid}] Sync failed: {e}")


async def flush_all_sandboxes() -> None:
    """Flush and sync all active sandboxes. Called before rebuilds and nightly."""
    try:
        rows = await sb_select(
            "openclaw_agents", "user_id,openclaw_agent_id", {}, schema="agents",
        )
    except Exception as e:
        log.error(f"[FLUSH] Failed to fetch sandboxes: {e}")
        return

    if not rows:
        log.info("[FLUSH] No active sandboxes.")
        return

    log.info(f"[FLUSH] Flushing {len(rows)} sandboxes...")
    for row in rows:
        await flush_and_sync_sandbox(row["user_id"], row["openclaw_agent_id"])
    log.info("[FLUSH] Done.")


async def consolidation_scheduler() -> None:
    """Background task: check once per hour, trigger at CONSOLIDATION_HOUR_UTC."""
    while True:
        now = datetime.now(timezone.utc)
        # Sleep until next hour boundary + 1 min buffer
        seconds_to_next_hour = (60 - now.minute) * 60 - now.second + 60
        await asyncio.sleep(seconds_to_next_hour)

        if datetime.now(timezone.utc).hour == CONSOLIDATION_HOUR_UTC:
            try:
                # Flush all sandboxes first, then consolidate
                await flush_all_sandboxes()
                await run_nightly_consolidation()
            except Exception as e:
                log.error(f"[CONSOLIDATE] Unhandled error: {e}", exc_info=True)


# ── Cron webhook receiver ────────────────────────────────────────────────
async def handle_cron_webhook(request: web.Request) -> web.Response:
    """Receive cron job results from sandboxes and deliver via Telegram."""
    # Auth check
    auth = request.headers.get("Authorization", "")
    if CRON_WEBHOOK_SECRET and auth != f"Bearer {CRON_WEBHOOK_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Extract message from cron finished event, user_id from query string
    summary = payload.get("summary", "")
    user_id = request.query.get("user_id", payload.get("user_id", ""))

    if not summary:
        return web.json_response({"error": "No summary in payload"}, status=400)

    if not user_id:
        return web.json_response({"error": "No user_id in payload"}, status=400)

    # Look up user's Telegram chat ID
    links = await sb_select(
        "telegram_links", "telegram_id",
        {"user_id": f"eq.{user_id}", "linked_at": "not.is.null"},
        schema="agents",
    )
    if not links:
        log.warning(f"[WEBHOOK] No telegram link for user {user_id[:8]}")
        return web.json_response({"error": "User not linked"}, status=404)

    chat_id = links[0]["telegram_id"]
    html_text = md_to_telegram_html(summary)

    try:
        await bot.send_message(chat_id, html_text, parse_mode="HTML")
        log.info(f"[WEBHOOK] Delivered cron message to {user_id[:8]}: {summary[:80]}")
    except Exception as e:
        log.error(f"[WEBHOOK] Failed to send to {user_id[:8]}: {e}")
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({"ok": True})


# ── Data query endpoint for sandbox agents ───────────────────────────────
# Allowed tables and their permitted columns (whitelist for security)
_QUERY_TABLES = {
    "biomarker_readings": "name,value,unit,recorded_at",
    "wearable_readings": "category,metric_type,value,unit,source,recorded_at",
    "workout_sessions": "workout_type,duration_minutes,calories,avg_heart_rate,max_heart_rate,strain,source,started_at",
    "composite_health_scores": "score,data_coverage_pct,domain_scores,score_date",
    "bio_age_scores": "bio_age,chronological_age,age_difference,biomarker_count,computed_at",
    "aging_velocity_scores": "velocity,span_days,data_points,computed_at",
    "daily_insights": "summary,sleep_summary,training_summary,recommendations,insight_date",
    "user_protocols": "name,goal,start_date,end_date,status",
    "lab_uploads": "filename,uploaded_at,status",
    "protocol_components": "title,category,dosage,unit,timing,frequency",
}


async def handle_data_query(request: web.Request) -> web.Response:
    """Query user's health data from Supabase. Called by sandbox agents."""
    auth = request.headers.get("Authorization", "")
    if CRON_WEBHOOK_SECRET and auth != f"Bearer {CRON_WEBHOOK_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    user_id = request.query.get("user_id", payload.get("user_id", ""))
    table = payload.get("table", "")
    limit = min(int(payload.get("limit", 50)), 200)  # Cap at 200
    order = payload.get("order", "")
    filters = payload.get("filters", {})

    if not user_id:
        return web.json_response({"error": "user_id required"}, status=400)
    if table not in _QUERY_TABLES:
        return web.json_response({"error": f"Table not allowed. Available: {list(_QUERY_TABLES.keys())}"}, status=400)

    select = _QUERY_TABLES[table]
    query_filters = {"user_id": f"eq.{user_id}"}

    # Allow safe filter operators
    for key, val in filters.items():
        if not isinstance(val, str):
            continue
        # Only allow safe PostgREST operators
        if any(val.startswith(op) for op in ("eq.", "gt.", "gte.", "lt.", "lte.", "like.", "ilike.")):
            query_filters[key] = val

    try:
        rows = await sb_select(
            table, select, query_filters,
            limit=limit,
            order=order or None,
        )
        log.info(f"[QUERY] {user_id[:8]} → {table}: {len(rows)} rows")
        return web.json_response({"table": table, "rows": rows, "count": len(rows)})
    except Exception as e:
        log.error(f"[QUERY] Error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def _aubrai_poll_and_deliver(request_id: str, user_id: str, question: str) -> None:
    """Background task: poll Aubrai until done, then deliver to Telegram."""
    uid = user_id[:8]

    # Poll for result (max 5 minutes)
    result_text = None
    for attempt in range(150):
        await asyncio.sleep(2)
        try:
            r = await http.post(
                f"{SUPABASE_URL}/functions/v1/aubrai-x402-status",
                headers={
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"requestId": request_id},
                timeout=httpx.Timeout(connect=10, read=15, write=10, pool=10),
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if data.get("status") == "completed":
                result_text = data.get("result", {}).get("text", "")
                break
            elif data.get("status") in ("failed", "error"):
                log.error(f"[AUBRAI] Failed for {uid}: {data.get('error', '')}")
                break
        except Exception as e:
            log.warning(f"[AUBRAI] Poll error: {e}")

    if not result_text:
        log.error(f"[AUBRAI] No result for {uid} after polling")
        # Send failure message to user
        links = await sb_select(
            "telegram_links", "telegram_id",
            {"user_id": f"eq.{user_id}", "linked_at": "not.is.null"},
            schema="agents",
        )
        if links:
            try:
                await bot.send_message(links[0]["telegram_id"], "Aubrai couldn't complete the research. Please try again.")
            except Exception:
                pass
        return

    log.info(f"[AUBRAI] Result for {uid}: {len(result_text)} chars")

    # Look up user's Telegram chat ID
    links = await sb_select(
        "telegram_links", "telegram_id",
        {"user_id": f"eq.{user_id}", "linked_at": "not.is.null"},
        schema="agents",
    )
    if not links:
        log.warning(f"[AUBRAI] No telegram link for {uid}")
        return

    chat_id = links[0]["telegram_id"]

    # Send Aubrai's response directly
    await send_long_message_to_chat(chat_id, f"🔬 Aubrai's Response\n\n{result_text}", use_html=False)


async def send_long_message_to_chat(chat_id: int, text: str, use_html: bool = True) -> None:
    """Send a long message to a chat ID (for proactive delivery)."""
    MAX = 4096
    if use_html:
        send_text = md_to_telegram_html(text)
        parse_mode = "HTML"
    else:
        send_text = text
        parse_mode = None

    if len(send_text) <= MAX:
        await bot.send_message(chat_id, send_text, parse_mode=parse_mode)
        return

    remaining = send_text
    while remaining:
        if len(remaining) <= MAX:
            await bot.send_message(chat_id, remaining, parse_mode=parse_mode)
            break
        split_at = remaining.rfind("\n\n", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = remaining.rfind("\n", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = remaining.rfind(". ", 0, MAX)
        if split_at < MAX * 0.5:
            split_at = MAX
        await bot.send_message(chat_id, remaining[:split_at], parse_mode=parse_mode)
        remaining = remaining[split_at:].lstrip()


async def handle_aubrai_submit(request: web.Request) -> web.Response:
    """Submit a question to Aubrai. Returns immediately, delivers result via Telegram."""
    auth = request.headers.get("Authorization", "")
    if CRON_WEBHOOK_SECRET and auth != f"Bearer {CRON_WEBHOOK_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    user_id = request.query.get("user_id", payload.get("user_id", ""))
    message = payload.get("message", "")
    if not user_id or not message:
        return web.json_response({"error": "user_id and message required"}, status=400)

    log.info(f"[AUBRAI] Submit from {user_id[:8]}: {message[:80]}")

    try:
        r = await http.post(
            f"{SUPABASE_URL}/functions/v1/aubrai-x402-chat-bot",
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "x-bot-auth": BOT_AUTH_SECRET,
            },
            json={"user_id": user_id, "message": message},
            timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
        )
        if r.status_code != 200:
            log.warning(f"[AUBRAI] Edge function: {r.status_code} {r.text[:200]}")
            return web.json_response({"error": "Aubrai request failed"}, status=r.status_code)

        request_id = r.json().get("requestId")
        log.info(f"[AUBRAI] Job queued: {request_id}")

        # Fire background task to poll and deliver
        asyncio.create_task(_aubrai_poll_and_deliver(request_id, user_id, message))

        return web.json_response({"status": "queued", "requestId": request_id})
    except Exception as e:
        log.error(f"[AUBRAI] Submit error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_flush(request: web.Request) -> web.Response:
    """Admin endpoint: flush all sandboxes before template rebuilds.
    Triggers agent memory write + QMD sync for every active sandbox."""
    auth = request.headers.get("Authorization", "")
    if not BOT_AUTH_SECRET or auth != f"Bearer {BOT_AUTH_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    log.info("[FLUSH] Manual flush triggered via /api/flush")
    await flush_all_sandboxes()
    return web.json_response({"status": "flushed"})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ── VitaApp integration endpoints ────────────────────────────────────────

async def handle_provision_agent(request: web.Request) -> web.Response:
    """POST /api/provision-agent — called by VitaApp when user connects Telegram.
    Creates a per-user Tinfoil sandbox + Telegram link token.
    Returns sandbox_url so VitaApp can send DEK directly to the sandbox."""
    auth = request.headers.get("Authorization", "")
    if BOT_AUTH_SECRET and auth != f"Bearer {BOT_AUTH_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    user_id = payload.get("user_id", "")
    if not user_id:
        return web.json_response({"error": "user_id required"}, status=400)

    uid = user_id[:8]

    # Verify Pro subscription
    tier = await get_user_tier(user_id)
    if tier != "pro":
        return web.json_response({
            "error": "pro_required",
            "message": "Pro subscription required for the personal AI agent.",
        }, status=403)

    # Create or get existing sandbox
    try:
        sandbox_info = await get_or_create_tinfoil_sandbox(user_id)
    except Exception as e:
        log.error(f"[{uid}] [PROVISION] Failed to create sandbox: {e}")
        return web.json_response({"error": "sandbox_creation_failed", "message": str(e)}, status=500)

    # Create a Telegram link token
    import secrets as secrets_mod
    link_token = secrets_mod.token_urlsafe(32)
    try:
        await sb_upsert(
            "telegram_links",
            {"user_id": user_id, "link_token": link_token},
            schema="agents",
            on_conflict="user_id",
        )
    except Exception as e:
        log.warning(f"[{uid}] [PROVISION] Failed to create link token: {e}")
        link_token = None

    log.info(f"[{uid}] [PROVISION] Agent provisioned: {sandbox_info['url']}")

    result = {
        "ok": True,
        "user_id": user_id,
        "sandbox_url": sandbox_info["url"],
        "container_id": sandbox_info.get("container_id", ""),
        "status": "ready",
    }
    if link_token:
        result["link_token"] = link_token
        result["telegram_link"] = f"https://t.me/VITAAPP_PERSONAL_AI_BOT?start={link_token}"

    return web.json_response(result)


async def handle_agent_status(request: web.Request) -> web.Response:
    """GET /api/agent-status?user_id=X — returns sandbox status.
    Called by VitaApp on the Telegram Agent settings page or when bot says to re-authorize."""
    auth = request.headers.get("Authorization", "")
    if BOT_AUTH_SECRET and auth != f"Bearer {BOT_AUTH_SECRET}":
        return web.json_response({"error": "Unauthorized"}, status=401)

    user_id = request.query.get("user_id", "")
    if not user_id:
        return web.json_response({"error": "user_id required"}, status=400)

    uid = user_id[:8]

    # Check if user has a sandbox
    record = await sb_select_one(
        "openclaw_agents", "user_id,tinfoil_container_id,tinfoil_container_url",
        {"user_id": f"eq.{user_id}"},
        schema="agents",
    )

    if not record or not record.get("tinfoil_container_url"):
        return web.json_response({
            "user_id": user_id,
            "provisioned": False,
            "sandbox_url": None,
            "status": "not_provisioned",
            "locked": True,
        })

    sandbox_url = record["tinfoil_container_url"]

    # Check if sandbox is running and locked/unlocked
    status = "unknown"
    locked = True
    try:
        r = await http.get(
            f"{sandbox_url}/health",
            headers=_tinfoil_headers(),
            timeout=httpx.Timeout(connect=5, read=5, write=5, pool=5),
        )
        if r.status_code == 200:
            health = r.json()
            status = "running"
            locked = health.get("unlockedUsers", 0) == 0
        else:
            status = "unreachable"
    except Exception:
        status = "unreachable"

    return web.json_response({
        "user_id": user_id,
        "provisioned": True,
        "sandbox_url": sandbox_url,
        "container_id": record.get("tinfoil_container_id", ""),
        "status": status,
        "locked": locked,
    })


# ── Main ─────────────────────────────────────────────────────────────────
async def main() -> None:
    log.info("Starting VITA Telegram bot (Python/aiogram + E2B)...")
    log.info(f"E2B template: {E2B_TEMPLATE}")
    log.info(f"QMD GPU: {QMD_GPU_URL}")
    log.info(f"Supermemory: {'enabled' if SUPERMEMORY_API_KEY else 'disabled'}")
    log.info(f"Consolidation: {CONSOLIDATION_HOUR_UTC}:00 UTC daily")
    log.info(f"Webhook server: port {WEBHOOK_PORT}")
    log.info(f"Tinfoil Admin: {'enabled' if TINFOIL_ADMIN_KEY else 'disabled'}")

    # Start consolidation scheduler as background task
    asyncio.create_task(consolidation_scheduler())

    # Start webhook HTTP server for cron delivery + VitaApp integration
    app = web.Application()
    app.router.add_post("/api/cron-webhook", handle_cron_webhook)
    app.router.add_post("/api/query", handle_data_query)
    app.router.add_post("/api/aubrai", handle_aubrai_submit)
    app.router.add_post("/api/flush", handle_flush)
    app.router.add_post("/api/provision-agent", handle_provision_agent)
    app.router.add_get("/api/agent-status", handle_agent_status)
    app.router.add_get("/health", handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WEBHOOK_PORT)
    await site.start()
    log.info(f"Webhook server listening on port {WEBHOOK_PORT}")

    # Start Telegram polling (blocking)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

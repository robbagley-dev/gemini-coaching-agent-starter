#!/usr/bin/env python3
"""
Dual-Engine AI Coaching Agent Starter Template with Feedback Store
------------------------------------------------------------------
A lightweight, zero-dependency Python backend for a multi-step generative AI coach.

Inference Strategy:
1. Primary:  Qwen (qwen3-vl-30b-a3b-instruct-mlx) on LM Studio via public Tailscale tunnel
2. Fallback: Google Gemini (gemini-2.5-flash) if LM Studio is unreachable
3. Offline:  Safe structured coaching guidance if both are unavailable

Feedback Mechanism:
- Local SQLite database (feedback.db) storing thumbs up / thumbs down ratings
- Structured suggestion capture with built-in PII privacy guardrails
"""

import json
import os
import re
import sqlite3
import ssl
import time
import uuid
from collections import defaultdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Load .env file manually if present (no external packages needed)
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Primary Engine: LM Studio Qwen via public Tailscale tunnel or local loopback
LM_STUDIO_URL = os.environ.get(
    "LM_STUDIO_URL",
    "https://mac-studio-2.tail299fc7.ts.net:8443/v1"
).rstrip("/")
QWEN_MODEL = os.environ.get("MODEL_NAME", "qwen3-vl-30b-a3b-instruct-mlx").strip()
# Required for the public mac-studio-2.tail299fc7.ts.net:8443 endpoint (auth-gated).
# Request a key from robbagley@ensign.edu, or run against your own LM Studio
# instance (LM_STUDIO_URL=http://127.0.0.1:1234/v1) with this left blank.
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "").strip()

# Fallback Engine: Google Gemini API (optional, used if Qwen is unreachable)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Server Settings
def _get_int_env(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    return int(val) if val.isdigit() else default

PORT = _get_int_env("PORT", 5050)
HOST = os.environ.get("HOST", "0.0.0.0")
RATE_LIMIT = _get_int_env("RATE_LIMIT_PER_MIN", 50)

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
STATIC_DIR = PUBLIC_DIR if PUBLIC_DIR.exists() else Path(__file__).resolve().parent / "static"


def resolve_db_path(root_dir=None, environ=None, home=None) -> Path:
    """Choose where feedback.db lives, keeping it out of cloud-synced folders."""
    root = Path(__file__).resolve().parent if root_dir is None else Path(root_dir)
    env = os.environ if environ is None else environ
    override = env.get("FEEDBACK_DB_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    if env.get("VERCEL"):
        return Path("/tmp") / "feedback.db"
    # A checkout under ~/Library/CloudStorage (OneDrive, Google Drive) syncs the
    # database on every write, so keep it on local disk instead.
    if "CloudStorage" in root.parts:
        base = Path.home() if home is None else Path(home)
        return base / "Local-Infra" / "app-data" / root.name / "feedback.db"
    return root / "feedback.db"


# Resolved after .env loads so FEEDBACK_DB_PATH can be set there.
DB_PATH = resolve_db_path()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. LOCAL FEEDBACK DATABASE (SQLITE)
# ==============================================================================

def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    mode TEXT,
                    rating TEXT NOT NULL,
                    question TEXT,
                    answer TEXT,
                    comment TEXT,
                    client_ip TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"[DB Init Error] {e}")

init_db()

def save_feedback(response_id: str, rating: str, comment: str = "", question: str = "", answer: str = "", mode: str = "", client_ip: str = ""):
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    record = {
        "created_at": now,
        "response_id": response_id,
        "mode": mode,
        "rating": rating,
        "question": question,
        "answer": answer,
        "comment": comment,
        "client_ip": client_ip,
    }
    print(f"[FEEDBACK] {json.dumps(record)}", flush=True)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO feedback (created_at, response_id, mode, rating, question, answer, comment, client_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, response_id, mode, rating, question, answer, comment, client_ip))
            conn.commit()
    except Exception as e:
        print(f"[Feedback Save Error] {e}")

# ==============================================================================
# 3. COACH SYSTEM PROMPT & PERSONA
# ==============================================================================
# Customize this prompt for your agent's specific focus and purpose!

SYSTEM_PROMPT = """You are a helpful, encouraging, and actionable AI Coach for students.
Your goal is to guide the user step-by-step through their goals with clear, constructive feedback.

Core Guidelines:
1. Be concise, warm, and conversational.
2. Ask only ONE focused follow-up question at a time to avoid overwhelming the user.
3. Preserve the user's authentic voice when refining their ideas or drafts.
4. Give specific praise for what works, and clear suggestions for improvement.
5. If the user asks for help brainstorming, give 2-3 distinct, realistic ideas.
"""

MODE_CONTEXTS = {
    "step1": "Mode: Step 1 (Discovery & Research). Help the user explore ideas, analyze key requirements, and clarify their target direction.",
    "step2": "Mode: Step 2 (Drafting & Core Message). Guide the user in building a compelling, clear message or draft. Highlight proof points.",
    "step3": "Mode: Step 3 (Practice & Role-Play). Act as a realistic partner/reviewer. Give realistic responses and ask one relevant follow-up at a time.",
    "step4": "Mode: Step 4 (Preparation & Strategic Questions). Help the user craft thoughtful, high-impact questions to ask in their upcoming opportunity.",
}

# ==============================================================================
# 4. PRIVACY FILTER & GUARDRAILS
# ==============================================================================

PII_PATTERNS = [
    # Social Security Numbers (e.g., 000-00-0000 or 000 00 0000)
    (re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"), "Social Security numbers"),
    # Credit Card Numbers (13 to 19 digits with dashes or spaces)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"), "payment card numbers"),
    # Password disclosures
    (re.compile(r"\b(?:password|pwd|passcode|secret_key)\s*[:=]\s*\S+", re.IGNORECASE), "passwords"),
]

def check_privacy(text: str) -> str | None:
    """Returns a warning message if sensitive PII is detected, else None."""
    for pattern, label in PII_PATTERNS:
        if pattern.search(text):
            return f"For your privacy and safety, please remove {label} or private identifiers before submitting."
    return None

# ==============================================================================
# 5. SLIDING-WINDOW RATE LIMITER (DEFAULT: 50 REQ/MIN)
# ==============================================================================

class SlidingWindowRateLimiter:
    def __init__(self, limit_per_minute: int = 50):
        self.limit = limit_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]
        if len(self.requests[client_ip]) >= self.limit:
            return False
        self.requests[client_ip].append(now)
        return True

RATE_LIMITER = SlidingWindowRateLimiter(limit_per_minute=RATE_LIMIT)

# ==============================================================================
# 6. INFERENCE ENGINES (PRIMARY: QWEN -> FALLBACK: GEMINI -> OFFLINE)
# ==============================================================================

def fallback_reply(message: str, mode: str) -> str:
    """Safe fallback response if both inference engines are unreachable."""
    fallbacks = {
        "step1": "Great start exploring this topic! Tell me more about what specific goal or organization you want to target.",
        "step2": "Let's work on your message. Share your main goal, one clear proof point or example, and what makes you unique.",
        "step3": "I'm ready to practice with you! Share your draft response and I will provide feedback and a realistic follow-up.",
        "step4": "Here is a strong starter question: 'What qualities help someone succeed most in this role?' How would you like to customize it?",
    }
    return fallbacks.get(mode, "Thanks for sharing! What specific aspect would you like to work on next?")


def query_qwen(message: str, mode: str, history: list[dict[str, str]]) -> str | None:
    """Queries LM Studio Qwen via OpenAI-compatible /v1/chat/completions."""
    mode_context = MODE_CONTEXTS.get(mode, "")
    system_content = f"{SYSTEM_PROMPT}\n\n{mode_context}".strip()

    messages = [{"role": "system", "content": system_content}]
    for item in history[-8:]:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1024,
    }

    url = f"{LM_STUDIO_URL}/chat/completions"
    req_data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

    request = Request(url, data=req_data, headers=headers, method="POST")
    with urlopen(request, timeout=35, context=ssl.create_default_context()) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip() or None
    return None


def query_gemini(message: str, mode: str, history: list[dict[str, str]]) -> str | None:
    """Queries Google Gemini GenerateContent API as fallback."""
    if not GEMINI_API_KEY:
        return None

    mode_context = MODE_CONTEXTS.get(mode, "")
    system_instruction = f"{SYSTEM_PROMPT}\n\n{mode_context}".strip()

    contents = []
    for item in history[-8:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role == "user" and content:
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content:
            contents.append({"role": "model", "parts": [{"text": content}]})

    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    req_data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    request = Request(url, data=req_data, headers=headers, method="POST")
    with urlopen(request, timeout=30, context=ssl.create_default_context()) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "").strip() or None
    return None


def ask_coach(message: str, mode: str, history: list[dict[str, str]]) -> tuple[str, bool, str]:
    """
    Dual-engine coordinator:
    1. Try Primary (LM Studio Qwen)
    2. Fallback to Gemini (if Qwen is down and key exists)
    3. Fallback to offline guidance
    Returns (reply_text, is_live_ai, engine_name).
    """
    # 1. Primary: LM Studio Qwen
    try:
        reply = query_qwen(message, mode, history)
        if reply:
            return reply, True, "qwen"
    except Exception as e:
        print(f"[Primary Qwen Unavailable] {e}")

    # 2. Fallback: Google Gemini
    if GEMINI_API_KEY:
        try:
            print("[Inference] Switching to Google Gemini fallback...")
            reply = query_gemini(message, mode, history)
            if reply:
                return reply, True, "gemini"
        except Exception as e:
            print(f"[Fallback Gemini Error] {e}")

    # 3. Final Fallback: Offline guidance
    return fallback_reply(message, mode), False, "fallback"


# ==============================================================================
# 7. HTTP REQUEST HANDLER
# ==============================================================================

class CoachHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, payload: dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Health & status endpoints
        if self.path in ("/healthz", "/api/status"):
            self._json({
                "status": "ok",
                "service": "AI Coach",
                "primary_engine": "LM Studio Qwen",
                "primary_model": QWEN_MODEL,
                "primary_endpoint": LM_STUDIO_URL,
                "fallback_engine": "Google Gemini" if GEMINI_API_KEY else "Static Fallback",
                "fallback_configured": bool(GEMINI_API_KEY),
                "rate_limit_per_min": RATE_LIMIT,
            })
            return
        super().do_GET()

    def do_POST(self):
        # ----------------------------------------------------------------------
        # Feedback Submission Endpoint
        # ----------------------------------------------------------------------
        if self.path == "/api/feedback":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(raw_body)
            except Exception:
                self._json({"error": "Invalid JSON payload."}, HTTPStatus.BAD_REQUEST)
                return

            response_id = str(data.get("response_id", "")).strip()
            rating = str(data.get("rating", "")).strip().lower()
            comment = str(data.get("comment", "")).strip()
            question = str(data.get("question", "")).strip()
            answer = str(data.get("answer", "")).strip()
            mode = str(data.get("mode", "")).strip()

            if rating not in ("up", "down"):
                self._json({"error": "Rating must be 'up' or 'down'."}, HTTPStatus.BAD_REQUEST)
                return

            if comment:
                warning = check_privacy(comment)
                if warning:
                    self._json({"error": warning}, HTTPStatus.BAD_REQUEST)
                    return

            client_ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
            try:
                save_feedback(response_id, rating, comment, question, answer, mode, client_ip)
                self._json({"status": "ok", "message": "Feedback saved."})
            except Exception as e:
                print(f"[Feedback Save Error] {e}")
                self._json({"error": "Failed to save feedback."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # ----------------------------------------------------------------------
        # Chat Generation Endpoint
        # ----------------------------------------------------------------------
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # 1. Rate Limiting Check
        client_ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
        if not RATE_LIMITER.is_allowed(client_ip):
            self._json(
                {"error": f"Rate limit of {RATE_LIMIT} req/min exceeded. Please wait a moment before sending another message."},
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            return

        # 2. Parse JSON body
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(raw_body)
        except Exception:
            self._json({"error": "Invalid JSON payload."}, HTTPStatus.BAD_REQUEST)
            return

        message = str(data.get("message", "")).strip()
        mode = str(data.get("mode", "step1")).strip()
        history = data.get("history", [])

        if not message:
            self._json({"error": "Message is required."}, HTTPStatus.BAD_REQUEST)
            return

        # 3. Privacy Guardrail Check
        privacy_warning = check_privacy(message)
        if privacy_warning:
            self._json({"error": privacy_warning}, HTTPStatus.BAD_REQUEST)
            return

        # 4. Generate Coach Response via Dual-Engine Coordinator
        reply, is_live, engine = ask_coach(message, mode, history)
        response_id = f"resp-{uuid.uuid4().hex[:12]}"
        self._json({
            "reply": reply,
            "live": is_live,
            "engine": engine,
            "response_id": response_id
        })


# ==============================================================================
# 8. SERVER ENTRYPOINT
# ==============================================================================

def main():
    server_address = (HOST, PORT)
    with ThreadingHTTPServer(server_address, CoachHandler) as httpd:
        print("================================================================")
        print(f"🚀 AI Coach Starter running at http://localhost:{PORT}")
        print(f"   Primary Engine:   LM Studio Qwen ({QWEN_MODEL})")
        print(f"   Primary Endpoint: {LM_STUDIO_URL}")
        print(f"   Fallback Engine:  {'Google Gemini (' + GEMINI_MODEL + ')' if GEMINI_API_KEY else 'Offline Fallback (Set GEMINI_API_KEY to enable Gemini)'}")
        print(f"   Rate Limit:       {RATE_LIMIT} req/min per IP")
        print(f"   Feedback Store:   SQLite ({DB_PATH.name})")
        print("================================================================")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()

"""
ContentAI — FastAPI backend (port 8001)
Google OAuth2 → JWT cookie → protected API routes
Run: uvicorn content_server:app --reload --port 8001
"""
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import base64
import shutil
import uuid

import httpx
import requests as _req
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

for _ep in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _ep.exists():
        load_dotenv(_ep)

sys.path.insert(0, str(Path(__file__).parent))
from content_db import (
    BlogPost, get_all_api_keys, get_api_key, get_db, get_or_create_user,
    init_db, save_api_key, save_post, save_draft,
)
from content_auth import create_token, get_current_user, require_user
from content_agent import run_content_pipeline, run_refine_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI         = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8001/auth/google/callback")
_GAUTH  = "https://accounts.google.com/o/oauth2/v2/auth"
_GTOKEN = "https://oauth2.googleapis.com/token"
_GINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"

app = FastAPI(title="ContentAI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
init_db()

# Serve uploaded images as static files
_UPLOADS = Path(__file__).parent / "uploads"
_UPLOADS.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_UPLOADS)), name="uploads")


# ── Keep-alive: ping self every 14 min so Render free tier never sleeps ───────
async def _keep_alive():
    import httpx
    await asyncio.sleep(60)
    url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not url:
        return
    if not url.startswith("http"):
        url = f"https://{url}"
    ping_url = f"{url}/api/health"
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(ping_url)
            print(f"[keep-alive] pinged {ping_url}")
        except Exception as e:
            print(f"[keep-alive] ping failed: {e}")
        await asyncio.sleep(14 * 60)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_keep_alive())


# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/content", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse((Path(__file__).parent / "content.html").read_text(encoding="utf-8"))


# ── Google OAuth ──────────────────────────────────────────────────────────────
@app.get("/auth/google")
async def auth_google():
    params = dict(client_id=GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI,
                  response_type="code", scope="openid email profile",
                  access_type="offline", prompt="select_account")
    return RedirectResponse(_GAUTH + "?" + urlencode(params))


@app.get("/auth/google/callback")
async def auth_callback(code: str, db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        tok = await client.post(_GTOKEN, data=dict(
            code=code, client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI, grant_type="authorization_code"
        ))
        tokens = tok.json()
        if "error" in tokens:
            raise HTTPException(400, tokens.get("error_description", "OAuth error"))
        info = (await client.get(_GINFO, headers={"Authorization": f"Bearer {tokens['access_token']}"})).json()

    user  = get_or_create_user(db, info)
    token = create_token(user.id)
    resp  = RedirectResponse("/")
    resp.set_cookie("content_token", token, httponly=True, max_age=30*24*3600, samesite="lax")
    return resp


@app.get("/auth/me")
async def auth_me(user=Depends(get_current_user)):
    if not user:
        return JSONResponse({"authenticated": False})
    return {"authenticated": True, "name": user.name, "email": user.email, "avatar": user.avatar}


@app.post("/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie("content_token")
    return {"success": True}


# ── API Keys ──────────────────────────────────────────────────────────────────
class KeySave(BaseModel):
    platform: str
    key: str

@app.post("/api/keys/save")
async def key_save(req: KeySave, db=Depends(get_db), user=Depends(require_user)):
    save_api_key(db, user.id, req.platform, req.key)
    return {"success": True}

@app.get("/api/keys/status")
async def key_status(db=Depends(get_db), user=Depends(get_current_user)):
    if not user:
        return {"connected": {"devto": False, "hashnode": False, "medium": False,
                              "linkedin": False, "twitter": False}}

    keys = get_all_api_keys(db, user.id)
    out = {p: bool(k) for p, k in keys.items()}

    # LinkedIn requires BOTH an access token and an author URN.
    li_token = keys.get("linkedin_token") or keys.get("linkedin", "")
    li_urn   = keys.get("linkedin_urn", "")
    if "|" in (keys.get("linkedin") or "") and not keys.get("linkedin_urn"):
        li_urn = li_urn or keys["linkedin"].partition("|")[2]

    # Twitter: 4 separate DB keys, legacy combined "ck|cs|at|ats", or all 4 env vars
    tw_ok = all((keys.get(k) or "").strip() for k in
                ("twitter_api_key", "twitter_api_secret",
                 "twitter_access_token", "twitter_access_secret"))
    if not tw_ok:
        tw = (keys.get("twitter") or "").strip()
        tw_ok = tw.count("|") == 3 and all(p.strip() for p in tw.split("|"))
    if not tw_ok:
        tw_ok = all(os.getenv(k, "").strip() for k in
                    ("TWITTER_API_KEY", "TWITTER_API_SECRET",
                     "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"))

    out["connected"] = {
        "devto":    bool(keys.get("devto")),
        "hashnode": bool(keys.get("hashnode")),
        "medium":   bool(keys.get("medium")),
        "linkedin": bool(li_token and li_urn),
        "twitter":  tw_ok,
    }
    return out


@app.get("/api/keys/load")
async def load_keys(db=Depends(get_db), user=Depends(get_current_user)):
    """Return full stored key values for settings pre-fill (requires auth)."""
    if not user:
        return {}
    return get_all_api_keys(db, user.id)


# ── Content Generation (SSE) ──────────────────────────────────────────────────
class ContentReq(BaseModel):
    topic: str
    style: str = "explainer"
    provider: str = "groq"
    model: Optional[str] = None
    api_base: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    euron_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

@app.post("/api/content/generate")
async def generate(req: ContentReq, db=Depends(get_db), user=Depends(get_current_user)):
    db_keys = get_all_api_keys(db, user.id) if user else {}

    groq_key   = req.groq_api_key   or db_keys.get("groq")   or os.getenv("GROQ_API_KEY", "")
    gemini_key = req.gemini_api_key or db_keys.get("gemini") or os.getenv("GEMINI_API_KEY", "")
    openai_key = req.openai_api_key or db_keys.get("openai") or os.getenv("OPENAI_API_KEY", "")
    nvidia_key = req.nvidia_api_key or db_keys.get("nvidia") or os.getenv("NVIDIA_API_KEY", "")
    euron_key  = req.euron_api_key  or db_keys.get("euron")  or os.getenv("EURON_API_KEY", "")
    tavily_key = req.tavily_api_key or db_keys.get("tavily") or os.getenv("TAVILY_API_KEY", "")

    if groq_key:   os.environ["GROQ_API_KEY"]   = groq_key
    if gemini_key: os.environ["GEMINI_API_KEY"]  = gemini_key
    if gemini_key: os.environ["GOOGLE_API_KEY"]  = gemini_key
    if openai_key: os.environ["OPENAI_API_KEY"]  = openai_key
    if nvidia_key: os.environ["NVIDIA_API_KEY"]  = nvidia_key
    if req.api_base: os.environ["OPENAI_API_BASE"] = req.api_base

    # Save keys to DB for logged-in users
    if user:
        if groq_key:   save_api_key(db, user.id, "groq",   groq_key)
        if gemini_key: save_api_key(db, user.id, "gemini", gemini_key)
        if openai_key: save_api_key(db, user.id, "openai", openai_key)
        if nvidia_key: save_api_key(db, user.id, "nvidia", nvidia_key)
        if euron_key:  save_api_key(db, user.id, "euron",  euron_key)
        if tavily_key: save_api_key(db, user.id, "tavily", tavily_key)

    async def stream():
        def sse(ev: str, data: dict) -> str:
            return f"event: {ev}\ndata: {json.dumps(data)}\n\n"

        try:
            # provider=="groq" is the app default — it now falls back through NVIDIA and
            # Gemini automatically, so only block if NONE of the three are configured.
            if req.provider == "groq" and not (groq_key or nvidia_key or gemini_key):
                yield sse("error", {"message":
                    "No LLM key configured. Add a Groq, NVIDIA, or Gemini API key in ⚙ Settings."}); return
            if req.provider == "gemini" and not gemini_key:
                yield sse("error", {"message": "Gemini API key required. Add it in ⚙ Settings."}); return
            if req.provider == "nvidia" and not nvidia_key:
                yield sse("error", {"message": "NVIDIA API key required. Add it in ⚙ Settings."}); return
            if req.provider in ("openai", "openai-compatible") and not openai_key:
                yield sse("error", {"message": "OpenAI API key required. Add it in ⚙ Settings."}); return
            if req.provider == "euron" and not euron_key:
                yield sse("error", {"message": "Euron API key required. Add it in ⚙ Settings."}); return
            if not tavily_key:
                yield sse("error", {"message": "Tavily API key required for web research. Add it in ⚙ Settings."}); return

            queue: asyncio.Queue = asyncio.Queue()

            async def cb(ev_type: str, data: dict):
                await queue.put((ev_type, data))

            task = asyncio.create_task(run_content_pipeline(
                topic=req.topic, style=req.style, provider=req.provider,
                model=req.model, api_base=req.api_base, tavily_api_key=tavily_key,
                event_cb=cb, euron_key=euron_key,
            ))

            while True:
                try:
                    ev_type, data = await asyncio.wait_for(queue.get(), timeout=180)
                    yield sse(ev_type, data)
                    if ev_type == "complete" and user:
                        # Auto-save draft to history
                        try:
                            save_draft(
                                db, user.id,
                                topic=req.topic,
                                title=data.get("title", req.topic),
                                cover_url=data.get("cover_url", ""),
                                markdown=data.get("markdown", ""),
                                tags=data.get("seo", {}).get("tags", []),
                                word_count=data.get("word_count", 0),
                            )
                        except Exception:
                            pass
                    if ev_type in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    yield sse("error", {"message": "Pipeline timed out after 3 minutes."}); break

            if not task.done():
                task.cancel()

        except Exception as e:
            yield sse("error", {"message": str(e), "details": traceback.format_exc()})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HITL Refine ──────────────────────────────────────────────────────────────
class RefineReq(BaseModel):
    topic: str
    markdown: str
    feedback: str
    provider: str = "groq"
    model: Optional[str] = None
    api_base: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    euron_api_key: Optional[str] = None

@app.post("/api/content/refine")
async def refine(req: RefineReq, db=Depends(get_db), user=Depends(get_current_user)):
    db_keys = get_all_api_keys(db, user.id) if user else {}
    groq_key   = req.groq_api_key   or db_keys.get("groq")   or os.getenv("GROQ_API_KEY", "")
    gemini_key = req.gemini_api_key or db_keys.get("gemini") or os.getenv("GEMINI_API_KEY", "")
    openai_key = req.openai_api_key or db_keys.get("openai") or os.getenv("OPENAI_API_KEY", "")
    nvidia_key = req.nvidia_api_key or db_keys.get("nvidia") or os.getenv("NVIDIA_API_KEY", "")
    euron_key  = req.euron_api_key  or db_keys.get("euron")  or os.getenv("EURON_API_KEY", "")
    if groq_key:   os.environ["GROQ_API_KEY"]  = groq_key
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key; os.environ["GOOGLE_API_KEY"] = gemini_key
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if nvidia_key: os.environ["NVIDIA_API_KEY"] = nvidia_key

    async def stream():
        def sse(ev, data): return f"event: {ev}\ndata: {json.dumps(data)}\n\n"
        try:
            queue: asyncio.Queue = asyncio.Queue()
            async def cb(ev_type, data): await queue.put((ev_type, data))
            task = asyncio.create_task(run_refine_pipeline(
                topic=req.topic, markdown=req.markdown, feedback=req.feedback,
                provider=req.provider, model=req.model, api_base=req.api_base,
                event_cb=cb, euron_key=euron_key,
            ))
            while True:
                try:
                    ev_type, data = await asyncio.wait_for(queue.get(), timeout=120)
                    yield sse(ev_type, data)
                    if ev_type in ("complete", "error"): break
                except asyncio.TimeoutError:
                    yield sse("error", {"message": "Refinement timed out."}); break
            if not task.done(): task.cancel()
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Publish ───────────────────────────────────────────────────────────────────
class PublishReq(BaseModel):
    platform: str
    markdown: str
    title: str
    tags: list = []
    cover_image: Optional[str] = None
    api_key: Optional[str] = None   # supplied inline if not in DB
    topic: Optional[str] = None

def _resolve_publish_key(platform: str, db_keys: dict, inline: Optional[str]) -> str:
    """Resolve a platform's credential. Priority: inline (one-shot) → DB (saved once) → .env.

    LinkedIn is special: it needs an access token AND an author URN, stored as two
    separate one-time credentials (linkedin_token / linkedin_urn). They are combined
    into the internal 'token|urn' format that _publish_linkedin expects.
    """
    if platform == "linkedin":
        token = (db_keys.get("linkedin_token")
                 or db_keys.get("linkedin")  # legacy combined "token|urn"
                 or os.getenv("LINKEDIN_ACCESS_TOKEN", ""))
        urn   = (db_keys.get("linkedin_urn")
                 or os.getenv("LINKEDIN_AUTHOR_URN", ""))
        # legacy combined value already carries the URN after a pipe
        if "|" in token and not db_keys.get("linkedin_urn"):
            token, _, legacy_urn = token.partition("|")
            urn = urn or legacy_urn
        token, urn = token.strip(), urn.strip()
        if not token or not urn:
            return ""
        return f"{token}|{urn}"

    if platform == "twitter":
        # Preferred: 4 separate one-time credentials saved from Settings
        db4 = [(db_keys.get(k) or "").strip() for k in
               ("twitter_api_key", "twitter_api_secret",
                "twitter_access_token", "twitter_access_secret")]
        if all(db4):
            return "|".join(db4)
        # Inline one-shot or legacy combined "ck|cs|at|ats"
        combined = (inline or db_keys.get("twitter") or "").strip()
        if combined:
            return combined
        # .env fallback
        env4 = [os.getenv(k, "").strip() for k in
                ("TWITTER_API_KEY", "TWITTER_API_SECRET",
                 "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET")]
        if all(env4):
            return "|".join(env4)
        return ""

    _env_fallback = {
        "devto":    "DEVTO_API_KEY",
        "hashnode": "HASHNODE_API_KEY",
        "medium":   "MEDIUM_API_KEY",
    }
    return (inline or db_keys.get(platform) or os.getenv(_env_fallback.get(platform, ""), "")).strip()


PLAT_DISPLAY = {"devto": "dev.to", "hashnode": "Hashnode", "linkedin": "LinkedIn",
                "medium": "Medium", "twitter": "X (Twitter)"}

# Platforms whose credentials are inherently multi-part / sensitive enough that we
# only ever resolve them from the DB (see _resolve_publish_key) — so publishing to
# them is only possible once a user is signed in and has saved them once.
_LOGIN_REQUIRED_PLATFORMS = {"linkedin", "twitter"}

@app.post("/api/content/publish")
async def publish(req: PublishReq, db: Session = Depends(get_db), user=Depends(get_current_user)):
    if not user and req.platform in _LOGIN_REQUIRED_PLATFORMS:
        raise HTTPException(401,
            f"Sign in with Google to publish to {PLAT_DISPLAY.get(req.platform, req.platform)}. "
            "This platform needs credentials saved securely to your account — "
            "dev.to, Hashnode & Medium can be used without signing in.")

    db_keys  = get_all_api_keys(db, user.id) if user else {}
    api_key  = _resolve_publish_key(req.platform, db_keys, req.api_key)

    if not api_key:
        if req.platform == "linkedin":
            raise HTTPException(400,
                "LinkedIn isn't connected. Add your Access Token and Author URN once "
                "in ⚙ Settings → Publishing Connections.")
        raise HTTPException(400,
            f"{req.platform} isn't connected. Add its API key once in ⚙ Settings → Publishing Connections."
            + ("" if user else " (No account? You can still publish — just paste a one-time key below.)"))

    # One-time persistence: if a key was supplied inline, remember it for next time.
    if user and req.api_key:
        save_api_key(db, user.id, req.platform, req.api_key)

    try:
        if req.platform == "devto":
            result = _publish_devto(req, api_key)
        elif req.platform == "hashnode":
            result = _publish_hashnode(req, api_key)
        elif req.platform == "linkedin":
            result = await _publish_linkedin(req, api_key)
        elif req.platform == "medium":
            result = _publish_medium(req, api_key)
        elif req.platform == "twitter":
            result = _publish_twitter(req, api_key)
        else:
            raise HTTPException(400, f"Platform '{req.platform}' not supported yet.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Publish error: {exc}") from exc

    saved_to_history = False
    if user and result.get("url"):
        save_post(db, user.id, req.topic or req.title, req.title, req.platform, result["url"],
                  cover_url=req.cover_image or "", markdown=req.markdown,
                  tags=req.tags, word_count=len(req.markdown.split()))
        saved_to_history = True

    result["saved_to_history"] = saved_to_history
    return result


def _mermaid_to_img(mermaid_code: str) -> str:
    """Convert a Mermaid diagram to a mermaid.ink image URL (base64-encoded)."""
    import base64 as _b64
    encoded = _b64.urlsafe_b64encode(mermaid_code.strip().encode()).decode().rstrip("=")
    return f"https://mermaid.ink/img/{encoded}"


def _replace_mermaid_blocks(markdown: str) -> str:
    """Replace ```mermaid ... ``` blocks with rendered image markdown for platforms that don't support Mermaid."""
    def _replace(m):
        code = m.group(1).strip()
        try:
            img_url = _mermaid_to_img(code)
            return f"![Diagram]({img_url})"
        except Exception:
            return ""  # drop unrenderable diagrams silently
    return re.sub(r"```mermaid\s*\n([\s\S]*?)```", _replace, markdown, flags=re.IGNORECASE)


_ATTRIBUTION = "\n\n---\n*✍️ Generated and published by [Quillr](https://contentai-utna.onrender.com) — AI blog writing, fully automated.*"

def _publish_devto(req: PublishReq, key: str) -> dict:
    tags = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in req.tags]
    tags = [t for t in tags if t][:4]
    # dev.to doesn't render Mermaid — convert diagrams to images first
    markdown = _replace_mermaid_blocks(req.markdown) + _ATTRIBUTION
    body: dict = {"article": {"title": req.title, "body_markdown": markdown,
                               "published": True, "tags": tags}}
    if req.cover_image:
        body["article"]["main_image"] = req.cover_image
    r = _req.post("https://dev.to/api/articles", json=body,
                  headers={"api-key": key, "Content-Type": "application/json"})
    if r.status_code in (200, 201):
        d = r.json()
        return {"success": True, "platform": "dev.to", "url": d.get("url", "https://dev.to")}
    raise HTTPException(r.status_code, r.text)


def _publish_hashnode(req: PublishReq, key: str) -> dict:
    # Key format: "token" or "token|publicationId"
    # Hashnode API v2 (gql.hashnode.com) requires publicationId.
    parts  = key.split("|", 1)
    token  = parts[0].strip()
    pub_id = parts[1].strip() if len(parts) > 1 else ""

    if not pub_id:
        raise HTTPException(400,
            "Hashnode requires a Publication ID. "
            "In Settings → Publishing Connections enter your token as: "
            "YOUR_TOKEN|YOUR_PUBLICATION_ID  "
            "(find your Publication ID in Hashnode Dashboard → Settings → General).")

    mutation = """
    mutation PublishPost($input: PublishPostInput!) {
      publishPost(input: $input) { post { url title } }
    }"""
    inp: dict = {
        "title": req.title,
        "contentMarkdown": req.markdown + _ATTRIBUTION,
        "publicationId": pub_id,
        "tags": [],
    }
    if req.cover_image:
        inp["coverImageOptions"] = {"coverImageURL": req.cover_image}

    _payload = {"query": mutation, "variables": {"input": inp}}
    _headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "User-Agent": "ContentAI/1.0",
    }
    try:
        # httpx preserves POST on 307/308 redirects (requests silently converts to GET)
        with httpx.Client(follow_redirects=True, timeout=30) as hx:
            resp = hx.post("https://gql.hashnode.com", json=_payload, headers=_headers)
    except Exception as exc:
        raise HTTPException(502, f"Could not reach Hashnode API: {exc}") from exc

    # Adapt to requests-style interface
    class _R:
        def __init__(self, r):
            self.status_code = r.status_code
            self.text = r.text
            self._r = r
        def json(self): return self._r.json()

    r = _R(resp)

    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code,
            f"Hashnode API returned {r.status_code}: {r.text[:300] or '(empty response)'}")

    raw = r.text.strip()
    if not raw:
        raise HTTPException(502, "Hashnode returned an empty response. Check your token.")

    if raw.lstrip().startswith("<"):
        raise HTTPException(502,
            "Hashnode returned an HTML page instead of JSON. "
            "This usually means the API endpoint has moved or your token is invalid. "
            f"Final URL reached: {str(resp.url)}")

    try:
        d = r.json()
    except ValueError:
        raise HTTPException(502,
            f"Hashnode returned non-JSON (URL: {resp.url}): {raw[:200]}")

    # GraphQL surfaces errors in the body even on HTTP 200
    gql_errors = d.get("errors")
    if gql_errors:
        msg = gql_errors[0].get("message", str(gql_errors[0]))
        raise HTTPException(400, f"Hashnode error: {msg}")

    post = (d.get("data") or {}).get("publishPost", {}).get("post") or {}
    if not post:
        raise HTTPException(400,
            "Hashnode returned no post data — double-check your token and Publication ID.")

    return {"success": True, "platform": "Hashnode", "url": post.get("url", "https://hashnode.com")}


# ─── LinkedIn ────────────────────────────────────────────────────────────────
# Key format stored in DB: "access_token|urn:li:person:XXXXXXX"
# Reuses the proven v2 UGC Posts API pattern from ResearchAI/server.py

_BOLD_CHARS = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇"
_PLAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BOLD_MAP    = str.maketrans(_PLAIN_CHARS, _BOLD_CHARS)

_LINKEDIN_POST_PROMPT = """\
You are a top-tier LinkedIn creator — your posts get 50k+ impressions because you write with the
specificity of a researcher, the clarity of a teacher, and the voice of someone who actually ships things.
Study how Justin Welsh, Lenny Rachitsky, and top AI researchers write on LinkedIn.

Convert this blog post into a LinkedIn post that stops the scroll.

Blog title: {title}
Blog content:
{content}

THE ANATOMY OF A WORLD-CLASS LINKEDIN POST:

**Line 1 (the hook — this is everything):**
One sentence. No warm-up. Options:
- A specific number that surprises: "OpenAI's o3 scored 87.5% on ARC-AGI. Human average: 85%."
- A counterintuitive truth: "The best engineers I know write less code, not more."
- A micro-story opener: "Last Tuesday I watched a junior dev ship in 2 hours what took me 2 days in 2019."
- A bold claim with stakes: "Most AI agents fail in production for one reason nobody talks about."
NEVER start with: "In this article", "I'm excited to share", "We all know", "Have you ever", "Today I".

**Lines 2-3 (blank line after hook, then the setup):**
Why does this matter? What's the tension or the cost of ignoring this? 2-3 short sentences max.

**The body (3-4 short paragraphs, separated by blank lines):**
- Each paragraph = one sharp idea. Max 3-4 lines.
- Mix formats: a paragraph, then maybe a mini-list, then a paragraph.
- Use concrete specifics — versions, numbers, names, exact outcomes.
- Write in second person ("you") like you're talking directly to one smart person.
- Short sentences land harder than long ones. Vary rhythm.

**Optional mini-list (3-5 items, only if it genuinely helps):**
Use → or numbered format. Each item must be specific and actionable, not vague.
Bad: "Use better prompts" → Good: "Prefix every system prompt with the output format you want"

**Closing (2-3 sentences):**
- One insight that reframes the whole post — the "aha" the reader takes away.
- End with a question that creates real debate. NOT "What do you think?" or "Agree?"
  Make it specific: "If you could only pick one of these, which would you bet on and why?"

**Hashtags (last line, separated by blank line):**
5-7 hashtags. Mix broad (#AI) with specific (#LLMOps #PromptEngineering). No made-up tags.

HARD RULES:
- Total: 1500–2500 characters (sweet spot for reach)
- ZERO of these words: leverage, utilize, paradigm, synergy, robust, scalable, seamless, holistic,
  transformative, game-changing, revolutionize, cutting-edge, unlock potential, excited to share,
  in today's world, it's worth noting, at the end of the day, delve, tapestry, moreover, furthermore
- NEVER use "It's not just X, it's Y" or "The result?" as a rhetorical reveal — those read as AI-written
- NO "Key takeaways:" headers (it's lazy and signals low-quality content to the algorithm)
- Vary sentence length hard — some 3 words, some long and winding. Uniform rhythm reads as AI-written.
- Take an actual position somewhere in the post — a mild disagreement with conventional wisdom, a specific
  thing that surprised you. Diplomatic even-handedness about everything is the #1 giveaway of AI writing.
- Use emojis sparingly and purposefully — 1-2 in the hook or closing for energy, NOT as a bullet prefix on every line
- Bold (**text**) maximum 3 times, only for the single most important phrase per section

Write the LinkedIn post now. Make it read like a specific person wrote it, not like a summary of the article."""


def _md_to_linkedin_text(text: str) -> str:
    """Clean LLM output for LinkedIn: strip markdown syntax, keep bold as unicode, preserve structure."""
    lines = []
    for line in text.split("\n"):
        # Drop code blocks entirely (not appropriate for LinkedIn)
        if line.strip().startswith("```"):
            continue
        # Headings → remove # prefix (LLM shouldn't add these but just in case)
        if re.match(r"^#{1,4}\s+", line):
            line = re.sub(r"^#{1,4}\s+", "", line).strip()
        # **bold** → unicode bold characters
        line = re.sub(r"\*\*(.+?)\*\*",
                      lambda m: m.group(1).translate(_BOLD_MAP), line)
        # *italic* → plain (no markdown on LinkedIn)
        line = re.sub(r"\*(.+?)\*", r"\1", line)
        # inline code → plain text
        line = re.sub(r"`([^`]+)`", r"\1", line)
        # strip image tags entirely
        line = re.sub(r"!\[.*?\]\(.*?\)", "", line)
        # links → keep label only
        line = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", line)
        # strip [IMAGE: ...] placeholders
        line = re.sub(r"\[IMAGE:.*?\]", "", line, flags=re.IGNORECASE)
        lines.append(line)
    # Collapse 3+ consecutive blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip()


def _summarise_markdown_for_linkedin(markdown: str, max_chars: int = 6000) -> str:
    """Extract the most insight-rich content from a full blog post for the LinkedIn prompt."""
    # Strip image placeholders, code blocks, and mermaid diagrams
    text = re.sub(r"```[\s\S]*?```", "", markdown)
    text = re.sub(r"\[IMAGE:.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Strip markdown syntax
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def _fetch_image_bytes(image_url: str) -> tuple:
    """Download cover image bytes. Handles local /uploads paths and remote URLs.
    Returns (bytes, content_type)."""
    if image_url.startswith("/"):
        p = Path(__file__).parent / image_url.lstrip("/")
        if not p.exists():
            raise RuntimeError(f"Local image not found: {image_url}")
        ext = p.suffix.lower().lstrip(".") or "png"
        return p.read_bytes(), f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
    r = _req.get(image_url, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
    return r.content, ctype


def _linkedin_upload_image(token: str, owner_urn: str, image_url: str) -> str:
    """Upload an image to LinkedIn's asset API. Returns the asset URN for UGC posts."""
    data, _ctype = _fetch_image_bytes(image_url)

    reg = _req.post(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        json={"registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": owner_urn,
            "serviceRelationships": [{
                "relationshipType": "OWNER",
                "identifier": "urn:li:userGeneratedContent",
            }],
        }},
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "X-Restli-Protocol-Version": "2.0.0"},
        timeout=30,
    )
    if reg.status_code not in (200, 201):
        raise RuntimeError(f"LinkedIn registerUpload failed ({reg.status_code}): {reg.text[:200]}")
    v = reg.json()["value"]
    upload_url = v["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
    asset_urn  = v["asset"]

    up = _req.put(upload_url, data=data,
                  headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"LinkedIn image upload failed ({up.status_code})")
    return asset_urn


async def _publish_linkedin(req: PublishReq, key: str) -> dict:
    """Post to LinkedIn UGC Posts API — same endpoint proven in ResearchAI."""
    parts  = key.split("|", 1)
    token  = parts[0].strip()
    author = parts[1].strip() if len(parts) > 1 else ""
    if not author:
        raise HTTPException(400,
            "LinkedIn key must be 'access_token|urn:li:person:XXXXX'. "
            "Get your URN from the LinkedIn Developer portal.")

    # Generate a LinkedIn-formatted post from the FULL blog using LLM
    try:
        from content_agent import get_llm_resilient as _agent_get_llm
        from langchain_core.messages import HumanMessage as _HM
        lk_llm = _agent_get_llm("groq", None)
        full_content = _summarise_markdown_for_linkedin(req.markdown)
        raw = lk_llm.invoke([_HM(content=_LINKEDIN_POST_PROMPT.format(
            title=req.title, content=full_content
        ))])
        linkedin_text = _md_to_linkedin_text(
            raw.content if hasattr(raw, "content") else str(raw)
        )[:3000]
    except Exception:
        plain = re.sub(r"[#*`!\[\]()>]", "", req.markdown[:2000]).strip()
        tags_str = " ".join(f"#{t}" for t in req.tags[:5]) if req.tags else ""
        linkedin_text = f"{req.title}\n\n{plain[:1800]}\n\n{tags_str}"[:3000]

    author_urn = author if author.startswith("urn:") else f"urn:li:person:{author}"

    # Upload cover image so the post carries the visual (falls back to text-only)
    asset_urn = None
    if req.cover_image:
        try:
            asset_urn = _linkedin_upload_image(token, author_urn, req.cover_image)
        except Exception as exc:
            print(f"[linkedin] image upload failed, posting text-only: {exc}")

    share_content: dict = {
        "shareCommentary": {"text": linkedin_text},
        "shareMediaCategory": "IMAGE" if asset_urn else "NONE",
    }
    if asset_urn:
        share_content["media"] = [{
            "status": "READY",
            "media": asset_urn,
            "title": {"text": req.title[:100]},
        }]

    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {"com.linkedin.ugc.ShareContent": share_content},
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    r = _req.post("https://api.linkedin.com/v2/ugcPosts", json=payload, headers=headers)
    if r.status_code in (200, 201):
        post_id = r.json().get("id", "")
        return {"success": True, "platform": "LinkedIn",
                "url": f"https://www.linkedin.com/feed/",
                "post_id": post_id}
    raise HTTPException(r.status_code, r.text)


# ─── LinkedIn preview endpoint ───────────────────────────────────────────────

class LinkedInPreviewReq(BaseModel):
    markdown: str
    title: str
    tags: list = []
    cover_image: Optional[str] = None

@app.post("/api/content/linkedin-preview")
async def linkedin_preview(req: LinkedInPreviewReq):
    """Generate the LinkedIn post text WITHOUT publishing. Used for the preview modal."""
    try:
        from content_agent import get_llm_resilient as _agent_get_llm
        from langchain_core.messages import HumanMessage as _HM
        lk_llm = _agent_get_llm("groq", None)
        full_content = _summarise_markdown_for_linkedin(req.markdown)
        raw = lk_llm.invoke([_HM(content=_LINKEDIN_POST_PROMPT.format(
            title=req.title, content=full_content
        ))])
        text = _md_to_linkedin_text(
            raw.content if hasattr(raw, "content") else str(raw)
        )[:3000]
    except Exception:
        plain = re.sub(r"[#*`!\[\]()>]", "", req.markdown[:2000]).strip()
        tags_str = " ".join(f"#{t}" for t in req.tags[:5]) if req.tags else ""
        text = f"{req.title}\n\n{plain[:1800]}\n\n{tags_str}"[:3000]

    return {"text": text, "char_count": len(text), "cover_image": req.cover_image or ""}


# ─── Medium ───────────────────────────────────────────────────────────────────
# Key: just the Medium Integration Token (Settings → Security → Integration Tokens)

def _publish_medium(req: PublishReq, key: str) -> dict:
    """Publish a blog post to Medium using the Integration Token."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Step 1: get the user's Medium ID
    me = _req.get("https://api.medium.com/v1/me", headers=headers)
    if me.status_code != 200:
        raise HTTPException(me.status_code,
            f"Medium auth failed — check your integration token. ({me.text})")
    user_id = me.json()["data"]["id"]

    # Step 2: sanitize tags (Medium allows up to 5, plain strings)
    tags = [re.sub(r"[^a-zA-Z0-9 ]", "", t)[:25] for t in req.tags][:5]

    # Step 3: publish
    body: dict = {
        "title":         req.title,
        "contentFormat": "markdown",
        "content":       req.markdown + _ATTRIBUTION,
        "tags":          tags,
        "publishStatus": "public",
    }
    if req.cover_image:
        # Medium doesn't accept a cover image via API directly;
        # embed it at the top of the markdown instead
        body["content"] = f"![cover]({req.cover_image})\n\n{req.markdown}"

    r = _req.post(f"https://api.medium.com/v1/users/{user_id}/posts",
                  json=body, headers=headers)
    if r.status_code in (200, 201):
        d    = r.json()
        post = d.get("data", {})
        return {"success": True, "platform": "Medium",
                "url": post.get("url", "https://medium.com")}
    raise HTTPException(r.status_code, r.text)


# ─── X (Twitter) ─────────────────────────────────────────────────────────────
# Key format stored in DB: "api_key|api_secret|access_token|access_token_secret"
# Long posts are auto-split into a thread (max 8 tweets); cover image on tweet 1.

_TWEET_LIMIT   = 275   # keep headroom under X's 280
_THREAD_MAX    = 8

_TWITTER_THREAD_PROMPT = """\
You are a viral X (Twitter) thread writer. Convert this blog post into a thread.

Blog title: {title}
Blog content:
{content}

RULES:
1. Write {max_tweets} tweets MAXIMUM. Fewer is fine if the content fits.
2. Each tweet MUST be under 270 characters (hard limit — count carefully).
3. Tweet 1 is the hook: a surprising number, bold claim, or curiosity gap. No "A thread 🧵" clichés
   as the whole tweet — the hook does the work, then end tweet 1 with " 🧵".
4. Number each tweet like "2/" at the start (skip numbering on tweet 1).
5. One idea per tweet. Vary tweet length hard — some are one short punchy line, some use the full limit.
6. Last tweet: one specific takeaway (a real opinion, not a recap) + a question that invites replies + 2-3 hashtags.
7. BANNED: leverage, utilize, paradigm, game-changer, revolutionize, "in today's world", delve, moreover,
   "it's not just X, it's Y", "the result?" as a rhetorical reveal.
8. Sound like a person tweeting a hot take, not an AI summarizing an article — commit to an opinion somewhere.
9. Separate each tweet with a line containing exactly: ---

Write the thread now."""


def _fallback_thread_split(title: str, markdown: str, tags: list) -> list:
    """No-LLM fallback: chunk cleaned text into tweets at sentence boundaries."""
    text = _summarise_markdown_for_linkedin(markdown, max_chars=2000)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    tweets, cur = [], title.strip()
    for s in sentences:
        if len(cur) + len(s) + 1 <= _TWEET_LIMIT - 10:
            cur = f"{cur} {s}".strip()
        else:
            tweets.append(cur)
            cur = s
        if len(tweets) >= _THREAD_MAX - 1:
            break
    if cur and len(tweets) < _THREAD_MAX:
        tweets.append(cur)
    if tags:
        tag_str = " ".join(f"#{t}" for t in tags[:3])
        if len(tweets[-1]) + len(tag_str) + 1 <= _TWEET_LIMIT:
            tweets[-1] = f"{tweets[-1]} {tag_str}"
    return [t[:_TWEET_LIMIT] for t in tweets if t.strip()]


def _generate_tweet_thread(req: PublishReq) -> list:
    """LLM-generate the thread; fall back to sentence chunking."""
    try:
        from content_agent import get_llm_resilient as _agent_get_llm
        from langchain_core.messages import HumanMessage as _HM
        llm = _agent_get_llm("groq", None)
        full_content = _summarise_markdown_for_linkedin(req.markdown)
        raw = llm.invoke([_HM(content=_TWITTER_THREAD_PROMPT.format(
            title=req.title, content=full_content, max_tweets=_THREAD_MAX))])
        text = raw.content if hasattr(raw, "content") else str(raw)
        tweets = [t.strip() for t in text.split("---") if t.strip()]
        # Clean markdown artefacts + enforce limits
        tweets = [re.sub(r"\*\*(.+?)\*\*", r"\1", t)[:_TWEET_LIMIT] for t in tweets]
        tweets = [t for t in tweets if t][:_THREAD_MAX]
        if tweets:
            return tweets
    except Exception as exc:
        print(f"[twitter] LLM thread generation failed, using fallback: {exc}")
    return _fallback_thread_split(req.title, req.markdown, req.tags)


def _publish_twitter(req: PublishReq, key: str) -> dict:
    """Post to X as a thread via tweepy. Cover image attached to the first tweet."""
    parts = [p.strip() for p in key.split("|")]
    if len(parts) != 4 or not all(parts):
        raise HTTPException(400,
            "X (Twitter) needs 4 credentials in one field: "
            "API_KEY|API_SECRET|ACCESS_TOKEN|ACCESS_TOKEN_SECRET  "
            "(from developer.x.com → your app → Keys and tokens).")
    ck, cs, at, ats = parts

    try:
        import tweepy
    except ImportError:
        raise HTTPException(500, "tweepy not installed on the server — run: pip3 install tweepy")

    tweets = _generate_tweet_thread(req)
    if not tweets:
        raise HTTPException(400, "Could not build any tweets from this post.")

    client = tweepy.Client(consumer_key=ck, consumer_secret=cs,
                           access_token=at, access_token_secret=ats)

    # Upload cover image (v1.1 media endpoint needs OAuth1)
    media_id = None
    if req.cover_image:
        try:
            import io
            auth = tweepy.OAuth1UserHandler(ck, cs, at, ats)
            api_v1 = tweepy.API(auth)
            data, ctype = _fetch_image_bytes(req.cover_image)
            ext = "jpg" if "jpeg" in ctype else ctype.split("/")[-1]
            media = api_v1.media_upload(filename=f"cover.{ext}", file=io.BytesIO(data))
            media_id = media.media_id
        except Exception as exc:
            print(f"[twitter] media upload failed, posting text-only: {exc}")

    # Post the chain
    first_id, prev_id, posted = None, None, 0
    try:
        for i, t in enumerate(tweets):
            kwargs: dict = {"text": t}
            if i == 0 and media_id:
                kwargs["media_ids"] = [media_id]
            if prev_id:
                kwargs["in_reply_to_tweet_id"] = prev_id
            resp = client.create_tweet(**kwargs)
            prev_id = resp.data["id"]
            if first_id is None:
                first_id = prev_id
            posted += 1
    except Exception as exc:
        if posted:
            return {"success": True, "platform": "X (Twitter)",
                    "url": f"https://x.com/i/status/{first_id}",
                    "tweets": posted,
                    "warning": f"Thread cut short after {posted}/{len(tweets)} tweets: {exc}"}
        err = str(exc)
        if "402" in err or "credits" in err.lower():
            raise HTTPException(402,
                "X now charges per post (Pay Per Use) and your account has no credits. "
                "Free option: use the thread preview in this dialog — copy each tweet and paste "
                "it on X manually. Or buy credits at developer.x.com → Billing.") from exc
        if "attached to a Project" in err:
            raise HTTPException(403,
                "Your X app isn't attached to a Project (required for API v2). "
                "Fix: developer.x.com → Dashboard → + Create Project → attach your app to it → "
                "regenerate your Access Token & Secret → update them in ⚙ Settings.") from exc
        if "oauth1 app permissions" in err.lower() or "not permitted" in err.lower():
            raise HTTPException(403,
                "Your X access token doesn't have write permission. "
                "Fix: developer.x.com → your app → User authentication settings → set 'Read and write' → "
                "regenerate your Access Token & Secret → update them in ⚙ Settings.") from exc
        raise HTTPException(502, f"X API error: {exc}") from exc

    return {"success": True, "platform": "X (Twitter)",
            "url": f"https://x.com/i/status/{first_id}", "tweets": posted}


class TwitterPreviewReq(BaseModel):
    markdown: str
    title: str
    tags: list = []

@app.post("/api/content/twitter-preview")
async def twitter_preview(req: TwitterPreviewReq):
    """Generate the X thread WITHOUT posting — used for the preview + manual copy-paste flow."""
    fake = PublishReq(platform="twitter", markdown=req.markdown,
                      title=req.title, tags=req.tags)
    tweets = _generate_tweet_thread(fake)
    return {"tweets": tweets, "count": len(tweets)}


# ── Image Upload ──────────────────────────────────────────────────────────────
_ALLOWED_IMG = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_BYTES    = 8 * 1024 * 1024   # 8 MB

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Accept a user-uploaded image and return its public URL."""
    if file.content_type not in _ALLOWED_IMG:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG/PNG/WEBP/GIF.")
    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, "Image too large — max 8 MB.")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    (Path(_UPLOADS) / filename).write_bytes(data)
    return {"url": f"/uploads/{filename}", "filename": filename}


# ── History ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(db: Session = Depends(get_db), user=Depends(get_current_user)):
    if not user:
        return []
    from content_db import BlogPost
    posts = (db.query(BlogPost)
               .filter_by(user_id=user.id)
               .order_by(BlogPost.created_at.desc())
               .limit(200)
               .all())

    # Group by normalised title → deduplicate, collect all platforms
    seen: dict = {}  # title → merged entry
    for p in posts:
        key = (p.title or p.topic or "").strip().lower()
        if key not in seen:
            seen[key] = {
                "id": p.id,
                "topic": p.topic,
                "title": p.title,
                "platforms": [],          # all platforms this post was sent to
                "urls": {},               # platform → url
                "cover_url": p.cover_url or "",
                "tags": (p.tags or "").split(",") if p.tags else [],
                "word_count": p.word_count or 0,
                "markdown": p.markdown or "",
                "created_at": p.created_at.isoformat(),
            }
        entry = seen[key]
        plat = p.platform or "draft"
        if plat not in entry["platforms"]:
            entry["platforms"].append(plat)
        if p.post_url:
            entry["urls"][plat] = p.post_url
        # Prefer the richest markdown / most recent cover
        if p.markdown and len(p.markdown) > len(entry.get("markdown", "")):
            entry["markdown"] = p.markdown
        if p.cover_url and not entry["cover_url"]:
            entry["cover_url"] = p.cover_url

    return list(seen.values())[:50]


@app.delete("/api/history/{post_id}")
async def delete_history(post_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    from content_db import BlogPost
    post = db.query(BlogPost).filter_by(id=post_id, user_id=user.id).first()
    if not post:
        raise HTTPException(404, "Post not found")
    db.delete(post)
    db.commit()
    return {"success": True}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "product": "ContentAI", "port": 8001}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("content_server:app", host="0.0.0.0", port=8001, reload=True)

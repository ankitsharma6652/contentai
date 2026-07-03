"""
Database models and helpers for ContentAI.
PostgreSQL (Neon) in production via DATABASE_URL env var.
SQLite fallback for local development.
Fernet encryption for stored API keys.
"""
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet

# ── Engine: Neon PostgreSQL in prod, SQLite locally ───────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL:
    # Neon gives postgres:// — SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    _IS_POSTGRES = True
else:
    DB_PATH = Path.home() / ".contentai" / "content.db"
    DB_PATH.parent.mkdir(exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    _IS_POSTGRES = False

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    google_id  = Column(String, unique=True, index=True, nullable=False)
    email      = Column(String, unique=True, index=True)
    name       = Column(String)
    avatar     = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, index=True, nullable=False)
    platform      = Column(String, nullable=False)
    encrypted_key = Column(Text, nullable=False)
    updated_at    = Column(DateTime, default=datetime.utcnow)


class BlogPost(Base):
    __tablename__ = "blog_posts"
    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, index=True)
    topic       = Column(String)
    title       = Column(String)
    platform    = Column(String)
    post_url    = Column(String)
    cover_url   = Column(String)
    markdown    = Column(Text)
    tags        = Column(String)          # comma-separated
    word_count  = Column(Integer)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ── Fernet (encryption) ───────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    key = os.getenv("FERNET_KEY", "")
    if not key:
        env_path = Path(__file__).parent / ".env"
        existing = env_path.read_text() if env_path.exists() else ""
        if "FERNET_KEY=" not in existing:
            key = Fernet.generate_key().decode()
            with open(env_path, "a") as f:
                f.write(f"\nFERNET_KEY={key}\n")
            os.environ["FERNET_KEY"] = key
        else:
            for line in existing.splitlines():
                if line.startswith("FERNET_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["FERNET_KEY"] = key
                    break
    raw = key.encode() if isinstance(key, str) else key
    return Fernet(raw)


def _encrypt(value: str) -> str:
    return _get_fernet().encrypt(value.encode()).decode()


def _decrypt(token: str) -> str:
    return _get_fernet().decrypt(token.encode()).decode()


# ── DB init + migrations ──────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)

    if _IS_POSTGRES:
        # PostgreSQL: use information_schema to check columns
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='blog_posts'"
            ))
            existing = {r[0] for r in result.fetchall()}
            for col, typedef in [
                ("cover_url",  "TEXT DEFAULT ''"),
                ("markdown",   "TEXT DEFAULT ''"),
                ("tags",       "TEXT DEFAULT ''"),
                ("word_count", "INTEGER DEFAULT 0"),
            ]:
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE blog_posts ADD COLUMN {col} {typedef}"))
            conn.commit()
    else:
        # SQLite: use PRAGMA
        with engine.connect() as conn:
            existing = {r[1] for r in conn.execute(text("PRAGMA table_info(blog_posts)")).fetchall()}
            for col, typedef in [
                ("cover_url",  "TEXT DEFAULT ''"),
                ("markdown",   "TEXT DEFAULT ''"),
                ("tags",       "TEXT DEFAULT ''"),
                ("word_count", "INTEGER DEFAULT 0"),
            ]:
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE blog_posts ADD COLUMN {col} {typedef}"))
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_user(db, info: dict) -> User:
    google_id = info.get("sub") or info.get("id")
    user = db.query(User).filter_by(google_id=google_id).first()
    if user:
        user.name   = info.get("name", user.name)
        user.avatar = info.get("picture", user.avatar)
        db.commit()
    else:
        user = User(
            google_id=google_id,
            email=info.get("email"),
            name=info.get("name"),
            avatar=info.get("picture"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def save_api_key(db, user_id: int, platform: str, key: str):
    enc = _encrypt(key)
    rec = db.query(ApiKey).filter_by(user_id=user_id, platform=platform).first()
    if rec:
        rec.encrypted_key = enc
        rec.updated_at    = datetime.utcnow()
    else:
        db.add(ApiKey(user_id=user_id, platform=platform, encrypted_key=enc))
    db.commit()


def get_api_key(db, user_id: int, platform: str) -> str:
    rec = db.query(ApiKey).filter_by(user_id=user_id, platform=platform).first()
    if not rec:
        return ""
    try:
        return _decrypt(rec.encrypted_key)
    except Exception:
        return ""


def get_all_api_keys(db, user_id: int) -> dict:
    recs = db.query(ApiKey).filter_by(user_id=user_id).all()
    out = {}
    for rec in recs:
        try:
            out[rec.platform] = _decrypt(rec.encrypted_key)
        except Exception:
            pass
    return out


def save_post(db, user_id: int, topic: str, title: str, platform: str, url: str,
              cover_url: str = "", markdown: str = "", tags: list = None, word_count: int = 0):
    db.add(BlogPost(
        user_id=user_id, topic=topic, title=title,
        platform=platform, post_url=url,
        cover_url=cover_url, markdown=markdown,
        tags=",".join(tags or []), word_count=word_count,
    ))
    db.commit()


def save_draft(db, user_id: int, topic: str, title: str, cover_url: str,
               markdown: str, tags: list, word_count: int):
    existing = (db.query(BlogPost)
                  .filter_by(user_id=user_id, topic=topic, platform="draft")
                  .first())
    if existing:
        existing.title      = title
        existing.cover_url  = cover_url
        existing.markdown   = markdown
        existing.tags       = ",".join(tags or [])
        existing.word_count = word_count
        existing.created_at = datetime.utcnow()
    else:
        db.add(BlogPost(
            user_id=user_id, topic=topic, title=title,
            platform="draft", post_url="",
            cover_url=cover_url, markdown=markdown,
            tags=",".join(tags or []), word_count=word_count,
        ))
    db.commit()

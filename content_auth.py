"""
JWT auth helpers for ContentAI.
Tokens stored in httpOnly cookies — never exposed to JS.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Cookie, Depends, HTTPException
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from content_db import get_db, User

SECRET_KEY = os.getenv("CONTENT_SECRET_KEY", "contentai-secret-change-in-prod")
ALGORITHM  = "HS256"
EXPIRE_DAYS = 30


def create_token(user_id: int) -> str:
    exp = datetime.utcnow() + timedelta(days=EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> Optional[int]:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(data["sub"])
    except (JWTError, KeyError, ValueError):
        return None


def get_current_user(
    content_token: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not content_token:
        return None
    uid = _decode_token(content_token)
    if uid is None:
        return None
    return db.query(User).filter_by(id=uid).first()


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in with Google first.")
    return user

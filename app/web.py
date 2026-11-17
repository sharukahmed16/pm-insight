"""Helpers shared by all pages: templates, login checks, flash messages."""
import os

from fastapi import Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ITEM_STATUSES, ITEM_TYPES, STATUS_LABELS, User

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
templates.env.globals.update(STATUS_LABELS=STATUS_LABELS, ITEM_STATUSES=ITEM_STATUSES,
                             ITEM_TYPES=ITEM_TYPES)


class LoginRequired(Exception):
    """Raised by require_user; main.py turns it into a redirect to /login."""


def render(request, name, status_code=200, **context):
    """Render a template with the logged-in user and any pending flash messages."""
    context["me"] = getattr(request.state, "user", None)
    context["flashes"] = request.session.pop("flash", [])
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def flash(request, message, kind="ok"):
    """A one-time message shown on the next page (kind: ok or error)."""
    request.session.setdefault("flash", []).append({"kind": kind, "text": message})


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency for every page that needs a logged-in user."""
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id else None
    if user is None:
        request.session.clear()
        raise LoginRequired()
    request.state.user = user
    return user


def require_manager(user: User = Depends(require_user)) -> User:
    if user.role != "manager":
        raise HTTPException(status_code=403, detail="Only a manager can do this.")
    return user


def get_or_404(db, model, object_id):
    obj = db.get(model, object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} {object_id} not found.")
    return obj


def to_int(value):
    """'' -> None, '5' -> 5 (form fields arrive as strings)."""
    value = (value or "").strip()
    return int(value) if value.isdigit() else None

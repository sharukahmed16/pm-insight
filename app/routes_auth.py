"""Login, registration, logout and the manager's user list."""
import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ROLES, User
from app.security import hash_password, verify_password
from app.web import flash, get_or_404, render, require_manager

router = APIRouter()
USERNAME = re.compile(r"^[a-z0-9_.]{3,30}$")
MIN_PASSWORD = 8


@router.get("/login")
def login_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/projects", status_code=303)
    return render(request, "login.html")


@router.post("/login")
def login(request: Request, username: str = Form(""), password: str = Form(""),
          db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == username.strip().lower()))
    if user is None or not verify_password(password, user.password_hash):
        # the same message either way, so the page does not reveal which usernames exist
        return render(request, "login.html", status_code=400, username=username,
                      error="Wrong username or password.")
    request.session.clear()               # a fresh session at every login
    request.session["user_id"] = user.id
    return RedirectResponse("/projects", status_code=303)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/register")
def register_page(request: Request):
    return render(request, "register.html", values={})


@router.post("/register")
def register(request: Request, full_name: str = Form(""), username: str = Form(""),
             password: str = Form(""), password2: str = Form(""),
             db: Session = Depends(get_db)):
    username = username.strip().lower()
    full_name = full_name.strip()
    errors = []
    if not full_name:
        errors.append("Enter your full name.")
    if not USERNAME.match(username):
        errors.append("Username: 3-30 characters, lower-case letters, digits, '.' or '_'.")
    elif db.scalar(select(User).where(User.username == username)):
        errors.append("That username is taken.")
    if len(password) < MIN_PASSWORD:
        errors.append(f"Password: at least {MIN_PASSWORD} characters.")
    elif password != password2:
        errors.append("The two passwords do not match.")
    if errors:
        return render(request, "register.html", status_code=400, errors=errors,
                      values={"full_name": full_name, "username": username})

    # the very first account becomes the manager, everyone after is a member
    first = db.scalar(select(func.count(User.id))) == 0
    user = User(username=username, full_name=full_name, password_hash=hash_password(password),
                role="manager" if first else "member")
    db.add(user)
    db.commit()
    request.session.clear()
    request.session["user_id"] = user.id
    flash(request, f"Welcome, {full_name}." + (" You are the manager." if first else ""))
    return RedirectResponse("/projects", status_code=303)


@router.get("/users")
def users_page(request: Request, me: User = Depends(require_manager), db: Session = Depends(get_db)):
    users = db.scalars(select(User).order_by(User.full_name)).all()
    return render(request, "users.html", users=users, roles=ROLES)


@router.post("/users/{user_id}/role")
def change_role(request: Request, user_id: int, role: str = Form(...),
                me: User = Depends(require_manager), db: Session = Depends(get_db)):
    user = get_or_404(db, User, user_id)
    if role not in ROLES:
        flash(request, "Unknown role.", "error")
    elif user.id == me.id:
        # a manager cannot demote themselves, so there is always at least one manager
        flash(request, "You cannot change your own role.", "error")
    else:
        user.role = role
        db.commit()
        flash(request, f"{user.full_name} is now a {role}.")
    return RedirectResponse("/users", status_code=303)

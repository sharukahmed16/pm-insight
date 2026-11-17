"""PM-Insight web application.

Start (from the repository root):   uvicorn app.main:app
Then open http://127.0.0.1:8000
"""
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app import routes_auth, routes_items, routes_projects
from app.config import SECRET_KEY
from app.web import LoginRequired, render

app = FastAPI(title="PM-Insight")

# The session is a signed cookie: the user can read it but cannot change it
# without SECRET_KEY. same_site="lax" stops other sites from sending our forms
# (cross-site request forgery). 8 hours, then log in again.
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", max_age=8 * 3600)

app.include_router(routes_auth.router)
app.include_router(routes_projects.router)
app.include_router(routes_items.router)


@app.exception_handler(LoginRequired)
def to_login(request: Request, exc: LoginRequired):
    return RedirectResponse("/login", status_code=303)


@app.exception_handler(StarletteHTTPException)
def error_page(request: Request, exc: StarletteHTTPException):
    return render(request, "error.html", status_code=exc.status_code,
                  code=exc.status_code, detail=exc.detail)

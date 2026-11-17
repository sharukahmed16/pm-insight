"""Projects, their backlog, and sprints (plan -> start -> close)."""
import re
from datetime import date

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.estimator import REFERENCE_PROJECTS
from app.models import Project, Sprint, User
from app.web import flash, get_or_404, render, require_manager, require_user

router = APIRouter()
PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def points(items):
    return sum(i.story_points or 0 for i in items)


def suggestion_stats(items):
    """How the model's suggestions compare with the points the team decided."""
    pairs = [(i.suggested_points, i.story_points) for i in items
             if i.suggested_points is not None and i.story_points is not None]
    if not pairs:
        return None
    return {"n": len(pairs),
            "exact": round(100 * sum(s == p for s, p in pairs) / len(pairs)),
            "mae": round(sum(abs(s - p) for s, p in pairs) / len(pairs), 2)}


# ------------------------------------------------------------------- projects
@router.get("/")
def home():
    return RedirectResponse("/projects", status_code=303)


@router.get("/projects")
def project_list(request: Request, me: User = Depends(require_user), db: Session = Depends(get_db)):
    projects = db.scalars(select(Project).order_by(Project.name)).all()
    return render(request, "projects.html", projects=projects)


@router.get("/projects/new")
def project_form(request: Request, me: User = Depends(require_manager)):
    return render(request, "project_new.html", values={}, references=REFERENCE_PROJECTS)


@router.post("/projects/new")
def project_create(request: Request, name: str = Form(""), key: str = Form(""),
                   description: str = Form(""), reference_project: str = Form(""),
                   me: User = Depends(require_manager), db: Session = Depends(get_db)):
    name, key = name.strip(), key.strip().upper()
    errors = []
    if not name:
        errors.append("Enter a project name.")
    if not PROJECT_KEY.match(key):
        errors.append("Key: 2-10 capital letters or digits, starting with a letter (e.g. PMI).")
    elif db.scalar(select(Project).where(Project.key == key)):
        errors.append(f"Key {key} is already used.")
    if reference_project not in REFERENCE_PROJECTS:
        errors.append("Choose the reference team whose estimates this project resembles.")
    if errors:
        return render(request, "project_new.html", status_code=400, errors=errors,
                      references=REFERENCE_PROJECTS,
                      values={"name": name, "key": key, "description": description,
                              "reference_project": reference_project})
    project = Project(name=name, key=key, description=description.strip(),
                      reference_project=reference_project, created_by_id=me.id)
    db.add(project)
    db.commit()
    flash(request, f"Project {key} created.")
    return RedirectResponse(f"/projects/{project.id}", status_code=303)


@router.get("/projects/{project_id}")
def project_page(request: Request, project_id: int, me: User = Depends(require_user),
                 db: Session = Depends(get_db)):
    project = get_or_404(db, Project, project_id)
    backlog = [i for i in project.items if i.sprint_id is None]
    closed = [s for s in project.sprints if s.status == "closed"]
    return render(request, "project.html", project=project, backlog=backlog,
                  open_sprints=[s for s in project.sprints if s.status != "closed"],
                  closed_sprints=closed, stats=suggestion_stats(project.items),
                  backlog_points=points(backlog), today=date.today().isoformat())


# -------------------------------------------------------------------- sprints
def parse_date(value):
    value = (value or "").strip()
    return date.fromisoformat(value) if value else None


@router.post("/projects/{project_id}/sprints")
def sprint_create(request: Request, project_id: int, name: str = Form(""), goal: str = Form(""),
                  start_date: str = Form(""), end_date: str = Form(""),
                  me: User = Depends(require_manager), db: Session = Depends(get_db)):
    project = get_or_404(db, Project, project_id)
    try:
        start, end = parse_date(start_date), parse_date(end_date)
    except ValueError:
        flash(request, "Dates must look like 2026-11-30.", "error")
        return RedirectResponse(f"/projects/{project.id}", status_code=303)
    if not name.strip():
        flash(request, "Enter a sprint name.", "error")
    elif start and end and end < start:
        flash(request, "The sprint cannot end before it starts.", "error")
    else:
        db.add(Sprint(project_id=project.id, name=name.strip(), goal=goal.strip(),
                      start_date=start, end_date=end))
        db.commit()
        flash(request, f"Sprint '{name.strip()}' planned.")
    return RedirectResponse(f"/projects/{project.id}", status_code=303)


@router.get("/sprints/{sprint_id}")
def sprint_board(request: Request, sprint_id: int, me: User = Depends(require_user),
                 db: Session = Depends(get_db)):
    sprint = get_or_404(db, Sprint, sprint_id)
    columns = {s: [i for i in sprint.items if i.status == s] for s in ("todo", "in_progress", "done")}
    return render(request, "sprint.html", sprint=sprint, project=sprint.project, columns=columns,
                  total_points=points(sprint.items), done_points=points(columns["done"]),
                  unestimated=sum(i.story_points is None for i in sprint.items))


@router.post("/sprints/{sprint_id}/start")
def sprint_start(request: Request, sprint_id: int, me: User = Depends(require_manager),
                 db: Session = Depends(get_db)):
    sprint = get_or_404(db, Sprint, sprint_id)
    active = [s for s in sprint.project.sprints if s.status == "active"]
    if sprint.status != "planned":
        flash(request, "Only a planned sprint can be started.", "error")
    elif active:
        flash(request, f"Close '{active[0].name}' first: one active sprint per project.", "error")
    else:
        sprint.status = "active"
        sprint.start_date = sprint.start_date or date.today()
        db.commit()
        flash(request, f"Sprint '{sprint.name}' started.")
    return RedirectResponse(f"/sprints/{sprint.id}", status_code=303)


@router.post("/sprints/{sprint_id}/close")
def sprint_close(request: Request, sprint_id: int, me: User = Depends(require_manager),
                 db: Session = Depends(get_db)):
    sprint = get_or_404(db, Sprint, sprint_id)
    if sprint.status != "active":
        flash(request, "Only an active sprint can be closed.", "error")
        return RedirectResponse(f"/sprints/{sprint.id}", status_code=303)
    done = [i for i in sprint.items if i.status == "done"]
    unfinished = [i for i in sprint.items if i.status != "done"]
    # snapshot the numbers first: unfinished items leave the sprint below
    sprint.committed_points = points(sprint.items)
    sprint.completed_points = points(done)
    for item in unfinished:
        item.sprint_id = None            # back to the backlog
    sprint.status = "closed"
    sprint.end_date = sprint.end_date or date.today()
    db.commit()
    flash(request, f"Sprint closed: {sprint.completed_points} of {sprint.committed_points} points done; "
                   f"{len(unfinished)} unfinished item(s) moved back to the backlog.")
    return RedirectResponse(f"/projects/{sprint.project_id}", status_code=303)

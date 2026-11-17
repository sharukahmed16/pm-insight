"""Work items: create (with the model's suggestion), view, edit, move on the board.

The quick-estimate page from the midterm demo is kept at /estimate.
"""
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import estimator
from app.db import get_db
from app.models import ITEM_STATUSES, ITEM_TYPES, Project, Sprint, StatusChange, User, WorkItem
from app.web import flash, get_or_404, render, require_user, to_int

router = APIRouter()


def try_suggest(title, description, reference_project):
    """(suggestion, None) or (None, reason). The app keeps working without the model."""
    try:
        return estimator.suggest(title, description, reference_project), None
    except estimator.ModelUnavailable as e:
        return None, str(e)


def form_choices(db, project):
    return {"users": db.scalars(select(User).order_by(User.full_name)).all(),
            "sprints": [s for s in project.sprints if s.status != "closed"],
            "scale": estimator.SCALE}


def read_item_form(db, project, title, description, item_type, assignee_id, sprint_id, story_points):
    """Validated values and a list of error messages."""
    values = {"title": title.strip(), "description": description.strip(), "item_type": item_type,
              "assignee_id": to_int(assignee_id), "sprint_id": to_int(sprint_id),
              "story_points": to_int(story_points)}
    errors = []
    if not values["title"]:
        errors.append("Enter a title.")
    if len(values["title"]) > 300:
        errors.append("The title is longer than 300 characters.")
    if item_type not in ITEM_TYPES:
        errors.append("Unknown item type.")
    if values["assignee_id"] is not None and db.get(User, values["assignee_id"]) is None:
        errors.append("Unknown assignee.")
    if values["sprint_id"] is not None:
        sprint = db.get(Sprint, values["sprint_id"])
        if sprint is None or sprint.project_id != project.id or sprint.status == "closed":
            errors.append("Choose an open sprint of this project.")
    if values["story_points"] is not None and values["story_points"] not in estimator.SCALE:
        errors.append("Story points must be one of " + ", ".join(map(str, estimator.SCALE)) + ".")
    return values, errors


# ----------------------------------------------------------------- create
@router.get("/projects/{project_id}/items/new")
def item_new_page(request: Request, project_id: int, sprint_id: str = "",
                  me: User = Depends(require_user), db: Session = Depends(get_db)):
    project = get_or_404(db, Project, project_id)
    values = {"item_type": "story", "sprint_id": to_int(sprint_id)}
    return render(request, "item_new.html", project=project, values=values, **form_choices(db, project))


@router.post("/projects/{project_id}/items/new")
def item_create(request: Request, project_id: int, action: str = Form("save"),
                title: str = Form(""), description: str = Form(""), item_type: str = Form("story"),
                assignee_id: str = Form(""), sprint_id: str = Form(""), story_points: str = Form(""),
                me: User = Depends(require_user), db: Session = Depends(get_db)):
    project = get_or_404(db, Project, project_id)
    values, errors = read_item_form(db, project, title, description, item_type,
                                    assignee_id, sprint_id, story_points)
    page = dict(project=project, values=values, **form_choices(db, project))

    if action == "suggest":
        if not values["title"]:
            return render(request, "item_new.html", status_code=400, errors=["Enter a title first."], **page)
        suggestion, problem = try_suggest(values["title"], values["description"], project.reference_project)
        if suggestion and values["story_points"] is None:
            values["story_points"] = suggestion["points"]      # pre-select; the team can change it
        return render(request, "item_new.html", suggestion=suggestion, problem=problem, **page)

    if errors:
        return render(request, "item_new.html", status_code=400, errors=errors, **page)

    # the suggestion is always recorded, even if nobody pressed "Suggest", so the
    # model can later be compared with what teams actually decided
    suggestion, _ = try_suggest(values["title"], values["description"], project.reference_project)
    item = WorkItem(project_id=project.id, created_by_id=me.id, **values,
                    suggested_points=suggestion["points"] if suggestion else None,
                    suggested_raw=suggestion["raw"] if suggestion else None)
    db.add(item)
    db.flush()                                    # gives item.id before the commit
    db.add(StatusChange(work_item_id=item.id, from_status=None, to_status="todo", changed_by_id=me.id))
    db.commit()
    flash(request, f"{item.key} created.")
    return RedirectResponse(f"/items/{item.id}", status_code=303)


# ------------------------------------------------------------ view / edit
@router.get("/items/{item_id}")
def item_page(request: Request, item_id: int, me: User = Depends(require_user),
              db: Session = Depends(get_db)):
    item = get_or_404(db, WorkItem, item_id)
    values = {k: getattr(item, k) for k in
              ("title", "description", "item_type", "assignee_id", "sprint_id", "story_points")}
    return render(request, "item.html", item=item, project=item.project, values=values,
                  **form_choices(db, item.project))


@router.post("/items/{item_id}")
def item_update(request: Request, item_id: int, title: str = Form(""), description: str = Form(""),
                item_type: str = Form("story"), assignee_id: str = Form(""), sprint_id: str = Form(""),
                story_points: str = Form(""), me: User = Depends(require_user),
                db: Session = Depends(get_db)):
    item = get_or_404(db, WorkItem, item_id)
    values, errors = read_item_form(db, item.project, title, description, item_type,
                                    assignee_id, sprint_id, story_points)
    if errors:
        return render(request, "item.html", status_code=400, errors=errors, item=item,
                      project=item.project, values=values, **form_choices(db, item.project))
    for field, value in values.items():
        setattr(item, field, value)
    db.commit()
    flash(request, f"{item.key} saved.")
    return RedirectResponse(f"/items/{item.id}", status_code=303)


@router.post("/items/{item_id}/status")
def item_status(request: Request, item_id: int, to_status: str = Form(...), back: str = Form("item"),
                me: User = Depends(require_user), db: Session = Depends(get_db)):
    item = get_or_404(db, WorkItem, item_id)
    if to_status not in ITEM_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status.")
    if to_status != item.status:
        db.add(StatusChange(work_item_id=item.id, from_status=item.status, to_status=to_status,
                            changed_by_id=me.id))
        item.status = to_status
        db.commit()
    # "back" is a fixed word, not a URL, so it cannot redirect anywhere outside the app
    if back == "board" and item.sprint_id:
        return RedirectResponse(f"/sprints/{item.sprint_id}", status_code=303)
    return RedirectResponse(f"/items/{item.id}", status_code=303)


# ------------------------------------------------------------ quick estimate
@router.get("/estimate")
def estimate_page(request: Request, me: User = Depends(require_user)):
    return render(request, "estimate.html", references=estimator.REFERENCE_PROJECTS, values={})


@router.post("/estimate")
def estimate(request: Request, reference_project: str = Form(""), title: str = Form(""),
             description: str = Form(""), me: User = Depends(require_user)):
    values = {"reference_project": reference_project, "title": title, "description": description}
    page = dict(references=estimator.REFERENCE_PROJECTS, values=values)
    if not title.strip() or reference_project not in estimator.REFERENCE_PROJECTS:
        return render(request, "estimate.html", status_code=400,
                      errors=["Choose a project and enter a title."], **page)
    suggestion, problem = try_suggest(title, description, reference_project)
    return render(request, "estimate.html", suggestion=suggestion, problem=problem, **page)

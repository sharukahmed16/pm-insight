"""Fill an empty database with a small demo: two users, two projects, a sprint.

Run from the repository root (after python -m app.init_db):  python -m app.seed_demo
Does nothing if any user already exists, so it never touches real data.

The demo accounts' password is random and written to logs/demo_accounts.txt
(logs/ is in .gitignore). Change or delete these accounts before real use.
"""
import os
import secrets
from datetime import date, timedelta

from app import models  # noqa: F401
from app.config import ROOT
from app.db import SessionLocal
from app.estimator import ModelUnavailable, suggest
from app.models import Project, Sprint, StatusChange, User, WorkItem
from app.security import hash_password

# (project key, title, description, team's points, status) - real issues from the
# held-out test split, the same ones as docs/demo_script.md
ISSUES = [
    ("STUDIO", "3.3.0 Studio Installer: Studio is installed in the wrong location",
     "The 3.3.0 Studio installer downloads the temporary files into the AppData folder. After the "
     "temp files have been obtained it moves the entire program into the Program Files folder. Once "
     "all the files have been moved it attempts to remove the temp files, but fails to remove the "
     "root folder.", 5, "done"),
    ("STUDIO", "Update content assist to support setting ActionBar properties via the Menu tag",
     "Alloy 1.4 adds support for setting Android ActionBar properties (title, subtitle, etc.) as "
     "attributes of the Menu tag. This change affects Android only.", 8, "in_progress"),
    ("STUDIO", "Add keyboard shortcut list to the Help menu", "", None, "todo"),
    ("DURA", "Retrieval Tool: Hidden password option",
     "DURACLOUD-759 added a hidden password for the Sync Tool. This task is to make use of the same "
     "environment variable to pick up the password for the Retrieval Tool.", 1, "todo"),
]


def main():
    db = SessionLocal()
    if db.query(User).count():
        print("Users already exist - demo data not added.")
        return
    password = os.environ.get("DEMO_PASSWORD") or secrets.token_urlsafe(9)
    manager = User(username="manager", full_name="Demo Manager", role="manager",
                   password_hash=hash_password(password))
    dev = User(username="developer", full_name="Demo Developer", role="member",
               password_hash=hash_password(password))
    db.add_all([manager, dev])
    db.flush()

    projects = {
        "STUDIO": Project(key="STUDIO", name="Studio Tools", reference_project="appceleratorstudio",
                          description="IDE features and installer.", created_by_id=manager.id),
        "DURA": Project(key="DURA", name="Cloud Sync", reference_project="duracloud",
                        description="Sync and retrieval tools.", created_by_id=manager.id),
    }
    db.add_all(projects.values())
    db.flush()
    sprint = Sprint(project_id=projects["STUDIO"].id, name="Sprint 1", goal="Installer and content assist",
                    status="active", start_date=date.today(), end_date=date.today() + timedelta(days=13))
    db.add(sprint)
    db.flush()

    for key, title, description, points, status in ISSUES:
        project = projects[key]
        try:
            s = suggest(title, description, project.reference_project)
        except ModelUnavailable:
            s = None
        item = WorkItem(project_id=project.id, title=title, description=description, story_points=points,
                        status=status, sprint_id=sprint.id if key == "STUDIO" and points else None,
                        assignee_id=dev.id, created_by_id=manager.id,
                        suggested_points=s["points"] if s else None, suggested_raw=s["raw"] if s else None)
        db.add(item)
        db.flush()
        db.add(StatusChange(work_item_id=item.id, from_status=None, to_status="todo", changed_by_id=manager.id))
        if status != "todo":
            db.add(StatusChange(work_item_id=item.id, from_status="todo", to_status=status, changed_by_id=dev.id))
    db.commit()

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "demo_accounts.txt"), "w", encoding="utf-8") as f:
        f.write(f"Demo accounts (local only - change before real use)\n"
                f"manager   / {password}\ndeveloper / {password}\n")
    print("Demo data added. Accounts are listed in logs/demo_accounts.txt")


if __name__ == "__main__":
    main()

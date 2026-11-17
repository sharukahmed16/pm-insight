"""Tests for the web application: login, roles, projects, sprints, work items.

They run against a throw-away in-memory SQLite database, not PostgreSQL, and
with a fake estimator, so they need neither the database server nor the model
files. One test (test_real_model_suggestion) uses the real model if it is there.

No test framework needed. Run from the repository root:

    python tests/test_app.py
"""
import os
import sys

# must be set before the app is imported: config.py reads them at import time
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app import estimator  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Sprint, User, WorkItem  # noqa: E402

REAL_SUGGEST = estimator.suggest
FAKE = {"points": 5, "raw": 4.6, "similar": [{"key": "X-1", "text": "a similar issue", "points": 5}]}


def fake_suggest(title, description, reference_project):
    return dict(FAKE)


def no_model(title, description, reference_project):
    raise estimator.ModelUnavailable("Model files not found: models/rf_project.joblib")


# ------------------------------------------------------------------ helpers
def fresh():
    """Empty database, fake estimator, new browser (cookie jar)."""
    estimator.suggest = fake_suggest
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return TestClient(app)


def register(c, username, password="password123"):
    return c.post("/register", data={"full_name": username.title(), "username": username,
                                     "password": password, "password2": password})


def login(c, username, password="password123"):
    return c.post("/login", data={"username": username, "password": password})


def manager_with_project(c, key="PMI"):
    register(c, "boss")
    r = c.post("/projects/new", data={"name": "Demo", "key": key, "description": "",
                                      "reference_project": "duracloud"})
    assert r.status_code == 200, r.text
    return int(r.url.path.rsplit("/", 1)[1])


def new_item(c, project_id, title="Add export button", **fields):
    data = {"action": "save", "title": title, "description": "", "item_type": "story",
            "assignee_id": "", "sprint_id": "", "story_points": ""}
    data.update({k: str(v) for k, v in fields.items()})
    r = c.post(f"/projects/{project_id}/items/new", data=data)
    assert r.status_code == 200, r.text
    return int(r.url.path.rsplit("/", 1)[1])


def db_get(model, object_id):
    with SessionLocal() as db:
        return db.get(model, object_id)


def history_of(item_id):
    with SessionLocal() as db:
        return [(h.from_status, h.to_status) for h in db.get(WorkItem, item_id).history]


# -------------------------------------------------------------------- tests
def test_pages_need_login():
    c = fresh()
    for path in ("/projects", "/estimate", "/users", "/items/1"):
        r = c.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login", path


def test_first_user_is_manager_then_members():
    c = fresh()
    register(c, "first")
    c.post("/logout")
    register(c, "second")
    with SessionLocal() as db:
        roles = {u.username: u.role for u in db.query(User)}
    assert roles == {"first": "manager", "second": "member"}, roles


def test_password_is_hashed_and_login_works():
    c = fresh()
    register(c, "alice", "correct horse")
    with SessionLocal() as db:
        stored = db.query(User).one().password_hash
    assert stored.startswith("pbkdf2_sha256$") and "correct horse" not in stored
    c.post("/logout")
    r = login(c, "alice", "wrong password")
    assert r.status_code == 400 and "Wrong username or password" in r.text
    r = login(c, "nobody", "correct horse")
    assert r.status_code == 400 and "Wrong username or password" in r.text
    r = login(c, "ALICE ", "correct horse")          # username is case- and space-insensitive
    assert r.status_code == 200 and r.url.path == "/projects"


def test_register_rejects_bad_input():
    c = fresh()
    register(c, "alice")
    c.post("/logout")
    for data, message in [
        ({"username": "alice"}, "taken"),
        ({"username": "x"}, "3-30 characters"),
        ({"username": "bob", "password": "short", "password2": "short"}, "at least 8"),
        ({"username": "bob", "password2": "different1"}, "do not match"),
    ]:
        form = {"full_name": "Bob", "username": "bob", "password": "password123", "password2": "password123"}
        form.update(data)
        r = c.post("/register", data=form)
        assert r.status_code == 400 and message in r.text, (data, r.status_code)


def test_member_cannot_do_manager_things():
    c = fresh()
    pid = manager_with_project(c)
    c.post("/logout")
    register(c, "dev")
    assert c.get("/projects/new").status_code == 403
    assert c.post("/projects/new", data={"name": "X", "key": "XX", "reference_project": "moodle"}).status_code == 403
    assert c.post(f"/projects/{pid}/sprints", data={"name": "S"}).status_code == 403
    assert c.get("/users").status_code == 403
    # but a member can add work items
    new_item(c, pid)


def test_project_key_rules():
    c = fresh()
    manager_with_project(c, "PMI")
    for key, message in [("PMI", "already used"), ("1AB", "2-10 capital"), ("A", "2-10 capital")]:
        r = c.post("/projects/new", data={"name": "X", "key": key, "reference_project": "moodle"})
        assert r.status_code == 400 and message in r.text, key
    r = c.post("/projects/new", data={"name": "X", "key": "NEW", "reference_project": "not-a-team"})
    assert r.status_code == 400 and "reference team" in r.text


def test_saving_records_suggestion_and_creation():
    c = fresh()
    pid = manager_with_project(c)
    iid = new_item(c, pid)                     # "Suggest" was never pressed
    item = db_get(WorkItem, iid)
    assert item.suggested_points == 5 and item.suggested_raw == 4.6
    assert item.story_points is None           # the team has not decided yet
    assert history_of(iid) == [(None, "todo")]


def test_suggest_preselects_points_but_saves_nothing():
    c = fresh()
    pid = manager_with_project(c)
    r = c.post(f"/projects/{pid}/items/new", data={"action": "suggest", "title": "Add export",
                                                   "item_type": "story"})
    assert r.status_code == 200
    assert "5 points" in r.text and "a similar issue" in r.text
    assert '<option value="5" selected>' in r.text
    with SessionLocal() as db:
        assert db.query(WorkItem).count() == 0


def test_app_works_without_the_model():
    c = fresh()
    pid = manager_with_project(c)
    estimator.suggest = no_model
    r = c.post(f"/projects/{pid}/items/new", data={"action": "suggest", "title": "Add export"})
    assert r.status_code == 200 and "No suggestion" in r.text
    iid = new_item(c, pid)
    assert db_get(WorkItem, iid).suggested_points is None


def test_status_changes_are_recorded():
    c = fresh()
    pid = manager_with_project(c)
    iid = new_item(c, pid)
    c.post(f"/items/{iid}/status", data={"to_status": "in_progress"})
    c.post(f"/items/{iid}/status", data={"to_status": "in_progress"})   # no change, no record
    c.post(f"/items/{iid}/status", data={"to_status": "done"})
    assert history_of(iid) == [(None, "todo"), ("todo", "in_progress"), ("in_progress", "done")]
    assert c.post(f"/items/{iid}/status", data={"to_status": "lost"}).status_code == 400


def test_item_edit_validation():
    c = fresh()
    pid = manager_with_project(c)
    iid = new_item(c, pid)
    base = {"title": "Renamed", "description": "", "item_type": "bug", "assignee_id": "",
            "sprint_id": "", "story_points": "8"}
    r = c.post(f"/items/{iid}", data=base)
    assert r.status_code == 200
    item = db_get(WorkItem, iid)
    assert (item.title, item.item_type, item.story_points) == ("Renamed", "bug", 8)
    for bad, message in [({"story_points": "7"}, "must be one of"), ({"title": " "}, "Enter a title"),
                         ({"assignee_id": "99"}, "Unknown assignee")]:
        r = c.post(f"/items/{iid}", data={**base, **bad})
        assert r.status_code == 400 and message in r.text, bad


def test_sprint_lifecycle():
    c = fresh()
    pid = manager_with_project(c)
    c.post(f"/projects/{pid}/sprints", data={"name": "Sprint 1", "start_date": "2026-11-17", "end_date": "2026-11-30"})
    c.post(f"/projects/{pid}/sprints", data={"name": "Sprint 2"})
    with SessionLocal() as db:
        s1, s2 = [s.id for s in db.query(Sprint).order_by(Sprint.id)]
    done = new_item(c, pid, "Finished work", sprint_id=s1, story_points=3)
    open_ = new_item(c, pid, "Unfinished work", sprint_id=s1, story_points=5)
    c.post(f"/items/{done}/status", data={"to_status": "done"})

    c.post(f"/sprints/{s1}/start")
    assert db_get(Sprint, s1).status == "active"
    r = c.post(f"/sprints/{s2}/start")
    assert "one active sprint per project" in r.text and db_get(Sprint, s2).status == "planned"

    c.post(f"/sprints/{s1}/close")
    s = db_get(Sprint, s1)
    assert (s.status, s.committed_points, s.completed_points) == ("closed", 8, 3)
    assert db_get(WorkItem, open_).sprint_id is None      # back to the backlog
    assert db_get(WorkItem, done).sprint_id == s1         # finished work stays in the sprint
    # a closed sprint cannot receive items or be started again
    r = c.post(f"/items/{open_}", data={"title": "x", "item_type": "story", "sprint_id": str(s1)})
    assert r.status_code == 400 and "open sprint" in r.text
    c.post(f"/sprints/{s1}/start")
    assert db_get(Sprint, s1).status == "closed"


def test_sprint_dates_are_checked():
    c = fresh()
    pid = manager_with_project(c)
    r = c.post(f"/projects/{pid}/sprints", data={"name": "S", "start_date": "2026-11-30", "end_date": "2026-11-01"})
    assert "cannot end before it starts" in r.text
    r = c.post(f"/projects/{pid}/sprints", data={"name": "S", "start_date": "30/11/2026"})
    assert "Dates must look like" in r.text
    with SessionLocal() as db:
        assert db.query(Sprint).count() == 0


def test_user_text_is_escaped():
    c = fresh()
    pid = manager_with_project(c)
    iid = new_item(c, pid, "<script>alert(1)</script>")
    for page in (f"/items/{iid}", f"/projects/{pid}"):
        html = c.get(page).text
        assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html, page


def test_manager_cannot_demote_self():
    c = fresh()
    register(c, "boss")
    with SessionLocal() as db:
        me = db.query(User).one().id
    r = c.post(f"/users/{me}/role", data={"role": "member"})
    assert "cannot change your own role" in r.text
    assert db_get(User, me).role == "manager"


def test_missing_things_give_404():
    c = fresh()
    register(c, "boss")
    for path in ("/projects/999", "/items/999", "/sprints/999"):
        r = c.get(path)
        assert r.status_code == 404 and "not found" in r.text, path


def test_real_model_suggestion():
    if not os.path.exists(estimator.FILES["model"]):
        print("      (skipped: models/rf_project.joblib not found)")
        return
    s = REAL_SUGGEST("Retrieval Tool: Hidden password option",
                     "Use the same environment variable to pick up the password for the Retrieval Tool.",
                     "duracloud")
    assert s["points"] in estimator.SCALE and s["raw"] >= 1
    assert len(s["similar"]) == 3 and all(x["key"].startswith("DURACLOUD") for x in s["similar"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:                     # also report crashes, not only failed asserts
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

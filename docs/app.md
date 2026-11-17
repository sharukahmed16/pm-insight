# The PM-Insight web application

A small project tracker with the week 9 estimation model built into the work-item form.

## Run it

1. `setup_app.bat` (once): installs SQLAlchemy, psycopg and itsdangerous, creates `.env`
   (you type your PostgreSQL password into it), creates the `pm_insight` database and its
   tables, adds demo data, runs the tests. Output: `logs\setup.log`, `logs\tests.log`.
2. `run_app.bat`: starts the server and opens http://127.0.0.1:8000.

Demo logins are in `logs\demo_accounts.txt`. The first account registered on an empty
database becomes the manager.

## Structure

| File | Role |
|---|---|
| `app/main.py` | creates the FastAPI app, the session cookie, the error pages |
| `app/config.py` | reads `.env` (database settings, session secret) |
| `app/db.py` | database engine, one session per request |
| `app/models.py` | the five tables (below) |
| `app/security.py` | password hashing (PBKDF2-SHA256, standard library) |
| `app/web.py` | shared helpers: templates, login checks, flash messages |
| `app/routes_auth.py` | login, registration, logout, user roles |
| `app/routes_projects.py` | projects, backlog, sprints (plan, start, close), velocity |
| `app/routes_items.py` | work items (create with suggestion, edit, status), quick estimate |
| `app/estimator.py` | loads the week 9 model once and returns a suggestion |
| `app/init_db.py`, `app/seed_demo.py` | create the database and tables; demo data |
| `app/templates/*.html` | the pages (Jinja2) |
| `tests/test_app.py` | 17 tests on an in-memory SQLite database |
| `setup_app.bat`, `run_app.bat` | one-time setup; start the server |

## Tables

```
users ──< projects (created_by)
projects ──< sprints ──< work_items >── users (assignee, created_by)
work_items ──< status_changes >── users (changed_by)
```

- **work_items** keeps two numbers: `suggested_points` (the model, recorded on every save)
  and `story_points` (the team's decision). The project page compares them, which shows
  how often teams accept the model's value.
- **status_changes** records every move between To do, In progress and Done, with who and
  when. This is the history a later delay-risk model would need.
- **sprints** store `committed_points` and `completed_points` when they are closed, because
  unfinished items go back to the backlog and would otherwise disappear from the numbers.

## How the model is used

- A project names one of the 16 dataset teams as its **reference team**. The model only
  knows those teams' point scales; the reference team decides which scale a new project's
  suggestions follow.
- "Suggest story points" shows the rounded suggestion, the raw model output and the three
  most similar past issues of the reference team, and pre-selects the value. The team can
  change it before saving.
- If the model files are missing, the app keeps working and says that no suggestion is
  available.

## Security decisions

| Risk | Handling |
|---|---|
| Stolen password table | PBKDF2-SHA256, 600,000 iterations, random salt per user |
| Guessing which usernames exist | the same error for an unknown user and a wrong password |
| Session tampering | signed cookie (SECRET_KEY in `.env`); new session at each login |
| Cross-site request forgery | session cookie is `SameSite=Lax`; all changes are POST |
| SQL injection | all queries go through the SQLAlchemy ORM (bound parameters) |
| Cross-site scripting | Jinja2 escapes all output; tested with a `<script>` title |
| Members doing manager actions | `require_manager` on project, sprint and role routes (403) |
| Secrets in git | `.env` and `logs/` are in `.gitignore` |

## Known limitations

- No migrations: `init_db` creates missing tables but does not change existing ones.
- Work-item keys use the database id, which is shared by all projects, so numbers skip
  inside a project (STUDIO-1, STUDIO-2, DURA-3, STUDIO-4). Jira numbers each project
  separately; that needs a per-project counter column.
- Registration is open; a real deployment would let only a manager add users.
- Every user can see and edit every project (no per-project membership).
- Runs on plain HTTP on the local machine; a deployment needs HTTPS (then `https_only=True`
  on the session cookie).
- The reference-team mapping is a workaround for the model having been trained on 16
  public projects; retraining on the company's own history would remove it.

## Checked by hand (17 Nov 2026, PostgreSQL 18, in the browser)

Log in as manager → project page (backlog, suggestion statistics) → new work item:
"Suggest" gave 8 points (raw 6.68) with three genuinely related Alloy content-assist
issues; saved with the team's 5 → board: moved to In progress → second project: planned,
started and closed a sprint (velocity 1 of 1) → quick estimate reproduces the demo-script
miss (Clover/SBT: 3 points, third similar issue 40) → users page → log out (protected
pages redirect to the login page).

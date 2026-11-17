"""Settings, read from the .env file in the repository root.

.env holds the database password and the session secret, so it is listed in
.gitignore and never committed. .env.example shows the expected keys.
An environment variable with the same name wins over the file (the tests use
this to point the app at a throw-away SQLite database).
"""
import os

from sqlalchemy.engine import URL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env")


def read_env_file(path=ENV_FILE):
    """KEY=value lines; blank lines and # comments are ignored."""
    values = {}
    if os.path.exists(path):
        # utf-8-sig: also accepts a file that Notepad saved with a byte-order mark
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip()
                    # PGPASSWORD="secret" and PGPASSWORD=secret mean the same
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                        value = value[1:-1]
                    values[key.strip()] = value
    return values


_file = read_env_file()


def setting(name, default=None):
    return os.environ.get(name, _file.get(name, default))


def database_url():
    """DATABASE_URL if given (tests), otherwise PostgreSQL from the PG* settings.

    URL.create escapes special characters in the password, which a hand-written
    "postgresql://user:pass@host/db" string would not.
    """
    if setting("DATABASE_URL"):
        return setting("DATABASE_URL")
    return URL.create(
        "postgresql+psycopg",
        username=setting("PGUSER", "postgres"),
        password=setting("PGPASSWORD"),
        host=setting("PGHOST", "localhost"),
        port=int(setting("PGPORT", "5432")),
        database=setting("PGDATABASE", "pm_insight"),
    )


SECRET_KEY = setting("SECRET_KEY")
if not SECRET_KEY:
    raise SystemExit("SECRET_KEY is missing from .env - run setup_app.bat, or copy .env.example to .env")

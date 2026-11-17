"""Create the PostgreSQL database (if missing) and all tables.

Run from the repository root:   python -m app.init_db
Safe to run again: existing tables and data are left alone. A column added
later to models.py is NOT added to an existing table (no migration tool);
for that, drop the database and run this again.
"""
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app import models  # noqa: F401  (registers the tables on Base.metadata)
from app.config import setting
from app.db import Base, engine


def create_database_if_missing():
    url = engine.url
    if url.get_backend_name() != "postgresql":
        return
    # connect to the built-in "postgres" database to create ours;
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database})
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
            print(f"Created database {url.database}")
    admin.dispose()


def explain(error):
    """Why the connection failed, without ever printing the password."""
    message = str(error.orig) if getattr(error, "orig", None) else str(error)
    print("\nCould not connect to PostgreSQL.")
    if "password authentication failed" in message:
        pw = setting("PGPASSWORD") or ""
        print(f"  The server is running, but it rejected the password for user '{engine.url.username}'.")
        if not pw:
            print("  PGPASSWORD in .env is empty.")
        else:
            print(f"  PGPASSWORD in .env has {len(pw)} characters.")
            if pw != pw.strip() or " " in pw:
                print("  It contains a space - check that this is really part of the password.")
        print("  Check it by logging in to pgAdmin or 'SQL Shell (psql)' as postgres with the same password.")
    elif "Connection refused" in message or "could not connect" in message.lower():
        print("  No PostgreSQL server answered on "
              f"{engine.url.host}:{engine.url.port}. Is the PostgreSQL service running?")
    else:
        print("  " + message.splitlines()[0])


if __name__ == "__main__":
    try:
        create_database_if_missing()
        Base.metadata.create_all(engine)
    except OperationalError as e:
        explain(e)
        sys.exit(1)
    print("Tables:", ", ".join(sorted(Base.metadata.tables)))

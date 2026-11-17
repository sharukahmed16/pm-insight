"""Database connection: one engine for the app, one session per request."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import database_url

url = database_url()
if str(url).startswith("sqlite"):
    # tests: an in-memory SQLite database must be shared by every connection
    engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
else:
    # pool_pre_ping: replace connections that PostgreSQL closed while idle
    engine = create_engine(url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: a session that is closed when the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

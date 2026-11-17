"""Database tables.

    users ──< projects (created_by)
    projects ──< sprints ──< work_items >── users (assignee)
    work_items ──< status_changes

Every status change of a work item is kept in status_changes. That history is
what a future delay-risk model would be trained on, and it is what the proposal
asked the tracker to record.
"""
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

ROLES = ("manager", "member")
SPRINT_STATUSES = ("planned", "active", "closed")
ITEM_TYPES = ("story", "bug", "task")
ITEM_STATUSES = ("todo", "in_progress", "done")
STATUS_LABELS = {"todo": "To do", "in_progress": "In progress", "done": "Done"}


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    full_name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    # the model only knows the story point scales of the 16 dataset teams, so a
    # project says which of them its estimates should follow
    reference_project: Mapped[str] = mapped_column(String(50))
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    sprints: Mapped[list["Sprint"]] = relationship(
        back_populates="project", order_by="Sprint.id", cascade="all, delete-orphan")
    items: Mapped[list["WorkItem"]] = relationship(
        back_populates="project", order_by="WorkItem.id", cascade="all, delete-orphan")


class Sprint(Base):
    __tablename__ = "sprints"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(100))
    goal: Mapped[str] = mapped_column(Text, default="")
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="planned")
    # filled when the sprint is closed, so velocity survives items moving back
    committed_points: Mapped[int | None] = mapped_column(Integer)
    completed_points: Mapped[int | None] = mapped_column(Integer)

    project: Mapped["Project"] = relationship(back_populates="sprints")
    items: Mapped[list["WorkItem"]] = relationship(back_populates="sprint", order_by="WorkItem.id")


class WorkItem(Base):
    __tablename__ = "work_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    sprint_id: Mapped[int | None] = mapped_column(ForeignKey("sprints.id"))  # None = backlog
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    item_type: Mapped[str] = mapped_column(String(20), default="story")
    status: Mapped[str] = mapped_column(String(20), default="todo")
    story_points: Mapped[int | None] = mapped_column(Integer)      # the team's decision
    suggested_points: Mapped[int | None] = mapped_column(Integer)  # what the model said
    suggested_raw: Mapped[float | None] = mapped_column(Float)     # before rounding
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    project: Mapped["Project"] = relationship(back_populates="items")
    sprint: Mapped[Optional["Sprint"]] = relationship(back_populates="items")
    assignee: Mapped[Optional["User"]] = relationship(foreign_keys=[assignee_id])
    created_by: Mapped["User"] = relationship(foreign_keys=[created_by_id])
    history: Mapped[list["StatusChange"]] = relationship(
        back_populates="item", order_by="StatusChange.id", cascade="all, delete-orphan")

    @property
    def key(self):
        return f"{self.project.key}-{self.id}"


class StatusChange(Base):
    __tablename__ = "status_changes"
    id: Mapped[int] = mapped_column(primary_key=True)
    work_item_id: Mapped[int] = mapped_column(ForeignKey("work_items.id"))
    from_status: Mapped[str | None] = mapped_column(String(20))   # None = created
    to_status: Mapped[str] = mapped_column(String(20))
    changed_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    item: Mapped["WorkItem"] = relationship(back_populates="history")
    changed_by: Mapped["User"] = relationship()

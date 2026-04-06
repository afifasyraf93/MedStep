import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base   = declarative_base()
ENGINE = None
Session = None


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    user_id       = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50), unique=True, nullable=False)
    email         = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), default="student")
    created_at    = Column(DateTime, default=datetime.utcnow)

    sessions  = relationship("UserSession", back_populates="user",
                              cascade="all, delete-orphan")
    histories = relationship("History", back_populates="user",
                              cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"


class UserSession(Base):
    __tablename__ = "sessions"

    session_id    = Column(Integer, primary_key=True, autoincrement=True)
    token         = Column(String(64), unique=True, nullable=False)
    login_time    = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    user_id       = Column(Integer, ForeignKey("users.user_id"),
                           nullable=False)

    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<Session {self.token[:8]}... user={self.user_id}>"


class History(Base):
    __tablename__ = "history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String(255), nullable=False)
    report     = Column(Text)
    findings   = Column(Text)
    impression = Column(Text)
    detections = Column(Text)   # JSON string of detection results
    user_id    = Column(Integer, ForeignKey("users.user_id"),
                        nullable=False)

    user = relationship("User", back_populates="histories")

    def __repr__(self):
        return f"<History {self.history_id} user={self.user_id}>"


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(db_path="data/medstep.db"):
    """Initialize database, create tables if not exist."""
    global ENGINE, Session

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    ENGINE  = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(bind=ENGINE)
    Base.metadata.create_all(ENGINE)

    print(f"Database initialized: {db_path}")
    return ENGINE, Session


def get_session():
    """Get a new database session."""
    if Session is None:
        init_db()
    return Session()
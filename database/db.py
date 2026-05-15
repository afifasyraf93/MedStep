from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///data/medstep.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__   = "users"

    user_id         = Column(Integer, primary_key=True, autoincrement=True)
    username        = Column(String(50), unique=True, nullable=False)
    email           = Column(String(100), unique=True, nullable=False)
    password_hash   = Column(String(255), nullable=False)
    role            = Column(String(20), nullable=False, default="student")
    created_at      = Column(DateTime, default=datetime.utcnow)
    analyses_count  = Column(Integer, default=0)
    streak_days     = Column(Integer, default=0)
    last_active     = Column(DateTime, nullable=True)

    sessions        = relationship("UserSession", back_populates="user")
    history         = relationship("History", back_populates="user")


class UserSession(Base):
    __tablename__   = "sessions"

    session_id      = Column(Integer, primary_key=True, autoincrement=True)
    token           = Column(String(64), unique=True, nullable=False)
    login_time      = Column(DateTime, default=datetime.utcnow)
    last_activity   = Column(DateTime, default=datetime.utcnow)

    user_id         = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    user            = relationship("User", back_populates="sessions")


class History(Base):
    __tablename__   = "history"

    history_id      = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    report          = Column(Text, nullable=True)

    user_id         = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    image_id        = Column(String, ForeignKey("cxr_cases.image_id"), nullable=True)

    case_name       = Column(String(100), nullable=True)
    patient_ref     = Column(String(100), nullable=True)
    notes           = Column(Text, nullable=True)
    detections      = Column(Text, nullable=True)
    heatmaps_dir    = Column(String(255), nullable=True)

    user            = relationship("User", back_populates="history")


class CXRCase(Base):
    __tablename__   = "cxr_cases"

    image_id        = Column(String(100), primary_key=True)
    patient_id      = Column(String(50), nullable=False)
    image_path      = Column(String(255), nullable=False)
    pneumonia       = Column(Integer)
    cardiomegaly    = Column(Integer)
    pleural_effusion= Column(Integer)
    pneumothorax    = Column(Integer)
    atelectasis     = Column(Integer)
    lung_mass       = Column(Integer)
    report_text     = Column(Text, nullable=True)
    embedding_index = Column(Integer, nullable=False, default=0)


def get_db():
    """Dependency — yields a DB session, closes after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # TODO: call Base.metadata.create_all(bind=engine)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
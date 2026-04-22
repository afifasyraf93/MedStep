import secrets
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database.db import User, UserSession

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SESSION_EXPIRE_HOURS = 24


def register(db: Session, username: str, email: str, password: str):
    """Hash password and insert new User row.
    Return the new User object, or raise ValueError if
    username/email already exists.
    """
    # TODO: check if username already taken (db.query(User).filter_by(...))
    if db.query(User).filter(User.username == username).first():
        raise ValueError("Username already taken")
    # TODO: check if email already taken
    if db.query(User).filter(User.email == email).first():
        raise ValueError("Email already registered")
    # TODO: hash password with pwd_context.hash(password)
    hashed = pwd_context.hash(password)
    # TODO: create User object and add/commit/refresh
    user = User(username=username, email=email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    # TODO: return user
    return user


def login(db: Session, username: str, password: str):
    """Verify credentials and create a session token.
    Return the token string, or raise ValueError on bad credentials.
    """
    # TODO: look up user by username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise ValueError("Invalid username or password")
    # TODO: verify password with pwd_context.verify(password, user.password_hash)
    if not pwd_context.verify(password, user.password_hash):
        raise ValueError("Invalid username or password")
    # TODO: generate token with secrets.token_urlsafe(32)
    token = secrets.token_urlsafe(32)
    # TODO: create UserSession object and add/commit
    session = UserSession(user_id=user.user_id, token=token)
    db.add(session)
    db.commit()
    db.refresh(session)
    # TODO: return token
    return token


def verify_session(db: Session, token: str):
    """Check token exists and is not expired.
    Update last_activity if valid.
    Return user_id, or raise ValueError if invalid/expired.
    """
    # TODO: query UserSession by token
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if not session:
        raise ValueError("Invalid Session Token")
    # TODO: check last_activity is within SESSION_EXPIRE_HOURS
    expiry = session.last_activity + timedelta(hours=SESSION_EXPIRE_HOURS)
    if datetime.utcnow() > expiry:
        db.delete(session)
        db.commit()
        raise ValueError("Invalid Session")
    # TODO: update last_activity to now and commit
    session.last_activity = datetime.utcnow()
    # TODO: return session.user_id
    return session.user_id


def logout(db: Session, token: str):
    """Delete the session row for this token."""
    # TODO: query UserSession by token and delete it
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if session:
        db.delete(session)
        db.commit()
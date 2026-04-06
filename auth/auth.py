import secrets
from datetime import datetime, timedelta
from passlib.context import CryptContext
from database.db import get_session, User, UserSession, init_db

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

# Session expiry
SESSION_EXPIRY_HOURS = 24


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


# ── Registration ──────────────────────────────────────────────────────────────

def register_user(username, email, password):
    """
    Register a new user.
    Returns (success, message)
    """
    db = get_session()
    try:
        # Check if username exists
        if db.query(User).filter_by(username=username).first():
            return False, "Username already taken"

        # Check if email exists
        if db.query(User).filter_by(email=email).first():
            return False, "Email already registered"

        # Validate inputs
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        if "@" not in email:
            return False, "Invalid email address"

        # Create user
        user = User(
            username      = username,
            email         = email,
            password_hash = hash_password(password),
            role          = "student"
        )
        db.add(user)
        db.commit()

        return True, "Registration successful"

    except Exception as e:
        db.rollback()
        return False, f"Registration failed: {str(e)}"
    finally:
        db.close()


# ── Login ─────────────────────────────────────────────────────────────────────

def login_user(username, password):
    """
    Authenticate user and create session.
    Returns (success, token_or_error_message)
    """
    db = get_session()
    try:
        user = db.query(User).filter_by(username=username).first()

        if not user:
            return False, "Username not found"

        if not verify_password(password, user.password_hash):
            return False, "Incorrect password"

        # Generate session token
        token = secrets.token_urlsafe(32)

        session = UserSession(
            token         = token,
            user_id       = user.user_id,
            login_time    = datetime.utcnow(),
            last_activity = datetime.utcnow()
        )
        db.add(session)
        db.commit()

        return True, token

    except Exception as e:
        db.rollback()
        return False, f"Login failed: {str(e)}"
    finally:
        db.close()


# ── Verification ──────────────────────────────────────────────────────────────

def verify_session(token):
    """
    Verify session token is valid and not expired.
    Updates last_activity if valid.
    Returns (valid, user_id_or_None)
    """
    if not token:
        return False, None

    db = get_session()
    try:
        session = db.query(UserSession).filter_by(token=token).first()

        if not session:
            return False, None

        # Check expiry
        expiry = session.last_activity + timedelta(
            hours=SESSION_EXPIRY_HOURS
        )
        if datetime.utcnow() > expiry:
            db.delete(session)
            db.commit()
            return False, None

        # Update last activity
        session.last_activity = datetime.utcnow()
        db.commit()

        return True, session.user_id

    except Exception as e:
        return False, None
    finally:
        db.close()


# ── Logout ────────────────────────────────────────────────────────────────────

def logout_user(token):
    """Delete session token."""
    db = get_session()
    try:
        session = db.query(UserSession).filter_by(token=token).first()
        if session:
            db.delete(session)
            db.commit()
        return True
    except Exception as e:
        db.rollback()
        return False
    finally:
        db.close()


# ── Get User ──────────────────────────────────────────────────────────────────

def get_user_by_id(user_id):
    """Return user dict by ID."""
    db = get_session()
    try:
        user = db.query(User).filter_by(user_id=user_id).first()
        if not user:
            return None
        return {
            "user_id":    user.user_id,
            "username":   user.username,
            "email":      user.email,
            "role":       user.role,
            "created_at": str(user.created_at)
        }
    finally:
        db.close()
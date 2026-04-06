from database.db import init_db
from auth.auth import register_user, login_user, verify_session, logout_user
from database.history import save_history, get_user_history

# Initialize database
init_db()
print("Database initialized\n")

# Clean slate — delete test user if exists from previous run
from database.db import get_session, User
db = get_session()
existing = db.query(User).filter_by(username="student1").first()
if existing:
    db.delete(existing)
    db.commit()
    print("Cleaned up existing test user")
db.close()

# Test registration
print("── Registration ──────────────────")
ok, msg = register_user("student1", "student1@test.com", "password123")
print(f"Register student1:   {ok} — {msg}")

ok, msg = register_user("student1", "other@test.com", "password123")
print(f"Register duplicate:  {ok} — {msg}")

ok, msg = register_user("ab", "short@test.com", "pass")
print(f"Register too short:  {ok} — {msg}")

# Test login
print("\n── Login ─────────────────────────")
ok, token = login_user("student1", "password123")
print(f"Login success: {ok} — token={token[:16] if ok else token}...")

ok2, msg2 = login_user("student1", "wrongpass")
print(f"Login wrong pass: {ok2} — {msg2}")

if not ok:
    print("\nERROR: Login failed, cannot continue tests")
    exit(1)

# Test session
print("\n── Session Verification ──────────")
valid, user_id = verify_session(token)
print(f"Valid session: {valid} — user_id={user_id}")

valid2, uid2 = verify_session("fake_token_123")
print(f"Invalid token: {valid2} — {uid2}")

# Test history
print("\n── History ───────────────────────")
history_id = save_history(
    user_id           = user_id,
    image_path        = "test/image.jpg",
    detection_results = {"pneumonia": {"probability": 0.7, "detected": True}},
    findings          = "Test findings",
    impression        = "Test impression",
    full_report       = "FINDINGS:\nTest\n\nIMPRESSION:\nTest"
)
print(f"Saved history: id={history_id}")

history = get_user_history(user_id)
print(f"Retrieved {len(history)} history entries")
if history:
    print(f"Latest: {history[0]['timestamp']} — "
          f"detected: {history[0]['detected']}")

# Test logout
print("\n── Logout ────────────────────────")
logout_user(token)
valid3, uid3 = verify_session(token)
print(f"After logout session valid: {valid3}")

print("\nAll auth tests complete ✓")
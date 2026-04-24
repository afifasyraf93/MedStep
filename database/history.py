from sqlalchemy.orm import Session
from database.db import History


def save_history(db: Session, user_id: int, report: str, image_id: str = None):
    """Create and save a new History row.
    Return the new History object.
    """
    # TODO: create History object with user_id, report, image_id
    history = History(user_id=user_id, report=report, image_id=image_id)
    # TODO: db.add, db.commit, db.refresh
    db.add(history)
    db.commit()
    db.refresh(history)
    # TODO: return history
    return history


def get_history(db: Session, user_id: int):
    """Return all History rows for this user, newest first."""
    # TODO: query History filtered by user_id
    history = db.query(History).filter(History.user_id == user_id)
    # TODO: order by timestamp descending — .order_by(History.timestamp.desc())
    history = history.order_by(History.timestamp.desc())
    # TODO: return .all()
    return history.all()


def delete_history(db: Session, history_id: int, user_id: int):
    """Delete a specific history row.
    Only deletes if it belongs to the requesting user (security check).
    Return True if deleted, False if not found.
    """
    # TODO: query History filtered by history_id AND user_id
    history = db.query(History).filter(History.history_id == history_id, History.user_id== user_id,).first()
    # TODO: if found, delete and commit, return True
    if history:
        db.delete(history)
        db.commit()
        return True
    # TODO: if not found, return False
    else:
        return False
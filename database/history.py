import json
from datetime import datetime
from database.db import get_session, History


def save_history(user_id, image_path, detection_results,
                  findings, impression, full_report):
    """
    Save analysis result to history.
    Returns history_id or None on failure.
    """
    db = get_session()
    try:
        entry = History(
            user_id    = user_id,
            image_path = image_path,
            report     = full_report,
            findings   = findings,
            impression = impression,
            detections = json.dumps(detection_results),
            timestamp  = datetime.utcnow()
        )
        db.add(entry)
        db.commit()
        return entry.history_id

    except Exception as e:
        db.rollback()
        print(f"Failed to save history: {e}")
        return None
    finally:
        db.close()


def get_user_history(user_id, limit=20):
    """
    Get analysis history for a user.
    Returns list of dicts, most recent first.
    """
    db = get_session()
    try:
        entries = (
            db.query(History)
            .filter_by(user_id=user_id)
            .order_by(History.timestamp.desc())
            .limit(limit)
            .all()
        )

        results = []
        for e in entries:
            try:
                detections = json.loads(e.detections) \
                    if e.detections else {}
            except Exception:
                detections = {}

            # Get detected pathologies for display
            detected = [
                k for k, v in detections.items()
                if isinstance(v, dict) and v.get("detected")
            ]

            results.append({
                "history_id": e.history_id,
                "timestamp":  e.timestamp.strftime("%Y-%m-%d %H:%M"),
                "image_path": e.image_path,
                "findings":   e.findings,
                "impression": e.impression,
                "report":     e.report,
                "detections": detections,
                "detected":   detected
            })

        return results

    finally:
        db.close()


def get_history_by_id(history_id, user_id):
    """
    Get single history entry.
    user_id ensures users can only access their own history.
    """
    db = get_session()
    try:
        entry = db.query(History).filter_by(
            history_id=history_id,
            user_id=user_id
        ).first()

        if not entry:
            return None

        return {
            "history_id": entry.history_id,
            "timestamp":  entry.timestamp.strftime("%Y-%m-%d %H:%M"),
            "image_path": entry.image_path,
            "findings":   entry.findings,
            "impression": entry.impression,
            "report":     entry.report,
            "detections": json.loads(entry.detections)
                          if entry.detections else {}
        }
    finally:
        db.close()
import csv
import json
from datetime import datetime
from pathlib import Path


class AlertManager:
    """Debounces alerts, saves evidence frames, and writes an audit log."""

    def __init__(self, alert_dir: Path, confirm_frames=6, cooldown_seconds=8):
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_frames = confirm_frames
        self.cooldown_seconds = cooldown_seconds
        self.pending = {}
        self.last_alert_time = {}
        self.sequence = 0

        self.csv_path = self.alert_dir.parent / "alerts.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "event_id", "severity", "incident_type",
                    "camera", "zone", "details", "evidence_file"
                ])

    def update(self, event_key, event: dict, frame):
        """Return a created alert dict once the event persists long enough."""
        now = datetime.now()
        state = self.pending.get(event_key, 0) + 1
        self.pending[event_key] = state

        last = self.last_alert_time.get(event_key)
        if state < self.confirm_frames:
            return None

        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None

        self.last_alert_time[event_key] = now
        self.pending[event_key] = 0
        self.sequence += 1

        event_id = f"NS-{now.strftime('%Y%m%d-%H%M%S')}-{self.sequence:03d}"
        evidence_name = f"{event_id}.jpg"
        evidence_path = self.alert_dir / evidence_name

        return self._save_alert(event_id, event, evidence_path, frame, now)

    def decay_missing(self, active_keys):
        """Slowly forget events that are no longer present."""
        for key in list(self.pending):
            if key not in active_keys:
                self.pending[key] = max(0, self.pending[key] - 2)
                if self.pending[key] == 0:
                    del self.pending[key]

    def _save_alert(self, event_id, event, evidence_path, frame, now):
        import cv2

        cv2.imwrite(str(evidence_path), frame)

        payload = {
            "timestamp": now.isoformat(timespec="seconds"),
            "event_id": event_id,
            **event,
            "evidence_file": str(evidence_path),
        }

        with open(evidence_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                payload["timestamp"],
                event_id,
                payload.get("severity", "WARNING"),
                payload.get("incident_type", ""),
                payload.get("camera", ""),
                payload.get("zone", ""),
                payload.get("details", ""),
                str(evidence_path),
            ])

        return payload

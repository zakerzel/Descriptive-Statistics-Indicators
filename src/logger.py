"""CSV logging utilities for the CW08 gesture system."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd


class GestureLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.columns = [
            "timestamp",
            "identity",
            "hand",
            "finger_count",
            "gesture",
            "confidence",
            "hands_detected",
            "total_fingers",
            "active_sequence_segments",
        ]
        if not self.csv_path.exists():
            self.reset()

    def reset(self):
        pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)

    def log_detection(self, identity, hand, finger_count, gesture, confidence, hands_detected, total_fingers, active_sequence_segments):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "identity": identity,
            "hand": hand,
            "finger_count": finger_count,
            "gesture": gesture,
            "confidence": round(float(confidence), 4),
            "hands_detected": hands_detected,
            "total_fingers": total_fingers,
            "active_sequence_segments": active_sequence_segments,
        }
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=False, index=False)


class SequenceAttemptLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.columns = ["timestamp", "identity", "score", "segment_count", "decision", "sequence"]
        if not self.csv_path.exists():
            self.reset()

    def reset(self):
        pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)

    def log_attempt(self, identity, score, segment_count, decision, sequence):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "identity": identity,
            "score": round(float(score), 4),
            "segment_count": int(segment_count),
            "decision": decision,
            "sequence": sequence,
        }
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=False, index=False)

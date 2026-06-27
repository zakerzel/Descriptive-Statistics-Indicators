"""Automatic gesture sequence identity recognition.

V8 rule: the hand used is part of the identity pattern. Any hand mismatch in a compared step makes it a different pattern.
"""

from pathlib import Path
import json
import time
import cv2


class AutoSequenceIdentityManager:
    def __init__(self, profiles_path: Path, max_profiles=8, min_segments=3, max_segments=12, stable_seconds=0.62, min_segment_duration=0.50, no_hand_finalize_seconds=1.20, recognition_threshold=0.66, new_profile_threshold=0.48):
        self.profiles_path = Path(profiles_path)
        self.max_profiles = max_profiles
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.stable_seconds = stable_seconds
        self.min_segment_duration = min_segment_duration
        self.no_hand_finalize_seconds = no_hand_finalize_seconds
        self.recognition_threshold = recognition_threshold
        self.new_profile_threshold = new_profile_threshold
        self.profiles = self._load_profiles()
        self.current_segments = []
        self._candidate_key = None
        self._candidate_started_at = None
        self._active_key = None
        self._active_started_at = None
        self._last_segment_end = None
        self._last_seen_hand_time = None
        self._pending_result = None

    def _load_profiles(self):
        if not self.profiles_path.exists():
            return {}
        try:
            return json.loads(self.profiles_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_profiles(self):
        self.profiles_path.parent.mkdir(exist_ok=True)
        self.profiles_path.write_text(json.dumps(self.profiles, indent=2), encoding="utf-8")

    def reset_all(self):
        self.clear_profiles()
        self.current_segments = []

    def clear_profiles(self):
        self.profiles = {}
        if self.profiles_path.exists():
            self.profiles_path.unlink()
        self._reset_current_sequence()

    def _reset_current_sequence(self):
        self.current_segments = []
        self._candidate_key = None
        self._candidate_started_at = None
        self._active_key = None
        self._active_started_at = None
        self._last_segment_end = None
        self._last_seen_hand_time = None

    def update(self, hand, finger_count, gesture, timestamp):
        self._last_seen_hand_time = timestamp
        key = f"{hand}:{finger_count}:{gesture}"
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate_started_at = timestamp
            return
        if self._candidate_started_at is None:
            self._candidate_started_at = timestamp
            return
        if timestamp - self._candidate_started_at < self.stable_seconds:
            return
        if self._active_key is None:
            self._active_key = key
            self._active_started_at = timestamp
            return
        if key != self._active_key:
            self._close_active_segment(timestamp)
            self._active_key = key
            self._active_started_at = timestamp
            if len(self.current_segments) >= self.max_segments:
                self._pending_result = self.finalize_sequence()

    def update_no_hand(self, timestamp):
        if self._last_seen_hand_time is None:
            return None
        if self.current_segments or self._active_key:
            if timestamp - self._last_seen_hand_time >= self.no_hand_finalize_seconds:
                return self.finalize_sequence()
        return None

    def consume_pending_result(self):
        result = self._pending_result
        self._pending_result = None
        return result

    def _close_active_segment(self, timestamp):
        if self._active_key is None or self._active_started_at is None:
            return
        duration = timestamp - self._active_started_at
        if duration < self.min_segment_duration:
            return
        hand, finger_count, gesture = self._active_key.split(":", 2)
        gap = 0.0 if self._last_segment_end is None else max(0.0, self._active_started_at - self._last_segment_end)
        segment = {"hand": hand, "finger_count": int(finger_count), "gesture": gesture, "duration": round(duration, 3), "gap": round(gap, 3)}
        if self.current_segments:
            prev = self.current_segments[-1]
            if prev["hand"] == segment["hand"] and prev["finger_count"] == segment["finger_count"] and prev["gesture"] == segment["gesture"]:
                prev["duration"] = round(prev["duration"] + duration, 3)
            else:
                self.current_segments.append(segment)
        else:
            self.current_segments.append(segment)
        self._last_segment_end = timestamp

    def finalize_sequence(self):
        self._close_active_segment(time.time())
        sequence = self.current_segments[: self.max_segments]
        if len(sequence) < self.min_segments:
            self._reset_current_sequence()
            return {"identity": "ignored", "score": 0.0, "segment_count": len(sequence), "decision": "too_short", "sequence": sequence, "sequence_string": self.sequence_to_string(sequence)}
        identity, score = self.match_sequence(sequence)
        if identity != "unknown":
            decision = "matched"
        elif len(self.profiles) < self.max_profiles and score < self.new_profile_threshold:
            identity = f"Person {len(self.profiles) + 1}"
            self.profiles[identity] = sequence
            self._save_profiles()
            decision = "new_profile_saved"
            score = 1.0
        else:
            decision = "unknown"
        result = {"identity": identity, "score": round(float(score), 4), "segment_count": len(sequence), "decision": decision, "sequence": sequence, "sequence_string": self.sequence_to_string(sequence)}
        self._reset_current_sequence()
        return result

    def match_sequence(self, sequence):
        if not self.profiles:
            return "unknown", 0.0
        best_identity = "unknown"
        best_score = 0.0
        for identity, reference in self.profiles.items():
            score = self._sequence_similarity(reference, sequence)
            if score > best_score:
                best_score = score
                best_identity = identity
        if best_score >= self.recognition_threshold:
            return best_identity, best_score
        return "unknown", best_score

    def _sequence_similarity(self, reference, unknown):
        max_len = max(len(reference), len(unknown), 1)
        n = min(len(reference), len(unknown))
        if n == 0:
            return 0.0
        for i in range(n):
            if reference[i]["hand"] != unknown[i]["hand"]:
                return 0.0
        gesture_scores = []
        count_scores = []
        timing_scores = []
        for i in range(n):
            ref = reference[i]
            unk = unknown[i]
            gesture_scores.append(1.0 if ref["gesture"] == unk["gesture"] else 0.0)
            count_scores.append(max(0.0, 1.0 - abs(int(ref["finger_count"]) - int(unk["finger_count"])) / 5.0))
            dsim = self._time_similarity(ref.get("duration", 0), unk.get("duration", 0))
            gsim = self._time_similarity(ref.get("gap", 0), unk.get("gap", 0))
            timing_scores.append(0.75 * dsim + 0.25 * gsim)
        length_penalty = n / max_len
        return float((0.55 * (sum(gesture_scores) / n) + 0.25 * (sum(count_scores) / n) + 0.20 * (sum(timing_scores) / n)) * length_penalty)

    def _time_similarity(self, ref_time, unk_time):
        ref_time = float(ref_time)
        unk_time = float(unk_time)
        if ref_time <= 0 and unk_time <= 0:
            return 1.0
        denom = max(ref_time, unk_time, 0.001)
        ratio = abs(ref_time - unk_time) / denom
        tolerance = 0.65
        return max(0.0, 1.0 - ratio / tolerance)

    def sequence_to_string(self, sequence):
        if not sequence:
            return ""
        return " -> ".join(f"{s['hand']}:{s['finger_count']}:{s['gesture']}" for s in sequence)

    def draw_sequence_panel(self, frame):
        if not self.current_segments:
            return
        x, y = 18, 155
        cv2.rectangle(frame, (10, 145), (960, 330), (10, 10, 10), -1)
        cv2.putText(frame, "Live gesture sequence", (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        y += 65
        for idx, seg in enumerate(self.current_segments[-6:], start=1):
            text = f"{idx}. {seg['hand']} | {seg['finger_count']} fingers | {seg['gesture']} | {seg['duration']}s"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1)
            y += 27

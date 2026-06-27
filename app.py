"""V8 application entry point."""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.auto_sequence_identity import AutoSequenceIdentityManager
from src.dashboard import draw_dashboard_panel
from src.hand_tracker import HandTracker
from src.indicators import compute_indicators
from src.logger import GestureLogger, SequenceAttemptLogger

DATA_DIR = Path("data")
SCREENSHOT_DIR = Path("screenshots")
LOG_PATH = DATA_DIR / "gesture_log.csv"
PROFILES_PATH = DATA_DIR / "sequence_profiles.json"
ATTEMPTS_PATH = DATA_DIR / "sequence_attempts.csv"
DATA_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)
WINDOW_NAME = "Gesture Recognition + Indicators V8"


def create_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def draw_header(frame, status):
    _, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 135), (18, 18, 18), -1)
    cv2.putText(frame, "Gesture Recognition + Statistical Indicators", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    cv2.putText(frame, f"Hands: {status['hands']}   Total fingers: {status['fingers']}   Live ID: {status['live_id']}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (0, 255, 255), 2)
    cv2.putText(frame, f"Sequence segments: {status['segments']}   Decision: {status['decision']}   Score: {status['score']:.2f}", (18, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)


def summarize_hands(hand_results):
    hands = len(hand_results)
    total_fingers = sum(hand["finger_count"] for hand in hand_results) if hand_results else 0
    if not hand_results:
        return hands, total_fingers, None
    best = max(hand_results, key=lambda item: item["confidence"])
    return hands, total_fingers, best


def main():
    cap = create_camera()
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = HandTracker(max_num_hands=2, smoothing_window=7)
    identity = AutoSequenceIdentityManager(PROFILES_PATH)
    gesture_logger = GestureLogger(LOG_PATH)
    attempt_logger = SequenceAttemptLogger(ATTEMPTS_PATH)
    last_log_time = 0.0
    live_id = "detecting"
    last_decision = "-"
    last_score = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1720, 900)
    print("V8 running. q=quit | s=screenshot | r=reset | p=clear profiles")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1120, 720))
        now = time.time()

        hand_results = tracker.process(frame)
        hands_count, total_fingers, primary = summarize_hands(hand_results)
        for hand in hand_results:
            tracker.draw(frame, hand)

        if primary:
            identity.update(primary["hand_label"], primary["finger_count"], primary["gesture"], now)
        else:
            result = identity.update_no_hand(now)
            if result:
                live_id = result["identity"]
                last_decision = result["decision"]
                last_score = result["score"]
                attempt_logger.log_attempt(result["identity"], result["score"], result["segment_count"], result["decision"], result["sequence_string"])

        result = identity.consume_pending_result()
        if result:
            live_id = result["identity"]
            last_decision = result["decision"]
            last_score = result["score"]
            attempt_logger.log_attempt(result["identity"], result["score"], result["segment_count"], result["decision"], result["sequence_string"])

        if now - last_log_time >= 0.55:
            for hand in hand_results:
                gesture_logger.log_detection(live_id, hand["hand_label"], hand["finger_count"], hand["gesture"], hand["confidence"], hands_count, total_fingers, len(identity.current_segments))
            last_log_time = now

        df = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
        attempts_df = pd.read_csv(ATTEMPTS_PATH) if ATTEMPTS_PATH.exists() else pd.DataFrame()
        indicators = compute_indicators(df, attempts_df)
        indicators["current_hands"] = hands_count
        indicators["current_total_fingers"] = total_fingers
        indicators["stored_profiles"] = len(identity.profiles)

        draw_header(frame, {"hands": hands_count, "fingers": total_fingers, "live_id": live_id, "segments": len(identity.current_segments), "decision": last_decision, "score": last_score})
        identity.draw_sequence_panel(frame)
        combined = np.hstack([frame, draw_dashboard_panel(indicators, height=frame.shape[0], width=600)])
        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            shot = SCREENSHOT_DIR / f"v8_screenshot_{int(time.time())}.png"
            cv2.imwrite(str(shot), combined)
            print(f"Screenshot saved: {shot}")
        if key == ord("r"):
            gesture_logger.reset()
            attempt_logger.reset()
            identity.reset_all()
            live_id = "detecting"
            last_decision = "reset"
            last_score = 0.0
        if key == ord("p"):
            identity.clear_profiles()
            live_id = "detecting"
            last_decision = "profiles cleared"
            last_score = 0.0

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

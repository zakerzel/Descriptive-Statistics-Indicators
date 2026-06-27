import cv2
import numpy as np


def draw_dashboard_panel(indicators, height=720, width=600):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    y = 36
    cv2.putText(panel, "REAL-TIME INDICATORS", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += 42
    lines = [
        f"Current hands: {indicators.get('current_hands', 0)}",
        f"Current total fingers: {indicators.get('current_total_fingers', 0)}",
        f"Stored profiles: {indicators.get('stored_profiles', 0)}",
        f"Total detections: {indicators['total_detections']}",
        "",
        "Central tendency",
        f"Most common gesture: {indicators['most_common_gesture']}",
        f"Most common fingers: {indicators['most_common_finger_count']}",
        "",
        "Rates / proportions",
        f"Open palm rate: {indicators['open_palm_rate']}%",
        f"Fist rate: {indicators['fist_rate']}%",
        f"Thumbs-up rate: {indicators['thumbs_up_rate']}%",
        f"Left hand rate: {indicators['left_hand_rate']}%",
        f"Right hand rate: {indicators['right_hand_rate']}%",
        "",
        "Dynamic sequence identity",
        f"Sequence attempts: {indicators['sequence_attempts']}",
        f"Profiles saved: {indicators['profiles_saved']}",
        f"Matched attempts: {indicators['matched_attempts']}",
        f"Ignored short: {indicators['ignored_short']}",
        f"Match rate: {indicators['match_rate']}%",
        f"Last ID: {indicators['last_identity']}",
        f"Last decision: {indicators['last_decision']}",
        f"Last score: {round(indicators['last_score'], 3)}",
    ]
    for line in lines:
        if line == "":
            y += 18
            continue
        color = (80, 220, 255) if line in ["Central tendency", "Rates / proportions", "Dynamic sequence identity"] else (255, 255, 255)
        thickness = 2 if color == (80, 220, 255) else 1
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, thickness, cv2.LINE_AA)
        y += 27
    cv2.putText(panel, "Automatic mode | q: quit | s: screenshot", (24, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1)
    cv2.putText(panel, "r: reset logs/profiles | p: clear profiles", (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1)
    return panel

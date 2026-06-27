"""Stable gesture classification using MediaPipe hand landmarks."""

import math
import numpy as np

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FINGER_PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
FINGER_DIP = {"index": 7, "middle": 11, "ring": 15, "pinky": 19}


def p(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def dist(a, b):
    return float(np.linalg.norm(p(a) - p(b)))


def angle(a, b, c):
    ba = p(a) - p(b)
    bc = p(c) - p(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    value = float(np.dot(ba, bc) / denom)
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def palm_scale(landmarks):
    return max(dist(landmarks[0], landmarks[9]), 1e-6)


def finger_extended(landmarks, finger):
    wrist = landmarks[0]
    mcp = landmarks[FINGER_MCP[finger]]
    pip = landmarks[FINGER_PIP[finger]]
    dip = landmarks[FINGER_DIP[finger]]
    tip = landmarks[FINGER_TIPS[finger]]
    scale = palm_scale(landmarks)
    tip_distance = dist(wrist, tip) / scale
    pip_distance = dist(wrist, pip) / scale
    mcp_distance = dist(wrist, mcp) / scale
    pip_angle = angle(mcp, pip, dip)
    dip_angle = angle(pip, dip, tip)
    straight = pip_angle > 135 and dip_angle > 125
    far = tip_distance > pip_distance + 0.16 and tip_distance > mcp_distance + 0.50
    vertical = tip.y < pip.y and tip.y < mcp.y and pip_angle > 115
    return bool((straight and far) or vertical)


def thumb_extended(landmarks, hand_label):
    wrist = landmarks[0]
    cmc = landmarks[1]
    mcp = landmarks[2]
    ip = landmarks[3]
    tip = landmarks[4]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    scale = palm_scale(landmarks)
    tip_distance = dist(wrist, tip) / scale
    ip_distance = dist(wrist, ip) / scale
    mcp_angle = angle(cmc, mcp, ip)
    ip_angle = angle(mcp, ip, tip)
    palm_width = max(dist(index_mcp, pinky_mcp), 1e-6)
    thumb_to_index = dist(tip, index_mcp) / palm_width
    straight = mcp_angle > 132 and ip_angle > 130
    far = tip_distance > ip_distance + 0.14
    separated = thumb_to_index > 0.60
    directional = tip.x < ip.x if hand_label.lower() == "right" else tip.x > ip.x
    return bool(straight and far and (separated or directional))


def get_finger_states(landmarks, hand_label):
    return {
        "thumb": thumb_extended(landmarks, hand_label),
        "index": finger_extended(landmarks, "index"),
        "middle": finger_extended(landmarks, "middle"),
        "ring": finger_extended(landmarks, "ring"),
        "pinky": finger_extended(landmarks, "pinky"),
    }


def classify_from_states(states):
    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]
    non_thumb = int(index) + int(middle) + int(ring) + int(pinky)
    total = non_thumb + int(thumb)
    if non_thumb == 0 and thumb:
        return "thumbs_up", 1
    if total == 0:
        return "fist", 0
    if index and middle and not ring and not pinky:
        return "peace", 2
    if index and middle and ring and not pinky:
        return "three_fingers", 3
    if index and middle and ring and pinky and thumb:
        return "open_palm", 5
    if index and middle and ring and pinky and not thumb:
        return "four_fingers", 4
    if non_thumb == 1 and index:
        return "one_finger", 1
    if total == 5:
        return "open_palm", 5
    return f"{total}_fingers", total

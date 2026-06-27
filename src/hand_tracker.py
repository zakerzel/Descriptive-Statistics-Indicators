"""MediaPipe hand tracking with temporal smoothing."""

from collections import Counter, deque
import cv2
import mediapipe as mp

from src.gesture_classifier import classify_from_states, get_finger_states


class HandTracker:
    def __init__(self, max_num_hands=2, smoothing_window=7):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self.history = {
            "Left": deque(maxlen=smoothing_window),
            "Right": deque(maxlen=smoothing_window),
            "Unknown": deque(maxlen=smoothing_window),
        }

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        output = []
        if not results.multi_hand_landmarks:
            return output
        handedness_list = results.multi_handedness or []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = "Unknown"
            confidence = 0.0
            if idx < len(handedness_list):
                info = handedness_list[idx].classification[0]
                hand_label = info.label
                confidence = float(info.score)
            states = get_finger_states(hand_landmarks.landmark, hand_label)
            raw_gesture, raw_count = classify_from_states(states)
            finger_count, gesture = self._smooth(hand_label, raw_count, raw_gesture)
            output.append({
                "hand_landmarks": hand_landmarks,
                "hand_label": hand_label,
                "confidence": confidence,
                "finger_states": states,
                "finger_count": finger_count,
                "gesture": gesture,
                "raw_finger_count": raw_count,
                "raw_gesture": raw_gesture,
            })
        return output

    def _smooth(self, hand_label, count, gesture):
        if hand_label not in self.history:
            hand_label = "Unknown"
        self.history[hand_label].append((count, gesture))
        counts = [item[0] for item in self.history[hand_label]]
        gestures = [item[1] for item in self.history[hand_label]]
        count_mode = Counter(counts).most_common(1)[0][0]
        gesture_mode = Counter(gestures).most_common(1)[0][0]
        gesture_to_count = {
            "fist": 0,
            "thumbs_up": 1,
            "one_finger": 1,
            "peace": 2,
            "three_fingers": 3,
            "four_fingers": 4,
            "open_palm": 5,
        }
        if gesture_mode in gesture_to_count:
            count_mode = gesture_to_count[gesture_mode]
        return count_mode, gesture_mode

    def draw(self, frame_bgr, hand_result):
        hand_landmarks = hand_result["hand_landmarks"]
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        h, w = frame_bgr.shape[:2]
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x1 = int(max(0, min(xs) * w - 12))
        y1 = int(max(0, min(ys) * h - 12))
        x2 = int(min(w, max(xs) * w + 12))
        y2 = int(min(h, max(ys) * h + 12))
        label = f"{hand_result['hand_label']}: {hand_result['gesture']} | {hand_result['finger_count']} fingers"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - 36)), (min(w, x1 + 470), y1), (0, 0, 0), -1)
        cv2.putText(frame_bgr, label, (x1 + 8, max(25, y1 - 11)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

    def close(self):
        self.hands.close()

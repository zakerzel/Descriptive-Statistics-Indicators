"""Statistical indicators for the live dashboard."""

import pandas as pd


def pct(num, den):
    return 0.0 if den == 0 else round((num / den) * 100, 2)


def compute_indicators(df, attempts_df=None):
    if df is None or df.empty:
        total = 0
        base = {
            "total_detections": 0,
            "current_hands": 0,
            "current_total_fingers": 0,
            "stored_profiles": 0,
            "most_common_gesture": "-",
            "most_common_finger_count": "-",
            "open_palm_rate": 0.0,
            "fist_rate": 0.0,
            "thumbs_up_rate": 0.0,
            "left_hand_rate": 0.0,
            "right_hand_rate": 0.0,
            "detections_by_identity": {},
            "gesture_distribution": {},
        }
    else:
        total = len(df)
        base = {
            "total_detections": total,
            "current_hands": 0,
            "current_total_fingers": 0,
            "stored_profiles": 0,
            "most_common_gesture": str(df["gesture"].mode().iloc[0]) if "gesture" in df and not df["gesture"].dropna().empty else "-",
            "most_common_finger_count": int(df["finger_count"].mode().iloc[0]) if "finger_count" in df and not df["finger_count"].dropna().empty else "-",
            "open_palm_rate": pct(int((df["gesture"] == "open_palm").sum()), total),
            "fist_rate": pct(int((df["gesture"] == "fist").sum()), total),
            "thumbs_up_rate": pct(int((df["gesture"] == "thumbs_up").sum()), total),
            "left_hand_rate": pct(int((df["hand"] == "Left").sum()), total),
            "right_hand_rate": pct(int((df["hand"] == "Right").sum()), total),
            "detections_by_identity": df["identity"].value_counts().head(4).to_dict() if "identity" in df else {},
            "gesture_distribution": df["gesture"].value_counts().head(4).to_dict() if "gesture" in df else {},
        }

    if attempts_df is None or attempts_df.empty:
        base.update({
            "sequence_attempts": 0,
            "profiles_saved": 0,
            "matched_attempts": 0,
            "ignored_short": 0,
            "match_rate": 0.0,
            "last_identity": "-",
            "last_score": 0.0,
            "last_decision": "-",
        })
        return base

    attempts = len(attempts_df)
    matched = int((attempts_df["decision"] == "matched").sum()) if "decision" in attempts_df else 0
    profiles = int((attempts_df["decision"] == "new_profile_saved").sum()) if "decision" in attempts_df else 0
    ignored = int((attempts_df["decision"] == "too_short").sum()) if "decision" in attempts_df else 0
    last = attempts_df.iloc[-1]
    base.update({
        "sequence_attempts": attempts,
        "profiles_saved": profiles,
        "matched_attempts": matched,
        "ignored_short": ignored,
        "match_rate": pct(matched, attempts),
        "last_identity": str(last.get("identity", "-")),
        "last_score": float(last.get("score", 0.0)),
        "last_decision": str(last.get("decision", "-")),
    })
    return base

# V8 — Gesture Recognition + Statistical Indicators

## Focus

This version is centered on the official activity requirements:

- Live hand detection.
- Finger counting from 0 to 5.
- Gesture classification.
- Left/right hand identification.
- CSV logging.
- Real-time indicators.
- Bonus: automatic gesture-sequence identity recognition.

## V8 key rule

For the identity bonus, the hand used is part of the identity pattern.

If a reference sequence uses the right hand and the repeated sequence uses the left hand in any step, it is treated as a different pattern.

This follows the activity requirement that each step must store and compare:

- gesture
- hand used
- timing

## V8 improvements

- Minimum valid sequence: 3 segments.
- Recommended sequence length: 4 segments.
- Maximum safety limit: 12 segments.
- Sequence finalizes automatically when the hand disappears.
- Strong hand-pattern matching: one hand mismatch prevents the match.
- Cleaner interface.
- Better finger-count consistency.
- Dynamic automatic identity assignment.
- CSV logs for detections and sequence attempts.

## Project structure

```text
Descriptive-Statistics-Indicators/
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── auto_sequence_identity.py
│   ├── dashboard.py
│   ├── gesture_classifier.py
│   ├── hand_tracker.py
│   ├── indicators.py
│   └── logger.py
├── data/
└── screenshots/
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `r` | Reset logs and profiles |
| `p` | Clear stored profiles |

## Recommended demo sequence

Use 3–5 clear gestures:

```text
open_palm -> thumbs_up -> fist -> one_finger
```

Then lower the hand for around 1 second so the sequence closes.

Repeat the same sequence with the same hand to identify the same person again.
Repeat it with the opposite hand to prove that it becomes a different pattern.

## Outputs

The app creates runtime evidence files in `data/`:

- `gesture_log.csv`
- `sequence_attempts.csv`
- `sequence_profiles.json`

Screenshots are saved in `screenshots/`.

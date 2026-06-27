# V8 — Gesture Recognition + Statistical Indicators

## How to use this project from GitHub

You can run this project directly from GitHub by cloning the repository:

```bash
git clone https://github.com/zakerzel/Descriptive-Statistics-Indicators.git
cd Descriptive-Statistics-Indicators
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

If you do not want to use Git, you can also download it manually:

```text
Code → Download ZIP → Extract folder → Open terminal inside the folder → Run python app.py
```

The main file is:

```text
app.py
```

The older experimental file was removed because the final V8 system is organized through `app.py` and the `src/` modules.

---

## Focus

This version is centered on the official activity requirements:

* Live hand detection.
* Finger counting from 0 to 5.
* Gesture classification.
* Left/right hand identification.
* CSV logging.
* Real-time statistical indicators.
* Bonus: automatic gesture-sequence identity recognition.

The system uses a webcam to detect hand gestures in real time and generates a clean dataset from the detections. That dataset is then used to calculate descriptive indicators such as gesture frequency, most common gesture, hand usage proportions, and sequence identity metrics.

---

## V8 key rule

For the identity bonus, the hand used is part of the identity pattern.

If a reference sequence uses the right hand and the repeated sequence uses the left hand in any step, it is treated as a different pattern.

Example:

```text
Person 1:
Right open_palm -> Right thumbs_up -> Right fist -> Right one_finger

Different pattern:
Left open_palm -> Left thumbs_up -> Left fist -> Left one_finger
```

This follows the activity requirement that each step must store and compare:

* gesture
* hand used
* timing

Therefore, the system does not identify a person only by the gesture order. It also considers which hand was used and the temporal structure of the sequence.

---

## V8 improvements

* Minimum valid sequence: 3 segments.
* Recommended sequence length: 4 segments.
* Maximum safety limit: 12 segments.
* Sequence finalizes automatically when the hand disappears.
* Strong hand-pattern matching: one hand mismatch prevents the match.
* Cleaner interface.
* Better finger-count consistency.
* Dynamic automatic identity assignment.
* CSV logs for detections and sequence attempts.
* Real-time dashboard with descriptive statistics and indicators.

---

## Project structure

```text
Descriptive-Statistics-Indicators/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── auto_sequence_identity.py
│   ├── dashboard.py
│   ├── gesture_classifier.py
│   ├── hand_tracker.py
│   ├── indicators.py
│   └── logger.py
├── data/
│   └── .gitkeep
└── screenshots/
    └── .gitkeep
```

---

## System workflow

```text
Webcam
  ↓
MediaPipe hand detection
  ↓
Finger counting + gesture classification
  ↓
Left/right hand identification
  ↓
Gesture sequence capture
  ↓
Automatic identity matching
  ↓
CSV logging
  ↓
Real-time statistical indicators dashboard
```

---

## Requirements

The project uses:

```text
opencv-python
mediapipe
pandas
numpy
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Run

From the project folder, run:

```bash
python app.py
```

A webcam window will open with:

* live hand detection,
* gesture labels,
* finger count,
* left/right hand detection,
* real-time indicators,
* automatic sequence identity recognition.

---

## Controls

| Key | Action                  |
| --- | ----------------------- |
| `q` | Quit                    |
| `s` | Save screenshot         |
| `r` | Reset logs and profiles |
| `p` | Clear stored profiles   |

The system works automatically. The controls are only for maintenance during testing or demonstration.

---

## Recommended demo sequence

Use 3–5 clear gestures:

```text
open_palm -> thumbs_up -> fist -> one_finger
```

Then lower the hand for around 1 second so the sequence closes.

Repeat the same sequence with the same hand to identify the same person again.

Repeat it with the opposite hand to prove that it becomes a different pattern.

---

## Suggested live demonstration

1. Press `r` to reset logs and profiles.
2. Perform a clear sequence with one hand.
3. Lower the hand for about 1 second.
4. The system saves the sequence as `Person 1`.
5. Perform a different sequence.
6. Lower the hand again.
7. The system saves it as `Person 2`.
8. Repeat the first sequence with the same hand.
9. The system should recognize it as `Person 1`.
10. Repeat the first sequence with the opposite hand.
11. The system should treat it as a different pattern.

---

## Output files

The app creates runtime evidence files in `data/`:

```text
data/gesture_log.csv
data/sequence_attempts.csv
data/sequence_profiles.json
```

Screenshots are saved in:

```text
screenshots/
```

These files are generated locally while the system is running.

---

## Logged data

Each detection is stored with information such as:

* timestamp,
* detected identity,
* hand,
* finger count,
* gesture,
* confidence,
* number of hands detected,
* total fingers detected,
* active sequence segments.

Each sequence attempt stores:

* timestamp,
* assigned identity,
* similarity score,
* number of sequence segments,
* decision,
* complete sequence string.

---

## Real-time indicators

The dashboard includes:

* current hands detected,
* current total fingers,
* stored profiles,
* total detections,
* most common gesture,
* most common finger count,
* open palm rate,
* fist rate,
* thumbs-up rate,
* left-hand rate,
* right-hand rate,
* sequence attempts,
* profiles saved,
* matched attempts,
* ignored short sequences,
* match rate,
* last detected identity,
* last decision,
* last score.

---

## Notes for better detection

For best results:

* Use a well-lit space.
* Keep the hand visible inside the camera frame.
* Hold each gesture for around 0.8 to 1 second.
* Use clear gestures.
* Avoid very fast transitions.
* Avoid extreme hand angles.
* Lower the hand briefly to finish the sequence.
* Use the same hand when trying to repeat a known identity pattern.

---

## Team

Beto-Saurio

Members:

* Gael
* Alberto
* Yazira
* Fabio

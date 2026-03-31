# 🏋️ Gym Rep Counter using Pose Estimation

A real-time AI-powered workout repetition counter that uses **MediaPipe Pose Estimation** and **OpenCV** to track body movements, count exercise reps automatically, and log your workout sessions — no wearable required.

---

## 📸 Features

- ✅ Real-time human pose detection via webcam
- ✅ **5 exercises** supported out of the box
- ✅ Automatic rep counting using joint angle analysis
- ✅ Live angle feedback with visual progress bar
- ✅ Rep goal tracker with on-screen progress
- ✅ Session timer & workout log saved to CSV
- ✅ On-screen exercise switcher with keyboard shortcuts
- ✅ Color-coded skeleton overlay with joint highlights

---

## 🏃 Supported Exercises

| # | Exercise        | Joints Tracked                        |
|---|-----------------|---------------------------------------|
| 1 | Bicep Curl      | Shoulder → Elbow → Wrist              |
| 2 | Squat           | Hip → Knee → Ankle                    |
| 3 | Push-Up         | Shoulder → Elbow → Wrist              |
| 4 | Shoulder Press  | Elbow → Shoulder → Hip                |
| 5 | Lateral Raise   | Hip → Shoulder → Elbow                |

---

## 🛠️ Tech Stack

| Library      | Purpose                              |
|--------------|--------------------------------------|
| `mediapipe`  | Pose landmark detection (33 points)  |
| `opencv-python` | Camera capture, drawing, UI       |
| `numpy`      | Angle calculation via dot product    |
| `csv / os`   | Workout log persistence              |

---

## 📁 Project Structure

```
gym-rep-counter/
├── gym_rep_counter.py      # Main application
├── workout_log.csv         # Auto-generated session log
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone / Download

```bash
git clone https://github.com/yourname/gym-rep-counter.git
cd gym-rep-counter
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

### 4. Run

```bash
python gym_rep_counter.py
```

> Make sure your webcam is connected and accessible.

---

## ⌨️ Keyboard Controls

| Key | Action                              |
|-----|-------------------------------------|
| `1` | Switch to Bicep Curl                |
| `2` | Switch to Squat                     |
| `3` | Switch to Push-Up                   |
| `4` | Switch to Shoulder Press            |
| `5` | Switch to Lateral Raise             |
| `R` | Reset rep count (saves current set) |
| `G` | Cycle rep goal (5 → 10 → 15 → 30)  |
| `Q` | Quit (saves final set)              |

---

## 📊 How Rep Counting Works

The system measures the **angle at the middle joint** of a three-point joint chain:

```
angle = arccos( (BA · BC) / (|BA| × |BC|) )
```

Each exercise has a configured **DOWN angle** and **UP angle**:

1. When the angle exceeds `DOWN` threshold → stage = "DOWN"
2. When the angle drops below `UP` threshold (from DOWN) → stage = "UP" → **Rep counted!**

This prevents double-counting and handles partial movements gracefully.

---

## 📝 Workout Log (CSV)

Every completed set is automatically saved to `workout_log.csv`:

```
Date,Time,Exercise,Reps,Duration(s)
2025-08-01,10:32:14,Bicep Curl,10,45
2025-08-01,10:33:22,Squat,12,60
```

---

## 🔧 Customization

### Add a new exercise

Edit the `EXERCISE_CONFIG` dictionary in `gym_rep_counter.py`:

```python
"Tricep Extension": {
    "joints": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "down": 170,
    "up": 50,
},
```

### Change detection sensitivity

```python
MATCH_THRESHOLD = 0.6   # Landmark visibility cutoff
```

### Use a different camera

```python
cap = cv2.VideoCapture(1)  # Change index to 1, 2, etc.
```

---

## 🚧 Limitations

- Works best with **good lighting** and a **clear background**
- Designed for **single-person** workouts
- Side-view is recommended for most exercises
- Accuracy depends on webcam quality and distance from camera

---

## 🔮 Future Improvements

- [ ] Multi-person support
- [ ] Voice feedback for rep counts
- [ ] Rep quality scoring (depth, tempo)
- [ ] Web dashboard for workout history
- [ ] Export to PDF reports

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Credits

Built using [MediaPipe](https://mediapipe.dev) by Google and [OpenCV](https://opencv.org).  
Inspired by AI fitness applications and computer vision education projects.

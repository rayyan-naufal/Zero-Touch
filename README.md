# Zero Touch

Zero Touch is a touchless human-computer interface and assistive tool for Windows. It enables users to navigate the mouse cursor and trigger mouse actions without physical contact, utilizing standard webcam input, MediaPipe computer vision pipelines, and low-latency Win32 event dispatching.

The system combines facial expression recognition, head-orientation tracking, and hand gesture estimation to deliver responsive control suited for accessibility needs, ergonomic hands-free setups, and presentation environments.

---

## Features

### Tracking Architecture
- **Dual-Mode Head Tracking**:
  - **Absolute Mode**: Direct angular mapping relative to a calibrated resting center.
  - **Relative / Joystick Mode**: Velocity-based cursor displacement when tilting outside a configurable deadzone, minimizing neck strain on large display surfaces.
- **One Euro Filter & Dynamic Smoothing**: Adaptive filtering suppresses high-frequency jitter during micro-movements while preserving rapid target tracking without phase lag.
- **Multimodal Control Pipeline**:
  - Independent toggling for cursor steering and click execution.
  - Decoupled hand and head tracking engines.

### Gesture and Action Triggers
- **Facial Triggers**:
  - Eye Aspect Ratio (EAR) computation with bilateral blink rejection for left and right wink detection.
  - Mouth Aspect Ratio (MAR) tracking for mouth-open actions.
  - Dwell clicking with visual countdown and radius stability detection.
- **Hand Gesture Triggers**:
  - Real-time hand landmark classification supporting Peace Sign, Pinch, Trio-Finger, and Fist gestures.
  - Configurable action mapping for Left Click, Right Click, Double Click, and Middle Click.
  - Hold-to-drag state handling for uninterrupted drag-and-drop operations.

### User Interface & Controls
- **Desktop Dashboard**: Built with CustomTkinter, offering Simple (standard controls) and Advanced (full parameter tuning) modes.
- **Telemetry HUD**: Real-time visualization of eye metrics, mouth openness, tracking vectors, and hand skeletons.
- **Global Hotkeys**: Non-blocking system-wide shortcuts to pause, calibrate, or toggle subsystem states from any foreground application.

---

## Architecture Overview

```
Webcam Frame (OpenCV)
  |
  +---> Face Mesh Ingestion (MediaPipe)
  |       |--> Landmark Extraction (Nose, Eyes, Lips)
  |       |--> Geometric Calculations (EAR, MAR, Rotation Deltas)
  |       +--> One Euro / EMA Smoothing Filter
  |
  +---> Hand Mesh Ingestion (MediaPipe Hands)
          |--> Joint Angles & Distance Ratios
          +--> Gesture Classifier (Peace, Pinch, Trio)
  |
  v
Arbitration & Controller (mouse_controller.py)
  |--> Debounce & Cooldown Engine
  |--> Win32 SendInput Event Dispatcher
  v
Operating System Cursor
```

---

## Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| Supported OS | Windows 10, Windows 11 (64-bit) |
| Runtime | Python 3.10+ (tested on Python 3.11) |
| Camera Input | Standard USB or integrated webcam (30+ FPS recommended) |
| Core Libraries | OpenCV, MediaPipe, CustomTkinter, Pynput, PyAutoGUI |
| Event Subsystem | Win32 API (`SendInput` via `ctypes`) |

---

## Project Structure

```
├── config.py                 # Configuration schema, presets, and persistent JSON store
├── gui.py                    # CustomTkinter desktop interface and telemetry dashboard
├── hand_tracker.py           # Hand landmark pipeline, gesture classifier, and filter
├── main.py                   # Application entry point, CLI arguments, and global hotkeys
├── mouse_controller.py       # Win32 cursor displacement, click debouncing, and dwell engine
├── tracker.py                # FaceMesh pipeline, head pose estimation, and metric logic
├── requirements.txt          # Project dependencies
├── run.bat                   # Windows batch launcher
├── test_facecontrol.py       # Automated unit test suite
├── dokumentasi/              # Performance analysis and latency optimization documentation
└── config.json               # Default application settings
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rayyan-naufal/Zero-Touch.git
   cd Zero-Touch
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation via test suite:
   ```bash
   python -m unittest discover
   ```

---

## Usage

### Starting the Application

Launch via command line:
```bash
python main.py
```

Or double-click `run.bat` on Windows.

### Default Shortcuts

| Key | Action |
| :--- | :--- |
| `F8` or `ESC` | Master Pause / Resume tracking |
| `F9` | Calibrate resting head position (global) |
| `C` | Calibrate resting head position (when window is focused) |
| `F6` | Toggle cursor movement subsystem |
| `F7` | Toggle click action subsystem |

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

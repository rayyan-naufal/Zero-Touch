# Implementation Plan - FaceControl (Head Pointer & Facial Expression Mouse for Windows)

Build a free, open-source Windows desktop application that mimics macOS Head Pointer and Facial Expression triggers. The application will track head motion via a standard webcam to move the cursor smoothly without jitter, detect facial expressions (winks, mouth open, smile) to trigger mouse clicks, and provide a desktop GUI to adjust sensitivity, calibrate center position, and toggle tracking.

---

## User Review Required

> [!IMPORTANT]
> **Webcam & Screen Resolution**: The application relies on a standard built-in or USB webcam (30+ FPS recommended) and requires camera permissions in Windows.
> **Dependency Installation**: The solution uses Python 3.11 with `customtkinter`, `opencv-python`, `mediapipe`, `pyautogui`, and `pynput`.

---

## Architecture & System Design

```mermaid
flowchart TD
    Webcam[Webcam Video Stream] --> FaceMesh[MediaPipe Face Mesh Detector]
    FaceMesh --> HeadPose[Head Motion & Pose Estimator]
    FaceMesh --> Gestures[Facial Feature & Blendshape Engine]
    
    subgraph Pointer Processing
        HeadPose --> Deadzone[Deadzone Filter]
        Deadzone --> EMA[EMA Jitter Smoothing Filter]
        EMA --> ScreenMap[Screen Coordinate Mapper]
    end
    
    subgraph Expression Processing
        Gestures --> EAR[Eye Aspect Ratio: Left/Right Wink]
        Gestures --> MAR[Mouth Aspect Ratio: Mouth Open/Smile]
        EAR --> Debounce[Gesture Hold & Debounce Filter]
        MAR --> Debounce
    end
    
    ScreenMap --> MouseCtrl[Mouse Controller: PyAutoGUI]
    Debounce --> MouseCtrl
    
    subgraph Desktop GUI (CustomTkinter)
        GUI[Live Video Canvas + Sliders + Calibrate Button]
        GUI <--> ConfigManager[Config & Preset Manager (config.json)]
        HotkeyListener[Global Hotkey: F8 / ESC] --> MouseCtrl
    end
```

---

## Proposed Changes

All files will be created in `c:\Rayyans\01_Rayyan\codes\FaceControl`:

### 1. Dependencies & Configuration

#### [NEW] `requirements.txt`
Dependencies required for computer vision, GUI, and input simulation:
```text
opencv-python>=4.8.0
mediapipe>=0.10.0
customtkinter>=5.2.0
pillow>=10.0.0
pyautogui>=0.9.54
pynput>=1.7.6
numpy>=1.24.0
```

#### [NEW] `config.py`
Configuration manager that loads and saves user settings to `config.json`:
- `cursor_speed`: Multiplier for head movement (default: `1.5`).
- `smoothing`: Exponential Moving Average factor $\alpha \in [0.1, 0.9]$ (default: `0.5`).
- `deadzone`: Threshold radius in normalized face coordinates where head movement is ignored (default: `0.015`).
- `left_click_trigger`: Selected facial gesture (default: `'Left Wink'`).
- `right_click_trigger`: Selected facial gesture (default: `'Mouth Open'`).
- `wink_threshold`: Eye Aspect Ratio threshold (default: `0.20`).
- `mouth_threshold`: Mouth Aspect Ratio threshold (default: `0.50`).
- `hotkey_toggle`: Global pause/resume key (default: `'f8'`).

---

### 2. Core Tracking & Gesture Engine

#### [NEW] `tracker.py`
Core vision module containing:
- **`FaceTracker` Class**:
  - Initializes `mp.solutions.face_mesh.FaceMesh` in stream mode (468 landmarks).
  - Landmark calculation:
    - **Nose tip** (Landmark 1) & **Nose bridge** (Landmark 168) for head orientation deltas relative to calibrated center.
    - **Left Eye EAR** (Landmarks 362, 385, 387, 263, 373, 380) vs **Right Eye EAR** (Landmarks 33, 160, 158, 133, 153, 144) to distinguish unilateral winks from bilateral natural blinks.
    - **Mouth Aspect Ratio (MAR)** (Landmarks 13, 14, 78, 308) to detect mouth opening without misfiring during normal talking.
  - **Smoothing Filter**: Applies Exponential Moving Average:
    $$P_{\text{smooth}} = \alpha \cdot P_{\text{current}} + (1 - \alpha) \cdot P_{\text{prev}}$$
  - **Center Calibration**: Sets resting head pose offset so the user sits comfortably.

#### [NEW] `mouse_controller.py`
Mouse execution engine:
- Controls physical cursor via `pyautogui` / Windows Win32 API.
- Debounces clicks (e.g. minimum 350ms cool-down) to avoid accidental rapid double-clicking.
- Supports click-and-drag when holding an expression for $>0.6\text{s}$.
- Includes a hard pause state toggled by global hotkeys.

---

### 3. Desktop User Interface

#### [NEW] `gui.py`
Modern desktop GUI built with `customtkinter`:
- **Left Panel (Live Feedback HUD)**:
  - Video feed displaying face mesh overlay.
  - Real-time gauge bars showing current Left EAR, Right EAR, and MAR metrics alongside trigger thresholds.
  - Visual badge displaying the last triggered action (`[LEFT CLICK]`, `[RIGHT CLICK]`, etc.).
- **Right Panel (Controls & Calibration)**:
  - Master tracking switch (`ON` / `OFF`).
  - Hotkey indicator (`Press F8 or ESC to Pause`).
  - Interactive sliders: **Cursor Speed**, **Smoothing (Jitter Reduction)**, **Deadzone**, **Wink Sensitivity**, **Mouth Sensitivity**.
  - Dropdown selectors: Map gestures to actions (Left Click, Right Click, Double Click, Middle Click).
  - **"Calibrate Resting Center"** button.
  - **"Save Settings"** button.

#### [NEW] `main.py`
Main application entry point orchestrating the camera thread, tracking loop, global keyboard listener, and GUI loop.

---

## Verification Plan

### Automated / Diagnostic Verification
1. **Module Smoke Test**:
   ```powershell
   python -c "import cv2, mediapipe, customtkinter, pyautogui, pynput; print('Dependencies OK')"
   ```
2. **Camera & Landmark Test**:
   Run a short headless check verifying that MediaPipe initializes and detects landmarks from the default webcam device (`cv2.VideoCapture(0)`).

### Manual Verification
1. **Camera Feed & GUI Appearance**:
   - Launch `python main.py`.
   - Verify the desktop window opens with live camera preview and dark-theme controls.
2. **Calibration**:
   - Sit comfortably, click "Calibrate Resting Center", and confirm the crosshair aligns with resting nose position.
3. **Cursor Movement**:
   - Turn tracking ON. Gently turn head up, down, left, right; ensure cursor glides smoothly across the desktop without jitter.
4. **Gesture Clicks**:
   - Wink left eye: Verify left click fires on screen.
   - Open mouth slightly: Verify right click context menu appears.
   - Natural blink: Verify both eyes blinking does NOT trigger accidental clicks.
5. **Emergency Pause**:
   - Press `F8` on the keyboard: Verify cursor tracking immediately freezes. Press `F8` again to resume.

"""Face and head tracking engine for Zero Touch.
Processes webcam video frames using MediaPipe Face Mesh, computes head motion deltas,
filters jitter using deadzones and exponential moving averages, and recognizes facial gestures.
"""

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig


class OneEuroFilter:
    """Adaptive 1€ Filter for real-time cursor motion.
    Suppresses micro-jitter when stationary (low speed) while maintaining
    zero lag during fast head movements (high speed).
    Reference: Casiez, Roussel, Vogel (CHI 2012) - 1€ Filter.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, t: float, x: float) -> float:
        dt = max(1e-4, t - self.t_prev)

        # Estimate derivative (speed) of the signal
        edx = (x - self.x_prev) / dt

        # Smooth derivative using d_cutoff
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = alpha_d * edx + (1.0 - alpha_d) * self.dx_prev

        # Dynamic cutoff based on instantaneous speed
        cutoff = self.min_cutoff + self.beta * abs(dx)

        # Smooth signal using dynamic cutoff
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        # Update historical state
        self.x_prev = x_hat
        self.dx_prev = dx
        self.t_prev = t
        return x_hat

    def reset(self, t0: float, x0: float, dx0: float = 0.0):
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)


class FaceTracker:
    """Detects facial landmarks, tracks head pose relative to calibrated center,
    and detects discrete facial gestures (winks, mouth open, smile).
    """

    # Landmark indices for MediaPipe Face Mesh (468 landmarks)
    NOSE_TIP = 1
    NOSE_BRIDGE = 168
    FOREHEAD = 10
    CHIN = 152

    # Eye landmarks: [outer, upper1, upper2, inner, lower1, lower2]
    # Left eye (relative to face): outer=263, inner=362
    LEFT_EYE = [263, 385, 387, 362, 373, 380]
    # Right eye (relative to face): outer=33, inner=133
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Mouth landmarks: top, bottom, left corner, right corner
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize MediaPipe Face Mesh (refine_landmarks=False for optimal CPU performance)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Thread safety lock for calibration, filters, and frame processing
        self._lock = threading.Lock()

        # Resting center calibration (normalized coordinates)
        self.resting_center: Optional[Tuple[float, float]] = None

        # Smoothed screen coordinates (One Euro Filter state)
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None
        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None

        # Temporal confirmation buffer for wink detection (Issue #3)
        self.candidate_gesture: Optional[str] = None
        self.gesture_frame_count: int = 0
        self.REQUIRED_WINK_FRAMES: int = 2  # Must persist >= 2 consecutive frames (~60ms)

        # Origin anchor state & Center calibration flag
        self.recalibrate_to_center: bool = False
        self.screen_origin_x: Optional[float] = None
        self.screen_origin_y: Optional[float] = None
        self.latest_dx: float = 0.0
        self.latest_dy: float = 0.0
        self.last_screen_w: int = 1920
        self.last_screen_h: int = 1080

        # Face absence counter for stale filter reset (Issue #7)
        self.face_lost_counter: int = 0
        self.MAX_LOST_FRAMES: int = 5

        # Previous frame metrics for telemetry
        self.latest_left_ear: float = 0.0
        self.latest_right_ear: float = 0.0
        self.latest_mar: float = 0.0
        self.latest_gesture: Optional[str] = None

    def calibrate_center(self, current_nose: Optional[Tuple[float, float]] = None):
        """Sets resting center to current nose position, or resets for automatic re-acquisition."""
        with self._lock:
            self.resting_center = current_nose
            self.recalibrate_to_center = True
            # Reset screen origin so next frame centers cursor at screen_w / 2.0
            self.screen_origin_x = None
            self.screen_origin_y = None
            self.latest_dx = 0.0
            self.latest_dy = 0.0
            # Reset adaptive filter on center recalibration
            self.filter_x = None
            self.filter_y = None
            self.smoothed_x = None
            self.smoothed_y = None
        if current_nose is not None:
            print(f"[FaceTracker] Resting center calibrated to: {self.resting_center}")
        else:
            print("[FaceTracker] Resting center reset for recalibration on next frame.")

    def reset_filters(
        self,
        current_pos: Optional[Tuple[float, float]] = None,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ):
        """Resets or re-seeds filter states for smooth transition on resume.
        Respects tracking_mode dimensionality:
        - Absolute mode: seeds screen pixel coordinates [0, screen_size] and aligns
          screen origin anchor with current_pos to eliminate cursor jumping.
        - Relative mode: resets angular deltas to 0.0 to prevent coordinate blowup,
          and updates smoothed cursor pos to current_pos.
        """
        with self._lock:
            now = time.perf_counter()
            beta = getattr(self.config, "filter_beta", 0.007)
            min_cutoff = max(0.2, 2.5 * (1.0 - getattr(self.config, "smoothing", 0.65)))
            sw = float(screen_w if screen_w is not None else getattr(self, "last_screen_w", 1920))
            sh = float(screen_h if screen_h is not None else getattr(self, "last_screen_h", 1080))
            speed = getattr(self.config, "cursor_speed", 2.2)

            if self.config.tracking_mode == "relative":
                # In Relative mode, filters smooth angular tilt delta (dx, dy ~ 0.0).
                # Resetting them to 0.0 prevents injecting large pixel coordinates.
                if self.filter_x is not None and self.filter_y is not None:
                    self.filter_x.reset(now, 0.0)
                    self.filter_y.reset(now, 0.0)
                else:
                    self.filter_x = OneEuroFilter(now, 0.0, min_cutoff=min_cutoff, beta=beta)
                    self.filter_y = OneEuroFilter(now, 0.0, min_cutoff=min_cutoff, beta=beta)

                if current_pos is not None:
                    self.smoothed_x = float(current_pos[0])
                    self.smoothed_y = float(current_pos[1])
                else:
                    self.smoothed_x = None
                    self.smoothed_y = None
            else:
                # In Absolute mode, filters operate in screen pixel space
                if current_pos is not None:
                    pos_x = float(current_pos[0])
                    pos_y = float(current_pos[1])
                    # Align screen origin anchor so current head pose maps directly to current_pos
                    dx = getattr(self, "latest_dx", 0.0)
                    dy = getattr(self, "latest_dy", 0.0)
                    self.screen_origin_x = pos_x - (dx * sw * speed)
                    self.screen_origin_y = pos_y - (dy * sh * speed)

                    if self.filter_x is not None and self.filter_y is not None:
                        self.filter_x.reset(now, pos_x)
                        self.filter_y.reset(now, pos_y)
                    else:
                        self.filter_x = OneEuroFilter(now, pos_x, min_cutoff=min_cutoff, beta=beta)
                        self.filter_y = OneEuroFilter(now, pos_y, min_cutoff=min_cutoff, beta=beta)
                    self.smoothed_x = pos_x
                    self.smoothed_y = pos_y
                else:
                    self.filter_x = None
                    self.filter_y = None
                    self.smoothed_x = None
                    self.smoothed_y = None

    def _euclidean_dist(self, p1: Any, p2: Any) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _calculate_ear(self, landmarks: List[Any], eye_indices: List[int]) -> float:
        """Calculates Eye Aspect Ratio (EAR) using 6 landmark points:
        EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
        """
        p0 = landmarks[eye_indices[0]]  # Outer corner
        p1 = landmarks[eye_indices[1]]  # Upper 1
        p2 = landmarks[eye_indices[2]]  # Upper 2
        p3 = landmarks[eye_indices[3]]  # Inner corner
        p4 = landmarks[eye_indices[4]]  # Lower 1
        p5 = landmarks[eye_indices[5]]  # Lower 2

        vertical_1 = self._euclidean_dist(p1, p5)
        vertical_2 = self._euclidean_dist(p2, p4)
        horizontal = self._euclidean_dist(p0, p3)

        if horizontal <= 1e-6:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _calculate_mar(self, landmarks: List[Any]) -> float:
        """Calculates Mouth Aspect Ratio (MAR):
        MAR = ||p_top - p_bottom|| / ||p_left - p_right||
        """
        top = landmarks[self.MOUTH_TOP]
        bottom = landmarks[self.MOUTH_BOTTOM]
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]

        vertical = self._euclidean_dist(top, bottom)
        horizontal = self._euclidean_dist(left, right)

        if horizontal <= 1e-6:
            return 0.0
        return vertical / horizontal

    def _calculate_smile(self, landmarks: List[Any]) -> float:
        """Calculates mouth width ratio normalized by outer eye distance."""
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]
        mouth_width = self._euclidean_dist(left, right)

        # Distance between outer corners of eyes as invariant face scale
        left_eye_outer = landmarks[self.LEFT_EYE[0]]
        right_eye_outer = landmarks[self.RIGHT_EYE[0]]
        face_scale = self._euclidean_dist(left_eye_outer, right_eye_outer)

        if face_scale <= 1e-6:
            return 0.0
        return mouth_width / face_scale

    def _detect_gesture(
        self, left_ear: float, right_ear: float, mar: float, smile_ratio: float = 0.0
    ) -> Optional[str]:
        """Recognizes gestures with temporal persistence buffer to reject blink rolling lag."""
        raw_gesture: Optional[str] = None
        wink_thresh = self.config.wink_threshold
        ear_diff = abs(left_ear - right_ear)

        # 1. Mouth Open
        if mar > self.config.mouth_threshold:
            raw_gesture = "Mouth Open"
        # 2. Smile
        elif smile_ratio > self.config.smile_threshold:
            raw_gesture = "Smile"
        # 3. Bilateral Blink check: if both eyes closed, reject wink!
        elif left_ear < wink_thresh and right_ear < wink_thresh:
            raw_gesture = None
        # 4. Unilateral Winks (robust against involuntary squinting of opposite eye)
        elif left_ear < wink_thresh and right_ear >= (wink_thresh - 0.01) and ear_diff >= 0.035:
            raw_gesture = "Left Wink"
        elif right_ear < wink_thresh and left_ear >= (wink_thresh - 0.01) and ear_diff >= 0.035:
            raw_gesture = "Right Wink"

        # Apply Temporal Confirmation Buffer for winks (Issue #3)
        if raw_gesture in ("Left Wink", "Right Wink"):
            if self.candidate_gesture == raw_gesture:
                self.gesture_frame_count += 1
            else:
                self.candidate_gesture = raw_gesture
                self.gesture_frame_count = 1

            if self.gesture_frame_count >= self.REQUIRED_WINK_FRAMES:
                return raw_gesture
            else:
                return None  # Wait for persistence confirmation
        else:
            self.candidate_gesture = raw_gesture
            self.gesture_frame_count = 1 if raw_gesture else 0
            return raw_gesture

    def process_frame(
        self,
        frame: np.ndarray,
        screen_w: int,
        screen_h: int,
        current_mouse_pos: Optional[Tuple[float, float]] = None,
        is_overridden: bool = False,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[str], np.ndarray, Dict[str, float]]:
        """Processes video frame.
        Returns:
            - target_cursor_pos: (x, y) screen coords or None
            - detected_gesture: name of detected action trigger or None
            - annotated_frame: frame with visual HUD overlays
            - telemetry: dict with live metrics (left_ear, right_ear, mar, smile)
        """
        h, w, _ = frame.shape
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe (zero-copy buffer optimization)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        target_cursor_pos = None
        detected_gesture = None
        telemetry = {"left_ear": 0.0, "right_ear": 0.0, "mar": 0.0, "smile": 0.0}

        if results.multi_face_landmarks:
            # Face is detected - reset absence counter (Issue #7)
            self.face_lost_counter = 0
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Head Pose / Nose Landmark
            nose_landmark = landmarks[self.NOSE_TIP]
            nose_x, nose_y = nose_landmark.x, nose_landmark.y

            # Calibrate center on initial detection if uncalibrated
            if self.resting_center is None:
                self.resting_center = (nose_x, nose_y)

            # 2. Aspect Ratios & Gestures (Decoupled: only if clicking enabled)
            if getattr(self.config, "clicking_enabled", True):
                raw_left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
                raw_right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)

                # In mirrored mode (flip_horizontal=True), user's physical right eye appears on screen right
                # (which MediaPipe Face Mesh detects as anatomical LEFT_EYE / landmark 263), and user's physical
                # left eye appears on screen left (MediaPipe anatomical RIGHT_EYE / landmark 33).
                if self.config.flip_horizontal:
                    phys_left_ear = raw_right_ear
                    phys_right_ear = raw_left_ear
                else:
                    phys_left_ear = raw_left_ear
                    phys_right_ear = raw_right_ear

                mar = self._calculate_mar(landmarks)
                smile_ratio = self._calculate_smile(landmarks)

                self.latest_left_ear = phys_left_ear
                self.latest_right_ear = phys_right_ear
                self.latest_mar = mar

                click_mode = getattr(self.config, "click_mode", "gestures").lower()
                if click_mode != "dwell":
                    detected_gesture = self._detect_gesture(phys_left_ear, phys_right_ear, mar, smile_ratio)
                else:
                    detected_gesture = None
                self.latest_gesture = detected_gesture

                telemetry = {
                    "left_ear": phys_left_ear,
                    "right_ear": phys_right_ear,
                    "mar": mar,
                    "smile": smile_ratio,
                }
            else:
                detected_gesture = None
                self.latest_gesture = None
                telemetry = {
                    "left_ear": 0.0,
                    "right_ear": 0.0,
                    "mar": 0.0,
                    "smile": 0.0,
                }

            # 3. Motion delta from resting center & Cursor calculation (Decoupled: only if pointer enabled)
            if getattr(self.config, "pointer_enabled", True):
                with self._lock:
                    self.last_screen_w = screen_w
                    self.last_screen_h = screen_h
                    ref_x, ref_y = self.resting_center
                    dx = nose_x - ref_x
                    dy = nose_y - ref_y

                    # Deadzone filter with smooth radial transition
                    dist = math.hypot(dx, dy)
                    dz = self.config.deadzone
                    if dist < dz:
                        dx = 0.0
                        dy = 0.0
                    else:
                        scale = (dist - dz) / dist
                        dx *= scale
                        dy *= scale

                    self.latest_dx = dx
                    self.latest_dy = dy

                    # 4. Map to Screen Coordinates & 5. One Euro Filter Adaptive Filtering
                    now = time.perf_counter()
                    beta = getattr(self.config, "filter_beta", 0.007)
                    min_cutoff = max(0.2, 2.5 * (1.0 - self.config.smoothing))

                    if is_overridden and current_mouse_pos is not None:
                        # Physical mouse has priority: synchronize tracker origin and filter states
                        # with the actual physical cursor position so head tracking seamlessly
                        # continues from this exact spot when override releases (Zero Cursor Jump).
                        curr_x, curr_y = current_mouse_pos
                        speed = self.config.cursor_speed
                        self.screen_origin_x = float(curr_x) - (dx * screen_w * speed)
                        self.screen_origin_y = float(curr_y) - (dy * screen_h * speed)
                        self.smoothed_x = float(curr_x)
                        self.smoothed_y = float(curr_y)

                        if self.config.tracking_mode == "relative":
                            if self.filter_x is not None and self.filter_y is not None:
                                self.filter_x.reset(now, 0.0)
                                self.filter_y.reset(now, 0.0)
                        else:
                            if self.filter_x is not None and self.filter_y is not None:
                                self.filter_x.reset(now, float(curr_x))
                                self.filter_y.reset(now, float(curr_y))
                            else:
                                self.filter_x = OneEuroFilter(now, float(curr_x), min_cutoff=min_cutoff, beta=beta)
                                self.filter_y = OneEuroFilter(now, float(curr_y), min_cutoff=min_cutoff, beta=beta)

                        target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                    elif self.config.tracking_mode == "relative" and current_mouse_pos is not None:
                        # Relative / Joystick mode: filter angular tilt delta (dx, dy) for smooth velocity
                        if self.filter_x is None or self.filter_y is None:
                            self.filter_x = OneEuroFilter(now, dx, min_cutoff=min_cutoff, beta=beta)
                            self.filter_y = OneEuroFilter(now, dy, min_cutoff=min_cutoff, beta=beta)
                            filtered_dx, filtered_dy = dx, dy
                        else:
                            self.filter_x.min_cutoff = min_cutoff
                            self.filter_y.min_cutoff = min_cutoff
                            self.filter_x.beta = beta
                            self.filter_y.beta = beta
                            filtered_dx = self.filter_x.filter(now, dx)
                            filtered_dy = self.filter_y.filter(now, dy)

                        curr_x, curr_y = current_mouse_pos
                        velocity_scale = self.config.cursor_speed * 45.0
                        raw_screen_x = curr_x + (filtered_dx * velocity_scale)
                        raw_screen_y = curr_y + (filtered_dy * velocity_scale)
                        raw_screen_x = max(0.0, min(screen_w - 1.0, raw_screen_x))
                        raw_screen_y = max(0.0, min(screen_h - 1.0, raw_screen_y))
                        self.smoothed_x = raw_screen_x
                        self.smoothed_y = raw_screen_y
                        target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                    else:
                        # Absolute mode: direct angular mapping relative to screen anchor
                        speed = self.config.cursor_speed
                        if self.recalibrate_to_center or self.screen_origin_x is None or self.screen_origin_y is None:
                            if self.recalibrate_to_center or current_mouse_pos is None:
                                self.screen_origin_x = screen_w / 2.0
                                self.screen_origin_y = screen_h / 2.0
                                self.recalibrate_to_center = False
                            else:
                                self.screen_origin_x = float(current_mouse_pos[0]) - (dx * screen_w * speed)
                                self.screen_origin_y = float(current_mouse_pos[1]) - (dy * screen_h * speed)

                        raw_screen_x = self.screen_origin_x + (dx * screen_w * speed)
                        raw_screen_y = self.screen_origin_y + (dy * screen_h * speed)
                        raw_screen_x = max(0.0, min(screen_w - 1.0, raw_screen_x))
                        raw_screen_y = max(0.0, min(screen_h - 1.0, raw_screen_y))

                        if self.filter_x is None or self.filter_y is None:
                            self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                            self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                            self.smoothed_x = raw_screen_x
                            self.smoothed_y = raw_screen_y
                        else:
                            self.filter_x.min_cutoff = min_cutoff
                            self.filter_y.min_cutoff = min_cutoff
                            self.filter_x.beta = beta
                            self.filter_y.beta = beta
                            self.smoothed_x = self.filter_x.filter(now, raw_screen_x)
                            self.smoothed_y = self.filter_y.filter(now, raw_screen_y)

                        target_cursor_pos = (self.smoothed_x, self.smoothed_y)
            else:
                target_cursor_pos = None

            # 6. Annotate Visual Feedback onto Frame
            frame = self._draw_visual_hud(frame, landmarks, w, h)
        else:
            # Face not detected: handle stale state cleanup (Issue #7)
            with self._lock:
                self.face_lost_counter += 1
                if self.face_lost_counter >= self.MAX_LOST_FRAMES:
                    self.smoothed_x = None
                    self.smoothed_y = None
                    self.filter_x = None
                    self.filter_y = None
                    self.candidate_gesture = None
                    self.gesture_frame_count = 0

        return target_cursor_pos, detected_gesture, frame, telemetry

    def _draw_visual_hud(self, frame: np.ndarray, landmarks: List[Any], w: int, h: int) -> np.ndarray:
        """Renders face tracking landmarks, crosshair, and calibration reference."""
        # Draw resting center crosshair
        if self.resting_center is not None:
            rc_x = int(self.resting_center[0] * w)
            rc_y = int(self.resting_center[1] * h)
            cv2.circle(frame, (rc_x, rc_y), 4, (0, 255, 255), -1)
            cv2.circle(frame, (rc_x, rc_y), int(self.config.deadzone * w), (0, 255, 255), 1)

        # Current nose position
        nose = landmarks[self.NOSE_TIP]
        n_x, n_y = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (n_x, n_y), 5, (0, 0, 255), -1)

        # Draw vector connecting center to current nose
        if self.resting_center is not None:
            rc_x = int(self.resting_center[0] * w)
            rc_y = int(self.resting_center[1] * h)
            cv2.line(frame, (rc_x, rc_y), (n_x, n_y), (0, 200, 0), 2)

        # Draw eye contours
        for idx in self.LEFT_EYE:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (255, 255, 0), -1)
        for idx in self.RIGHT_EYE:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (255, 255, 0), -1)

        # Draw mouth landmarks
        for idx in [self.MOUTH_TOP, self.MOUTH_BOTTOM, self.MOUTH_LEFT, self.MOUTH_RIGHT]:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

        return frame

    def close(self):
        """Releases MediaPipe resources."""
        with self._lock:
            if hasattr(self, "face_mesh") and self.face_mesh:
                self.face_mesh.close()

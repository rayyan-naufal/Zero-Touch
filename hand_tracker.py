"""Hand tracking and gesture recognition engine for Zero Touch.
Processes webcam video frames using MediaPipe Hands, classifies hand gestures
(Peace Sign, Pinch Index, Pinch Middle, Index Point, Fist, Open Palm),
filters coordinate jitter with OneEuroFilter for finger pointer mode,
and renders visual skeleton HUD telemetry.
"""

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig
from tracker import OneEuroFilter


class HandTracker:
    """Detects hand landmarks, classifies invariant hand gestures,
    and calculates finger pointer coordinates.
    """

    # Hand Landmark Constants
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Skeletal connections for rendering
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = threading.Lock()

        # MediaPipe Hands (model_complexity=0 for minimum CPU latency)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        # Filters for finger pointer mode
        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None
        self.screen_origin_x: Optional[float] = None
        self.screen_origin_y: Optional[float] = None

        # Smart Sensitivity state machine (Windows-inspired Precision Pointer & Dynamic Ballistics)
        self.prev_finger_pos: Optional[Tuple[float, float]] = None
        self.prev_finger_time: float = 0.0
        self.filtered_velocity: float = 0.0
        self.accum_screen_x: Optional[float] = None
        self.accum_screen_y: Optional[float] = None

        # Temporal confirmation buffer for gestures
        self.candidate_gesture: Optional[str] = None
        self.gesture_frame_count: int = 0
        self.REQUIRED_GESTURE_FRAMES: int = 2

        # Telemetry
        self.latest_gesture: Optional[str] = None
        self.latest_handedness: Optional[str] = None
        self.latest_pinch_dist: float = 1.0

        # Thumb Trigger state machine
        self.thumb_is_open: bool = False
        self.thumb_state: str = "IDLE"  # "IDLE", "PENDING_TAP", "HOLDING_DRAG"
        self.thumb_opened_time: Optional[float] = None
        self.latest_thumb_status: str = "TUCKED"

    @staticmethod
    def _dist(p1: Any, p2: Any) -> float:
        """3D Euclidean distance with downweighted z depth to mitigate sensor noise."""
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = (getattr(p1, "z", 0.0) - getattr(p2, "z", 0.0)) * 0.4
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _dist_2d(p1: Any, p2: Any) -> float:
        """2D image-plane Euclidean distance."""
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx * dx + dy * dy)

    def _compute_dynamic_gain(self, velocity: float) -> float:
        """Evaluates non-linear ballistic response curve G(v) based on active profile.
        Velocity is in normalized screen units / second.
        """
        profile = getattr(self.config, "hand_sensitivity_profile", "dynamic").lower()
        if profile == "linear":
            return 1.0

        if profile == "smooth":
            g_min, g_max = 0.60, 1.75
            v_mid, gamma = 0.35, 1.20
        else:  # "dynamic" (default, Windows Precision Mouse curve)
            g_min, g_max = 0.40, 2.40
            v_mid, gamma = 0.28, 1.65

        v = max(0.0, float(velocity))
        v_pow = math.pow(v, gamma)
        v_mid_pow = math.pow(v_mid, gamma)
        gain = g_min + (g_max - g_min) * (v_pow / (v_mid_pow + v_pow + 1e-9))
        return gain

    @staticmethod
    def _apply_continuous_soft_deadzone(
        delta_x: float, delta_y: float, deadband: float
    ) -> Tuple[float, float, float]:
        """Applies C^0 continuous soft deadzone radial ramp to eliminate physiological tremor
        without introducing abrupt step jumps.
        Returns: (shaped_dx, shaped_dy, effective_magnitude)
        """
        mag = math.hypot(delta_x, delta_y)
        if mag <= deadband or mag < 1e-7:
            return 0.0, 0.0, 0.0

        # Continuous soft ramp scaling: ramps smoothly from 0.0 at deadband boundary
        diff = mag - deadband
        ramp_factor = min(1.0, diff / (deadband + 1e-9))
        eff_mag = diff * ramp_factor
        scale = eff_mag / mag
        return delta_x * scale, delta_y * scale, eff_mag

    def _is_finger_extended(self, landmarks: List[Any], tip_idx: int, pip_idx: int, mcp_idx: int, wrist: Any) -> bool:
        """Checks if a finger is extended using both wrist and knuckle (MCP) metrics."""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]

        # Primary check: Tip further from wrist than PIP
        dist_tip_wrist = self._dist_2d(tip, wrist)
        dist_pip_wrist = self._dist_2d(pip, wrist)

        # Knuckle check: Tip distance from MCP vs PIP distance from MCP
        dist_tip_mcp = self._dist_2d(tip, mcp)
        dist_pip_mcp = self._dist_2d(pip, mcp)

        return (dist_tip_wrist > dist_pip_wrist) and (dist_tip_mcp > dist_pip_mcp * 0.95)

    def _classify_gesture(self, landmarks: List[Any]) -> Tuple[Optional[str], float]:
        """Classifies hand gesture based on geometric, rotation-invariant metrics.
        Returns: (gesture_name, pinch_distance_normalized)
        """
        wrist = landmarks[self.WRIST]
        palm_scale = max(1e-4, self._dist_2d(wrist, landmarks[self.MIDDLE_MCP]))

        # Knuckle-anchored finger extension tests
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, wrist)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, wrist)
        ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, wrist)
        pinky_ext = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, wrist)

        # Pinch distances normalized by palm scale
        thumb_tip = landmarks[self.THUMB_TIP]
        idx_tip = landmarks[self.INDEX_TIP]
        mid_tip = landmarks[self.MIDDLE_TIP]

        pinch_idx_dist = self._dist(thumb_tip, idx_tip) / palm_scale
        pinch_mid_dist = self._dist(thumb_tip, mid_tip) / palm_scale

        is_3finger = getattr(self.config, "control_preset", "") == "3-Finger Control"
        if is_3finger:
            # 3-Finger Control Preset:
            # 1. Open Palm (All 4 fingers extended) -> Freeze cursor / idle
            if idx_ext and mid_ext and ring_ext and pinky_ext:
                return "Open Palm", pinch_idx_dist

            # 2. Fist / All 3 active fingers folded -> Freeze cursor / idle
            if not idx_ext and not mid_ext and not ring_ext:
                return "Fist", pinch_idx_dist

            # 3. 3-Finger Move (Index, Middle, Ring extended, Pinky folded) -> Move cursor
            if idx_ext and mid_ext and ring_ext and not pinky_ext:
                return "3-Finger Move", pinch_idx_dist

            # 4. Telunjuk ditekuk (Index folded, Middle & Ring extended, Pinky folded) -> Left Click
            if not idx_ext and mid_ext and ring_ext and not pinky_ext:
                return "3-Finger Left Click", pinch_idx_dist

            # 5. Manis ditekuk (Ring folded, Index & Middle extended, Pinky folded) -> Right Click
            if idx_ext and mid_ext and not ring_ext and not pinky_ext:
                return "3-Finger Right Click (Ring)", pinch_idx_dist

            # 6. Tengah ditekuk (Middle folded, Index & Ring extended, Pinky folded) -> Right Click
            if idx_ext and not mid_ext and ring_ext and not pinky_ext:
                return "3-Finger Right Click (Middle)", pinch_idx_dist

            return None, pinch_idx_dist

        # Pinch threshold normalized
        pinch_threshold = getattr(self.config, "hand_pinch_threshold", 0.08) * 3.2

        # 1. Pinch Index (Thumb + Index touching)
        if pinch_idx_dist < pinch_threshold:
            return "Pinch Index", pinch_idx_dist

        # 2. Pinch Middle (Thumb + Middle touching)
        if pinch_mid_dist < pinch_threshold:
            return "Pinch Middle", pinch_mid_dist

        # 3. Peace Sign (Index & Middle extended, Ring & Pinky folded, separated in V-shape)
        if idx_ext and mid_ext and not ring_ext and not pinky_ext:
            v_sep = self._dist_2d(idx_tip, mid_tip) / palm_scale
            if v_sep > 0.15:
                return "Peace Sign", pinch_idx_dist

        # 4. Pointing (Only index finger extended)
        if idx_ext and not mid_ext and not ring_ext and not pinky_ext:
            return "Pointing", pinch_idx_dist

        # 5. Fist (All 4 non-thumb fingers folded)
        if not idx_ext and not mid_ext and not ring_ext and not pinky_ext:
            return "Fist", pinch_idx_dist

        # 6. Open Palm (All 4 fingers extended)
        if idx_ext and mid_ext and ring_ext and pinky_ext:
            return "Open Palm", pinch_idx_dist

        return None, pinch_idx_dist

    def reset_filters(
        self,
        current_pos: Optional[Tuple[float, float]] = None,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ):
        """Resets OneEuroFilter and Smart Sensitivity accumulator states for finger tracking."""
        with self._lock:
            now = time.perf_counter()
            beta = getattr(self.config, "filter_beta", 0.007)
            min_cutoff = max(0.2, 2.5 * (1.0 - getattr(self.config, "smoothing", 0.65)))
            self.prev_finger_pos = None
            self.prev_finger_time = 0.0
            self.filtered_velocity = 0.0

            if current_pos is not None:
                self.screen_origin_x = float(current_pos[0])
                self.screen_origin_y = float(current_pos[1])
                self.accum_screen_x = float(current_pos[0])
                self.accum_screen_y = float(current_pos[1])
                self.filter_x = OneEuroFilter(now, current_pos[0], min_cutoff=min_cutoff, beta=beta)
                self.filter_y = OneEuroFilter(now, current_pos[1], min_cutoff=min_cutoff, beta=beta)
                self.smoothed_x = float(current_pos[0])
                self.smoothed_y = float(current_pos[1])
            else:
                self.screen_origin_x = None
                self.screen_origin_y = None
                self.accum_screen_x = None
                self.accum_screen_y = None
                self.filter_x = None
                self.filter_y = None
                self.smoothed_x = None
                self.smoothed_y = None

    def _update_finger_cursor(
        self,
        tracking_tip: Any,
        is_overridden: bool,
        current_mouse_pos: Optional[Tuple[float, float]],
        screen_w: int,
        screen_h: int,
        now: float,
    ) -> Tuple[float, float]:
        """Calculates smooth cursor position from tracking finger tip (middle finger in 3-finger mode, index finger otherwise)."""
        curr_finger = (tracking_tip.x, tracking_tip.y)
        speed = getattr(self.config, "hand_cursor_speed", 2.0)
        smart_sens_enabled = getattr(self.config, "hand_smart_sensitivity", True)

        beta = getattr(self.config, "filter_beta", 0.007)
        min_cutoff = max(0.2, 2.5 * (1.0 - getattr(self.config, "smoothing", 0.65)))

        with self._lock:
            if is_overridden and current_mouse_pos is not None:
                curr_x, curr_y = current_mouse_pos
                self.accum_screen_x = float(curr_x)
                self.accum_screen_y = float(curr_y)
                dx_raw = (tracking_tip.x - 0.5) * speed
                dy_raw = (tracking_tip.y - 0.5) * speed
                self.screen_origin_x = float(curr_x) - (dx_raw * screen_w)
                self.screen_origin_y = float(curr_y) - (dy_raw * screen_h)
                self.smoothed_x = float(curr_x)
                self.smoothed_y = float(curr_y)
                self.prev_finger_pos = curr_finger
                self.prev_finger_time = now
                self.filtered_velocity = 0.0
                if self.filter_x is not None:
                    self.filter_x.reset(now, float(curr_x))
                if self.filter_y is not None:
                    self.filter_y.reset(now, float(curr_y))
                return (self.smoothed_x, self.smoothed_y)
            elif smart_sens_enabled:
                # --- In-App Smart Sensitivity with Dynamic Ballistics ---
                # 1. Cold-start / Reacquisition initialization
                if self.prev_finger_pos is None or self.accum_screen_x is None:
                    self.prev_finger_pos = curr_finger
                    self.prev_finger_time = now
                    self.filtered_velocity = 0.0
                    if current_mouse_pos is not None:
                        self.accum_screen_x = float(current_mouse_pos[0])
                        self.accum_screen_y = float(current_mouse_pos[1])
                    else:
                        self.accum_screen_x = screen_w / 2.0
                        self.accum_screen_y = screen_h / 2.0
                    raw_screen_x = self.accum_screen_x
                    raw_screen_y = self.accum_screen_y
                    self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                    self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                    self.smoothed_x = raw_screen_x
                    self.smoothed_y = raw_screen_y
                else:
                    dt = now - self.prev_finger_time
                    delta_x = curr_finger[0] - self.prev_finger_pos[0]
                    delta_y = curr_finger[1] - self.prev_finger_pos[1]
                    delta_dist = math.hypot(delta_x, delta_y)

                    # 2. Frame Hitch Guard & Delta Ceiling Clamp
                    if dt > 0.15 or dt <= 0.001 or delta_dist > 0.12:
                        self.prev_finger_pos = curr_finger
                        self.prev_finger_time = now
                        self.filtered_velocity = 0.0
                        raw_screen_x = self.accum_screen_x
                        raw_screen_y = self.accum_screen_y
                    else:
                        # 3. Continuous Soft Deadzone Ramp
                        deadband = getattr(self.config, "hand_tremor_deadband", 0.0015)
                        shaped_dx, shaped_dy, eff_mag = self._apply_continuous_soft_deadzone(
                            delta_x, delta_y, deadband
                        )

                        # 4. Instantaneous Velocity & 1st-order Low-Pass Filter (EMA)
                        v_instant = eff_mag / dt
                        self.filtered_velocity = 0.35 * v_instant + 0.65 * self.filtered_velocity

                        # 5. Dynamic Gain from Ballistic Curve
                        gain = self._compute_dynamic_gain(self.filtered_velocity)

                        # 6. Scaled Delta Integration
                        scale_factor = speed * gain
                        delta_screen_x = shaped_dx * scale_factor * screen_w
                        delta_screen_y = shaped_dy * scale_factor * screen_h

                        # 7. Integrator Anti-Windup Clamping
                        self.accum_screen_x = max(0.0, min(screen_w - 1.0, self.accum_screen_x + delta_screen_x))
                        self.accum_screen_y = max(0.0, min(screen_h - 1.0, self.accum_screen_y + delta_screen_y))

                        raw_screen_x = self.accum_screen_x
                        raw_screen_y = self.accum_screen_y

                        self.prev_finger_pos = curr_finger
                        self.prev_finger_time = now

                    # 8. Adaptive OneEuroFilter smoothing
                    if self.filter_x is None or self.filter_y is None:
                        self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                        self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                        self.smoothed_x = raw_screen_x
                        self.smoothed_y = raw_screen_y
                    else:
                        self.smoothed_x = self.filter_x.filter(now, raw_screen_x)
                        self.smoothed_y = self.filter_y.filter(now, raw_screen_y)

                return (self.smoothed_x, self.smoothed_y)
            else:
                # --- Linear Raw Mode Bypass ---
                dx_raw = (idx_tip.x - 0.5) * speed
                dy_raw = (idx_tip.y - 0.5) * speed
                if self.screen_origin_x is None or self.screen_origin_y is None:
                    if current_mouse_pos is not None:
                        self.screen_origin_x = float(current_mouse_pos[0]) - (dx_raw * screen_w)
                        self.screen_origin_y = float(current_mouse_pos[1]) - (dy_raw * screen_h)
                    else:
                        self.screen_origin_x = screen_w / 2.0
                        self.screen_origin_y = screen_h / 2.0

                raw_screen_x = self.screen_origin_x + (dx_raw * screen_w)
                raw_screen_y = self.screen_origin_y + (dy_raw * screen_h)
                raw_screen_x = max(0.0, min(screen_w - 1.0, raw_screen_x))
                raw_screen_y = max(0.0, min(screen_h - 1.0, raw_screen_y))

                if self.filter_x is None or self.filter_y is None:
                    self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                    self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                    self.smoothed_x = raw_screen_x
                    self.smoothed_y = raw_screen_y
                else:
                    self.smoothed_x = self.filter_x.filter(now, raw_screen_x)
                    self.smoothed_y = self.filter_y.filter(now, raw_screen_y)

                return (self.smoothed_x, self.smoothed_y)

    def process_frame(
        self,
        frame: np.ndarray,
        screen_w: int,
        screen_h: int,
        current_mouse_pos: Optional[Tuple[float, float]] = None,
        is_overridden: bool = False,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[str], np.ndarray, Dict[str, Any]]:
        """Processes RGB video frame for hand tracking.
        Returns:
            - target_cursor_pos: (x, y) if finger pointer mode active, else None
            - detected_gesture: confirmed gesture name or None
            - annotated_frame: frame with visual skeleton HUD
            - telemetry: dictionary containing hand metrics
        """
        if not getattr(self.config, "hand_tracking_enabled", True):
            return None, None, frame, {
                "hand_detected": False,
                "handedness": None,
                "hand_gesture": None,
                "pinch_distance": 1.0,
            }

        # MediaPipe expects RGB frame
        if frame.shape[2] == 3 and not hasattr(frame, "_is_rgb"):
            # Check if input is already RGB or BGR
            # In our pipeline, frame is usually BGR from cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        target_cursor_pos = None
        confirmed_gesture = None
        effective_gesture = None
        handedness_label = None
        pinch_dist = 1.0

        if results.multi_hand_landmarks:
            is_mirrored = getattr(self.config, "flip_horizontal", True)
            right_only = getattr(self.config, "hand_right_only", True)

            chosen_idx = None
            detected_handedness = []

            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                raw_label = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    raw_label = results.multi_handedness[idx].classification[0].label
                # In MediaPipe Hands, label assumes mirrored selfie input.
                # If mirrored, raw_label is physical hand. If un-mirrored, inverted.
                phys_hand = raw_label if is_mirrored else ("Right" if raw_label == "Left" else "Left")
                detected_handedness.append(phys_hand)
                if right_only and phys_hand == "Right" and chosen_idx is None:
                    chosen_idx = idx

            if right_only:
                if chosen_idx is not None:
                    # Right hand actively found among detected hands
                    landmarks = results.multi_hand_landmarks[chosen_idx].landmark
                    handedness_label = "Right"
                    self.latest_handedness = "Right"
                else:
                    # Only Left hand(s) detected -> Ignore control & gestures
                    ignored_hand = detected_handedness[0] if detected_handedness else "Left"
                    self.latest_handedness = f"{ignored_hand} (Ignored)"
                    with self._lock:
                        if self.thumb_state == "HOLDING_DRAG":
                            effective_gesture = "Release Drag"
                        self.thumb_state = "IDLE"
                        self.thumb_opened_time = None
                        self.thumb_is_open = False
                        self.latest_thumb_status = "IGNORED"
                        self.candidate_gesture = None
                        self.gesture_frame_count = 0
                        self.latest_gesture = None
                        self.filter_x = None
                        self.filter_y = None
                        self.smoothed_x = None
                        self.smoothed_y = None
                        self.screen_origin_x = None
                        self.screen_origin_y = None
                        self.prev_finger_pos = None
                        self.prev_finger_time = 0.0
                        self.filtered_velocity = 0.0
                        self.accum_screen_x = None
                        self.accum_screen_y = None

                    # Draw visual skeleton in muted/ignored style
                    first_landmarks = results.multi_hand_landmarks[0].landmark
                    frame = self._draw_hand_hud(frame, first_landmarks, None, f"{ignored_hand} (Ignored)", is_ignored=True)
                    telemetry = {
                        "hand_detected": True,
                        "hand_ignored": True,
                        "handedness": f"{ignored_hand} (Ignored)",
                        "hand_gesture": None,
                        "pinch_distance": 1.0,
                        "thumb_status": "IGNORED",
                        "thumb_is_open": False,
                    }
                    return None, effective_gesture, frame, telemetry
            else:
                # Any hand allowed
                chosen_idx = 0
                landmarks = results.multi_hand_landmarks[0].landmark
                handedness_label = detected_handedness[0] if detected_handedness else "Hand"
                self.latest_handedness = handedness_label

            wrist = landmarks[self.WRIST]
            palm_scale = max(1e-4, self._dist_2d(wrist, landmarks[self.MIDDLE_MCP]))

            # 1. Gesture Classification
            raw_gesture, pinch_dist = self._classify_gesture(landmarks)
            self.latest_pinch_dist = pinch_dist

            # 2. Thumb Openness Metric & Hysteresis
            thumb_open_ratio = self._dist_2d(landmarks[self.THUMB_TIP], landmarks[self.INDEX_MCP]) / palm_scale
            open_thresh = getattr(self.config, "hand_thumb_open_ratio", 0.65)
            close_thresh = getattr(self.config, "hand_thumb_close_ratio", 0.48)

            if not self.thumb_is_open:
                if thumb_open_ratio >= open_thresh:
                    self.thumb_is_open = True
            else:
                if thumb_open_ratio <= close_thresh:
                    self.thumb_is_open = False

            # 3. Thumb State Machine
            now = time.perf_counter()
            thumb_action = None
            thumb_trigger_enabled = getattr(self.config, "hand_thumb_trigger_enabled", True)

            if thumb_trigger_enabled:
                if raw_gesture in ("Fist", "Open Palm"):
                    # Whole hand open or fist cancels pending tap to prevent false clicks
                    if self.thumb_state == "HOLDING_DRAG":
                        thumb_action = "Release Drag"
                    self.thumb_state = "IDLE"
                    self.thumb_opened_time = None
                elif self.thumb_is_open and self.thumb_state == "IDLE":
                    self.thumb_opened_time = now
                    self.thumb_state = "PENDING_TAP"
                elif self.thumb_is_open and self.thumb_state == "PENDING_TAP":
                    hold_thresh = getattr(self.config, "hand_thumb_hold_threshold", 1.0)
                    if (now - (self.thumb_opened_time or now)) >= hold_thresh:
                        self.thumb_state = "HOLDING_DRAG"
                        thumb_action = "Hold Left Click"
                elif self.thumb_is_open and self.thumb_state == "HOLDING_DRAG":
                    thumb_action = "Hold Left Click"
                elif not self.thumb_is_open and self.thumb_state in ("PENDING_TAP", "HOLDING_DRAG"):
                    if self.thumb_state == "PENDING_TAP":
                        thumb_action = "Left Click"
                    elif self.thumb_state == "HOLDING_DRAG":
                        thumb_action = "Release Drag"
                    self.thumb_state = "IDLE"
                    self.thumb_opened_time = None
            else:
                if self.thumb_state == "HOLDING_DRAG":
                    thumb_action = "Release Drag"
                self.thumb_state = "IDLE"
                self.thumb_opened_time = None

            self.latest_thumb_status = "DRAGGING" if self.thumb_state == "HOLDING_DRAG" else ("OPEN" if self.thumb_is_open else "TUCKED")

            # 4. Temporal Confirmation Buffer for other gestures
            if raw_gesture:
                if self.candidate_gesture == raw_gesture:
                    self.gesture_frame_count += 1
                else:
                    self.candidate_gesture = raw_gesture
                    self.gesture_frame_count = 1

                if self.gesture_frame_count >= self.REQUIRED_GESTURE_FRAMES:
                    confirmed_gesture = raw_gesture
            else:
                self.candidate_gesture = None
                self.gesture_frame_count = 0
                confirmed_gesture = None

            # Effective gesture prioritization
            if thumb_trigger_enabled:
                effective_gesture = thumb_action
            else:
                effective_gesture = confirmed_gesture

            self.latest_gesture = effective_gesture or confirmed_gesture

            # 5. Finger Pointer Calculation (if pointer_source is "finger")
            pointer_src = getattr(self.config, "pointer_source", "finger")
            if pointer_src == "finger" and getattr(self.config, "pointer_enabled", True):
                idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, wrist)
                mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, wrist)
                ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, wrist)
                pinky_ext = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, wrist)

                is_3finger = getattr(self.config, "control_preset", "") == "3-Finger Control"

                if is_3finger:
                    # 3-Finger Control Preset:
                    if raw_gesture in ("3-Finger Left Click", "3-Finger Right Click (Ring)", "3-Finger Right Click (Middle)"):
                        # Clicking action: anchor cursor at current position to prevent wobble while curling finger
                        if self.smoothed_x is not None and self.smoothed_y is not None:
                            target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                        elif current_mouse_pos is not None:
                            target_cursor_pos = (float(current_mouse_pos[0]), float(current_mouse_pos[1]))
                        else:
                            target_cursor_pos = None
                    elif raw_gesture in ("Fist", "Open Palm") or not (idx_ext and mid_ext and ring_ext and not pinky_ext):
                        # All 3 folded (Fist), Open Palm (all 4 open), or non-moving postures: freeze cursor
                        target_cursor_pos = None
                        with self._lock:
                            self.screen_origin_x = None
                            self.screen_origin_y = None
                            self.prev_finger_pos = None
                            self.prev_finger_time = 0.0
                            self.filtered_velocity = 0.0
                            self.accum_screen_x = None
                            self.accum_screen_y = None
                    else:
                        # 3 fingers extended: move cursor smoothly using index fingertip below
                        target_cursor_pos = "PROCEED_TO_MOVE"
                else:
                    # Freeze cursor if:
                    # 1. Fist (all fingers folded)
                    # 2. Open Palm (all fingers shown)
                    # 3. Multi-finger open (middle & ring extended)
                    # 4. Index finger not extended
                    if raw_gesture in ("Fist", "Open Palm") or (mid_ext and ring_ext) or (not idx_ext):
                        # Freeze cursor and clear origin anchor & accumulator
                        target_cursor_pos = None
                        with self._lock:
                            self.screen_origin_x = None
                            self.screen_origin_y = None
                            self.prev_finger_pos = None
                            self.prev_finger_time = 0.0
                            self.filtered_velocity = 0.0
                            self.accum_screen_x = None
                            self.accum_screen_y = None
                    elif confirmed_gesture == "Peace Sign" or (raw_gesture == "Peace Sign" and getattr(self.config, "hand_peace_action", "None") != "None"):
                        # Peace Sign triggers click: anchor cursor at current position to prevent wobble
                        if self.smoothed_x is not None and self.smoothed_y is not None:
                            target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                        elif current_mouse_pos is not None:
                            target_cursor_pos = (float(current_mouse_pos[0]), float(current_mouse_pos[1]))
                        else:
                            target_cursor_pos = None

                if target_cursor_pos == "PROCEED_TO_MOVE" or (not is_3finger and idx_ext and not mid_ext and not ring_ext and not pinky_ext):
                    # Pointing / Move cursor smoothly:
                    # In 3-Finger Control mode, track middle fingertip to prevent cursor drift when curling index finger to click.
                    tracking_tip = landmarks[self.MIDDLE_TIP] if is_3finger else landmarks[self.INDEX_TIP]
                    curr_finger = (tracking_tip.x, tracking_tip.y)
                    speed = getattr(self.config, "hand_cursor_speed", 2.0)
                    smart_sens_enabled = getattr(self.config, "hand_smart_sensitivity", True)

                    beta = getattr(self.config, "filter_beta", 0.007)
                    min_cutoff = max(0.2, 2.5 * (1.0 - getattr(self.config, "smoothing", 0.65)))

                    with self._lock:
                        if is_overridden and current_mouse_pos is not None:
                            curr_x, curr_y = current_mouse_pos
                            self.accum_screen_x = float(curr_x)
                            self.accum_screen_y = float(curr_y)
                            dx_raw = (tracking_tip.x - 0.5) * speed
                            dy_raw = (tracking_tip.y - 0.5) * speed
                            self.screen_origin_x = float(curr_x) - (dx_raw * screen_w)
                            self.screen_origin_y = float(curr_y) - (dy_raw * screen_h)
                            self.smoothed_x = float(curr_x)
                            self.smoothed_y = float(curr_y)
                            self.prev_finger_pos = curr_finger
                            self.prev_finger_time = now
                            self.filtered_velocity = 0.0
                            if self.filter_x is not None:
                                self.filter_x.reset(now, float(curr_x))
                            if self.filter_y is not None:
                                self.filter_y.reset(now, float(curr_y))
                            target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                        elif smart_sens_enabled:
                            # --- In-App Smart Sensitivity with Dynamic Ballistics ---
                            # 1. Cold-start / Reacquisition initialization
                            if self.prev_finger_pos is None or self.accum_screen_x is None:
                                self.prev_finger_pos = curr_finger
                                self.prev_finger_time = now
                                self.filtered_velocity = 0.0
                                if current_mouse_pos is not None:
                                    self.accum_screen_x = float(current_mouse_pos[0])
                                    self.accum_screen_y = float(current_mouse_pos[1])
                                else:
                                    self.accum_screen_x = screen_w / 2.0
                                    self.accum_screen_y = screen_h / 2.0
                                raw_screen_x = self.accum_screen_x
                                raw_screen_y = self.accum_screen_y
                                self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                                self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                                self.smoothed_x = raw_screen_x
                                self.smoothed_y = raw_screen_y
                            else:
                                dt = now - self.prev_finger_time
                                delta_x = curr_finger[0] - self.prev_finger_pos[0]
                                delta_y = curr_finger[1] - self.prev_finger_pos[1]
                                delta_dist = math.hypot(delta_x, delta_y)

                                # 2. Frame Hitch Guard & Delta Ceiling Clamp
                                if dt > 0.15 or dt <= 0.001 or delta_dist > 0.12:
                                    self.prev_finger_pos = curr_finger
                                    self.prev_finger_time = now
                                    self.filtered_velocity = 0.0
                                    raw_screen_x = self.accum_screen_x
                                    raw_screen_y = self.accum_screen_y
                                else:
                                    # 3. Continuous Soft Deadzone Ramp
                                    deadband = getattr(self.config, "hand_tremor_deadband", 0.0015)
                                    shaped_dx, shaped_dy, eff_mag = self._apply_continuous_soft_deadzone(
                                        delta_x, delta_y, deadband
                                    )

                                    # 4. Instantaneous Velocity & 1st-order Low-Pass Filter (EMA)
                                    v_instant = eff_mag / dt
                                    self.filtered_velocity = 0.35 * v_instant + 0.65 * self.filtered_velocity

                                    # 5. Dynamic Gain from Ballistic Curve
                                    gain = self._compute_dynamic_gain(self.filtered_velocity)

                                    # 6. Scaled Delta Integration
                                    scale_factor = speed * gain
                                    delta_screen_x = shaped_dx * scale_factor * screen_w
                                    delta_screen_y = shaped_dy * scale_factor * screen_h

                                    # 7. Integrator Anti-Windup Clamping
                                    self.accum_screen_x = max(0.0, min(screen_w - 1.0, self.accum_screen_x + delta_screen_x))
                                    self.accum_screen_y = max(0.0, min(screen_h - 1.0, self.accum_screen_y + delta_screen_y))

                                    raw_screen_x = self.accum_screen_x
                                    raw_screen_y = self.accum_screen_y

                                    self.prev_finger_pos = curr_finger
                                    self.prev_finger_time = now

                                    # 8. Adaptive OneEuroFilter smoothing
                                    if self.filter_x is None or self.filter_y is None:
                                        self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                                        self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                                        self.smoothed_x = raw_screen_x
                                        self.smoothed_y = raw_screen_y
                                    else:
                                        self.smoothed_x = self.filter_x.filter(now, raw_screen_x)
                                        self.smoothed_y = self.filter_y.filter(now, raw_screen_y)

                            target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                        else:
                            # --- Linear Raw Mode Bypass ---
                            dx_raw = (tracking_tip.x - 0.5) * speed
                            dy_raw = (tracking_tip.y - 0.5) * speed
                            if self.screen_origin_x is None or self.screen_origin_y is None:
                                if current_mouse_pos is not None:
                                    self.screen_origin_x = float(current_mouse_pos[0]) - (dx_raw * screen_w)
                                    self.screen_origin_y = float(current_mouse_pos[1]) - (dy_raw * screen_h)
                                else:
                                    self.screen_origin_x = screen_w / 2.0
                                    self.screen_origin_y = screen_h / 2.0

                            raw_screen_x = self.screen_origin_x + (dx_raw * screen_w)
                            raw_screen_y = self.screen_origin_y + (dy_raw * screen_h)
                            raw_screen_x = max(0.0, min(screen_w - 1.0, raw_screen_x))
                            raw_screen_y = max(0.0, min(screen_h - 1.0, raw_screen_y))

                            if self.filter_x is None or self.filter_y is None:
                                self.filter_x = OneEuroFilter(now, raw_screen_x, min_cutoff=min_cutoff, beta=beta)
                                self.filter_y = OneEuroFilter(now, raw_screen_y, min_cutoff=min_cutoff, beta=beta)
                                self.smoothed_x = raw_screen_x
                                self.smoothed_y = raw_screen_y
                            else:
                                self.smoothed_x = self.filter_x.filter(now, raw_screen_x)
                                self.smoothed_y = self.filter_y.filter(now, raw_screen_y)

                            target_cursor_pos = (self.smoothed_x, self.smoothed_y)
                elif target_cursor_pos is None:
                    # Non-pointing pose: freeze cursor
                    with self._lock:
                        self.screen_origin_x = None
                        self.screen_origin_y = None
                        self.prev_finger_pos = None
                        self.prev_finger_time = 0.0
                        self.filtered_velocity = 0.0
                        self.accum_screen_x = None
                        self.accum_screen_y = None

            # 6. Draw Hand Skeleton & Visual Annotations
            frame = self._draw_hand_hud(frame, landmarks, effective_gesture or confirmed_gesture, handedness_label)
        else:
            with self._lock:
                # Tracking Loss Guard: emit Release Drag if holding drag, reset state cleanly
                if self.thumb_state == "HOLDING_DRAG":
                    effective_gesture = "Release Drag"
                self.thumb_state = "IDLE"
                self.thumb_opened_time = None
                self.thumb_is_open = False
                self.latest_thumb_status = "NONE"
                self.candidate_gesture = None
                self.gesture_frame_count = 0
                self.latest_gesture = None
                self.latest_handedness = None
                self.filter_x = None
                self.filter_y = None
                self.smoothed_x = None
                self.smoothed_y = None
                self.screen_origin_x = None
                self.screen_origin_y = None
                self.prev_finger_pos = None
                self.prev_finger_time = 0.0
                self.filtered_velocity = 0.0
                self.accum_screen_x = None
                self.accum_screen_y = None

        telemetry = {
            "hand_detected": bool(results.multi_hand_landmarks),
            "handedness": handedness_label,
            "hand_gesture": effective_gesture or confirmed_gesture,
            "pinch_distance": pinch_dist,
            "thumb_status": self.latest_thumb_status,
            "thumb_is_open": self.thumb_is_open,
        }

        return target_cursor_pos, effective_gesture, frame, telemetry

    def _draw_hand_hud(
        self,
        frame: np.ndarray,
        landmarks: List[Any],
        gesture: Optional[str],
        handedness: Optional[str],
        is_ignored: bool = False,
    ) -> np.ndarray:
        """Draws aesthetic hand skeletal lines, joints, and status banner."""
        h, w, _ = frame.shape
        points = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        # Draw bones / connections
        bone_color = (60, 60, 80) if is_ignored else (80, 230, 110)
        for p1_idx, p2_idx in self.CONNECTIONS:
            pt1 = points[p1_idx]
            pt2 = points[p2_idx]
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # Draw joints
        for idx, (px, py) in enumerate(points):
            if is_ignored:
                cv2.circle(frame, (px, py), 3, (100, 100, 130), -1)
            elif idx in (self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP):
                # Highlight active fingertips
                cv2.circle(frame, (px, py), 5, (0, 242, 254), -1)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
            else:
                cv2.circle(frame, (px, py), 3, (0, 180, 255), -1)

        # Draw gesture banner above wrist
        wrist_pt = points[self.WRIST]
        text_x = max(10, min(w - 220, wrist_pt[0] - 60))
        text_y = max(30, wrist_pt[1] - 25)

        if is_ignored:
            label_text = f"{handedness}: IGNORED"
            bg_color = (30, 30, 120)  # Dark reddish for ignored
            text_color = (130, 180, 250)
        else:
            thumb_info = f" | T:{self.latest_thumb_status}" if getattr(self.config, "hand_thumb_trigger_enabled", True) else ""
            label_text = f"{handedness or 'HAND'}: {gesture if gesture else 'POINT'}{thumb_info}"
            if self.latest_thumb_status == "DRAGGING":
                bg_color = (20, 80, 180)  # Orange/Gold for active drag
            elif gesture:
                bg_color = (20, 80, 20)   # Green for gesture action
            else:
                bg_color = (40, 40, 40)   # Dark gray for resting pointing
            text_color = (0, 255, 180) if (gesture or self.latest_thumb_status != "TUCKED") else (200, 200, 200)

        cv2.rectangle(frame, (text_x - 6, text_y - 20), (text_x + len(label_text) * 10 + 10, text_y + 6), bg_color, -1)
        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

        return frame

    def close(self):
        """Releases MediaPipe Hands resources."""
        with self._lock:
            if hasattr(self, "hands") and self.hands:
                self.hands.close()

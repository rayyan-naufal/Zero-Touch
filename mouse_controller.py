"""Mouse controller module for Zero Touch.
Provides low-latency cursor positioning, click trigger debouncing,
drag-and-drop state management, and thread-safe pause toggling.
"""

import ctypes
import math
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple
import pyautogui

from config import AppConfig

# Disable PyAutoGUI delay for instantaneous response
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# Windows Win32 API constants for direct cursor control
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    user32 = ctypes.windll.user32
    try:
        import winsound
    except ImportError:
        winsound = None

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
else:
    winsound = None


class MouseController:
    """Handles pointer motion, click actions, debouncing, and drag states."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()

        # Thread-safe states (Reentrant Lock for atomic nested calls)
        self._lock = threading.RLock()
        self.is_paused = False  # Master pause (both pointer & clicking)
        self.is_pointer_paused = False  # Head pointer pause only
        self.is_clicking_paused = False  # Click actions pause only
        self.is_dragging = False
        self.active_hand_gesture: Optional[str] = None
        self.is_hand_holding_drag: bool = False

        # Debouncing and timing timestamps
        self.last_action_time = 0.0
        self.active_hold_gesture: Optional[str] = None
        self.hold_start_time = 0.0
        self.pending_action: Optional[str] = None

        # Dwell Control State Machine (macOS Hover-to-Click)
        self.dwell_anchor: Optional[Tuple[float, float]] = None
        self.dwell_start_time: Optional[float] = None
        self.dwell_triggered: bool = False

        # Physical mouse / trackpad override state
        self.last_physical_mouse_time = 0.0
        self.is_overridden = False
        self.physical_move_threshold = 4.0  # Pixels movement to consider intentional user input
        self.last_programmatic_pos: Optional[Tuple[int, int]] = None
        try:
            self.last_programmatic_pos = self.get_physical_cursor_pos()
        except Exception:
            self.last_programmatic_pos = None

        # Optional GUI notification callback: callback(action_name: str)
        self.action_callback: Optional[Callable[[str], None]] = None

    def set_action_callback(self, callback: Callable[[str], None]):
        """Sets a callback function to notify GUI when an action is triggered."""
        self.action_callback = callback

    def get_physical_cursor_pos(self) -> Tuple[int, int]:
        """Queries the current operating system cursor position."""
        if IS_WINDOWS:
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            return int(pt.x), int(pt.y)
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def check_physical_mouse_override(self, cached_pos: Optional[Tuple[int, int]] = None) -> bool:
        """Detects physical mouse/trackpad movement and manages override state.
        Returns True if head tracking pointer is currently suppressed by physical mouse.
        """
        if not getattr(self.config, "mouse_override", True):
            if self.is_overridden:
                self.is_overridden = False
            return False

        now = time.perf_counter()
        curr_x, curr_y = cached_pos if cached_pos is not None else self.get_physical_cursor_pos()

        if self.last_programmatic_pos is not None:
            dx = curr_x - self.last_programmatic_pos[0]
            dy = curr_y - self.last_programmatic_pos[1]
            dist = math.hypot(dx, dy)
            if dist >= self.physical_move_threshold:
                # Deliberate user physical mouse / trackpad movement detected
                self.last_physical_mouse_time = now
                if not self.is_overridden:
                    self.is_overridden = True
                    if self.action_callback:
                        self.action_callback("MOUSE OVERRIDE")

        if self.is_overridden:
            timeout = getattr(self.config, "override_timeout", 0.8)
            if (now - self.last_physical_mouse_time) < timeout:
                # Keep last programmatic pos synced with physical cursor so user can keep moving smoothly
                self.last_programmatic_pos = (curr_x, curr_y)
                return True
            else:
                # Physical override expired; return control to head tracking
                self.is_overridden = False
                self.last_programmatic_pos = (curr_x, curr_y)
                return False

        return False

    def update_override_state(self, cached_pos: Optional[Tuple[int, int]] = None) -> Tuple[bool, bool]:
        """Evaluates override status and detects if tracking just resumed.
        Returns: (is_currently_overridden, just_resumed)
        """
        was_overridden = self.is_overridden
        is_now_overridden = self.check_physical_mouse_override(cached_pos=cached_pos)
        just_resumed = was_overridden and not is_now_overridden
        return is_now_overridden, just_resumed

    def toggle_pause(self) -> bool:
        """Toggles master pause state. Returns new pause state."""
        with self._lock:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.is_overridden = False
                if self.is_dragging or self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self._reset_dwell_unlocked()
                self.active_hold_gesture = None
                self.pending_action = None
                self.active_hand_gesture = None
            status = self.is_paused

        if self.action_callback:
            self.action_callback("PAUSED" if status else "RESUMED")
        return status

    def set_paused(self, paused: bool):
        """Explicitly sets master pause state."""
        with self._lock:
            self.is_paused = paused
            if self.is_paused:
                self.is_overridden = False
                if self.is_dragging or self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self._reset_dwell_unlocked()
                self.active_hold_gesture = None
                self.pending_action = None
                self.active_hand_gesture = None

    def toggle_pointer_pause(self) -> bool:
        """Toggles head pointer tracking pause state independently. Returns new state."""
        with self._lock:
            self.is_pointer_paused = not self.is_pointer_paused
            if self.is_pointer_paused:
                self.is_overridden = False
            status = self.is_pointer_paused

        if self.action_callback:
            self.action_callback("POINTER PAUSED" if status else "POINTER ACTIVE")
        return status

    def set_pointer_paused(self, paused: bool):
        """Explicitly sets head pointer pause state."""
        with self._lock:
            self.is_pointer_paused = paused
            if self.is_pointer_paused:
                self.is_overridden = False

    def toggle_clicking_pause(self) -> bool:
        """Toggles clicking actions pause state independently. Returns new state."""
        with self._lock:
            self.is_clicking_paused = not self.is_clicking_paused
            if self.is_clicking_paused:
                if self.is_dragging or self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self._reset_dwell_unlocked()
                self.active_hold_gesture = None
                self.pending_action = None
                self.active_hand_gesture = None
            status = self.is_clicking_paused

        if self.action_callback:
            self.action_callback("CLICKS PAUSED" if status else "CLICKS ACTIVE")
        return status

    def set_clicking_paused(self, paused: bool):
        """Explicitly sets clicking pause state."""
        with self._lock:
            self.is_clicking_paused = paused
            if self.is_clicking_paused:
                if self.is_dragging or self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self._reset_dwell_unlocked()
                self.active_hold_gesture = None
                self.pending_action = None
                self.active_hand_gesture = None

    def reset_dwell(self):
        """Resets dwell timer, anchor, and trigger state safely."""
        with self._lock:
            self._reset_dwell_unlocked()

    def _reset_dwell_unlocked(self):
        self.dwell_anchor = None
        self.dwell_start_time = None
        self.dwell_triggered = False

    def check_dwell(self, cached_pos: Optional[Tuple[int, int]] = None) -> Tuple[Optional[str], float]:
        """Evaluates stationary dwell timing using actual system cursor coordinates.
        Returns:
            (action_triggered_or_None, dwell_progress_ratio)
            progress_ratio: 0.0 to 1.0 (for live HUD feedback)
        """
        with self._lock:
            click_mode = getattr(self.config, "click_mode", "gestures").lower()
            if (
                self.is_paused
                or self.is_clicking_paused
                or not getattr(self.config, "clicking_enabled", True)
                or not getattr(self.config, "dwell_enabled", False)
                or click_mode == "gestures"  # Issue 2: Bypass dwell when click_mode is gestures only
                or self.is_dragging  # Absolute suppression during drag operations
                or self.is_hand_holding_drag
            ):
                self._reset_dwell_unlocked()
                return None, 0.0

            if cached_pos is not None:
                curr_x, curr_y = cached_pos
            else:
                curr_x, curr_y = self.get_physical_cursor_pos()
            now = time.perf_counter()

            # Safe initialization on first check or after resume
            if self.dwell_anchor is None or self.dwell_start_time is None:
                self.dwell_anchor = (float(curr_x), float(curr_y))
                self.dwell_start_time = now
                self.dwell_triggered = False
                return None, 0.0

            dist = math.hypot(curr_x - self.dwell_anchor[0], curr_y - self.dwell_anchor[1])
            dwell_radius = getattr(self.config, "dwell_radius", 20)

            # Cursor shifted beyond tolerance radius -> reset anchor and timer
            if dist > dwell_radius:
                self.dwell_anchor = (float(curr_x), float(curr_y))
                self.dwell_start_time = now
                self.dwell_triggered = False
                return None, 0.0

            # Cursor remains stationary within tolerance radius
            elapsed = now - self.dwell_start_time
            dwell_time = max(0.2, getattr(self.config, "dwell_time", 0.8))
            progress = min(1.0, elapsed / dwell_time)

            if not self.dwell_triggered:
                if elapsed >= dwell_time:
                    # Latch trigger: execute action and lock out until cursor moves out
                    self.dwell_triggered = True
                    action = getattr(self.config, "dwell_action", "Left Click")
                    self._execute_action_unlocked(action)
                    return action, 1.0
                return None, progress
            else:
                # Already fired, wait for user to move outside dwell radius
                return None, 1.0

    def trigger_switch_action(self, switch_action: str) -> bool:
        """Executes an assistive keyboard switch action (macOS Alternate Actions)."""
        with self._lock:
            if (
                self.is_paused
                or self.is_clicking_paused
                or not getattr(self.config, "clicking_enabled", True)
            ):
                return False

            self._reset_dwell_unlocked()

            if switch_action == "left":
                self._execute_action_unlocked("Left Click")
                return True
            elif switch_action == "right":
                self._execute_action_unlocked("Right Click")
                return True
            elif switch_action == "drag":
                if self.is_dragging:
                    self._release_drag_unlocked()
                else:
                    self._start_drag_unlocked()
                return True
            elif switch_action == "double":
                self._execute_action_unlocked("Double Click")
                return True
            elif switch_action == "middle":
                self._execute_action_unlocked("Middle Click")
                return True
        return False

    def move_to(self, x: float, y: float, cached_pos: Optional[Tuple[int, int]] = None) -> bool:
        """Moves cursor to absolute screen coordinates (clamped).
        Suppressed if paused, head pointer disabled/paused, or physical mouse is active.
        Returns True if cursor moved, False if suppressed.
        """
        if (
            self.is_paused
            or self.is_pointer_paused
            or not getattr(self.config, "pointer_enabled", True)
        ):
            return False

        if self.check_physical_mouse_override(cached_pos=cached_pos):
            return False

        clamped_x = max(0, min(self.screen_width - 1, int(x)))
        clamped_y = max(0, min(self.screen_height - 1, int(y)))

        # Skip redundant OS API dispatch if position is identical to last programmatic pos
        if self.last_programmatic_pos is not None:
            if self.last_programmatic_pos[0] == clamped_x and self.last_programmatic_pos[1] == clamped_y:
                return True

        if IS_WINDOWS:
            user32.SetCursorPos(clamped_x, clamped_y)
        else:
            pyautogui.moveTo(clamped_x, clamped_y)

        self.last_programmatic_pos = (clamped_x, clamped_y)
        return True

    def move_rel(self, dx: float, dy: float, cached_pos: Optional[Tuple[int, int]] = None) -> bool:
        """Moves cursor relative to current position.
        Suppressed if paused, head pointer disabled/paused, or physical mouse is active.
        Returns True if cursor moved, False if suppressed.
        """
        if (
            self.is_paused
            or self.is_pointer_paused
            or not getattr(self.config, "pointer_enabled", True)
        ):
            return False

        if self.check_physical_mouse_override(cached_pos=cached_pos):
            return False

        current_x, current_y = cached_pos if cached_pos is not None else self.get_physical_cursor_pos()
        new_x = max(0, min(self.screen_width - 1, int(current_x + dx)))
        new_y = max(0, min(self.screen_height - 1, int(current_y + dy)))

        if IS_WINDOWS:
            user32.SetCursorPos(new_x, new_y)
        else:
            pyautogui.moveTo(new_x, new_y)

        self.last_programmatic_pos = (new_x, new_y)
        return True

    def process_gesture(self, detected_gesture: Optional[str]):
        """Evaluates detected facial gesture using a robust state machine:
        - Brief gesture release triggers a single click.
        - Sustained gesture (> drag_hold_duration) triggers click-and-drag.
        - Releasing after drag drops the item without firing extra clicks.
        """
        with self._lock:
            click_mode = getattr(self.config, "click_mode", "gestures").lower()
            if (
                self.is_paused
                or self.is_clicking_paused
                or not getattr(self.config, "clicking_enabled", True)
                or click_mode == "dwell"  # Issue 2: Bypass gestures when click_mode is dwell only
            ):
                return

            now = time.perf_counter()

            # Handle release or disappearance of gesture
            if detected_gesture is None:
                if self.active_hold_gesture is not None:
                    hold_duration = now - self.hold_start_time
                    if self.is_dragging:
                        self._release_drag_unlocked()
                        self.last_action_time = now
                    elif self.pending_action is not None:
                        # Fire single action on release if cooldown elapsed and within max gesture hold
                        if hold_duration <= self.config.gesture_timeout and (now - self.last_action_time) >= self.config.click_cooldown:
                            self._execute_action_unlocked(self.pending_action)
                            self.last_action_time = now
                        self.pending_action = None

                    self.active_hold_gesture = None
                    self.hold_start_time = 0.0
                return

            # Map current gesture to configured action
            action = self._map_gesture_to_action(detected_gesture)
            if not action or action == "None":
                # If gesture maps to None, treat as release
                self.process_gesture(None)
                return

            # Reset dwell to avoid conflicting triggers
            self._reset_dwell_unlocked()

            # New gesture began
            if self.active_hold_gesture != detected_gesture:
                if self.is_dragging:
                    self._release_drag_unlocked()
                self.active_hold_gesture = detected_gesture
                self.hold_start_time = now
                self.pending_action = action
            else:
                # Gesture is being held continuously
                hold_duration = now - self.hold_start_time
                if (
                    hold_duration >= self.config.drag_hold_duration
                    and not self.is_dragging
                    and action == "Left Click"
                ):
                    self._start_drag_unlocked()
                    self.pending_action = None  # Prevent firing a click when released after drag
                elif hold_duration > self.config.gesture_timeout and not self.is_dragging:
                    # Fatigue guard: cancel pending click if gesture held too long without drag
                    self.pending_action = None

    def _map_gesture_to_action(self, gesture: str) -> Optional[str]:
        if gesture == self.config.left_click_trigger:
            return "Left Click"
        elif gesture == self.config.right_click_trigger:
            return "Right Click"
        elif gesture == self.config.double_click_trigger:
            return "Double Click"
        elif gesture == self.config.middle_click_trigger:
            return "Middle Click"
        return None

    def release_drag(self):
        """Public thread-safe method to release any active drag lock (e.g. on preset change or pause)."""
        with self._lock:
            if self.is_dragging or self.is_hand_holding_drag:
                self._release_drag_unlocked()
                self.is_hand_holding_drag = False
            self.active_hand_gesture = None

    def _map_hand_gesture_to_action(self, gesture: str) -> Optional[str]:
        if gesture in ("Hold Left Click", "Release Drag", "Left Click", "Right Click", "Double Click", "Middle Click"):
            return gesture
        if gesture == "3-Finger Left Click":
            return "Left Click"
        elif gesture in ("3-Finger Right Click (Ring)", "3-Finger Right Click"):
            return "Right Click"
        elif gesture == "3-Finger Right Click (Middle)":
            return getattr(self.config, "hand_three_finger_middle_action", "Right Click")
        elif gesture in ("Open Palm", "3-Finger Move"):
            return "None"
        elif gesture == "Peace Sign":
            return getattr(self.config, "hand_peace_action", "Left Click")
        elif gesture == "Pinch Index":
            return getattr(self.config, "hand_pinch_index_action", "Hold Left Click")
        elif gesture == "Pinch Middle":
            return getattr(self.config, "hand_pinch_middle_action", "Right Click")
        elif gesture == "Pointing":
            return getattr(self.config, "hand_point_action", "None")
        elif gesture == "Fist":
            return getattr(self.config, "hand_fist_action", "None")
        return None

    def process_hand_gesture(self, detected_gesture: Optional[str]):
        """Evaluates detected hand gesture with support for Hold Left Click / Drag and discrete clicks."""
        with self._lock:
            click_mode = getattr(self.config, "click_mode", "gestures").lower()
            if (
                self.is_paused
                or self.is_clicking_paused
                or not getattr(self.config, "clicking_enabled", True)
                or not getattr(self.config, "hand_tracking_enabled", True)
                or click_mode == "dwell"
            ):
                if self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                self.active_hand_gesture = None
                return

            now = time.perf_counter()

            # When gesture is released / None
            if detected_gesture is None:
                if self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self.active_hand_gesture = None
                return

            action = self._map_hand_gesture_to_action(detected_gesture)
            if not action or action == "None":
                if self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self.active_hand_gesture = None
                return

            # Explicit Release Drag handling from thumb trigger
            if action in ("Release Drag", "Drop"):
                if self.is_hand_holding_drag:
                    self._release_drag_unlocked()
                    self.is_hand_holding_drag = False
                self.active_hand_gesture = None
                return

            # Reset dwell timer to prevent collision
            self._reset_dwell_unlocked()

            # Case A: Sustained Hold Action ("Hold Left Click")
            if action in ("Hold Left Click", "Drag", "Hold Click"):
                if not self.is_hand_holding_drag:
                    self._start_drag_unlocked()
                    self.is_hand_holding_drag = True
                self.active_hand_gesture = detected_gesture
                return

            # If transitioning from hold action to another gesture, release hold
            if self.is_hand_holding_drag:
                self._release_drag_unlocked()
                self.is_hand_holding_drag = False

            # Case B: Discrete Actions ("Left Click", "Right Click", "Double Click", "Middle Click")
            # Fire once when gesture begins (debounced with click_cooldown)
            if self.active_hand_gesture != detected_gesture:
                if (now - self.last_action_time) >= self.config.click_cooldown:
                    self._execute_action_unlocked(action)
                    self.last_action_time = now
                    self.active_hand_gesture = detected_gesture


    def _execute_action_unlocked(self, action: str):
        """Executes the specific mouse action while internal lock is already held."""
        if action == "Left Click":
            if IS_WINDOWS:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            else:
                pyautogui.click(button="left")
        elif action == "Right Click":
            if IS_WINDOWS:
                user32.mouse_event(MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            else:
                pyautogui.click(button="right")
        elif action == "Double Click":
            if IS_WINDOWS:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                def _dispatch_second_click():
                    time.sleep(0.05)
                    with self._lock:
                        if not self.is_paused and not self.is_clicking_paused:
                            user32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                threading.Thread(target=_dispatch_second_click, daemon=True).start()
            else:
                threading.Thread(target=pyautogui.doubleClick, daemon=True).start()
        elif action == "Middle Click":
            if IS_WINDOWS:
                user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN | MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            else:
                pyautogui.middleClick()

        self._play_feedback_sound()

        if self.action_callback:
            self.action_callback(action)

    def _execute_action(self, action: str):
        """Executes the specific mouse action with lock acquisition."""
        with self._lock:
            self._execute_action_unlocked(action)

    def _play_feedback_sound(self):
        """Plays non-blocking audio feedback if enabled (MessageBeep is natively async in Win32)."""
        if getattr(self.config, "audio_feedback", False) and winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass

    def _start_drag_unlocked(self):
        self.is_dragging = True
        if IS_WINDOWS:
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        else:
            pyautogui.mouseDown(button="left")
        if self.action_callback:
            self.action_callback("DRAG START")

    def _release_drag_unlocked(self):
        self.is_dragging = False
        self.is_hand_holding_drag = False
        if IS_WINDOWS:
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            pyautogui.mouseUp(button="left")
        if self.action_callback:
            self.action_callback("DRAG RELEASE")

    def _start_drag(self):
        """Initiates mouse drag state."""
        with self._lock:
            self._start_drag_unlocked()

    def _release_drag(self):
        """Concludes mouse drag state."""
        with self._lock:
            self._release_drag_unlocked()

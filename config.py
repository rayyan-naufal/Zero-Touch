"""Configuration management module for Zero Touch.
Handles persistent user preferences, gesture assignments, and sensitivity tuning.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class AppConfig:
    # UX Display Mode ("simple" or "advanced")
    ui_mode: str = "simple"  # "simple" (Ringkas) or "advanced" (Lanjutan)

    # Tracking pointer settings (Head Pointer Subsystem)
    pointer_enabled: bool = True  # Whether head tracking moves the cursor
    cursor_speed: float = 2.2
    smoothing: float = 0.65  # Base smoothing factor (0.0 to 1.0)
    filter_beta: float = 0.007  # Speed sensitivity coefficient for One Euro Filter (CHI 2012)
    deadzone: float = 0.010  # Normalized radius threshold for head stillness
    tracking_mode: str = "absolute"  # "absolute" or "relative"
    flip_horizontal: bool = True  # Mirror webcam feed for intuitive movement
    mouse_override: bool = True  # Automatically yield cursor control to physical mouse/trackpad
    override_timeout: float = 0.8  # Seconds of physical mouse inactivity before resuming head tracking

    # Clicking & Alternate Actions Subsystem
    clicking_enabled: bool = True  # Decoupled master toggle for all clicking actions
    click_mode: str = "gestures"  # "gestures", "dwell", "hybrid"

    # Gesture thresholds
    wink_threshold: float = 0.20  # EAR below this is considered winked
    mouth_threshold: float = 0.50  # MAR above this is considered open mouth
    smile_threshold: float = 0.45  # Smile metric threshold

    # Gesture mapping
    left_click_trigger: str = "Left Wink"
    right_click_trigger: str = "Right Wink"
    double_click_trigger: str = "None"
    middle_click_trigger: str = "Mouth Open"

    # Dwell Control Settings (macOS Hover-to-Click)
    dwell_enabled: bool = False
    dwell_time: float = 0.8  # Seconds stationary before triggering action
    dwell_radius: int = 20  # Pixel tolerance radius for stationary cursor
    dwell_action: str = "Left Click"  # "Left Click", "Right Click", "Double Click"

    # Assistive Switches (Physical Keyboard Hotkeys)
    hotkey_switch_left: str = ""  # e.g., "space" or "f1"
    hotkey_switch_right: str = ""  # e.g., "f2"
    hotkey_switch_drag: str = ""  # e.g., "f3"

    # Timing & Debouncing
    click_cooldown: float = 0.35  # Seconds between click events to avoid double firing
    drag_hold_duration: float = 0.65  # Hold gesture longer than this to engage drag
    gesture_timeout: float = 1.5  # Max duration to hold gesture before cancelling click (fatigue guard)

    # Feedback & Hardware Shortcuts
    audio_feedback: bool = False  # Discrete click chime/beep
    visual_feedback: bool = True  # Toast/badge feedback
    camera_index: int = 0
    hotkey_toggle: str = "f8"  # Master Pause (both pointer & clicking)
    hotkey_calibrate: str = "c"  # Resting center calibration
    hotkey_toggle_pointer: str = "f6"  # Toggle Head Pointer only
    hotkey_toggle_clicks: str = "f7"  # Toggle Click Actions only

    # Hand Gesture Tracking Subsystem
    control_preset: str = "Thumb Trigger"  # Active control preset
    hand_tracking_enabled: bool = True  # Master toggle for hand gesture tracking
    hand_right_only: bool = True  # Whether to only allow right hand control (ignores left hand)
    pointer_source: str = "finger"  # "finger" (index finger tip pointer) or "head" (head pointer)
    hand_thumb_trigger_enabled: bool = True  # Thumb Tap (<1s) for Left Click, Hold (>=1s) for Drag
    hand_thumb_hold_threshold: float = 1.0  # Seconds to hold thumb open before engaging drag
    hand_thumb_open_ratio: float = 0.65  # Hysteresis open threshold (dist to index MCP / palm scale)
    hand_thumb_close_ratio: float = 0.48  # Hysteresis close threshold
    hand_pinch_threshold: float = 0.08  # Normalized Euclidean distance threshold for pinch detection
    hand_peace_action: str = "None"  # Action triggered by Peace sign
    hand_pinch_index_action: str = "None"  # Action triggered by Thumb+Index pinch
    hand_pinch_middle_action: str = "None"  # Action triggered by Thumb+Middle pinch
    hand_point_action: str = "None"  # Pointing index finger moves cursor
    hand_fist_action: str = "None"  # Fist does nothing (freeze cursor)
    hand_cursor_speed: float = 2.0  # Speed multiplier when pointer_source is "finger"
    hand_three_finger_middle_action: str = "Right Click"  # "Right Click" or "Middle Click" for 3-finger middle curl

    # In-App Smart Sensitivity Settings (Windows Trackpad / Precision Mouse Inspired)
    hand_smart_sensitivity: bool = True  # Master toggle for in-app smart sensitivity
    hand_sensitivity_profile: str = "dynamic"  # "dynamic" (Windows Mouse), "smooth" (Trackpad), "linear" (Raw)
    hand_tremor_deadband: float = 0.0015  # Continuous soft deadzone radius for physiological tremor suppression

    @property
    def enable_head_pointer(self) -> bool:
        """Backward compatible alias for pointer_enabled."""
        return self.pointer_enabled

    @enable_head_pointer.setter
    def enable_head_pointer(self, value: bool):
        self.pointer_enabled = value

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["enable_head_pointer"] = self.pointer_enabled
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        data_copy = dict(data)
        # Migrate legacy key enable_head_pointer to pointer_enabled if needed
        if "enable_head_pointer" in data_copy and "pointer_enabled" not in data_copy:
            data_copy["pointer_enabled"] = data_copy["enable_head_pointer"]
        filtered = {k: v for k, v in data_copy.items() if k in valid_keys}
        return cls(**filtered)


# Backward compatibility wrapper for constructor calls with enable_head_pointer
_orig_appconfig_init = AppConfig.__init__


def _appconfig_init(self, *args, **kwargs):
    if "enable_head_pointer" in kwargs:
        if "pointer_enabled" not in kwargs:
            kwargs["pointer_enabled"] = kwargs.pop("enable_head_pointer")
        else:
            kwargs.pop("enable_head_pointer")
    _orig_appconfig_init(self, *args, **kwargs)


AppConfig.__init__ = _appconfig_init


CONTROL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Thumb Trigger": {
        "pointer_source": "finger",
        "hand_tracking_enabled": True,
        "hand_thumb_trigger_enabled": True,
        "hand_thumb_hold_threshold": 1.0,
        "hand_peace_action": "None",
        "hand_pinch_index_action": "None",
        "hand_pinch_middle_action": "None",
        "hand_point_action": "None",
        "hand_fist_action": "None",
    },
    "Peace & Fist (Classic)": {
        "pointer_source": "finger",
        "hand_tracking_enabled": True,
        "hand_thumb_trigger_enabled": False,
        "hand_peace_action": "Left Click",
        "hand_pinch_index_action": "None",
        "hand_pinch_middle_action": "None",
        "hand_point_action": "None",
        "hand_fist_action": "None",
    },
    "Pinch & Point": {
        "pointer_source": "finger",
        "hand_tracking_enabled": True,
        "hand_thumb_trigger_enabled": False,
        "hand_pinch_index_action": "Hold Left Click",
        "hand_pinch_middle_action": "Right Click",
        "hand_peace_action": "None",
        "hand_point_action": "None",
        "hand_fist_action": "None",
    },
    "3-Finger Control": {
        "pointer_source": "finger",
        "hand_tracking_enabled": True,
        "hand_thumb_trigger_enabled": False,
        "hand_peace_action": "None",
        "hand_pinch_index_action": "None",
        "hand_pinch_middle_action": "None",
        "hand_point_action": "None",
        "hand_fist_action": "None",
    },
    "Head & Face (Handsfree)": {
        "pointer_source": "head",
        "pointer_enabled": True,
        "hand_tracking_enabled": False,
        "hand_thumb_trigger_enabled": False,
    },
    "Custom": {},
}


class ConfigManager:
    """Manages loading, saving, and defaults for application configuration."""

    DEFAULT_FILENAME = "config.json"

    def __init__(self, filepath: str = DEFAULT_FILENAME):
        self.filepath = filepath
        self.config = self.load()

    def load(self) -> AppConfig:
        """Loads configuration from JSON file or returns defaults if absent or corrupt."""
        if not os.path.exists(self.filepath):
            default_cfg = AppConfig()
            self.save(default_cfg)
            return default_cfg

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig.from_dict(data)
        except Exception as e:
            print(f"[ConfigManager] Warning: Failed to load {self.filepath} ({e}). Falling back to defaults.")
            return AppConfig()

    def save(self, config: AppConfig = None) -> bool:
        """Saves current or provided configuration to disk."""
        if config is not None:
            self.config = config

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4)
            return True
        except Exception as e:
            print(f"[ConfigManager] Error: Failed to save config to {self.filepath}: {e}")
            return False

    def reset_defaults(self) -> AppConfig:
        """Resets config to default values and saves to disk."""
        self.config = AppConfig()
        self.save(self.config)
        return self.config

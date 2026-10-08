"""Main entry point for Zero Touch application.
Orchestrates configuration, tracker, mouse controller, global hotkeys listener, and GUI.
"""

import argparse
import ctypes
import sys
import threading
from pynput import keyboard

# Enable Windows Per-Monitor DPI awareness to prevent coordinate scaling mismatch
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from config import ConfigManager
from gui import FaceControlApp
from hand_tracker import HandTracker
from mouse_controller import MouseController
from tracker import FaceTracker


def _normalize_key(key) -> str:
    """Normalizes pynput key representation to lowercase string name."""
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    if hasattr(key, "name") and key.name:
        return key.name.lower()
    return str(key).replace("Key.", "").lower()


def setup_global_hotkeys(app: FaceControlApp) -> keyboard.Listener:
    """Sets up a non-blocking background listener for OS-wide global hotkeys & assistive switches."""

    def on_press(key):
        try:
            # F8 or ESC: Master Pause / Resume
            if key in (keyboard.Key.f8, keyboard.Key.esc):
                app.after(0, app.toggle_tracking_hotkey)
            # F6: Head Pointer Toggle
            elif key == keyboard.Key.f6:
                app.after(0, app.toggle_pointer_hotkey)
            # F7: Click Actions Toggle
            elif key == keyboard.Key.f7:
                app.after(0, app.toggle_clicks_hotkey)
            else:
                k_str = _normalize_key(key)
                cfg = app.config
                cal_key = (getattr(cfg, "hotkey_calibrate", "c") or "c").lower()

                # Dedicated F9 or configured function key for global OS calibration (prevents hijacking normal 'c' typing)
                if key == keyboard.Key.f9 or (cal_key != "c" and k_str == cal_key):
                    app.after(0, app.calibrate_center)
                else:
                    # Check Global Assistive Switches (Alternate Pointer Actions)
                    sw_left = (getattr(cfg, "hotkey_switch_left", "") or "").lower()
                    sw_right = (getattr(cfg, "hotkey_switch_right", "") or "").lower()
                    sw_drag = (getattr(cfg, "hotkey_switch_drag", "") or "").lower()

                    if sw_left and k_str == sw_left:
                        app.mouse.trigger_switch_action("left")
                    elif sw_right and k_str == sw_right:
                        app.mouse.trigger_switch_action("right")
                    elif sw_drag and k_str == sw_drag:
                        app.mouse.trigger_switch_action("drag")
        except Exception as e:
            print(f"[HotkeyListener] Error processing key event: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def run_headless_diagnostic(duration_sec: float = 3.0) -> bool:
    """Runs a headless diagnostic test to verify tracker initialization and mock frame pipeline."""
    import time
    import numpy as np

    print("[Diagnostic] Running headless smoke test...")
    config_manager = ConfigManager("test_config.json")
    tracker = FaceTracker(config_manager.config)
    mouse = MouseController(config_manager.config)

    # Generate synthetic black frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    target_pos, gesture, annotated, telemetry = tracker.process_frame(dummy_frame, 1920, 1080)

    print(f"[Diagnostic] Headless process_frame output: target_pos={target_pos}, gesture={gesture}")
    print(f"[Diagnostic] Telemetry: {telemetry}")

    # Test HandTracker headless processing
    hand_tracker = HandTracker(config_manager.config)
    hand_pos, hand_gest, hand_annot, hand_telemetry = hand_tracker.process_frame(dummy_frame, 1920, 1080)
    print(f"[Diagnostic] Headless HandTracker output: hand_pos={hand_pos}, hand_gest={hand_gest}")
    print(f"[Diagnostic] Hand Telemetry: {hand_telemetry}")
    hand_tracker.close()

    # Test pause toggle
    assert not mouse.is_paused, "Initial pause state should be False"
    mouse.toggle_pause()
    assert mouse.is_paused, "Pause toggle should switch state to True"
    mouse.toggle_pause()
    assert not mouse.is_paused, "Pause toggle should revert state to False"

    tracker.close()
    print("[Diagnostic] Smoke test completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Zero Touch: Touchless Head & Hand Mouse")
    parser.add_argument("--test", action="store_true", help="Run automated headless diagnostic test and exit")
    args = parser.parse_args()

    if args.test:
        success = run_headless_diagnostic()
        sys.exit(0 if success else 1)

    print("=====================================================")
    print("  Zero Touch - Touchless Head & Hand Mouse")
    print("  Controls:")
    print("    - F8 / ESC : Pause / Resume Tracking")
    print("    - F9 / C   : Calibrate Resting Head Center (F9 Global, C Window)")
    print("    - F6       : Toggle Head Pointer Motion")
    print("    - F7       : Toggle Clicking Subsystem")
    print("=====================================================")

    config_manager = ConfigManager()
    tracker = FaceTracker(config_manager.config)
    hand_tracker = HandTracker(config_manager.config)
    mouse_controller = MouseController(config_manager.config)

    app = FaceControlApp(config_manager, tracker, mouse_controller, hand_tracker=hand_tracker)
    hotkey_listener = setup_global_hotkeys(app)

    try:
        app.mainloop()
    finally:
        print("[Zero Touch] Shutting down...")
        if hotkey_listener and hotkey_listener.is_alive():
            hotkey_listener.stop()


if __name__ == "__main__":
    main()

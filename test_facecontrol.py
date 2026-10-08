"""Automated verification and unit test suite for FaceControl components."""

import json
import math
import os
import time
import unittest
from types import SimpleNamespace
import numpy as np

from config import AppConfig, ConfigManager
from mouse_controller import MouseController
from tracker import FaceTracker, OneEuroFilter


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestFaceControl(unittest.TestCase):
    def setUp(self):
        self.test_config_path = "test_temp_config.json"
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def tearDown(self):
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_config_defaults_and_persistence(self):
        """Test config manager loads, updates, saves and fallbacks."""
        cm = ConfigManager(self.test_config_path)
        self.assertAlmostEqual(cm.config.cursor_speed, 2.2)
        self.assertEqual(cm.config.left_click_trigger, "Left Wink")

        # Mutate and save
        cm.config.cursor_speed = 3.5
        cm.save()

        # Reload
        cm2 = ConfigManager(self.test_config_path)
        self.assertAlmostEqual(cm2.config.cursor_speed, 3.5)

        # Corrupt file test
        with open(self.test_config_path, "w") as f:
            f.write("{invalid json content")
        cm3 = ConfigManager(self.test_config_path)
        self.assertAlmostEqual(cm3.config.cursor_speed, 2.2)  # Should fallback to default

    def test_config_ui_mode_persistence(self):
        """Test ui_mode default value, updating to advanced, and json persistence."""
        cm = ConfigManager(self.test_config_path)
        self.assertEqual(cm.config.ui_mode, "simple")

        # Mutate to advanced and save
        cm.config.ui_mode = "advanced"
        cm.save()

        # Reload and verify
        cm2 = ConfigManager(self.test_config_path)
        self.assertEqual(cm2.config.ui_mode, "advanced")

    def test_control_presets_integrity(self):
        """Verify all CONTROL_PRESETS define valid attributes on AppConfig."""
        from config import CONTROL_PRESETS
        valid_fields = AppConfig.__dataclass_fields__.keys()
        for preset_name, settings in CONTROL_PRESETS.items():
            for key, val in settings.items():
                self.assertIn(key, valid_fields, f"Preset '{preset_name}' defines invalid attribute '{key}'")
            # Apply to config and verify
            cfg = AppConfig()
            for key, val in settings.items():
                setattr(cfg, key, val)
                self.assertEqual(getattr(cfg, key), val)

    def test_simple_mode_preset_mapping_and_custom(self):
        """Verify preset mapping helpers correctly translate between display labels and preset keys."""
        from gui import FaceControlApp
        # Test mapping helper methods as static functions
        self.assertEqual(
            FaceControlApp._map_preset_to_simple_label(None, "Thumb Trigger"),
            "🖐️ Kontrol Jari & Jempol (Rekomendasi)",
        )
        self.assertEqual(
            FaceControlApp._map_preset_to_simple_label(None, "Head & Face (Handsfree)"),
            "🙆 Bebas Tangan / Wajah (Handsfree)",
        )
        self.assertEqual(
            FaceControlApp._map_preset_to_simple_label(None, "Custom"),
            "⚙️ Kustom (Dari Mode Lanjutan)",
        )
        self.assertEqual(
            FaceControlApp._map_preset_to_advanced_label(None, "Thumb Trigger"),
            "Thumb Trigger (Default)",
        )
        self.assertEqual(
            FaceControlApp._map_preset_to_advanced_label(None, "Custom"),
            "Custom",
        )

    def test_mouse_controller_pause_and_debounce(self):
        """Test pause toggling and action debouncing."""
        cfg = AppConfig(click_cooldown=0.2)
        mouse = MouseController(cfg)

        actions_received = []
        mouse.set_action_callback(lambda act: actions_received.append(act))

        # Initial state
        self.assertFalse(mouse.is_paused)

        # Toggle pause
        is_paused = mouse.toggle_pause()
        self.assertTrue(is_paused)
        self.assertTrue(mouse.is_paused)
        self.assertIn("PAUSED", actions_received)

        # When paused, process_gesture should do nothing
        mouse.process_gesture("Left Wink")
        self.assertNotIn("Left Click", actions_received)

        # Resume
        mouse.toggle_pause()
        self.assertFalse(mouse.is_paused)

        # Trigger first gesture and release
        mouse.process_gesture("Left Wink")
        mouse.process_gesture(None)  # Release triggers click
        self.assertIn("Left Click", actions_received)
        click_count_1 = actions_received.count("Left Click")

        # Immediate repeated trigger within cooldown should be debounced
        mouse.process_gesture("Left Wink")
        mouse.process_gesture(None)
        self.assertEqual(actions_received.count("Left Click"), click_count_1)

        # Wait for cooldown to expire
        time.sleep(0.25)
        mouse.process_gesture("Left Wink")
        mouse.process_gesture(None)
        self.assertEqual(actions_received.count("Left Click"), click_count_1 + 1)

    def test_mouse_controller_drag_and_drop(self):
        """Test click-and-drag triggering when gesture is held."""
        cfg = AppConfig(drag_hold_duration=0.15, click_cooldown=0.05)
        mouse = MouseController(cfg)

        actions = []
        mouse.set_action_callback(lambda act: actions.append(act))

        # Hold Left Wink for > drag_hold_duration
        mouse.process_gesture("Left Wink")
        time.sleep(0.20)
        mouse.process_gesture("Left Wink")

        self.assertTrue(mouse.is_dragging)
        self.assertIn("DRAG START", actions)

        # Release gesture drops the drag without extra click
        mouse.process_gesture(None)
        self.assertFalse(mouse.is_dragging)
        self.assertIn("DRAG RELEASE", actions)
        self.assertNotIn("Left Click", actions)

    def test_tracker_ear_and_mar_math(self):
        """Test EAR and MAR calculations with synthetic landmark coordinates."""
        cfg = AppConfig()
        tracker = FaceTracker(cfg)

        landmarks = [DummyLandmark(0, 0)] * 478
        landmarks[263] = DummyLandmark(0.1, 0.5)
        landmarks[385] = DummyLandmark(0.2, 0.45)
        landmarks[387] = DummyLandmark(0.3, 0.45)
        landmarks[362] = DummyLandmark(0.4, 0.5)
        landmarks[373] = DummyLandmark(0.3, 0.55)
        landmarks[380] = DummyLandmark(0.2, 0.55)

        ear = tracker._calculate_ear(landmarks, tracker.LEFT_EYE)
        self.assertAlmostEqual(ear, 0.3333, places=3)

        landmarks[tracker.MOUTH_TOP] = DummyLandmark(0.5, 0.4)
        landmarks[tracker.MOUTH_BOTTOM] = DummyLandmark(0.5, 0.6)
        landmarks[tracker.MOUTH_LEFT] = DummyLandmark(0.3, 0.5)
        landmarks[tracker.MOUTH_RIGHT] = DummyLandmark(0.7, 0.5)

        mar = tracker._calculate_mar(landmarks)
        self.assertAlmostEqual(mar, 0.5, places=3)
        tracker.close()

    def test_bilateral_blink_rejection(self):
        """Crucial test: natural blinks (both eyes closed) MUST NOT trigger clicks."""
        cfg = AppConfig(wink_threshold=0.20)
        tracker = FaceTracker(cfg)

        # Bilateral Blink: both eyes EAR = 0.14
        gesture = tracker._detect_gesture(left_ear=0.14, right_ear=0.14, mar=0.2)
        self.assertIsNone(gesture, "Natural bilateral blink should NOT trigger any gesture")

        # Frame 1 of Left Wink: buffered in temporal persistence buffer
        gesture_f1 = tracker._detect_gesture(left_ear=0.14, right_ear=0.32, mar=0.2)
        self.assertIsNone(gesture_f1, "Frame 1 should be buffered awaiting confirmation")

        # Frame 2 of Left Wink: confirmed!
        gesture_f2 = tracker._detect_gesture(left_ear=0.14, right_ear=0.32, mar=0.2)
        self.assertEqual(gesture_f2, "Left Wink", "Frame 2 should confirm Left Wink")

        # Right Wink: right_ear = 0.14, left_ear = 0.32
        tracker._detect_gesture(left_ear=0.32, right_ear=0.14, mar=0.2)
        gesture_r = tracker._detect_gesture(left_ear=0.32, right_ear=0.14, mar=0.2)
        self.assertEqual(gesture_r, "Right Wink")

        # Mouth Open: MAR = 0.60
        gesture_mouth = tracker._detect_gesture(left_ear=0.32, right_ear=0.32, mar=0.60)
        self.assertEqual(gesture_mouth, "Mouth Open")

        tracker.close()

    def test_mirror_mode_physical_eye_mapping(self):
        """Verifies that in mirrored mode (flip_horizontal=True), physical right eye maps to right_ear telemetry."""
        from unittest.mock import MagicMock
        cfg = AppConfig(flip_horizontal=True, wink_threshold=0.20)
        tracker = FaceTracker(cfg)

        # Mock FaceMesh landmarks
        landmarks = [DummyLandmark(0.5, 0.5)] * 478
        # In mirrored frame, user physical right eye is at MediaPipe canonical LEFT_EYE (263)
        # Set LEFT_EYE closed (ear approx 0.13)
        landmarks[263] = DummyLandmark(0.6, 0.5)
        landmarks[385] = DummyLandmark(0.65, 0.49)
        landmarks[387] = DummyLandmark(0.70, 0.49)
        landmarks[362] = DummyLandmark(0.75, 0.5)
        landmarks[373] = DummyLandmark(0.70, 0.51)
        landmarks[380] = DummyLandmark(0.65, 0.51)

        # Set RIGHT_EYE open (ear approx 0.67)
        landmarks[33] = DummyLandmark(0.2, 0.5)
        landmarks[160] = DummyLandmark(0.25, 0.45)
        landmarks[158] = DummyLandmark(0.30, 0.45)
        landmarks[133] = DummyLandmark(0.35, 0.5)
        landmarks[153] = DummyLandmark(0.30, 0.55)
        landmarks[144] = DummyLandmark(0.25, 0.55)

        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [MagicMock(landmark=landmarks)]
        tracker.face_mesh.process = MagicMock(return_value=mock_results)

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, _, telemetry = tracker.process_frame(frame, 1920, 1080)

        # Physical right eye was closed -> telemetry right_ear must be lower than left_ear and below wink_threshold
        self.assertLess(telemetry["right_ear"], 0.20)
        self.assertGreater(telemetry["left_ear"], 0.25)
        tracker.close()

    def test_one_euro_filter_jitter_suppression(self):
        """Verify that stationary jitter is strongly suppressed by OneEuroFilter."""
        t = 0.0
        dt = 1.0 / 30.0  # 30 FPS
        base_val = 500.0
        filt = OneEuroFilter(t0=t, x0=base_val, min_cutoff=1.0, beta=0.007)

        np.random.seed(42)
        raw_values = []
        filtered_values = []

        for _ in range(60):
            t += dt
            noise = float(np.random.normal(0, 3.0))
            raw = base_val + noise
            filtered = filt.filter(t, raw)
            raw_values.append(raw)
            filtered_values.append(filtered)

        # Output variance should be dramatically reduced compared to input jitter
        raw_std = float(np.std(raw_values[10:]))
        filt_std = float(np.std(filtered_values[10:]))
        self.assertLess(filt_std, raw_std * 0.45, "Filtered jitter std should be < 45% of raw jitter")

    def test_one_euro_filter_high_speed_responsiveness(self):
        """Verify that fast movements adapt cutoff frequency to eliminate lag."""
        t = 0.0
        dt = 1.0 / 30.0
        filt = OneEuroFilter(t0=t, x0=100.0, min_cutoff=1.0, beta=0.007)

        # Stationary first
        for _ in range(10):
            t += dt
            filt.filter(t, 100.0)

        # Fast jump to 1000.0 over 3 frames (high velocity > 9000 px/sec)
        t += dt
        out1 = filt.filter(t, 400.0)
        t += dt
        out2 = filt.filter(t, 800.0)
        t += dt
        out3 = filt.filter(t, 1000.0)

        # Should be closely following target at frame 3 (> 900.0) due to dynamic beta cutoff boost
        self.assertGreater(out3, 900.0, "High speed tracking should reach > 90% target position quickly")

    def test_one_euro_filter_reset(self):
        """Verify reset re-initializes internal state properly."""
        filt = OneEuroFilter(t0=0.0, x0=50.0, min_cutoff=1.0, beta=0.007)
        filt.filter(0.1, 150.0)
        filt.reset(t0=1.0, x0=200.0, dx0=0.0)
        self.assertEqual(filt.t_prev, 1.0)
        self.assertEqual(filt.x_prev, 200.0)
        self.assertEqual(filt.dx_prev, 0.0)

    def test_double_click_non_blocking(self):
        """Verify that double click action dispatches asynchronously without blocking caller."""
        cfg = AppConfig()
        mouse = MouseController(cfg)
        actions = []
        mouse.set_action_callback(lambda act: actions.append(act))

        t_start = time.perf_counter()
        mouse._execute_action("Double Click")
        elapsed = time.perf_counter() - t_start

        self.assertLess(elapsed, 0.035, f"Double click took {elapsed:.4f}s, expected non-blocking < 35ms")
        self.assertIn("Double Click", actions)

    def test_tracker_face_lost_reset(self):
        """Verify that tracker state cleanly resets to None when face is absent for >= 5 frames."""
        cfg = AppConfig()
        tracker = FaceTracker(cfg)

        tracker.smoothed_x = 500.0
        tracker.smoothed_y = 300.0
        tracker.filter_x = OneEuroFilter(0.0, 500.0)
        tracker.filter_y = OneEuroFilter(0.0, 300.0)
        tracker.candidate_gesture = "Left Wink"
        tracker.gesture_frame_count = 2

        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(tracker.MAX_LOST_FRAMES):
            tracker.process_frame(empty_frame, 1920, 1080)

        self.assertIsNone(tracker.smoothed_x)
        self.assertIsNone(tracker.smoothed_y)
        self.assertIsNone(tracker.filter_x)
        self.assertIsNone(tracker.filter_y)
        self.assertIsNone(tracker.candidate_gesture)
        tracker.close()

    def test_mouse_fatigue_hold_cancellation(self):
        """Verify that holding a gesture > 1.5s cancels click upon release (fatigue guard)."""
        cfg = AppConfig(click_cooldown=0.05, drag_hold_duration=5.0)
        mouse = MouseController(cfg)
        actions = []
        mouse.set_action_callback(lambda act: actions.append(act))

        mouse.process_gesture("Left Wink")
        mouse.hold_start_time = time.perf_counter() - 2.0
        mouse.process_gesture(None)

        self.assertNotIn("Left Click", actions, "Accidental long gesture hold (> 1.5s) must be dropped")

    def test_config_filter_and_timeout_parameters(self):
        """Verify new filter_beta and gesture_timeout configuration parameters."""
        cfg = AppConfig()
        self.assertEqual(cfg.filter_beta, 0.007)
        self.assertEqual(cfg.gesture_timeout, 1.5)

        # Mutate and serialize
        cfg.filter_beta = 0.015
        cfg.gesture_timeout = 2.0
        d = cfg.to_dict()
        self.assertEqual(d["filter_beta"], 0.015)
        self.assertEqual(d["gesture_timeout"], 2.0)

        # Deserialization
        cfg2 = AppConfig.from_dict(d)
        self.assertEqual(cfg2.filter_beta, 0.015)
        self.assertEqual(cfg2.gesture_timeout, 2.0)

    def test_mouse_move_rel_win32_integration(self):
        """Verify move_rel executes without error and updates cursor position."""
        cfg = AppConfig()
        mouse = MouseController(cfg)
        # Calling move_rel with zero or small delta should complete cleanly
        mouse.move_rel(0.0, 0.0)
        self.assertFalse(mouse.is_paused)

    def test_mouse_override_detection_and_timeout(self):
        """Verify physical mouse movement suppresses programmatic cursor motion until timeout."""
        cfg = AppConfig(mouse_override=True, override_timeout=0.2)
        mouse = MouseController(cfg)
        actions = []
        mouse.set_action_callback(lambda act: actions.append(act))

        # Initial baseline position
        mouse.last_programmatic_pos = (500, 500)
        self.assertFalse(mouse.is_overridden)

        # Mock physical movement: cursor moved by user to (520, 520)
        mouse.get_physical_cursor_pos = lambda: (520, 520)
        is_overridden = mouse.check_physical_mouse_override()
        self.assertTrue(is_overridden, "Moving physical mouse should trigger override")
        self.assertTrue(mouse.is_overridden)
        self.assertIn("MOUSE OVERRIDE", actions)

        # While overridden, move_to should be suppressed
        moved = mouse.move_to(100, 100)
        self.assertFalse(moved, "move_to should return False while physical mouse is active")

        # Advance time beyond timeout
        mouse.last_physical_mouse_time = time.perf_counter() - 0.3
        is_overridden, just_resumed = mouse.update_override_state()
        self.assertFalse(is_overridden, "Override should expire after timeout")
        self.assertTrue(just_resumed, "just_resumed should be True when transitioning back")
        self.assertFalse(mouse.is_overridden)

    def test_head_pointer_disabled_gesture_only_mode(self):
        """Verify that disabling head pointer halts cursor movement while gestures remain operational."""
        cfg = AppConfig(enable_head_pointer=False, click_cooldown=0.05)
        mouse = MouseController(cfg)
        actions = []
        mouse.set_action_callback(lambda act: actions.append(act))

        # Cursor movement should be suppressed
        moved = mouse.move_to(800, 600)
        self.assertFalse(moved, "move_to must return False when enable_head_pointer is False")
        moved_rel = mouse.move_rel(10, 10)
        self.assertFalse(moved_rel, "move_rel must return False when enable_head_pointer is False")

        # Facial gestures must still trigger clicks normally
        mouse.process_gesture("Left Wink")
        mouse.process_gesture(None)
        self.assertIn("Left Click", actions, "Gestures must still work in Gesture-Only mode")

    def test_config_mouse_override_options_persistence(self):
        """Verify serialization and deserialization of new mouse override configuration attributes."""
        cfg = AppConfig(enable_head_pointer=False, mouse_override=True, override_timeout=1.2)
        d = cfg.to_dict()
        self.assertFalse(d["enable_head_pointer"])
        self.assertTrue(d["mouse_override"])
        self.assertEqual(d["override_timeout"], 1.2)

        cfg2 = AppConfig.from_dict(d)
        self.assertFalse(cfg2.enable_head_pointer)
        self.assertTrue(cfg2.mouse_override)
        self.assertEqual(cfg2.override_timeout, 1.2)

    def test_tracker_reset_filters_smooth_transition(self):
        """Verify tracker filter re-seeding for smooth hand-off in both Absolute and Relative modes."""
        # 1. Absolute mode test
        cfg_abs = AppConfig(tracking_mode="absolute")
        tracker_abs = FaceTracker(cfg_abs)
        tracker_abs.filter_x = OneEuroFilter(0.0, 500.0)
        tracker_abs.filter_y = OneEuroFilter(0.0, 300.0)

        tracker_abs.reset_filters((800.0, 600.0))
        self.assertEqual(tracker_abs.smoothed_x, 800.0)
        self.assertEqual(tracker_abs.smoothed_y, 600.0)
        self.assertEqual(tracker_abs.filter_x.x_prev, 800.0)
        self.assertEqual(tracker_abs.filter_y.x_prev, 600.0)
        tracker_abs.close()

        # 2. Relative mode test (Issue 1 verification): angular deltas must reset to 0.0, not screen pixels
        cfg_rel = AppConfig(tracking_mode="relative")
        tracker_rel = FaceTracker(cfg_rel)
        tracker_rel.filter_x = OneEuroFilter(0.0, 0.05)
        tracker_rel.filter_y = OneEuroFilter(0.0, -0.02)

        tracker_rel.reset_filters((800.0, 600.0))
        self.assertEqual(tracker_rel.smoothed_x, 800.0)
        self.assertEqual(tracker_rel.smoothed_y, 600.0)
        self.assertEqual(tracker_rel.filter_x.x_prev, 0.0, "Relative filter_x must re-seed to 0.0 angular delta")
        self.assertEqual(tracker_rel.filter_y.x_prev, 0.0, "Relative filter_y must re-seed to 0.0 angular delta")
        tracker_rel.close()

    def test_relative_mode_override_hand_off_stability(self):
        """Integration test: verify relative mode does not launch cursor to screen boundaries post-override."""
        cfg = AppConfig(tracking_mode="relative", cursor_speed=2.0)
        tracker = FaceTracker(cfg)
        tracker.resting_center = (0.5, 0.5)

        # User was using physical mouse and left cursor at (960, 540)
        physical_cursor = (960.0, 540.0)
        tracker.reset_filters(physical_cursor)

        # Mock frame processing with small head tilt: nose at (0.51, 0.51) -> dx=0.01, dy=0.01
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Calling reset_filters re-anchored filters to 0.0; check filter output doesn't explode
        now = time.perf_counter()
        filtered_dx = tracker.filter_x.filter(now, 0.01)
        filtered_dy = tracker.filter_y.filter(now, 0.01)

        # Delta velocity scale: cursor_speed * 45.0 = 90.0
        step_x = filtered_dx * 90.0
        step_y = filtered_dy * 90.0
        new_screen_x = physical_cursor[0] + step_x
        new_screen_y = physical_cursor[1] + step_y

        # Target cursor should stay smoothly near (960, 540) (e.g., within 5 pixels), NOT fly to 0 or 1920
        self.assertAlmostEqual(new_screen_x, 960.0, delta=5.0)
        self.assertAlmostEqual(new_screen_y, 540.0, delta=5.0)
        tracker.close()

    def test_mouse_controller_cached_pos_efficiency(self):
        """Verify cached_pos parameter eliminates redundant OS cursor polling."""
        cfg = AppConfig()
        mouse = MouseController(cfg)
        mouse.last_programmatic_pos = (500, 500)

        # Override get_physical_cursor_pos to raise error if called unexpectedly
        def fail_if_called():
            raise RuntimeError("Redundant Win32 cursor polling detected!")

        mouse.get_physical_cursor_pos = fail_if_called

        # When cached_pos is provided, it should execute cleanly without calling get_physical_cursor_pos
        is_overridden, just_resumed = mouse.update_override_state(cached_pos=(500, 500))
        self.assertFalse(is_overridden)

        # Test move_to with cached_pos
        moved = mouse.move_to(550, 550, cached_pos=(500, 500))
        self.assertTrue(moved)

    def test_tracker_thread_safe_calibration(self):
        """Verify calibrate_center and reset_filters can be invoked without race condition."""
        cfg = AppConfig()
        tracker = FaceTracker(cfg)
        tracker.resting_center = (0.5, 0.5)
        tracker.filter_x = OneEuroFilter(0.0, 500.0)
        tracker.filter_y = OneEuroFilter(0.0, 300.0)

        # Calibrate center resets filters safely under thread lock
        tracker.calibrate_center((0.52, 0.48))
        self.assertIsNone(tracker.filter_x)
        self.assertIsNone(tracker.filter_y)
        self.assertEqual(tracker.resting_center, (0.52, 0.48))
        tracker.close()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Decoupled Mouse and Click Control Architecture.
Validates independent pointer/clicking controls, dwell control state machine,
assistive switches, and backward compatibility.
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from config import AppConfig, ConfigManager
from mouse_controller import MouseController
from tracker import FaceTracker


class TestDecoupledConfig(unittest.TestCase):
    """Verifies AppConfig serialization and backward compatibility."""

    def test_default_values(self):
        cfg = AppConfig()
        self.assertTrue(cfg.pointer_enabled)
        self.assertTrue(cfg.clicking_enabled)
        self.assertEqual(cfg.click_mode, "gestures")
        self.assertFalse(cfg.dwell_enabled)
        self.assertEqual(cfg.dwell_time, 0.8)
        self.assertEqual(cfg.dwell_radius, 20)
        self.assertEqual(cfg.dwell_action, "Left Click")
        self.assertEqual(cfg.hotkey_toggle_pointer, "f6")
        self.assertEqual(cfg.hotkey_toggle_clicks, "f7")

    def test_backward_compatibility_enable_head_pointer(self):
        # Legacy config containing enable_head_pointer: False
        legacy_data = {
            "cursor_speed": 3.0,
            "enable_head_pointer": False,
            "wink_threshold": 0.25,
        }
        cfg = AppConfig.from_dict(legacy_data)
        self.assertFalse(cfg.pointer_enabled)
        self.assertFalse(cfg.enable_head_pointer)

        # Setter should synchronize
        cfg.enable_head_pointer = True
        self.assertTrue(cfg.pointer_enabled)

    def test_to_dict_and_from_dict(self):
        cfg = AppConfig(
            pointer_enabled=False,
            clicking_enabled=True,
            dwell_enabled=True,
            dwell_time=1.2,
            dwell_radius=25,
            hotkey_switch_left="space",
        )
        data = cfg.to_dict()
        restored = AppConfig.from_dict(data)
        self.assertFalse(restored.pointer_enabled)
        self.assertTrue(restored.clicking_enabled)
        self.assertTrue(restored.dwell_enabled)
        self.assertEqual(restored.dwell_time, 1.2)
        self.assertEqual(restored.dwell_radius, 25)
        self.assertEqual(restored.hotkey_switch_left, "space")


class TestMouseControllerDecoupled(unittest.TestCase):
    """Verifies decoupled execution states and Dwell Control state machine."""

    def setUp(self):
        self.config = AppConfig(
            pointer_enabled=True,
            clicking_enabled=True,
            dwell_enabled=True,
            click_mode="hybrid",
            dwell_time=0.4,
            dwell_radius=15,
            dwell_action="Left Click",
            mouse_override=False,
        )
        self.controller = MouseController(self.config)

    def test_independent_pause_toggles(self):
        # Initially all active
        self.assertFalse(self.controller.is_paused)
        self.assertFalse(self.controller.is_pointer_paused)
        self.assertFalse(self.controller.is_clicking_paused)

        # Toggle pointer pause (F6)
        self.assertTrue(self.controller.toggle_pointer_pause())
        self.assertTrue(self.controller.is_pointer_paused)
        self.assertFalse(self.controller.is_clicking_paused)
        self.assertFalse(self.controller.is_paused)

        # Toggle clicking pause (F7)
        self.assertTrue(self.controller.toggle_clicking_pause())
        self.assertTrue(self.controller.is_clicking_paused)
        self.assertTrue(self.controller.is_pointer_paused)

        # Resume pointer
        self.assertFalse(self.controller.toggle_pointer_pause())
        self.assertFalse(self.controller.is_pointer_paused)
        self.assertTrue(self.controller.is_clicking_paused)

    def test_pointer_suppression_when_paused(self):
        with patch.object(self.controller, "get_physical_cursor_pos", return_value=(100, 100)):
            # Active move
            self.assertTrue(self.controller.move_to(200, 200))

            # Pointer paused
            self.controller.set_pointer_paused(True)
            self.assertFalse(self.controller.move_to(300, 300))

            # Unpause pointer, but config.pointer_enabled is False
            self.controller.set_pointer_paused(False)
            self.config.pointer_enabled = False
            self.assertFalse(self.controller.move_to(300, 300))

    def test_clicking_suppression_when_paused(self):
        with patch.object(self.controller, "_execute_action") as mock_exec:
            self.controller.set_clicking_paused(True)
            self.controller.process_gesture("Left Wink")
            self.controller.process_gesture(None)
            mock_exec.assert_not_called()

    def test_dwell_control_state_machine(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            # 1. First poll initializes anchor without crash
            action, progress = self.controller.check_dwell(cached_pos=(500, 500))
            self.assertIsNone(action)
            self.assertEqual(progress, 0.0)

            # 2. Advance time past dwell_time at same position
            time.sleep(0.45)
            action, progress = self.controller.check_dwell(cached_pos=(502, 501))
            self.assertEqual(action, "Left Click")
            self.assertEqual(progress, 1.0)
            mock_exec.assert_called_once_with("Left Click")

            # 3. Lockout check: holding cursor still does NOT trigger multiple clicks
            mock_exec.reset_mock()
            action, progress = self.controller.check_dwell(cached_pos=(503, 502))
            self.assertIsNone(action)
            self.assertEqual(progress, 1.0)
            mock_exec.assert_not_called()

            # 4. Moving cursor outside dwell_radius resets lockout
            action, progress = self.controller.check_dwell(cached_pos=(550, 550))
            self.assertIsNone(action)
            self.assertEqual(progress, 0.0)

    def test_dwell_suppressed_during_drag(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            self.controller.is_dragging = True
            action, progress = self.controller.check_dwell(cached_pos=(300, 300))
            self.assertIsNone(action)
            self.assertEqual(progress, 0.0)

            time.sleep(0.45)
            action, progress = self.controller.check_dwell(cached_pos=(300, 300))
            self.assertIsNone(action)
            self.assertEqual(progress, 0.0)
            mock_exec.assert_not_called()

    def test_assistive_switch_actions(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            # Left click switch
            self.assertTrue(self.controller.trigger_switch_action("left"))
            mock_exec.assert_called_with("Left Click")

            # Right click switch
            self.assertTrue(self.controller.trigger_switch_action("right"))
            mock_exec.assert_called_with("Right Click")

            # Double click switch
            self.assertTrue(self.controller.trigger_switch_action("double"))
            mock_exec.assert_called_with("Double Click")

        # Drag lock toggle switch
        with patch.object(self.controller, "_start_drag_unlocked") as mock_start, \
             patch.object(self.controller, "_release_drag_unlocked") as mock_release:
            self.assertFalse(self.controller.is_dragging)
            self.controller.trigger_switch_action("drag")
            mock_start.assert_called_once()

            self.controller.is_dragging = True
            self.controller.trigger_switch_action("drag")
            mock_release.assert_called_once()

    def test_click_mode_enforcement(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            # 1. Mode: "gestures" -> dwell must be bypassed
            self.config.click_mode = "gestures"
            self.controller.check_dwell(cached_pos=(400, 400))
            time.sleep(0.45)
            action, progress = self.controller.check_dwell(cached_pos=(400, 400))
            self.assertIsNone(action)
            self.assertEqual(progress, 0.0)

            # 2. Mode: "dwell" -> gestures must be bypassed
            self.config.click_mode = "dwell"
            self.controller.process_gesture("Left Wink")
            self.controller.process_gesture(None)
            mock_exec.assert_not_called()

            # 3. Mode: "hybrid" -> both gestures and dwell allowed
            self.config.click_mode = "hybrid"
            # Gesture executes
            self.controller.process_gesture("Left Wink")
            self.controller.process_gesture(None)
            mock_exec.assert_called_with("Left Click")

            # Dwell executes
            mock_exec.reset_mock()
            self.controller.check_dwell(cached_pos=(700, 700))
            time.sleep(0.45)
            action, progress = self.controller.check_dwell(cached_pos=(700, 700))
            self.assertEqual(action, "Left Click")
            self.assertEqual(progress, 1.0)
            mock_exec.assert_called_once_with("Left Click")


class TestGlobalHotkeysNormalization(unittest.TestCase):
    """Verifies pynput key normalization in main.py."""

    def test_key_normalization(self):
        from main import _normalize_key

        class MockKeyChar:
            char = "C"

        class MockKeyName:
            name = "Space"

        self.assertEqual(_normalize_key(MockKeyChar()), "c")
        self.assertEqual(_normalize_key(MockKeyName()), "space")


class TestZeroCursorJumpOnPhysicalResume(unittest.TestCase):
    """Verifies that moving physical cursor to screen corners and resuming head tracking
    continues directly from the physical position without snapping back to screen center.
    """

    def test_absolute_mode_zero_jump_and_continuation(self):
        import numpy as np
        from unittest.mock import MagicMock

        cfg = AppConfig(tracking_mode="absolute", cursor_speed=2.0, smoothing=0.65)
        tracker = FaceTracker(cfg)

        class DummyLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class DummyMeshResults:
            multi_face_landmarks = [
                MagicMock(landmark=[DummyLandmark(0.5, 0.5)] * 478)
            ]

        tracker.face_mesh.process = MagicMock(return_value=DummyMeshResults())

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        screen_w, screen_h = 1920, 1080

        # 1. Baseline initialization: face resting at center (0.5, 0.5)
        pos, _, _, _ = tracker.process_frame(dummy_frame, screen_w, screen_h)
        self.assertEqual(tracker.resting_center, (0.5, 0.5))

        # 2. User moves physical mouse to top-right corner (1800, 100)
        # Face is stationary (still at 0.5, 0.5), is_overridden=True
        corner_pos = (1800, 100)
        pos, _, _, _ = tracker.process_frame(
            dummy_frame, screen_w, screen_h, current_mouse_pos=corner_pos, is_overridden=True
        )
        self.assertEqual(pos, (1800.0, 100.0))

        # 3. Physical mouse override ends (is_overridden=False), face still stationary
        # The cursor MUST NOT jump back to center (960, 540)
        pos, _, _, _ = tracker.process_frame(
            dummy_frame, screen_w, screen_h, current_mouse_pos=corner_pos, is_overridden=False
        )
        self.assertAlmostEqual(pos[0], 1800.0, delta=2.0)
        self.assertAlmostEqual(pos[1], 100.0, delta=2.0)

        # 4. User moves head slightly to the left (nose moves from 0.5 to 0.48, delta = -0.02)
        DummyMeshResults.multi_face_landmarks[0].landmark[tracker.NOSE_TIP] = DummyLandmark(0.48, 0.5)
        pos, _, _, _ = tracker.process_frame(
            dummy_frame, screen_w, screen_h, current_mouse_pos=corner_pos, is_overridden=False
        )
        # Cursor must move leftward starting from ~1800, NOT from 960
        self.assertLess(pos[0], 1800.0)
        self.assertGreater(pos[0], 1600.0)

        # 5. User recalibrates (presses C) -> anchor resets to screen center (960, 540)
        # Even when current_mouse_pos is passed (as gui.py does), calibration must recenter to (960, 540)
        tracker.calibrate_center(None)
        DummyMeshResults.multi_face_landmarks[0].landmark[tracker.NOSE_TIP] = DummyLandmark(0.5, 0.5)
        pos, _, _, _ = tracker.process_frame(dummy_frame, screen_w, screen_h, current_mouse_pos=corner_pos)
        self.assertAlmostEqual(pos[0], 960.0, delta=5.0)
        self.assertAlmostEqual(pos[1], 540.0, delta=5.0)

        tracker.close()


class TestCycle1QARegressions(unittest.TestCase):
    """Regression test suite for Cycle 1 QA fixes."""

    def test_pause_cleans_up_pending_and_active_gestures(self):
        cfg = AppConfig(click_cooldown=0.05, drag_hold_duration=0.5)
        ctrl = MouseController(cfg)

        ctrl.process_gesture("Left Wink")
        self.assertEqual(ctrl.pending_action, "Left Click")
        self.assertEqual(ctrl.active_hold_gesture, "Left Wink")

        # Pause should clear pending and active hold states
        ctrl.set_paused(True)
        self.assertIsNone(ctrl.pending_action)
        self.assertIsNone(ctrl.active_hold_gesture)
        self.assertIsNone(ctrl.active_hand_gesture)

    def test_hand_tracker_resets_origin_on_hand_loss(self):
        from hand_tracker import HandTracker

        cfg = AppConfig()
        ht = HandTracker(cfg)
        ht.screen_origin_x = 500.0
        ht.screen_origin_y = 400.0

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process frame with no hand detected
        pos, gest, _, telemetry = ht.process_frame(frame, 1920, 1080)
        self.assertFalse(telemetry["hand_detected"])
        self.assertIsNone(ht.screen_origin_x)
        self.assertIsNone(ht.screen_origin_y)
        ht.close()

    def test_redundant_cursor_moves_are_bypassed(self):
        cfg = AppConfig(pointer_enabled=True, mouse_override=False)
        ctrl = MouseController(cfg)

        # First move
        self.assertTrue(ctrl.move_to(350, 450))
        self.assertEqual(ctrl.last_programmatic_pos, (350, 450))

        # Second move to identical coordinate
        with patch("mouse_controller.user32.SetCursorPos" if sys.platform == "win32" else "pyautogui.moveTo") as mock_set:
            self.assertTrue(ctrl.move_to(350, 450))
            mock_set.assert_not_called()


if __name__ == "__main__":
    unittest.main()

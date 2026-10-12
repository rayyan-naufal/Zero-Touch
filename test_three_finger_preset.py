"""Unit tests for the 3-Finger Control (Trio Finger Mode) Preset in FaceControl.
Verifies cursor motion, left click, right click (ring/middle), and idle freeze states.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from config import CONTROL_PRESETS, AppConfig
from hand_tracker import HandTracker
from mouse_controller import MouseController


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestThreeFingerPreset(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.config.control_preset = "3-Finger Control"
        self.config.pointer_source = "finger"
        self.config.pointer_enabled = True
        self.config.hand_tracking_enabled = True
        self.config.hand_thumb_trigger_enabled = False
        self.config.hand_smart_sensitivity = True
        self.config.hand_cursor_speed = 2.0
        self.config.click_cooldown = 0.35

        self.tracker = HandTracker(self.config)
        self.mouse = MouseController(self.config)

    def _create_synthetic_landmarks(self, idx_ext: bool, mid_ext: bool, ring_ext: bool, pinky_ext: bool):
        """Creates synthetic MediaPipe 21 landmarks with exact finger extension flags."""
        landmarks = [DummyLandmark(0.5, 0.5, 0.0)] * 21
        wrist = DummyLandmark(0.5, 0.8, 0.0)
        landmarks[self.tracker.WRIST] = wrist
        landmarks[self.tracker.MIDDLE_MCP] = DummyLandmark(0.5, 0.5, 0.0)

        # Thumb resting / tucked
        landmarks[self.tracker.THUMB_CMC] = DummyLandmark(0.45, 0.7, 0.0)
        landmarks[self.tracker.THUMB_MCP] = DummyLandmark(0.42, 0.65, 0.0)
        landmarks[self.tracker.THUMB_IP] = DummyLandmark(0.40, 0.60, 0.0)
        landmarks[self.tracker.THUMB_TIP] = DummyLandmark(0.38, 0.58, 0.0)

        # Helper to set finger landmarks
        def set_finger(tip_idx, pip_idx, mcp_idx, base_x, is_extended):
            landmarks[mcp_idx] = DummyLandmark(base_x, 0.50, 0.0)
            if is_extended:
                landmarks[pip_idx] = DummyLandmark(base_x, 0.35, 0.0)
                landmarks[tip_idx] = DummyLandmark(base_x, 0.20, 0.0)
            else:
                landmarks[pip_idx] = DummyLandmark(base_x, 0.55, 0.0)
                landmarks[tip_idx] = DummyLandmark(base_x, 0.65, 0.0)

        set_finger(self.tracker.INDEX_TIP, self.tracker.INDEX_PIP, self.tracker.INDEX_MCP, 0.35, idx_ext)
        set_finger(self.tracker.MIDDLE_TIP, self.tracker.MIDDLE_PIP, self.tracker.MIDDLE_MCP, 0.50, mid_ext)
        set_finger(self.tracker.RING_TIP, self.tracker.RING_PIP, self.tracker.RING_MCP, 0.65, ring_ext)
        set_finger(self.tracker.PINKY_TIP, self.tracker.PINKY_PIP, self.tracker.PINKY_MCP, 0.80, pinky_ext)

        return landmarks

    def test_preset_configuration_definition(self):
        """Verifies that 3-Finger Control exists in CONTROL_PRESETS dictionary."""
        self.assertIn("3-Finger Control", CONTROL_PRESETS)
        preset_cfg = CONTROL_PRESETS["3-Finger Control"]
        self.assertEqual(preset_cfg["pointer_source"], "finger")
        self.assertTrue(preset_cfg["hand_tracking_enabled"])
        self.assertFalse(preset_cfg["hand_thumb_trigger_enabled"])

    def test_three_finger_move_classification_and_motion(self):
        """When Index, Middle, and Ring are extended (Pinky folded), cursor moves and no click fires."""
        landmarks = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=False)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "3-Finger Move")

        # Verify mouse action mapping is None
        action = self.mouse._map_hand_gesture_to_action(gesture)
        self.assertIn(action, (None, "None"))

        # Verify frame processing produces valid cursor position
        class DummyResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, detected_g, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 960.0, places=2)
        self.assertAlmostEqual(pos[1], 540.0, places=2)

    def test_three_finger_cursor_tracks_middle_finger_not_index(self):
        """In 3-Finger mode, cursor tracks middle finger tip. Moving index finger alone must not move cursor."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: Stable 3-finger pose
        landmarks1 = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=False)
        class DummyResult1:
            multi_hand_landmarks = [MagicMock(landmark=landmarks1)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult1())
        pos1, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
        self.assertIsNotNone(pos1)

        # Frame 2: Index finger lowers/moves down (simulating user beginning to curl index finger for left click)
        # Middle finger remains completely steady at base_x=0.50, tip_y=0.20
        landmarks2 = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=False)
        # Lower only the index finger tip
        landmarks2[self.tracker.INDEX_TIP] = DummyLandmark(0.35, 0.35, 0.0)

        class DummyResult2:
            multi_hand_landmarks = [MagicMock(landmark=landmarks2)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult2())
        time.sleep(0.02)  # small delta time
        pos2, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(pos1[0], pos1[1]))

        # Since middle finger did not move, cursor should remain at the same position (no drift)
        self.assertAlmostEqual(pos2[0], pos1[0], delta=1.0)
        self.assertAlmostEqual(pos2[1], pos1[1], delta=1.0)

        # Frame 3: Now move the middle finger tip to the right
        landmarks3 = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=False)
        landmarks3[self.tracker.MIDDLE_TIP] = DummyLandmark(0.55, 0.20, 0.0)

        class DummyResult3:
            multi_hand_landmarks = [MagicMock(landmark=landmarks3)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult3())
        time.sleep(0.02)
        pos3, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(pos2[0], pos2[1]))
        # Cursor should now have moved right (x increases)
        self.assertGreater(pos3[0], pos2[0])

    def test_index_curled_triggers_left_click(self):
        """When Index is curled while Middle and Ring are extended, Left Click is classified and mapped."""
        landmarks = self._create_synthetic_landmarks(idx_ext=False, mid_ext=True, ring_ext=True, pinky_ext=False)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "3-Finger Left Click")

        action = self.mouse._map_hand_gesture_to_action(gesture)
        self.assertEqual(action, "Left Click")

    def test_ring_curled_triggers_right_click(self):
        """When Ring is curled while Index and Middle are extended, Right Click is classified and mapped."""
        landmarks = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=False, pinky_ext=False)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "3-Finger Right Click (Ring)")

        action = self.mouse._map_hand_gesture_to_action(gesture)
        self.assertEqual(action, "Right Click")

    def test_middle_curled_triggers_right_click(self):
        """When Middle is curled while Index and Ring are extended, Right Click is classified and mapped."""
        landmarks = self._create_synthetic_landmarks(idx_ext=True, mid_ext=False, ring_ext=True, pinky_ext=False)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "3-Finger Right Click (Middle)")

        action = self.mouse._map_hand_gesture_to_action(gesture)
        self.assertEqual(action, "Right Click")

    def test_middle_curled_custom_middle_action(self):
        """Verifies that middle curl can map to Middle Click if configured."""
        self.config.hand_three_finger_middle_action = "Middle Click"
        action = self.mouse._map_hand_gesture_to_action("3-Finger Right Click (Middle)")
        self.assertEqual(action, "Middle Click")

    def test_all_curled_fist_freezes_cursor(self):
        """When all 3 fingers are folded (Fist), cursor is frozen (None) and no click fires."""
        landmarks = self._create_synthetic_landmarks(idx_ext=False, mid_ext=False, ring_ext=False, pinky_ext=False)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Fist")

        class DummyResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
        self.assertIsNone(pos, "Cursor must freeze when all 3 fingers are curled")
        self.assertIsNone(self.tracker.accum_screen_x)

    def test_all_extended_open_palm_freezes_cursor(self):
        """When all fingers including pinky are open (Open Palm), cursor is frozen (None)."""
        landmarks = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=True)
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Open Palm")

        class DummyResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
        self.assertIsNone(pos, "Cursor must freeze when all fingers are extended (Open Palm)")

    def test_click_anti_wobble_anchors_cursor_position(self):
        """Verifies that while performing a click gesture, cursor position is anchored without wobble."""
        landmarks_move = self._create_synthetic_landmarks(idx_ext=True, mid_ext=True, ring_ext=True, pinky_ext=False)

        class DummyResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks_move)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyResult())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Move to (500, 400)
        self.tracker.smoothed_x = 500.0
        self.tracker.smoothed_y = 400.0

        # Now curl index finger for Left Click
        landmarks_click = self._create_synthetic_landmarks(idx_ext=False, mid_ext=True, ring_ext=True, pinky_ext=False)
        DummyResult.multi_hand_landmarks = [MagicMock(landmark=landmarks_click)]

        pos, _, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(500, 400))
        self.assertEqual(pos, (500.0, 400.0), "Cursor must anchor at current smoothed position during click")

    @patch.object(MouseController, "_execute_action_unlocked")
    def test_mouse_controller_debounced_clicks(self, mock_execute):
        """Verifies discrete clicks are fired once upon gesture start and debounced."""
        self.mouse.process_hand_gesture("3-Finger Left Click")
        self.assertEqual(mock_execute.call_count, 1)
        mock_execute.assert_called_with("Left Click")

        # Immediate next frame with same gesture: should not double-click
        self.mouse.process_hand_gesture("3-Finger Left Click")
        self.assertEqual(mock_execute.call_count, 1)

        # Transition back to moving
        self.mouse.process_hand_gesture("3-Finger Move")
        self.assertIsNone(self.mouse.active_hand_gesture)

        # Click again after cooldown: should fire
        time.sleep(0.01)
        self.mouse.last_action_time = time.perf_counter() - 0.50
        self.mouse.process_hand_gesture("3-Finger Right Click (Ring)")
        self.assertEqual(mock_execute.call_count, 2)


if __name__ == "__main__":
    unittest.main()

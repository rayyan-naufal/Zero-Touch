"""Unit tests for Hand Tracking, Gesture Classification, and Mouse Integration.
Verifies landmark geometry, temporal persistence buffers, Hold Left Click / Drag state machine,
discrete clicks, and configuration persistence.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from config import AppConfig, ConfigManager, CONTROL_PRESETS
from hand_tracker import HandTracker
from mouse_controller import MouseController


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def build_mock_hand_landmarks(gesture: str) -> list:
    """Constructs 21 synthetic hand landmarks for precise geometric testing.
    Wrist is placed at (0.5, 0.8), palm MCPs at y=0.55.
    """
    pts = [MockLandmark(0.5, 0.8) for _ in range(21)]

    # MCP (Knuckle) bases
    pts[HandTracker.WRIST] = MockLandmark(0.5, 0.8)
    pts[HandTracker.THUMB_CMC] = MockLandmark(0.42, 0.72)
    pts[HandTracker.THUMB_MCP] = MockLandmark(0.38, 0.65)
    pts[HandTracker.THUMB_IP] = MockLandmark(0.35, 0.58)
    pts[HandTracker.THUMB_TIP] = MockLandmark(0.32, 0.52)

    pts[HandTracker.INDEX_MCP] = MockLandmark(0.44, 0.55)
    pts[HandTracker.MIDDLE_MCP] = MockLandmark(0.50, 0.55)
    pts[HandTracker.RING_MCP] = MockLandmark(0.56, 0.55)
    pts[HandTracker.PINKY_MCP] = MockLandmark(0.62, 0.58)

    # By default, fold all non-thumb fingers into palm
    for mcp_idx, pip_idx, dip_idx, tip_idx, x_col in [
        (HandTracker.INDEX_MCP, HandTracker.INDEX_PIP, HandTracker.INDEX_DIP, HandTracker.INDEX_TIP, 0.44),
        (HandTracker.MIDDLE_MCP, HandTracker.MIDDLE_PIP, HandTracker.MIDDLE_DIP, HandTracker.MIDDLE_TIP, 0.50),
        (HandTracker.RING_MCP, HandTracker.RING_PIP, HandTracker.RING_DIP, HandTracker.RING_TIP, 0.56),
        (HandTracker.PINKY_MCP, HandTracker.PINKY_PIP, HandTracker.PINKY_DIP, HandTracker.PINKY_TIP, 0.62),
    ]:
        pts[pip_idx] = MockLandmark(x_col, 0.65)
        pts[dip_idx] = MockLandmark(x_col, 0.70)
        pts[tip_idx] = MockLandmark(x_col, 0.73)  # Folded: closer to wrist than PIP

    if gesture == "Peace Sign":
        # Index extended: TIP higher/further from wrist than PIP
        pts[HandTracker.INDEX_PIP] = MockLandmark(0.40, 0.45)
        pts[HandTracker.INDEX_DIP] = MockLandmark(0.38, 0.35)
        pts[HandTracker.INDEX_TIP] = MockLandmark(0.35, 0.25)
        # Middle extended (V-shape separation)
        pts[HandTracker.MIDDLE_PIP] = MockLandmark(0.52, 0.45)
        pts[HandTracker.MIDDLE_DIP] = MockLandmark(0.55, 0.35)
        pts[HandTracker.MIDDLE_TIP] = MockLandmark(0.58, 0.25)

    elif gesture == "Pinch Index":
        # Thumb tip and Index tip touching at (0.42, 0.48)
        pts[HandTracker.THUMB_TIP] = MockLandmark(0.42, 0.48)
        pts[HandTracker.INDEX_TIP] = MockLandmark(0.422, 0.482)

    elif gesture == "Pinch Middle":
        # Thumb tip and Middle tip touching at (0.45, 0.48)
        pts[HandTracker.THUMB_TIP] = MockLandmark(0.45, 0.48)
        pts[HandTracker.MIDDLE_TIP] = MockLandmark(0.452, 0.482)

    elif gesture == "Pointing":
        # Only Index extended
        pts[HandTracker.INDEX_PIP] = MockLandmark(0.44, 0.45)
        pts[HandTracker.INDEX_DIP] = MockLandmark(0.44, 0.35)
        pts[HandTracker.INDEX_TIP] = MockLandmark(0.44, 0.25)

    elif gesture == "Fist":
        # All fingers folded close to wrist
        pass

    elif gesture == "Open Palm":
        # All fingers extended
        for pip_idx, dip_idx, tip_idx, x_col in [
            (HandTracker.INDEX_PIP, HandTracker.INDEX_DIP, HandTracker.INDEX_TIP, 0.44),
            (HandTracker.MIDDLE_PIP, HandTracker.MIDDLE_DIP, HandTracker.MIDDLE_TIP, 0.50),
            (HandTracker.RING_PIP, HandTracker.RING_DIP, HandTracker.RING_TIP, 0.56),
            (HandTracker.PINKY_PIP, HandTracker.PINKY_DIP, HandTracker.PINKY_TIP, 0.62),
        ]:
            pts[pip_idx] = MockLandmark(x_col, 0.45)
            pts[dip_idx] = MockLandmark(x_col, 0.35)
            pts[tip_idx] = MockLandmark(x_col, 0.25)

    return pts


class TestHandConfig(unittest.TestCase):
    """Verifies AppConfig serialization, deserialization, and backward compatibility for hand tracking."""

    def test_default_hand_config_values(self):
        cfg = AppConfig()
        self.assertTrue(cfg.hand_tracking_enabled)
        self.assertEqual(cfg.control_preset, "Thumb Trigger")
        self.assertEqual(cfg.pointer_source, "finger")
        self.assertTrue(cfg.hand_thumb_trigger_enabled)
        self.assertEqual(cfg.hand_thumb_hold_threshold, 1.0)
        self.assertEqual(cfg.hand_peace_action, "None")
        self.assertEqual(cfg.hand_pinch_index_action, "None")
        self.assertEqual(cfg.hand_pinch_middle_action, "None")
        self.assertEqual(cfg.hand_point_action, "None")
        self.assertEqual(cfg.hand_fist_action, "None")
        self.assertEqual(cfg.hand_cursor_speed, 2.0)
        self.assertTrue(cfg.hand_right_only)

    def test_hand_config_serialization(self):
        cfg = AppConfig(
            hand_tracking_enabled=False,
            pointer_source="head",
            control_preset="Custom",
            hand_thumb_trigger_enabled=False,
            hand_pinch_threshold=0.10,
            hand_peace_action="Right Click",
            hand_pinch_index_action="Left Click",
        )
        d = cfg.to_dict()
        restored = AppConfig.from_dict(d)
        self.assertFalse(restored.hand_tracking_enabled)
        self.assertEqual(restored.pointer_source, "head")
        self.assertEqual(restored.control_preset, "Custom")
        self.assertFalse(restored.hand_thumb_trigger_enabled)
        self.assertEqual(restored.hand_pinch_threshold, 0.10)
        self.assertEqual(restored.hand_peace_action, "Right Click")
        self.assertEqual(restored.hand_pinch_index_action, "Left Click")

    def test_backward_compatibility_missing_hand_fields(self):
        legacy = {
            "cursor_speed": 2.5,
            "pointer_enabled": True,
            "clicking_enabled": True,
        }
        cfg = AppConfig.from_dict(legacy)
        self.assertTrue(cfg.hand_tracking_enabled)
        self.assertEqual(cfg.pointer_source, "finger")
        self.assertEqual(cfg.control_preset, "Thumb Trigger")
        self.assertEqual(cfg.hand_peace_action, "None")


class TestHandGestureClassification(unittest.TestCase):
    """Verifies invariant geometric classification and temporal confirmation buffer."""

    def setUp(self):
        self.config = AppConfig()
        self.tracker = HandTracker(self.config)

    def tearDown(self):
        self.tracker.close()

    def test_classify_peace_sign(self):
        landmarks = build_mock_hand_landmarks("Peace Sign")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Peace Sign")

    def test_classify_pinch_index(self):
        landmarks = build_mock_hand_landmarks("Pinch Index")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Pinch Index")
        self.assertLess(dist, 0.08 * 3.2)

    def test_classify_pinch_middle(self):
        landmarks = build_mock_hand_landmarks("Pinch Middle")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Pinch Middle")

    def test_classify_pointing(self):
        landmarks = build_mock_hand_landmarks("Pointing")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Pointing")

    def test_classify_fist(self):
        landmarks = build_mock_hand_landmarks("Fist")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Fist")

    def test_classify_open_palm(self):
        landmarks = build_mock_hand_landmarks("Open Palm")
        gesture, dist = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Open Palm")

    def test_3d_spatial_distance_with_depth(self):
        p1 = MockLandmark(0.0, 0.0, z=0.0)
        p2 = MockLandmark(0.3, 0.4, z=0.0)
        # Standard 2D 3-4-5 triangle
        self.assertAlmostEqual(self.tracker._dist(p1, p2), 0.5, places=4)
        self.assertAlmostEqual(self.tracker._dist_2d(p1, p2), 0.5, places=4)

        # In 3D with downweighted z=1.2: dz = 1.2 * 0.4 = 0.48
        # sqrt(0.3^2 + 0.4^2 + 0.48^2) = sqrt(0.25 + 0.2304) = sqrt(0.4804) = 0.6931089
        p3 = MockLandmark(0.3, 0.4, z=1.2)
        self.assertAlmostEqual(self.tracker._dist(p1, p3), 0.6931089, places=4)
        self.assertAlmostEqual(self.tracker._dist_2d(p1, p3), 0.5, places=4)


class TestMouseControllerHandIntegration(unittest.TestCase):
    """Verifies MouseController hand gesture execution, Hold Left Click state machine, and discrete actions."""

    def setUp(self):
        self.config = AppConfig(
            hand_tracking_enabled=True,
            hand_peace_action="Left Click",
            hand_pinch_index_action="Hold Left Click",
            hand_pinch_middle_action="Right Click",
            click_cooldown=0.1,
            mouse_override=False,
        )
        self.controller = MouseController(self.config)

    def test_peace_sign_discrete_click(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            # First frame of Peace Sign fires click
            self.controller.process_hand_gesture("Peace Sign")
            mock_exec.assert_called_once_with("Left Click")

            # Holding Peace Sign continuously does NOT spam clicks
            mock_exec.reset_mock()
            self.controller.process_hand_gesture("Peace Sign")
            mock_exec.assert_not_called()

            # Releasing gesture resets state
            self.controller.process_hand_gesture(None)
            self.assertIsNone(self.controller.active_hand_gesture)

            # Showing Peace Sign again after cooldown fires click again
            time.sleep(0.12)
            self.controller.process_hand_gesture("Peace Sign")
            mock_exec.assert_called_once_with("Left Click")

    def test_pinch_index_hold_left_click_and_release(self):
        with patch.object(self.controller, "_start_drag_unlocked") as mock_start, \
             patch.object(self.controller, "_release_drag_unlocked") as mock_release:
            
            self.assertFalse(self.controller.is_hand_holding_drag)

            # 1. Pinch Index engages Hold Left Click / Drag
            self.controller.process_hand_gesture("Pinch Index")
            self.assertTrue(self.controller.is_hand_holding_drag)
            mock_start.assert_called_once()
            mock_release.assert_not_called()

            # 2. Maintaining pinch keeps hold active without restarting
            mock_start.reset_mock()
            self.controller.process_hand_gesture("Pinch Index")
            self.assertTrue(self.controller.is_hand_holding_drag)
            mock_start.assert_not_called()
            mock_release.assert_not_called()

            # 3. Releasing pinch (fingers separate) drops/releases drag
            self.controller.process_hand_gesture(None)
            self.assertFalse(self.controller.is_hand_holding_drag)
            mock_release.assert_called_once()

    def test_drag_release_on_master_pause(self):
        with patch.object(self.controller, "_start_drag_unlocked"), \
             patch.object(self.controller, "_release_drag_unlocked") as mock_release:

            # Start hold drag
            self.controller.process_hand_gesture("Pinch Index")
            self.assertTrue(self.controller.is_hand_holding_drag)

            # Master pause toggled
            self.controller.toggle_pause()
            self.assertTrue(self.controller.is_paused)
            self.assertFalse(self.controller.is_hand_holding_drag)
            mock_release.assert_called_once()

    def test_rapid_transition_does_not_swallow_gesture(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            # 1. Action 1 fires at t=0
            self.controller.process_hand_gesture("Peace Sign")
            mock_exec.assert_called_once_with("Left Click")

            # 2. Immediate transition to Pinch Middle within cooldown period (< 0.1s)
            mock_exec.reset_mock()
            self.controller.process_hand_gesture("Pinch Middle")
            # Should not fire yet due to cooldown
            mock_exec.assert_not_called()

            # 3. Once cooldown elapses, continuing or repeating Pinch Middle executes cleanly
            time.sleep(0.12)
            self.controller.process_hand_gesture("Pinch Middle")
            mock_exec.assert_called_once_with("Right Click")

    def test_hand_tracking_disabled_suppression(self):
        with patch.object(self.controller, "_execute_action_unlocked") as mock_exec:
            self.config.hand_tracking_enabled = False
            self.controller.process_hand_gesture("Peace Sign")
            mock_exec.assert_not_called()


class TestFingerPointerGestures(unittest.TestCase):
    """Verifies that in finger pointer mode:
    - Fist does nothing (target_cursor_pos is None)
    - Pointing moves cursor (target_cursor_pos is computed)
    - Peace sign anchors cursor and triggers Left Click
    """

    def setUp(self):
        self.config = AppConfig(
            pointer_source="finger",
            pointer_enabled=True,
            hand_tracking_enabled=True,
            hand_thumb_trigger_enabled=False,  # Classic mode testing
            hand_peace_action="Left Click",
            hand_fist_action="None",
            hand_point_action="None",
        )
        self.tracker = HandTracker(self.config)

    def tearDown(self):
        self.tracker.close()

    def test_fist_does_not_move_cursor(self):
        landmarks = build_mock_hand_landmarks("Fist")
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Fist")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock(landmark=landmarks)]
        mock_results.multi_handedness = None

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pos, gest, _, _ = self.tracker.process_frame(frame, 1920, 1080)
            self.assertIsNone(pos, "Fist must not move the cursor")

    def test_open_palm_does_not_move_cursor(self):
        landmarks = build_mock_hand_landmarks("Open Palm")
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Open Palm")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock(landmark=landmarks)]
        mock_results.multi_handedness = None

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pos, gest, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNone(pos, "Open Palm (all fingers extended) must not move the cursor")

    def test_pointing_moves_cursor(self):
        landmarks = build_mock_hand_landmarks("Pointing")
        gesture, _ = self.tracker._classify_gesture(landmarks)
        self.assertEqual(gesture, "Pointing")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock(landmark=landmarks)]
        mock_results.multi_handedness = None

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pos, gest, _, _ = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNotNone(pos, "Pointing must move the cursor")
            self.assertIsInstance(pos, tuple)
            self.assertEqual(len(pos), 2)


class TestThumbTriggerStateMachine(unittest.TestCase):
    """Verifies Thumb Tap (<1s) for Left Click and Hold (>=1s) for Drag / Drop."""

    def setUp(self):
        self.config = AppConfig(
            pointer_source="finger",
            pointer_enabled=True,
            hand_tracking_enabled=True,
            hand_thumb_trigger_enabled=True,
            hand_thumb_hold_threshold=1.0,
            hand_cursor_speed=2.0,
        )
        self.tracker = HandTracker(self.config)

    def tearDown(self):
        self.tracker.close()

    def _build_frame_with_thumb(self, thumb_open: bool):
        landmarks = build_mock_hand_landmarks("Pointing")
        if thumb_open:
            # Move thumb tip far from index MCP to exceed open ratio (0.65)
            landmarks[HandTracker.THUMB_TIP] = MockLandmark(0.10, 0.55)
        else:
            # Tuck thumb tip close to index MCP to fall below close ratio (0.48)
            landmarks[HandTracker.THUMB_TIP] = MockLandmark(0.43, 0.55)

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [MagicMock(landmark=landmarks)]
        mock_results.multi_handedness = None
        return mock_results

    def test_thumb_tap_click_under_one_second(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Start tucked
        mock_closed = self._build_frame_with_thumb(thumb_open=False)
        with patch.object(self.tracker.hands, "process", return_value=mock_closed):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertFalse(self.tracker.thumb_is_open)
            self.assertEqual(self.tracker.thumb_state, "IDLE")
            self.assertIsNone(gest)

        # 2. Open thumb (<1s) -> PENDING_TAP
        mock_open = self._build_frame_with_thumb(thumb_open=True)
        with patch.object(self.tracker.hands, "process", return_value=mock_open):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertTrue(self.tracker.thumb_is_open)
            self.assertEqual(self.tracker.thumb_state, "PENDING_TAP")
            self.assertIsNone(gest)

        # 3. Close thumb within 1s -> Emits Left Click
        with patch.object(self.tracker.hands, "process", return_value=mock_closed):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertFalse(self.tracker.thumb_is_open)
            self.assertEqual(self.tracker.thumb_state, "IDLE")
            self.assertEqual(gest, "Left Click")

    def test_thumb_hold_drag_over_one_second(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_open = self._build_frame_with_thumb(thumb_open=True)
        mock_closed = self._build_frame_with_thumb(thumb_open=False)

        # 1. Open thumb
        with patch.object(self.tracker.hands, "process", return_value=mock_open):
            self.tracker.process_frame(frame, 1920, 1080)
            self.assertEqual(self.tracker.thumb_state, "PENDING_TAP")

            # 2. Advance time past hold threshold (1.0s)
            self.tracker.thumb_opened_time = time.perf_counter() - 1.2
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertEqual(self.tracker.thumb_state, "HOLDING_DRAG")
            self.assertEqual(gest, "Hold Left Click")

        # 3. Close thumb after hold -> Emits Release Drag
        with patch.object(self.tracker.hands, "process", return_value=mock_closed):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertEqual(self.tracker.thumb_state, "IDLE")
            self.assertEqual(gest, "Release Drag")

    def test_tracking_loss_releases_drag(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_open = self._build_frame_with_thumb(thumb_open=True)
        with patch.object(self.tracker.hands, "process", return_value=mock_open):
            self.tracker.process_frame(frame, 1920, 1080)
            self.tracker.thumb_opened_time = time.perf_counter() - 1.2
            self.tracker.process_frame(frame, 1920, 1080)
            self.assertEqual(self.tracker.thumb_state, "HOLDING_DRAG")

        # Hand lost (no landmarks)
        mock_lost = MagicMock()
        mock_lost.multi_hand_landmarks = None
        with patch.object(self.tracker.hands, "process", return_value=mock_lost):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080)
            self.assertEqual(gest, "Release Drag")
            self.assertEqual(self.tracker.thumb_state, "IDLE")


class TestControlPresets(unittest.TestCase):
    """Verifies CONTROL_PRESETS structure and configuration application."""

    def test_presets_exist(self):
        expected_presets = [
            "Thumb Trigger",
            "Peace & Fist (Classic)",
            "Pinch & Point",
            "Head & Face (Handsfree)",
            "Custom",
        ]
        for name in expected_presets:
            self.assertIn(name, CONTROL_PRESETS)

    def test_thumb_trigger_preset_values(self):
        p = CONTROL_PRESETS["Thumb Trigger"]
        self.assertEqual(p["pointer_source"], "finger")
        self.assertTrue(p["hand_thumb_trigger_enabled"])
        self.assertEqual(p["hand_thumb_hold_threshold"], 1.0)
        self.assertEqual(p["hand_peace_action"], "None")

    def test_peace_classic_preset_values(self):
        p = CONTROL_PRESETS["Peace & Fist (Classic)"]
        self.assertEqual(p["pointer_source"], "finger")
        self.assertFalse(p["hand_thumb_trigger_enabled"])
        self.assertEqual(p["hand_peace_action"], "Left Click")


class TestHandednessSelection(unittest.TestCase):
    """Verifies Right Hand Only filtering, Left Hand suppression, and dual-hand priority."""

    def setUp(self):
        self.config = AppConfig()
        self.config.hand_tracking_enabled = True
        self.config.hand_right_only = True
        self.config.flip_horizontal = True
        self.tracker = HandTracker(self.config)

    def _make_hand(self, gesture: str, label: str):
        landmarks = build_mock_hand_landmarks(gesture)
        hand_lm = MagicMock(landmark=landmarks)
        h_class = MagicMock()
        h_class.classification = [MagicMock(label=label)]
        return hand_lm, h_class

    def test_default_config_hand_right_only(self):
        cfg = AppConfig()
        self.assertTrue(cfg.hand_right_only)

    def test_right_hand_accepted_when_right_only_enabled(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_lm, h_class = self._make_hand("Pointing", "Right")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]
        mock_results.multi_handedness = [h_class]

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNotNone(pos)
            self.assertFalse(tele.get("hand_ignored", False))
            self.assertEqual(tele.get("handedness"), "Right")

    def test_left_hand_ignored_when_right_only_enabled(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_lm, h_class = self._make_hand("Pointing", "Left")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]
        mock_results.multi_handedness = [h_class]

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNone(pos, "Left hand must not produce cursor position when hand_right_only is True")
            self.assertTrue(tele.get("hand_ignored"))
            self.assertIn("Ignored", tele.get("handedness", ""))
            self.assertIsNone(tele.get("hand_gesture"))

    def test_left_hand_accepted_when_right_only_disabled(self):
        import numpy as np
        self.config.hand_right_only = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_lm, h_class = self._make_hand("Pointing", "Left")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]
        mock_results.multi_handedness = [h_class]

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNotNone(pos, "Left hand must be accepted when hand_right_only is False")
            self.assertFalse(tele.get("hand_ignored", False))
            self.assertEqual(tele.get("handedness"), "Left")

    def test_right_hand_prioritized_when_both_hands_detected(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        left_lm, left_class = self._make_hand("Fist", "Left")
        right_lm, right_class = self._make_hand("Pointing", "Right")

        # Left hand first in list, Right hand second
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [left_lm, right_lm]
        mock_results.multi_handedness = [left_class, right_class]

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNotNone(pos, "Right hand must be prioritized over Left hand")
            self.assertFalse(tele.get("hand_ignored", False))
            self.assertEqual(tele.get("handedness"), "Right")

    def test_unmirrored_handedness_mapping(self):
        import numpy as np
        self.config.flip_horizontal = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # When flip_horizontal=False, raw MediaPipe "Left" corresponds to physical "Right"
        hand_lm, h_class = self._make_hand("Pointing", "Left")

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]
        mock_results.multi_handedness = [h_class]

        with patch.object(self.tracker.hands, "process", return_value=mock_results):
            pos, gest, _, tele = self.tracker.process_frame(frame, 1920, 1080, current_mouse_pos=(960, 540))
            self.assertIsNotNone(pos)
            self.assertEqual(tele.get("handedness"), "Right")


if __name__ == "__main__":
    unittest.main()

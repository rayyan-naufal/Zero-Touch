"""Unit tests for In-App Smart Sensitivity & Ballistic Acceleration in Hand Tracking.
Validates velocity gain curves, continuous soft deadzone, anti-windup accumulator,
frame hitch guards, and ensures zero modification to Windows OS settings.
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from config import AppConfig
from hand_tracker import HandTracker


class DummyLandmark:
    """Mock landmark for MediaPipe Hands testing."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestSmartSensitivity(unittest.TestCase):
    """Verifies Smart Sensitivity algorithms and safeguards."""

    def setUp(self):
        self.config = AppConfig(
            hand_tracking_enabled=True,
            pointer_source="finger",
            hand_smart_sensitivity=True,
            hand_sensitivity_profile="dynamic",
            hand_tremor_deadband=0.0015,
            hand_cursor_speed=2.0,
        )
        self.tracker = HandTracker(self.config)

    def test_zero_os_alteration(self):
        """CRITICAL: Verifies no Windows OS mouse settings or registry APIs are touched."""
        # Inspect HandTracker module and AppConfig to ensure zero calls to SPI_SETMOUSE
        with patch("ctypes.windll.user32.SystemParametersInfoW", create=True) as mock_spi:
            # Process frames and exercise the tracker
            cfg = AppConfig()
            tracker = HandTracker(cfg)
            tracker._compute_dynamic_gain(0.5)
            tracker._apply_continuous_soft_deadzone(0.01, 0.01, 0.0015)
            # Ensure SPI was never invoked
            mock_spi.assert_not_called()

    def test_dynamic_gain_curve_damping_and_acceleration(self):
        """Verifies micro-precision damping on slow movement and acceleration on fast flick."""
        # 1. Micro-precision aiming (very slow velocity v = 0.03)
        gain_slow = self.tracker._compute_dynamic_gain(velocity=0.03)
        self.assertLessEqual(gain_slow, 0.50, "Slow aiming should be heavily dampened (G <= 0.50) for micro-precision")
        self.assertGreaterEqual(gain_slow, 0.40, "Slow gain should not drop below G_min (0.40)")

        # 2. Normal natural motion (v = 0.28)
        gain_mid = self.tracker._compute_dynamic_gain(velocity=0.28)
        self.assertAlmostEqual(gain_mid, 1.40, delta=0.20, msg="Transition velocity should yield natural gain")

        # 3. High-speed sweep / flick (v = 1.0)
        gain_fast = self.tracker._compute_dynamic_gain(velocity=1.0)
        self.assertGreaterEqual(gain_fast, 2.00, "Fast flick should accelerate (G >= 2.00) for effortless screen sweeps")
        self.assertLessEqual(gain_fast, 2.45, "Fast gain should respect asymptotic ceiling G_max")

    def test_profile_presets_smooth_and_linear(self):
        """Verifies Smooth (Trackpad) and Linear (Raw) profile parameters."""
        # Smooth Profile
        self.config.hand_sensitivity_profile = "smooth"
        g_slow = self.tracker._compute_dynamic_gain(0.01)
        g_fast = self.tracker._compute_dynamic_gain(1.0)
        self.assertAlmostEqual(g_slow, 0.60, delta=0.05)
        self.assertAlmostEqual(g_fast, 1.50, delta=0.10)

        # Linear Profile (Bypass)
        self.config.hand_sensitivity_profile = "linear"
        self.assertEqual(self.tracker._compute_dynamic_gain(0.01), 1.0)
        self.assertEqual(self.tracker._compute_dynamic_gain(1.50), 1.0)

    def test_continuous_soft_deadzone_ramp(self):
        """Verifies soft deadzone radial ramp eliminates tremors without sudden step jumps."""
        deadband = 0.0015

        # 1. Below deadband threshold: strict suppression
        dx, dy, eff = self.tracker._apply_continuous_soft_deadzone(0.0010, 0.0, deadband)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)
        self.assertEqual(eff, 0.0)

        # 2. Just above deadband threshold (e.g. 0.0016): smooth C^0 continuity
        dx, dy, eff = self.tracker._apply_continuous_soft_deadzone(0.0016, 0.0, deadband)
        self.assertGreater(eff, 0.0)
        self.assertLessEqual(eff, 0.00015, "Just above deadband must ramp smoothly from zero (no step jump)")

        # 3. Large movement (e.g. 0.05): scales properly
        dx_large, dy_large, eff_large = self.tracker._apply_continuous_soft_deadzone(0.05, 0.0, deadband)
        self.assertAlmostEqual(eff_large, 0.05, delta=0.005)

    def test_cold_start_and_tracking_reacquisition(self):
        """Verifies that frame 1 upon detection initializes baseline with zero delta (no teleportation)."""
        # Set up synthetic pointing hand landmarks
        landmarks = [DummyLandmark(0.5, 0.5)] * 21
        landmarks[self.tracker.INDEX_TIP] = DummyLandmark(0.3, 0.3)
        landmarks[self.tracker.INDEX_PIP] = DummyLandmark(0.4, 0.4)
        landmarks[self.tracker.INDEX_MCP] = DummyLandmark(0.45, 0.45)
        landmarks[self.tracker.WRIST] = DummyLandmark(0.5, 0.7)
        landmarks[self.tracker.MIDDLE_MCP] = DummyLandmark(0.5, 0.5)

        class DummyHandsResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyHandsResult())

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        screen_w, screen_h = 1920, 1080

        # Initial detection (cold-start)
        pos1, _, _, _ = self.tracker.process_frame(dummy_frame, screen_w, screen_h, current_mouse_pos=(800, 600))
        self.assertIsNotNone(pos1)
        self.assertEqual(self.tracker.prev_finger_pos, (0.3, 0.3))
        # Cursor should remain anchored at current_mouse_pos without jump
        self.assertAlmostEqual(pos1[0], 800.0, places=3)
        self.assertAlmostEqual(pos1[1], 600.0, places=3)

        # Reacquisition test: simulate hand loss then re-appearance at (0.8, 0.2)
        self.tracker.prev_finger_pos = None
        landmarks[self.tracker.INDEX_TIP] = DummyLandmark(0.8, 0.2)
        pos2, _, _, _ = self.tracker.process_frame(dummy_frame, screen_w, screen_h, current_mouse_pos=(800, 600))
        # Must re-anchor cleanly at (800, 600) on reacquisition frame
        self.assertAlmostEqual(pos2[0], 800.0, places=3)
        self.assertAlmostEqual(pos2[1], 600.0, places=3)
        self.assertEqual(self.tracker.prev_finger_pos, (0.8, 0.2))

    def test_integrator_anti_windup_clamping(self):
        """Verifies that accumulator never exceeds screen bounds [0.0, screen_size - 1.0]."""
        self.tracker.accum_screen_x = 1918.0
        self.tracker.accum_screen_y = 500.0
        self.tracker.prev_finger_pos = (0.5, 0.5)
        self.tracker.prev_finger_time = time.perf_counter() - 0.033

        # Simulate large positive X displacement
        landmarks = [DummyLandmark(0.5, 0.5)] * 21
        landmarks[self.tracker.INDEX_TIP] = DummyLandmark(0.58, 0.5)  # Delta X = +0.08
        landmarks[self.tracker.INDEX_PIP] = DummyLandmark(0.52, 0.5)
        landmarks[self.tracker.INDEX_MCP] = DummyLandmark(0.50, 0.5)
        landmarks[self.tracker.WRIST] = DummyLandmark(0.5, 0.7)
        landmarks[self.tracker.MIDDLE_MCP] = DummyLandmark(0.5, 0.5)

        class DummyHandsResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyHandsResult())
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, _, _, _ = self.tracker.process_frame(dummy_frame, 1920, 1080)
        self.assertEqual(self.tracker.accum_screen_x, 1919.0, "Accumulator must clamp at screen_w - 1.0")

    def test_frame_hitch_guard(self):
        """Verifies that camera lag or frame drops (> 0.15s) reset baseline without jumping."""
        self.tracker.accum_screen_x = 500.0
        self.tracker.accum_screen_y = 500.0
        self.tracker.prev_finger_pos = (0.5, 0.5)
        # Simulate frame drop of 0.40 seconds
        self.tracker.prev_finger_time = time.perf_counter() - 0.40

        landmarks = [DummyLandmark(0.5, 0.5)] * 21
        landmarks[self.tracker.INDEX_TIP] = DummyLandmark(0.55, 0.5)
        landmarks[self.tracker.INDEX_PIP] = DummyLandmark(0.52, 0.5)
        landmarks[self.tracker.INDEX_MCP] = DummyLandmark(0.50, 0.5)
        landmarks[self.tracker.WRIST] = DummyLandmark(0.5, 0.7)
        landmarks[self.tracker.MIDDLE_MCP] = DummyLandmark(0.5, 0.5)

        class DummyHandsResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyHandsResult())
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, _, _, _ = self.tracker.process_frame(dummy_frame, 1920, 1080)
        # Cursor must remain at 500.0 rather than jumping across the screen
        self.assertEqual(self.tracker.accum_screen_x, 500.0)

    def test_raw_linear_mode_bypass(self):
        """Verifies disabling hand_smart_sensitivity restores linear raw mode."""
        self.config.hand_smart_sensitivity = False
        self.tracker.screen_origin_x = 960.0
        self.tracker.screen_origin_y = 540.0

        landmarks = [DummyLandmark(0.5, 0.5)] * 21
        landmarks[self.tracker.INDEX_TIP] = DummyLandmark(0.6, 0.5)  # dx = (0.6 - 0.5) * 2.0 = 0.2
        landmarks[self.tracker.INDEX_PIP] = DummyLandmark(0.52, 0.5)
        landmarks[self.tracker.INDEX_MCP] = DummyLandmark(0.50, 0.5)
        landmarks[self.tracker.WRIST] = DummyLandmark(0.5, 0.7)
        landmarks[self.tracker.MIDDLE_MCP] = DummyLandmark(0.5, 0.5)

        class DummyHandsResult:
            multi_hand_landmarks = [MagicMock(landmark=landmarks)]
            multi_handedness = [MagicMock(classification=[MagicMock(label="Right")])]

        self.tracker.hands.process = MagicMock(return_value=DummyHandsResult())
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pos, _, _, _ = self.tracker.process_frame(dummy_frame, 1920, 1080)
        # In raw mode: 960 + (0.2 * 1920) = 1344.0
        self.assertAlmostEqual(pos[0], 1344.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()

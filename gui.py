"""Modern desktop user interface for Zero Touch built with CustomTkinter.
Displays live webcam video with annotated face mesh & hand tracking, telemetry gauges,
interactive sensitivity sliders, and gesture action mappings.
Features both Simple Mode (Ringkas) and Advanced Mode (Lanjutan).
"""

import ctypes
import queue
import sys
import threading
import time
from typing import Callable, Optional
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from config import AppConfig, ConfigManager, CONTROL_PRESETS
from hand_tracker import HandTracker
from mouse_controller import MouseController
from tracker import FaceTracker

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Global Win32 POINT structure for zero-overhead cursor polling in relative mode
if sys.platform == "win32":
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class FaceControlApp(ctk.CTk):
    """Main CustomTkinter desktop window for Zero Touch."""

    def __init__(
        self,
        config_manager: ConfigManager,
        tracker: FaceTracker,
        mouse_controller: MouseController,
        hand_tracker: Optional[HandTracker] = None,
    ):
        super().__init__()

        self.config_manager = config_manager
        self.config = config_manager.config
        self.tracker = tracker
        self.mouse = mouse_controller
        self.hand_tracker = hand_tracker if hand_tracker is not None else HandTracker(self.config)

        # Window configuration
        self.title("Zero Touch — Touchless Head & Hand Mouse")
        self.geometry("1020x720")
        self.minsize(980, 660)

        # Bring window to foreground on launch
        self.lift()
        self.attributes("-topmost", True)
        self.after(800, lambda: self.attributes("-topmost", False))
        self.focus_force()

        # Connect mouse action callback to UI badge
        self.mouse.set_action_callback(self._on_action_triggered)
        self.last_action_badge_time = 0.0
        self._last_dwell_progress = -1.0
        self._last_hotkey_time = 0.0
        self._hotkey_cooldown = 0.3

        # Video feed states
        self.is_running = True
        self.tracking_enabled = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_thread: Optional[threading.Thread] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Build GUI Layout
        self._build_layout()

        # Global Window Hotkey Bindings (macOS style shortcuts)
        self.bind_all("<F8>", lambda e: self.toggle_tracking_hotkey())
        self.bind_all("<Escape>", lambda e: self.toggle_tracking_hotkey())
        self.bind_all("<F6>", lambda e: self.toggle_pointer_hotkey())
        self.bind_all("<F7>", lambda e: self.toggle_clicks_hotkey())
        self.bind_all("<c>", lambda e: self.calibrate_center())
        self.bind_all("<C>", lambda e: self.calibrate_center())
        self.bind_all("<Key>", self._on_key_press)

        # Handle window closing protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start camera capture thread
        self._start_camera_stream()

        # Start GUI update loop
        self.after(20, self._update_gui_loop)

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)  # Left panel (Video & Telemetry)
        self.grid_columnconfigure(1, weight=2)  # Right panel (Settings & Controls)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT PANEL: Video Feed & Telemetry HUD
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, corner_radius=12, fg_color="#18181b")
        self.left_panel.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # Video Title & Status Header
        video_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        video_header.pack(padx=14, pady=(12, 6), fill="x")

        self.video_title = ctk.CTkLabel(
            video_header,
            text="🎥 Live Tracking HUD",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f4f4f5",
        )
        self.video_title.pack(side="left")

        self.lbl_stream_status = ctk.CTkLabel(
            video_header,
            text="● LIVE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4ade80",
            fg_color="#064e3b",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.lbl_stream_status.pack(side="right")

        # Video Canvas / Label
        self.video_canvas = ctk.CTkLabel(
            self.left_panel,
            text="Menghubungkan ke kamera...",
            width=480,
            height=360,
            corner_radius=8,
            fg_color="#09090b",
        )
        self.video_canvas.pack(padx=14, pady=4, fill="both", expand=True)

        # Telemetry Frame
        self.telemetry_frame = ctk.CTkFrame(self.left_panel, fg_color="#202025", corner_radius=10)
        self.telemetry_frame.pack(padx=14, pady=(8, 12), fill="x")

        # Action Badge (Standby, Left Click, Dragging, etc.)
        self.badge_label = ctk.CTkLabel(
            self.telemetry_frame,
            text="[ STANDBY ]",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80",
            fg_color="#14532d",
            corner_radius=8,
            height=32,
        )
        self.badge_label.pack(padx=12, pady=(10, 8), fill="x")

        # --- Simple Mode Telemetry Container ---
        self.simple_telemetry_container = ctk.CTkFrame(self.telemetry_frame, fg_color="transparent")
        simple_stat_box = ctk.CTkFrame(self.simple_telemetry_container, fg_color="#18181b", corner_radius=8)
        simple_stat_box.pack(fill="x", padx=4, pady=2)
        simple_stat_box.grid_columnconfigure((0, 1), weight=1)

        self.lbl_simple_face = ctk.CTkLabel(
            simple_stat_box,
            text="👤 Wajah: ⚪ Mencari...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#a1a1aa",
        )
        self.lbl_simple_face.grid(row=0, column=0, padx=8, pady=6, sticky="w")

        self.lbl_simple_hand = ctk.CTkLabel(
            simple_stat_box,
            text="🖐️ Tangan: ⚪ Tidak Terdeteksi",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#a1a1aa",
        )
        self.lbl_simple_hand.grid(row=0, column=1, padx=8, pady=6, sticky="e")

        self.lbl_simple_ptr_src = ctk.CTkLabel(
            simple_stat_box,
            text="🎯 Pointer: Jari Telunjuk",
            font=ctk.CTkFont(size=11),
            text_color="#38bdf8",
        )
        self.lbl_simple_ptr_src.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="w")

        # --- Advanced Mode Telemetry Container ---
        self.advanced_telemetry_container = ctk.CTkFrame(self.telemetry_frame, fg_color="transparent")

        gauge_container = ctk.CTkFrame(self.advanced_telemetry_container, fg_color="transparent")
        gauge_container.pack(fill="x", padx=4, pady=(0, 4))
        gauge_container.grid_columnconfigure((1, 4), weight=1)

        # Row 0: Left Eye EAR & Right Eye EAR
        ctk.CTkLabel(gauge_container, text="L-EAR:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, sticky="w", padx=(2, 4), pady=2
        )
        self.bar_left_ear = ctk.CTkProgressBar(gauge_container, height=10)
        self.bar_left_ear.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self.lbl_left_ear = ctk.CTkLabel(gauge_container, text="0.00", width=34, font=ctk.CTkFont(size=11))
        self.lbl_left_ear.grid(row=0, column=2, padx=(2, 8), pady=2)

        ctk.CTkLabel(gauge_container, text="R-EAR:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=3, sticky="w", padx=(2, 4), pady=2
        )
        self.bar_right_ear = ctk.CTkProgressBar(gauge_container, height=10)
        self.bar_right_ear.grid(row=0, column=4, sticky="ew", padx=2, pady=2)
        self.lbl_right_ear = ctk.CTkLabel(gauge_container, text="0.00", width=34, font=ctk.CTkFont(size=11))
        self.lbl_right_ear.grid(row=0, column=5, padx=2, pady=2)

        # Row 1: Mouth MAR & Dwell Hover
        ctk.CTkLabel(gauge_container, text="M-MAR:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w", padx=(2, 4), pady=2
        )
        self.bar_mar = ctk.CTkProgressBar(gauge_container, height=10)
        self.bar_mar.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.lbl_mar = ctk.CTkLabel(gauge_container, text="0.00", width=34, font=ctk.CTkFont(size=11))
        self.lbl_mar.grid(row=1, column=2, padx=(2, 8), pady=2)

        ctk.CTkLabel(gauge_container, text="Dwell:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=3, sticky="w", padx=(2, 4), pady=2
        )
        self.bar_dwell = ctk.CTkProgressBar(gauge_container, height=10)
        self.bar_dwell.set(0.0)
        self.bar_dwell.grid(row=1, column=4, sticky="ew", padx=2, pady=2)
        self.lbl_dwell = ctk.CTkLabel(gauge_container, text="0%", width=34, font=ctk.CTkFont(size=11))
        self.lbl_dwell.grid(row=1, column=5, padx=2, pady=2)

        # Advanced Hand Tracking Telemetry Bar
        hand_status_frame = ctk.CTkFrame(self.advanced_telemetry_container, fg_color="#18181b", corner_radius=6)
        hand_status_frame.pack(fill="x", padx=4, pady=(2, 6))
        hand_status_frame.grid_columnconfigure((0, 1), weight=1)

        self.lbl_hand_status = ctk.CTkLabel(
            hand_status_frame, text="Hand: [ NO HAND ]", font=ctk.CTkFont(size=11, weight="bold"), text_color="#a1a1aa"
        )
        self.lbl_hand_status.grid(row=0, column=0, sticky="w", padx=8, pady=3)

        self.lbl_hand_gesture = ctk.CTkLabel(
            hand_status_frame, text="Gesture: [ NONE ]", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8"
        )
        self.lbl_hand_gesture.grid(row=0, column=1, sticky="e", padx=8, pady=3)

        # ==========================================
        # RIGHT PANEL: Settings & Mode Switcher
        # ==========================================
        self.right_panel = ctk.CTkFrame(self, corner_radius=12, fg_color="#18181b")
        self.right_panel.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="nsew")

        # Top Control Bar (Mode Switcher + Global Master Switches)
        top_bar = ctk.CTkFrame(self.right_panel, fg_color="#202025", corner_radius=10)
        top_bar.pack(padx=10, pady=(10, 6), fill="x")

        # Segmented Button: Mode Ringkas vs Mode Lanjutan
        cur_mode = getattr(self.config, "ui_mode", "simple")
        self.seg_ui_mode = ctk.CTkSegmentedButton(
            top_bar,
            values=["⚡ Mode Ringkas", "🛠️ Mode Lanjutan"],
            command=self._on_switch_ui_mode,
            font=ctk.CTkFont(size=12, weight="bold"),
            selected_color="#2563eb",
            selected_hover_color="#1d4ed8",
            unselected_color="#18181b",
            unselected_hover_color="#27272a",
            height=32,
        )
        self.seg_ui_mode.set("⚡ Mode Ringkas" if cur_mode == "simple" else "🛠️ Mode Lanjutan")
        self.seg_ui_mode.pack(padx=10, pady=(10, 8), fill="x")

        # Global Master Active Switch & Center Calibration
        quick_action_row = ctk.CTkFrame(top_bar, fg_color="transparent")
        quick_action_row.pack(padx=10, pady=(0, 10), fill="x")
        quick_action_row.grid_columnconfigure(0, weight=1)
        quick_action_row.grid_columnconfigure(1, weight=1)

        self.switch_tracking = ctk.CTkSwitch(
            quick_action_row,
            text="Pelacakan Aktif (F8)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_tracking_switch,
        )
        self.switch_tracking.select()
        self.switch_tracking.grid(row=0, column=0, sticky="w", padx=4)

        self.btn_calibrate = ctk.CTkButton(
            quick_action_row,
            text="🎯 Pusatkan Kursor [C]",
            command=self.calibrate_center,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.btn_calibrate.grid(row=0, column=1, sticky="e", padx=4)

        # Mode Container Frames
        self.mode_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.mode_container.pack(padx=6, pady=(0, 6), fill="both", expand=True)

        self.simple_mode_frame = ctk.CTkFrame(self.mode_container, fg_color="transparent")
        self.advanced_mode_frame = ctk.CTkFrame(self.mode_container, fg_color="transparent")

        # Build content for both modes
        self._build_simple_mode_view(self.simple_mode_frame)
        self._build_advanced_mode_view(self.advanced_mode_frame)

        # Activate initial view and synchronize states
        self._apply_ui_mode_view(cur_mode)

    def _create_card(self, parent, title: Optional[str] = None, subtitle: Optional[str] = None) -> ctk.CTkFrame:
        """Creates a standardized card container with modern padding, border, and headers."""
        card = ctk.CTkFrame(parent, fg_color="#202025", corner_radius=10, border_width=1, border_color="#2b2b36")
        card.pack(padx=6, pady=5, fill="x")
        if title:
            lbl_title = ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#60a5fa",
                anchor="w",
            )
            lbl_title.pack(padx=12, pady=(8, 2), anchor="w")
        if subtitle:
            lbl_sub = ctk.CTkLabel(
                card,
                text=subtitle,
                font=ctk.CTkFont(size=11),
                text_color="#94a3b8",
                anchor="w",
                justify="left",
            )
            lbl_sub.pack(padx=12, pady=(0, 6), anchor="w")
        return card

    # =========================================================================
    # SIMPLE MODE (MODE RINGKAS) VIEW
    # =========================================================================
    def _build_simple_mode_view(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Card 1: Control Style Presets
        card_preset = self._create_card(
            scroll,
            title="🎮 Gaya Kendali Utama",
            subtitle="Pilih cara yang paling mudah & nyaman untuk mengontrol mouse.",
        )
        preset_simple_options = [
            "🖐️ Kontrol Jari & Jempol (Rekomendasi)",
            "🙆 Bebas Tangan / Wajah (Handsfree)",
            "✌️ Gestur Santai (Peace & Kepal)",
            "🤏 Cubit & Arahkan (Pinch & Point)",
            "3️⃣ Tiga Jari (3-Finger Control)",
            "⚙️ Kustom (Dari Mode Lanjutan)",
        ]
        cur_preset = getattr(self.config, "control_preset", "Thumb Trigger")
        cur_s_display = self._map_preset_to_simple_label(cur_preset)

        self.opt_simple_preset = ctk.CTkOptionMenu(
            card_preset,
            values=preset_simple_options,
            command=self._on_select_simple_preset,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
        )
        self.opt_simple_preset.set(cur_s_display)
        self.opt_simple_preset.pack(padx=12, pady=(4, 6), fill="x")

        self.lbl_simple_preset_desc = ctk.CTkLabel(
            card_preset,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#38bdf8",
            wraplength=380,
            justify="left",
            anchor="w",
        )
        self.lbl_simple_preset_desc.pack(padx=12, pady=(0, 10), fill="x")
        self._update_simple_preset_hint(cur_s_display)

        # Card 2: Speed and Smoothness
        card_speed = self._create_card(
            scroll,
            title="⚡ Kecepatan & Kehalusan Gerak",
            subtitle="Atur kelincahan pergerakan mouse agar pas dengan tangan/kepala.",
        )
        cur_spd = self.config.hand_cursor_speed if getattr(self.config, "pointer_source", "head") == "finger" else self.config.cursor_speed
        self.simple_slider_speed = self._add_slider(
            "Kecepatan Kursor (Speed)",
            1.0,
            5.0,
            cur_spd,
            self._on_simple_speed_change,
            parent=card_speed,
        )
        self.simple_slider_smoothing = self._add_slider(
            "Kehalusan Gerakan (Stabilizer)",
            0.15,
            0.90,
            self.config.smoothing,
            self._on_simple_smoothing_change,
            parent=card_speed,
        )

        # Card 3: Easy Helper Toggles
        card_helpers = self._create_card(
            scroll,
            title="✨ Fitur Kemudahan & Bantuan",
            subtitle="Opsi otomatis untuk pengalaman yang lebih santai.",
        )

        self.switch_simple_dwell = ctk.CTkSwitch(
            card_helpers,
            text="Klik Otomatis saat Berhenti (Dwell Click)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_simple_dwell,
        )
        if getattr(self.config, "dwell_enabled", False):
            self.switch_simple_dwell.select()
        else:
            self.switch_simple_dwell.deselect()
        self.switch_simple_dwell.pack(padx=12, pady=6, anchor="w")

        self.switch_simple_override = ctk.CTkSwitch(
            card_helpers,
            text="Serahkan ke Mouse Fisik saat Disentuh",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_simple_override,
        )
        if getattr(self.config, "mouse_override", True):
            self.switch_simple_override.select()
        else:
            self.switch_simple_override.deselect()
        self.switch_simple_override.pack(padx=12, pady=6, anchor="w")

        self.switch_simple_audio = ctk.CTkSwitch(
            card_helpers,
            text="Suara Nada 'Beep' saat Klik Terpicu",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_simple_audio,
        )
        if getattr(self.config, "audio_feedback", False):
            self.switch_simple_audio.select()
        else:
            self.switch_simple_audio.deselect()
        self.switch_simple_audio.pack(padx=12, pady=(6, 10), anchor="w")

        # Card 4: Persistence
        card_save = self._create_card(scroll)
        btn_box = ctk.CTkFrame(card_save, fg_color="transparent")
        btn_box.pack(padx=10, pady=8, fill="x")
        btn_box.grid_columnconfigure((0, 1), weight=1)

        btn_save = ctk.CTkButton(
            btn_box,
            text="💾 Simpan Pengaturan",
            command=self._save_settings,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
        )
        btn_save.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        btn_reset = ctk.CTkButton(
            btn_box,
            text="↺ Kembalikan Default",
            command=self._reset_defaults,
            fg_color="#3f3f46",
            hover_color="#27272a",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
        )
        btn_reset.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.lbl_simple_save_status = ctk.CTkLabel(
            card_save, text="", font=ctk.CTkFont(size=12), text_color="#22c55e"
        )
        self.lbl_simple_save_status.pack(padx=10, pady=(0, 6))

    def _map_preset_to_simple_label(self, preset_name: str) -> str:
        if preset_name == "Thumb Trigger":
            return "🖐️ Kontrol Jari & Jempol (Rekomendasi)"
        elif preset_name == "Head & Face (Handsfree)":
            return "🙆 Bebas Tangan / Wajah (Handsfree)"
        elif preset_name == "Peace & Fist (Classic)":
            return "✌️ Gestur Santai (Peace & Kepal)"
        elif preset_name == "Pinch & Point":
            return "🤏 Cubit & Arahkan (Pinch & Point)"
        elif preset_name == "3-Finger Control":
            return "3️⃣ Tiga Jari (3-Finger Control)"
        elif preset_name == "Custom":
            return "⚙️ Kustom (Dari Mode Lanjutan)"
        return "🖐️ Kontrol Jari & Jempol (Rekomendasi)"

    def _map_preset_to_advanced_label(self, preset_name: str) -> str:
        if preset_name == "Thumb Trigger":
            return "Thumb Trigger (Default)"
        elif preset_name == "3-Finger Control":
            return "3-Finger Control (Index/Mid/Ring)"
        elif preset_name == "Peace & Fist (Classic)":
            return "Peace & Fist (Classic)"
        elif preset_name == "Pinch & Point":
            return "Pinch & Point"
        elif preset_name == "Head & Face (Handsfree)":
            return "Head & Face (Handsfree)"
        return "Custom"

    def _update_simple_preset_hint(self, label: str):
        if "Jari & Jempol" in label or "Thumb" in label:
            desc = "💡 Arahkan jari telunjuk untuk gerakan mouse. Ketuk jempol untuk klik kiri, tahan untuk seret."
        elif "Bebas Tangan" in label or "Handsfree" in label:
            desc = "💡 Gerakkan kepala perlahan untuk arahkan mouse. Kedipkan mata untuk klik kiri/kanan."
        elif "Peace" in label:
            desc = "💡 Arahkan tangan untuk gerak kursor. Buat simbol peace (✌️) untuk klik kiri."
        elif "Pinch" in label:
            desc = "💡 Telunjuk menggerakkan kursor. Cubit telunjuk untuk tahan/seret, cubit tengah untuk klik kanan."
        elif "Tiga Jari" in label or "3-Finger" in label:
            desc = "💡 Tekuk jari telunjuk untuk klik kiri, tekuk jari tengah untuk klik kanan."
        elif "Kustom" in label or "Custom" in label:
            desc = "💡 Pengaturan kustom yang diatur dari Mode Lanjutan sedang aktif."
        else:
            desc = "💡 Preset kustom aktif."
        if hasattr(self, "lbl_simple_preset_desc"):
            self.lbl_simple_preset_desc.configure(text=desc)

    def _on_select_simple_preset(self, val: str):
        if "Jari & Jempol" in val:
            preset_name = "Thumb Trigger"
        elif "Bebas Tangan" in val:
            preset_name = "Head & Face (Handsfree)"
        elif "Peace" in val:
            preset_name = "Peace & Fist (Classic)"
        elif "Cubit" in val:
            preset_name = "Pinch & Point"
        elif "Tiga Jari" in val:
            preset_name = "3-Finger Control"
        elif "Kustom" in val:
            preset_name = "Custom"
        else:
            preset_name = "Thumb Trigger"

        self._update_simple_preset_hint(val)
        if preset_name != "Custom":
            self._apply_control_preset(preset_name)
        else:
            self.config.control_preset = "Custom"
            if hasattr(self, "opt_control_preset"):
                self.opt_control_preset.set("Custom")

    def _on_simple_speed_change(self, val: float):
        fval = float(val)
        # Synchronize both head cursor speed and hand finger speed so it works in any preset
        self._update_config_attr("cursor_speed", fval)
        self._update_config_attr("hand_cursor_speed", fval)
        if hasattr(self, "slider_speed"):
            self._set_slider_value(self.slider_speed, fval)
        if hasattr(self, "slider_hand_speed"):
            self._set_slider_value(self.slider_hand_speed, fval)

    def _on_simple_smoothing_change(self, val: float):
        fval = float(val)
        self._update_config_attr("smoothing", fval)
        if hasattr(self, "slider_smoothing"):
            self._set_slider_value(self.slider_smoothing, fval)

    def _on_toggle_simple_dwell(self):
        is_on = bool(self.switch_simple_dwell.get())
        self._update_config_attr("dwell_enabled", is_on)
        self.mouse.reset_dwell()
        if hasattr(self, "switch_dwell"):
            if is_on:
                self.switch_dwell.select()
            else:
                self.switch_dwell.deselect()

    def _on_toggle_simple_override(self):
        is_on = bool(self.switch_simple_override.get())
        self._update_config_attr("mouse_override", is_on)
        if not is_on:
            self.mouse.is_overridden = False
        if hasattr(self, "switch_mouse_override"):
            if is_on:
                self.switch_mouse_override.select()
            else:
                self.switch_mouse_override.deselect()

    def _on_toggle_simple_audio(self):
        is_on = bool(self.switch_simple_audio.get())
        self._update_config_attr("audio_feedback", is_on)
        if hasattr(self, "switch_audio"):
            if is_on:
                self.switch_audio.select()
            else:
                self.switch_audio.deselect()

    # =========================================================================
    # ADVANCED MODE (MODE LANJUTAN) VIEW
    # =========================================================================
    def _build_advanced_mode_view(self, parent):
        # Subswitch Frame (Head Pointer F6, Click Actions F7)
        subswitch_frame = ctk.CTkFrame(parent, fg_color="#202025", corner_radius=8)
        subswitch_frame.pack(padx=4, pady=(2, 6), fill="x")
        subswitch_frame.grid_columnconfigure((0, 1), weight=1)

        self.switch_head_pointer = ctk.CTkSwitch(
            subswitch_frame,
            text="Gerak Kursor / Pointer (F6)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_head_pointer,
        )
        if getattr(self.config, "pointer_enabled", True):
            self.switch_head_pointer.select()
        else:
            self.switch_head_pointer.deselect()
        self.switch_head_pointer.grid(row=0, column=0, sticky="w", padx=10, pady=6)

        self.switch_click_actions = ctk.CTkSwitch(
            subswitch_frame,
            text="Aksi Klik (F7)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_click_actions,
        )
        if getattr(self.config, "clicking_enabled", True):
            self.switch_click_actions.select()
        else:
            self.switch_click_actions.deselect()
        self.switch_click_actions.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        # Tabview for advanced settings
        self.tabview = ctk.CTkTabview(parent, corner_radius=10)
        self.tabview.pack(padx=2, pady=(0, 4), fill="both", expand=True)

        tab_ptr = self.tabview.add("🎯 Pointer")
        tab_clk = self.tabview.add("🖱️ Clicking")
        tab_hand = self.tabview.add("🖐️ Hand")
        tab_sys = self.tabview.add("⚙️ System")

        self._build_pointer_tab(tab_ptr)
        self._build_clicking_tab(tab_clk)
        self._build_hand_tab(tab_hand)
        self._build_system_tab(tab_sys)

    def _on_advanced_cursor_speed_change(self, val: float):
        fval = float(val)
        self._update_config_attr("cursor_speed", fval)
        if getattr(self.config, "pointer_source", "head") == "head" and hasattr(self, "simple_slider_speed"):
            self._set_slider_value(self.simple_slider_speed, fval)

    def _on_advanced_hand_speed_change(self, val: float):
        fval = float(val)
        self._update_config_attr("hand_cursor_speed", fval)
        if getattr(self.config, "pointer_source", "head") == "finger" and hasattr(self, "simple_slider_speed"):
            self._set_slider_value(self.simple_slider_speed, fval)

    def _on_advanced_smoothing_change(self, val: float):
        fval = float(val)
        self._update_config_attr("smoothing", fval)
        if hasattr(self, "simple_slider_smoothing"):
            self._set_slider_value(self.simple_slider_smoothing, fval)

    def _build_pointer_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Card 1: Pointer Motion & Source
        card_motion = self._create_card(
            scroll,
            title="Gerak Pointer & Sumber",
            subtitle="Pilih sumber pelacakan dan mode pergerakan kursor.",
        )
        cur_src = getattr(self.config, "pointer_source", "head")
        self.opt_pointer_source = self._add_dropdown(
            "Pointer Source:",
            ["Head Pointer (Default)", "Finger Tip (Index)"],
            "Head Pointer (Default)" if cur_src == "head" else "Finger Tip (Index)",
            self._on_select_pointer_source,
            parent=card_motion,
        )

        self.opt_tracking_mode = self._add_dropdown(
            "Tracking Mode:",
            ["Absolute (Direct)", "Relative (Joystick)"],
            "Absolute (Direct)" if self.config.tracking_mode == "absolute" else "Relative (Joystick)",
            lambda val: self._update_config_attr("tracking_mode", "relative" if "Relative" in val else "absolute"),
            parent=card_motion,
        )

        self.slider_speed = self._add_slider(
            "Cursor Speed",
            1.0,
            5.0,
            self.config.cursor_speed,
            self._on_advanced_cursor_speed_change,
            parent=card_motion,
        )

        self.slider_smoothing = self._add_slider(
            "Smoothing (Jitter Reduction)",
            0.1,
            0.95,
            self.config.smoothing,
            self._on_advanced_smoothing_change,
            parent=card_motion,
        )

        # Card 2: Filter Tuning
        card_filter = self._create_card(
            scroll,
            title="Filter Sensitivitas & Deadzone",
            subtitle="Penyetelan One Euro Filter dan ambang batas diam.",
        )
        self.slider_beta = self._add_slider(
            "Filter Sensitivity (Beta)",
            0.001,
            0.020,
            getattr(self.config, "filter_beta", 0.007),
            lambda v: self._update_config_attr("filter_beta", float(v)),
            parent=card_filter,
        )

        self.slider_deadzone = self._add_slider(
            "Rest Deadzone",
            0.002,
            0.035,
            self.config.deadzone,
            lambda v: self._update_config_attr("deadzone", float(v)),
            parent=card_filter,
        )

        # Card 3: Mouse Override
        card_override = self._create_card(
            scroll,
            title="Physical Mouse Override",
            subtitle="Menyerahkan kontrol ke mouse fisik saat digerakkan.",
        )
        self.switch_mouse_override = ctk.CTkSwitch(
            card_override,
            text="Physical Mouse Override",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_mouse_override,
        )
        if getattr(self.config, "mouse_override", True):
            self.switch_mouse_override.select()
        else:
            self.switch_mouse_override.deselect()
        self.switch_mouse_override.pack(padx=10, pady=(6, 4), anchor="w")

        self.slider_override_delay = self._add_slider(
            "Override Resume Delay (s)",
            0.2,
            2.5,
            getattr(self.config, "override_timeout", 0.8),
            lambda v: self._update_config_attr("override_timeout", float(v)),
            parent=card_override,
        )

    def _on_select_pointer_source(self, val: str):
        src = "head" if "Head" in val else "finger"
        self._update_config_attr("pointer_source", src)
        # Re-sync speed in simple mode to match newly active source
        if hasattr(self, "simple_slider_speed"):
            active_spd = self.config.hand_cursor_speed if src == "finger" else self.config.cursor_speed
            self._set_slider_value(self.simple_slider_speed, active_spd)

    def _build_clicking_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Card 1: Primary Click Mode
        card_mode = self._create_card(scroll, title="Metode Klik Primer")
        self.opt_click_mode = self._add_dropdown(
            "Primary Click Mode:",
            ["Gestures", "Dwell", "Hybrid"],
            getattr(self.config, "click_mode", "gestures").capitalize(),
            lambda val: self._update_config_attr("click_mode", val.lower()),
            parent=card_mode,
        )

        # Card 2: Dwell Control
        card_dwell = self._create_card(
            scroll,
            title="Dwell Control (Hover-to-Click)",
            subtitle="Memicu klik otomatis saat kursor berhenti.",
        )
        self.switch_dwell = ctk.CTkSwitch(
            card_dwell,
            text="Enable Dwell Hover Click",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_dwell,
        )
        if getattr(self.config, "dwell_enabled", False):
            self.switch_dwell.select()
        else:
            self.switch_dwell.deselect()
        self.switch_dwell.pack(padx=10, pady=(6, 4), anchor="w")

        self.opt_dwell_action = self._add_dropdown(
            "Dwell Action:",
            ["Left Click", "Right Click", "Double Click"],
            getattr(self.config, "dwell_action", "Left Click"),
            lambda val: self._update_config_attr("dwell_action", val),
            parent=card_dwell,
        )

        self.slider_dwell_time = self._add_slider(
            "Dwell Delay (seconds)",
            0.3,
            2.5,
            getattr(self.config, "dwell_time", 0.8),
            lambda v: self._update_config_attr("dwell_time", float(v)),
            parent=card_dwell,
        )

        self.slider_dwell_radius = self._add_slider(
            "Dwell Jitter Tolerance (px)",
            5.0,
            50.0,
            float(getattr(self.config, "dwell_radius", 20)),
            lambda v: self._update_config_attr("dwell_radius", int(float(v))),
            parent=card_dwell,
        )

        # Card 3: Facial Expression Thresholds
        card_face = self._create_card(
            scroll,
            title="Ambang Ekspresi Wajah",
            subtitle="Sensitivitas kedipan mata, buka mulut, dan senyuman.",
        )
        self.slider_wink = self._add_slider(
            "Wink Threshold",
            0.12,
            0.30,
            self.config.wink_threshold,
            lambda v: self._update_config_attr("wink_threshold", float(v)),
            parent=card_face,
        )

        self.slider_mouth = self._add_slider(
            "Mouth Open Threshold",
            0.25,
            0.75,
            self.config.mouth_threshold,
            lambda v: self._update_config_attr("mouth_threshold", float(v)),
            parent=card_face,
        )

        self.slider_smile = self._add_slider(
            "Smile Threshold",
            0.30,
            0.70,
            getattr(self.config, "smile_threshold", 0.45),
            lambda v: self._update_config_attr("smile_threshold", float(v)),
            parent=card_face,
        )

        # Card 4: Facial Action Mapping
        card_map = self._create_card(
            scroll,
            title="Pemetaan Aksi Gestur Wajah",
            subtitle="Petakan ekspresi wajah ke aksi klik mouse.",
        )
        gesture_options = ["Left Wink", "Right Wink", "Mouth Open", "Smile", "None"]
        self.opt_left_click = self._add_dropdown(
            "Left Click Action:",
            gesture_options,
            self.config.left_click_trigger,
            lambda val: self._update_config_attr("left_click_trigger", val),
            parent=card_map,
        )

        self.opt_right_click = self._add_dropdown(
            "Right Click Action:",
            gesture_options,
            self.config.right_click_trigger,
            lambda val: self._update_config_attr("right_click_trigger", val),
            parent=card_map,
        )

        self.opt_double_click = self._add_dropdown(
            "Double Click Action:",
            gesture_options,
            getattr(self.config, "double_click_trigger", "None"),
            lambda val: self._update_config_attr("double_click_trigger", val),
            parent=card_map,
        )

        self.opt_mid_click = self._add_dropdown(
            "Middle Click Action:",
            gesture_options,
            self.config.middle_click_trigger,
            lambda val: self._update_config_attr("middle_click_trigger", val),
            parent=card_map,
        )

    def _build_hand_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Card 1: Presets & Subsystem
        card_preset = self._create_card(scroll, title="Preset Kontrol Tangan")
        preset_options = [
            "Thumb Trigger (Default)",
            "3-Finger Control (Index/Mid/Ring)",
            "Peace & Fist (Classic)",
            "Pinch & Point",
            "Head & Face (Handsfree)",
            "Custom",
        ]
        cur_preset = getattr(self.config, "control_preset", "Thumb Trigger")
        cur_display = self._map_preset_to_advanced_label(cur_preset)

        self.opt_control_preset = self._add_dropdown(
            "Active Control Preset:",
            preset_options,
            cur_display,
            self._on_select_control_preset,
            parent=card_preset,
        )

        # Card 2: Thumb Trigger
        card_thumb = self._create_card(
            scroll,
            title="Thumb Trigger (Klik & Seret)",
            subtitle="Ketuk jempol untuk klik kiri, tahan untuk seret.",
        )
        self.switch_thumb_trigger = ctk.CTkSwitch(
            card_thumb,
            text="Enable Thumb Click & Hold Drag",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_thumb_trigger,
        )
        if getattr(self.config, "hand_thumb_trigger_enabled", True):
            self.switch_thumb_trigger.select()
        else:
            self.switch_thumb_trigger.deselect()
        self.switch_thumb_trigger.pack(padx=10, pady=(6, 4), anchor="w")

        self.slider_thumb_hold = self._add_slider(
            "Thumb Drag Hold Delay (s)",
            0.4,
            2.0,
            getattr(self.config, "hand_thumb_hold_threshold", 1.0),
            lambda v: self._update_thumb_hold_threshold(float(v)),
            parent=card_thumb,
        )

        # Card 3: Hand Tracking Subsystem
        card_subsys = self._create_card(
            scroll,
            title="Subsistem Pelacakan Tangan",
            subtitle="Deteksi tangan dan sensitivitas cubitan/kecepatan.",
        )
        self.switch_hand_tracking = ctk.CTkSwitch(
            card_subsys,
            text="Enable Hand Tracking Subsystem",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_hand_tracking,
        )
        if getattr(self.config, "hand_tracking_enabled", True):
            self.switch_hand_tracking.select()
        else:
            self.switch_hand_tracking.deselect()
        self.switch_hand_tracking.pack(padx=10, pady=(6, 4), anchor="w")

        self.switch_hand_right_only = ctk.CTkSwitch(
            card_subsys,
            text="Right Hand Only (Ignore Left)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_hand_right_only,
        )
        if getattr(self.config, "hand_right_only", True):
            self.switch_hand_right_only.select()
        else:
            self.switch_hand_right_only.deselect()
        self.switch_hand_right_only.pack(padx=10, pady=(2, 6), anchor="w")

        self.slider_pinch_threshold = self._add_slider(
            "Pinch Sensitivity (Distance)",
            0.04,
            0.15,
            getattr(self.config, "hand_pinch_threshold", 0.08),
            lambda v: self._update_config_attr("hand_pinch_threshold", float(v)),
            parent=card_subsys,
        )

        self.slider_hand_speed = self._add_slider(
            "Finger Pointer Speed",
            1.0,
            4.0,
            getattr(self.config, "hand_cursor_speed", 2.0),
            self._on_advanced_hand_speed_change,
            parent=card_subsys,
        )

        # Card 4: Smart Sensitivity
        card_smart = self._create_card(
            scroll,
            title="Smart Sensitivity & Tremor Filter",
            subtitle="Stabilisasi kursor ala Windows Precision Trackpad.",
        )
        self.switch_smart_sens = ctk.CTkSwitch(
            card_smart,
            text="Enable Smart Sensitivity (Precision)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_smart_sens,
        )
        if getattr(self.config, "hand_smart_sensitivity", True):
            self.switch_smart_sens.select()
        else:
            self.switch_smart_sens.deselect()
        self.switch_smart_sens.pack(padx=10, pady=(6, 4), anchor="w")

        prof_options = ["Dynamic (Windows Precision)", "Smooth (Trackpad)", "Linear (Raw)"]
        cur_prof = getattr(self.config, "hand_sensitivity_profile", "dynamic").lower()
        if cur_prof == "smooth":
            prof_display = "Smooth (Trackpad)"
        elif cur_prof == "linear":
            prof_display = "Linear (Raw)"
        else:
            prof_display = "Dynamic (Windows Precision)"

        self.opt_sens_profile = self._add_dropdown(
            "Pointer Profile:",
            prof_options,
            prof_display,
            self._on_select_sens_profile,
            parent=card_smart,
        )

        self.slider_tremor_deadband = self._add_slider(
            "Tremor Suppression (Deadband)",
            0.0005,
            0.0040,
            getattr(self.config, "hand_tremor_deadband", 0.0015),
            lambda v: self._update_config_attr("hand_tremor_deadband", float(v)),
            parent=card_smart,
        )

        if not getattr(self.config, "hand_smart_sensitivity", True):
            self.opt_sens_profile.configure(state="disabled")
            self.slider_tremor_deadband.configure(state="disabled")

        # Card 5: Hand Action Mapping
        card_hand_map = self._create_card(
            scroll,
            title="Pemetaan Gestur Tangan",
            subtitle="Petakan gestur jari tertentu ke aksi mouse.",
        )
        action_options = ["Left Click", "Hold Left Click", "Right Click", "Double Click", "Middle Click", "None"]

        def _on_hand_action_change(attr, val):
            self._update_config_attr(attr, val)
            self.config.control_preset = "Custom"
            if hasattr(self, "opt_control_preset"):
                self.opt_control_preset.set("Custom")
            if hasattr(self, "opt_simple_preset"):
                self.opt_simple_preset.set("⚙️ Kustom (Dari Mode Lanjutan)")
                self._update_simple_preset_hint("⚙️ Kustom (Dari Mode Lanjutan)")

        self.opt_hand_peace = self._add_dropdown(
            "Peace Sign (✌️):",
            action_options,
            getattr(self.config, "hand_peace_action", "None"),
            lambda val: _on_hand_action_change("hand_peace_action", val),
            parent=card_hand_map,
        )

        self.opt_hand_pinch_idx = self._add_dropdown(
            "Pinch Index (🤏):",
            action_options,
            getattr(self.config, "hand_pinch_index_action", "None"),
            lambda val: _on_hand_action_change("hand_pinch_index_action", val),
            parent=card_hand_map,
        )

        self.opt_hand_pinch_mid = self._add_dropdown(
            "Pinch Middle (🤏):",
            action_options,
            getattr(self.config, "hand_pinch_middle_action", "None"),
            lambda val: _on_hand_action_change("hand_pinch_middle_action", val),
            parent=card_hand_map,
        )

        self.opt_hand_point = self._add_dropdown(
            "Pointing (👉):",
            action_options,
            getattr(self.config, "hand_point_action", "None"),
            lambda val: _on_hand_action_change("hand_point_action", val),
            parent=card_hand_map,
        )

        self.opt_hand_fist = self._add_dropdown(
            "Fist (✊):",
            action_options,
            getattr(self.config, "hand_fist_action", "None"),
            lambda val: _on_hand_action_change("hand_fist_action", val),
            parent=card_hand_map,
        )

    def _build_system_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Card 1: Assistive Keyboard Switches
        card_sw = self._create_card(
            scroll,
            title="Assistive Keyboard Switches",
            subtitle="Gunakan tombol keyboard fisik sebagai sakelar alternatif.",
        )
        switch_keys = ["None", "space", "f1", "f2", "f3", "f4", "Control_L", "Alt_L"]
        cur_left = getattr(self.config, "hotkey_switch_left", "") or "None"
        cur_right = getattr(self.config, "hotkey_switch_right", "") or "None"
        cur_drag = getattr(self.config, "hotkey_switch_drag", "") or "None"

        self.opt_switch_left = self._add_dropdown(
            "Left Click Key:",
            switch_keys,
            cur_left,
            lambda val: self._update_config_attr("hotkey_switch_left", "" if val == "None" else val),
            parent=card_sw,
        )

        self.opt_switch_right = self._add_dropdown(
            "Right Click Key:",
            switch_keys,
            cur_right,
            lambda val: self._update_config_attr("hotkey_switch_right", "" if val == "None" else val),
            parent=card_sw,
        )

        self.opt_switch_drag = self._add_dropdown(
            "Drag Lock Key:",
            switch_keys,
            cur_drag,
            lambda val: self._update_config_attr("hotkey_switch_drag", "" if val == "None" else val),
            parent=card_sw,
        )

        # Card 2: Feedback & Alerts
        card_fb = self._create_card(
            scroll,
            title="Feedback & Notifikasi",
            subtitle="Indikasi suara dan visual ketika aksi terpicu.",
        )
        self.switch_audio = ctk.CTkSwitch(
            card_fb,
            text="Audio Click Beep",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_audio_feedback,
        )
        if getattr(self.config, "audio_feedback", False):
            self.switch_audio.select()
        else:
            self.switch_audio.deselect()
        self.switch_audio.pack(padx=10, pady=(6, 4), anchor="w")

        self.switch_visual = ctk.CTkSwitch(
            card_fb,
            text="Visual HUD Feedback",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_toggle_visual_feedback,
        )
        if getattr(self.config, "visual_feedback", True):
            self.switch_visual.select()
        else:
            self.switch_visual.deselect()
        self.switch_visual.pack(padx=10, pady=(4, 8), anchor="w")

        # Card 3: Persistence
        card_p = self._create_card(scroll)
        btn_frame = ctk.CTkFrame(card_p, fg_color="transparent")
        btn_frame.pack(padx=10, pady=8, fill="x")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_save = ctk.CTkButton(
            btn_frame,
            text="Simpan Pengaturan",
            command=self._save_settings,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
        )
        self.btn_save.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.btn_reset = ctk.CTkButton(
            btn_frame,
            text="↺ Kembalikan Default",
            command=self._reset_defaults,
            fg_color="#3f3f46",
            hover_color="#27272a",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
        )
        self.btn_reset.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.lbl_save_status = ctk.CTkLabel(
            card_p, text="", font=ctk.CTkFont(size=12), text_color="#22c55e"
        )
        self.lbl_save_status.pack(padx=10, pady=(0, 6))

    # =========================================================================
    # MODE SWITCHING & VIEW SYNCHRONIZATION
    # =========================================================================
    def _on_switch_ui_mode(self, mode_label: str):
        mode_val = "simple" if "Ringkas" in mode_label else "advanced"
        self._apply_ui_mode_view(mode_val)

    def _sync_all_controls(self):
        """Synchronizes all controls between Simple Mode and Advanced Mode to eliminate state drift."""
        cur_src = getattr(self.config, "pointer_source", "head")
        cur_spd = self.config.hand_cursor_speed if cur_src == "finger" else self.config.cursor_speed
        if hasattr(self, "simple_slider_speed"):
            self._set_slider_value(self.simple_slider_speed, cur_spd)
        if hasattr(self, "slider_speed"):
            self._set_slider_value(self.slider_speed, self.config.cursor_speed)
        if hasattr(self, "slider_hand_speed"):
            self._set_slider_value(self.slider_hand_speed, getattr(self.config, "hand_cursor_speed", 2.0))

        if hasattr(self, "simple_slider_smoothing"):
            self._set_slider_value(self.simple_slider_smoothing, self.config.smoothing)
        if hasattr(self, "slider_smoothing"):
            self._set_slider_value(self.slider_smoothing, self.config.smoothing)

        # Switches synchronization
        is_dwell = getattr(self.config, "dwell_enabled", False)
        if hasattr(self, "switch_simple_dwell"):
            self.switch_simple_dwell.select() if is_dwell else self.switch_simple_dwell.deselect()
        if hasattr(self, "switch_dwell"):
            self.switch_dwell.select() if is_dwell else self.switch_dwell.deselect()

        is_override = getattr(self.config, "mouse_override", True)
        if hasattr(self, "switch_simple_override"):
            self.switch_simple_override.select() if is_override else self.switch_simple_override.deselect()
        if hasattr(self, "switch_mouse_override"):
            self.switch_mouse_override.select() if is_override else self.switch_mouse_override.deselect()

        is_audio = getattr(self.config, "audio_feedback", False)
        if hasattr(self, "switch_simple_audio"):
            self.switch_simple_audio.select() if is_audio else self.switch_simple_audio.deselect()
        if hasattr(self, "switch_audio"):
            self.switch_audio.select() if is_audio else self.switch_audio.deselect()

        if hasattr(self, "switch_head_pointer"):
            self.switch_head_pointer.select() if getattr(self.config, "pointer_enabled", True) else self.switch_head_pointer.deselect()
        if hasattr(self, "switch_click_actions"):
            self.switch_click_actions.select() if getattr(self.config, "clicking_enabled", True) else self.switch_click_actions.deselect()

        cur_preset = getattr(self.config, "control_preset", "Thumb Trigger")
        if hasattr(self, "opt_simple_preset"):
            simple_lbl = self._map_preset_to_simple_label(cur_preset)
            self.opt_simple_preset.set(simple_lbl)
            self._update_simple_preset_hint(simple_lbl)
        if hasattr(self, "opt_control_preset"):
            self.opt_control_preset.set(self._map_preset_to_advanced_label(cur_preset))

    def _apply_ui_mode_view(self, mode_val: str):
        is_simple = (mode_val == "simple")
        self.config.ui_mode = "simple" if is_simple else "advanced"

        # Sync all controls between modes
        self._sync_all_controls()

        if is_simple:
            self.advanced_mode_frame.pack_forget()
            self.simple_mode_frame.pack(fill="both", expand=True)

            self.advanced_telemetry_container.pack_forget()
            self.simple_telemetry_container.pack(fill="x", padx=4, pady=(0, 4))
        else:
            self.simple_mode_frame.pack_forget()
            self.advanced_mode_frame.pack(fill="both", expand=True)

            self.simple_telemetry_container.pack_forget()
            self.advanced_telemetry_container.pack(fill="x", padx=4, pady=(0, 4))

    # =========================================================================
    # UI COMPONENT HELPERS
    # =========================================================================
    def _add_section_header(self, title: str, parent=None):
        target = parent if parent is not None else self.right_panel
        lbl = ctk.CTkLabel(
            target,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd",
        )
        lbl.pack(padx=10, pady=(8, 4), anchor="w")

    def _add_slider(
        self,
        name: str,
        from_: float,
        to: float,
        current: float,
        callback: Callable[[float], None],
        parent=None,
    ) -> ctk.CTkSlider:
        target = parent if parent is not None else self.right_panel
        row = ctk.CTkFrame(target, fg_color="transparent")
        row.pack(padx=10, pady=2, fill="x")
        row.grid_columnconfigure(0, weight=1)

        val_label = ctk.CTkLabel(row, text=f"{current:.3f}", width=45, font=ctk.CTkFont(size=12))
        lbl = ctk.CTkLabel(row, text=name, font=ctk.CTkFont(size=12), anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        val_label.grid(row=0, column=1, sticky="e")

        def on_change(val):
            val_label.configure(text=f"{val:.3f}")
            callback(val)

        slider = ctk.CTkSlider(
            target,
            from_=from_,
            to=to,
            number_of_steps=100,
            command=on_change,
        )
        slider.set(current)
        slider.val_label = val_label
        slider.pack(padx=10, pady=(0, 6), fill="x")
        return slider

    def _set_slider_value(self, slider: ctk.CTkSlider, value: float):
        """Sets slider thumb position and updates associated numeric label."""
        if slider is None:
            return
        slider.set(value)
        if hasattr(slider, "val_label") and slider.val_label:
            slider.val_label.configure(text=f"{value:.3f}")

    def _add_dropdown(
        self, label_text: str, values: list, current_val: str, callback: Callable[[str], None], parent=None
    ) -> ctk.CTkOptionMenu:
        target = parent if parent is not None else self.right_panel
        frame = ctk.CTkFrame(target, fg_color="transparent")
        frame.pack(padx=10, pady=3, fill="x")
        frame.grid_columnconfigure(1, weight=1)

        lbl = ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(size=12), width=120, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")

        opt = ctk.CTkOptionMenu(frame, values=values, command=callback)
        opt.set(current_val)
        opt.grid(row=0, column=1, sticky="ew")
        return opt

    def _update_config_attr(self, attr_name: str, value: any):
        setattr(self.config, attr_name, value)

    def _save_settings(self):
        self.config_manager.save(self.config)
        msg = "Pengaturan berhasil disimpan ke config.json!"
        if hasattr(self, "lbl_save_status"):
            self.lbl_save_status.configure(text=msg, text_color="#4ade80")
            self.after(2500, lambda: self.lbl_save_status.configure(text=""))
        if hasattr(self, "lbl_simple_save_status"):
            self.lbl_simple_save_status.configure(text=msg, text_color="#4ade80")
            self.after(2500, lambda: self.lbl_simple_save_status.configure(text=""))

    def _reset_defaults(self):
        self.config = self.config_manager.reset_defaults()
        self.tracker.config = self.config
        self.mouse.config = self.config
        if hasattr(self, "hand_tracker") and self.hand_tracker:
            self.hand_tracker.config = self.config

        # Update Advanced UI sliders
        if hasattr(self, "slider_speed"):
            self._set_slider_value(self.slider_speed, self.config.cursor_speed)
            self._set_slider_value(self.slider_smoothing, self.config.smoothing)
            self._set_slider_value(self.slider_beta, self.config.filter_beta)
            self._set_slider_value(self.slider_deadzone, self.config.deadzone)
            self._set_slider_value(self.slider_wink, self.config.wink_threshold)
            self._set_slider_value(self.slider_mouth, self.config.mouth_threshold)
            self._set_slider_value(self.slider_smile, getattr(self.config, "smile_threshold", 0.45))
            self._set_slider_value(self.slider_override_delay, getattr(self.config, "override_timeout", 0.8))
            self._set_slider_value(self.slider_dwell_time, getattr(self.config, "dwell_time", 0.8))
            self._set_slider_value(self.slider_dwell_radius, float(getattr(self.config, "dwell_radius", 20)))

        if hasattr(self, "opt_tracking_mode"):
            self.opt_tracking_mode.set("Absolute (Direct)" if self.config.tracking_mode == "absolute" else "Relative (Joystick)")
            self.opt_click_mode.set(getattr(self.config, "click_mode", "gestures").capitalize())
            self.opt_dwell_action.set(getattr(self.config, "dwell_action", "Left Click"))
            self.opt_left_click.set(self.config.left_click_trigger)
            self.opt_right_click.set(self.config.right_click_trigger)
            self.opt_double_click.set(getattr(self.config, "double_click_trigger", "None"))
            self.opt_mid_click.set(self.config.middle_click_trigger)

        if hasattr(self, "switch_thumb_trigger"):
            self.switch_thumb_trigger.select()
        if hasattr(self, "slider_thumb_hold"):
            self._set_slider_value(self.slider_thumb_hold, 1.0)
        if hasattr(self, "slider_pinch_threshold"):
            self._set_slider_value(self.slider_pinch_threshold, getattr(self.config, "hand_pinch_threshold", 0.08))
            self._set_slider_value(self.slider_hand_speed, getattr(self.config, "hand_cursor_speed", 2.0))
            self.opt_pointer_source.set("Head Pointer (Default)" if getattr(self.config, "pointer_source", "head") == "head" else "Finger Tip (Index)")
            self.opt_hand_peace.set(getattr(self.config, "hand_peace_action", "None"))
            self.opt_hand_pinch_idx.set(getattr(self.config, "hand_pinch_index_action", "None"))
            self.opt_hand_pinch_mid.set(getattr(self.config, "hand_pinch_middle_action", "None"))
            self.opt_hand_point.set(getattr(self.config, "hand_point_action", "None"))
            self.opt_hand_fist.set(getattr(self.config, "hand_fist_action", "None"))
            if getattr(self.config, "hand_tracking_enabled", True):
                self.switch_hand_tracking.select()
            else:
                self.switch_hand_tracking.deselect()
            if hasattr(self, "switch_hand_right_only"):
                if getattr(self.config, "hand_right_only", True):
                    self.switch_hand_right_only.select()
                else:
                    self.switch_hand_right_only.deselect()

        if hasattr(self, "switch_smart_sens"):
            if getattr(self.config, "hand_smart_sensitivity", True):
                self.switch_smart_sens.select()
            else:
                self.switch_smart_sens.deselect()
            if hasattr(self, "opt_sens_profile"):
                cur_p = getattr(self.config, "hand_sensitivity_profile", "dynamic").lower()
                self.opt_sens_profile.set("Smooth (Trackpad)" if cur_p == "smooth" else ("Linear (Raw)" if cur_p == "linear" else "Dynamic (Windows Precision)"))
            if hasattr(self, "slider_tremor_deadband"):
                self._set_slider_value(self.slider_tremor_deadband, getattr(self.config, "hand_tremor_deadband", 0.0015))
            self._on_toggle_smart_sens()

        # Synchronize all controls between Simple and Advanced
        self._sync_all_controls()

        msg = "Pengaturan telah dikembalikan ke default!"
        if hasattr(self, "lbl_save_status"):
            self.lbl_save_status.configure(text=msg, text_color="#fbbf24")
            self.after(2500, lambda: self.lbl_save_status.configure(text=""))
        if hasattr(self, "lbl_simple_save_status"):
            self.lbl_simple_save_status.configure(text=msg, text_color="#fbbf24")
            self.after(2500, lambda: self.lbl_simple_save_status.configure(text=""))

    def _on_toggle_thumb_trigger(self):
        is_on = bool(self.switch_thumb_trigger.get())
        self._update_config_attr("hand_thumb_trigger_enabled", is_on)
        self.config.control_preset = "Custom"
        if hasattr(self, "opt_control_preset"):
            self.opt_control_preset.set("Custom")
        if hasattr(self, "opt_simple_preset"):
            self.opt_simple_preset.set("⚙️ Kustom (Dari Mode Lanjutan)")
            self._update_simple_preset_hint("⚙️ Kustom (Dari Mode Lanjutan)")
        self._on_action_triggered("THUMB ON" if is_on else "THUMB OFF")

    def _update_thumb_hold_threshold(self, val: float):
        self._update_config_attr("hand_thumb_hold_threshold", float(val))
        self.config.control_preset = "Custom"
        if hasattr(self, "opt_control_preset"):
            self.opt_control_preset.set("Custom")
        if hasattr(self, "opt_simple_preset"):
            self.opt_simple_preset.set("⚙️ Kustom (Dari Mode Lanjutan)")

    def _on_select_control_preset(self, val: str):
        if "Thumb" in val:
            preset_name = "Thumb Trigger"
        elif "3-Finger" in val or "Three" in val:
            preset_name = "3-Finger Control"
        elif "Peace" in val:
            preset_name = "Peace & Fist (Classic)"
        elif "Pinch" in val:
            preset_name = "Pinch & Point"
        elif "Head" in val:
            preset_name = "Head & Face (Handsfree)"
        else:
            preset_name = "Custom"
        self._apply_control_preset(preset_name)

    def _apply_control_preset(self, preset_name: str):
        if hasattr(self.mouse, "release_drag"):
            self.mouse.release_drag()

        self.config.control_preset = preset_name
        if preset_name in CONTROL_PRESETS and CONTROL_PRESETS[preset_name]:
            for k, v in CONTROL_PRESETS[preset_name].items():
                setattr(self.config, k, v)

        # Update Simple Mode preset selector & hint
        if hasattr(self, "opt_simple_preset"):
            simple_lbl = self._map_preset_to_simple_label(preset_name)
            self.opt_simple_preset.set(simple_lbl)
            self._update_simple_preset_hint(simple_lbl)

        # Update Advanced Preset selector
        if hasattr(self, "opt_control_preset"):
            self.opt_control_preset.set(self._map_preset_to_advanced_label(preset_name))

        # Update Advanced GUI controls
        if hasattr(self, "opt_pointer_source"):
            cur_src = getattr(self.config, "pointer_source", "head")
            self.opt_pointer_source.set("Head Pointer (Default)" if cur_src == "head" else "Finger Tip (Index)")

        if hasattr(self, "switch_hand_tracking"):
            if getattr(self.config, "hand_tracking_enabled", True):
                self.switch_hand_tracking.select()
            else:
                self.switch_hand_tracking.deselect()

        if hasattr(self, "switch_thumb_trigger"):
            if getattr(self.config, "hand_thumb_trigger_enabled", True):
                self.switch_thumb_trigger.select()
            else:
                self.switch_thumb_trigger.deselect()

        if hasattr(self, "slider_thumb_hold"):
            self._set_slider_value(self.slider_thumb_hold, getattr(self.config, "hand_thumb_hold_threshold", 1.0))

        if hasattr(self, "opt_hand_peace"):
            self.opt_hand_peace.set(getattr(self.config, "hand_peace_action", "None"))
            self.opt_hand_pinch_idx.set(getattr(self.config, "hand_pinch_index_action", "None"))
            self.opt_hand_pinch_mid.set(getattr(self.config, "hand_pinch_middle_action", "None"))
            self.opt_hand_point.set(getattr(self.config, "hand_point_action", "None"))
            self.opt_hand_fist.set(getattr(self.config, "hand_fist_action", "None"))

        if hasattr(self, "switch_head_pointer"):
            if getattr(self.config, "pointer_enabled", True):
                self.switch_head_pointer.select()
            else:
                self.switch_head_pointer.deselect()

        # Re-sync speed sliders
        cur_src = getattr(self.config, "pointer_source", "head")
        active_spd = self.config.hand_cursor_speed if cur_src == "finger" else self.config.cursor_speed
        if hasattr(self, "simple_slider_speed"):
            self._set_slider_value(self.simple_slider_speed, active_spd)

        self.config_manager.save(self.config)
        self._on_action_triggered(f"PRESET: {preset_name}")

    def calibrate_center(self):
        """Calibrates resting center to current nose position and centers the cursor."""
        self.tracker.calibrate_center(None)  # Resets filter state and auto-recenters on next detected frame
        sw, sh = self.mouse.screen_width, self.mouse.screen_height
        self.mouse.move_to(sw / 2.0, sh / 2.0)
        self._on_action_triggered("CALIBRATED")

    def _on_toggle_hand_tracking(self):
        is_on = bool(self.switch_hand_tracking.get())
        self._update_config_attr("hand_tracking_enabled", is_on)
        self._on_action_triggered("HAND ON" if is_on else "HAND OFF")

    def _on_toggle_hand_right_only(self):
        is_on = bool(self.switch_hand_right_only.get())
        self._update_config_attr("hand_right_only", is_on)
        self._on_action_triggered("RIGHT HAND ONLY" if is_on else "ANY HAND")

    def _on_toggle_smart_sens(self):
        is_on = bool(self.switch_smart_sens.get())
        self._update_config_attr("hand_smart_sensitivity", is_on)
        state = "normal" if is_on else "disabled"
        if hasattr(self, "opt_sens_profile"):
            self.opt_sens_profile.configure(state=state)
        if hasattr(self, "slider_tremor_deadband"):
            self.slider_tremor_deadband.configure(state=state)
        self._on_action_triggered("SMART SENS ON" if is_on else "SMART SENS OFF")

    def _on_select_sens_profile(self, val: str):
        if "Smooth" in val:
            profile = "smooth"
        elif "Linear" in val:
            profile = "linear"
        else:
            profile = "dynamic"
        self._update_config_attr("hand_sensitivity_profile", profile)

    def _on_toggle_tracking_switch(self):
        is_on = bool(self.switch_tracking.get())
        self.tracking_enabled = is_on
        self.mouse.set_paused(not is_on)
        self._on_action_triggered("TRACKING ON" if is_on else "PAUSED")

    def _on_toggle_head_pointer(self):
        is_on = bool(self.switch_head_pointer.get())
        self._update_config_attr("pointer_enabled", is_on)
        self._update_config_attr("enable_head_pointer", is_on)
        self.mouse.set_pointer_paused(not is_on)
        self._on_action_triggered("POINTER ON" if is_on else "GESTURE ONLY")

    def _on_toggle_click_actions(self):
        is_on = bool(self.switch_click_actions.get())
        self._update_config_attr("clicking_enabled", is_on)
        self.mouse.set_clicking_paused(not is_on)
        self._on_action_triggered("CLICKS ON" if is_on else "CLICKS PAUSED")

    def _on_toggle_dwell(self):
        is_on = bool(self.switch_dwell.get())
        self._update_config_attr("dwell_enabled", is_on)
        self.mouse.reset_dwell()
        if hasattr(self, "switch_simple_dwell"):
            self.switch_simple_dwell.select() if is_on else self.switch_simple_dwell.deselect()

    def _on_toggle_mouse_override(self):
        is_on = bool(self.switch_mouse_override.get())
        self._update_config_attr("mouse_override", is_on)
        if not is_on:
            self.mouse.is_overridden = False
        if hasattr(self, "switch_simple_override"):
            self.switch_simple_override.select() if is_on else self.switch_simple_override.deselect()

    def _on_toggle_audio_feedback(self):
        is_on = bool(self.switch_audio.get())
        self._update_config_attr("audio_feedback", is_on)
        if hasattr(self, "switch_simple_audio"):
            self.switch_simple_audio.select() if is_on else self.switch_simple_audio.deselect()

    def _on_toggle_visual_feedback(self):
        is_on = bool(self.switch_visual.get())
        self._update_config_attr("visual_feedback", is_on)

    def toggle_tracking_hotkey(self):
        """Called by F8 / ESC hotkey listener with debounce protection."""
        now = time.perf_counter()
        if (now - self._last_hotkey_time) < self._hotkey_cooldown:
            return
        self._last_hotkey_time = now
        is_paused = self.mouse.toggle_pause()
        self.tracking_enabled = not is_paused
        self.after(0, lambda: self.switch_tracking.select() if not is_paused else self.switch_tracking.deselect())

    def toggle_pointer_hotkey(self):
        """Called by F6 hotkey listener with debounce protection."""
        now = time.perf_counter()
        if (now - self._last_hotkey_time) < self._hotkey_cooldown:
            return
        self._last_hotkey_time = now
        is_paused = self.mouse.toggle_pointer_pause()
        is_active = not is_paused
        self._update_config_attr("pointer_enabled", is_active)
        self.after(0, lambda: self.switch_head_pointer.select() if is_active else self.switch_head_pointer.deselect())

    def toggle_clicks_hotkey(self):
        """Called by F7 hotkey listener with debounce protection."""
        now = time.perf_counter()
        if (now - self._last_hotkey_time) < self._hotkey_cooldown:
            return
        self._last_hotkey_time = now
        is_paused = self.mouse.toggle_clicking_pause()
        is_active = not is_paused
        self._update_config_attr("clicking_enabled", is_active)
        self.after(0, lambda: self.switch_click_actions.select() if is_active else self.switch_click_actions.deselect())

    def _on_key_press(self, event):
        """Dispatches assistive switch actions from physical keyboard events."""
        key = event.keysym.lower()
        char = event.char.lower() if event.char else ""

        sw_left = getattr(self.config, "hotkey_switch_left", "").lower()
        sw_right = getattr(self.config, "hotkey_switch_right", "").lower()
        sw_drag = getattr(self.config, "hotkey_switch_drag", "").lower()

        if sw_left and (key == sw_left or char == sw_left):
            self.mouse.trigger_switch_action("left")
        elif sw_right and (key == sw_right or char == sw_right):
            self.mouse.trigger_switch_action("right")
        elif sw_drag and (key == sw_drag or char == sw_drag):
            self.mouse.trigger_switch_action("drag")

    def _on_action_triggered(self, action_name: str):
        """Thread-safe trigger for HUD action badge update."""
        if threading.current_thread() is threading.main_thread():
            self._update_badge_ui(action_name)
        else:
            self.after(0, lambda a=action_name: self._update_badge_ui(a))

    def _update_badge_ui(self, action_name: str):
        self.last_action_badge_time = time.time()
        badge_colors = {
            "Left Click": ("#14532d", "#4ade80"),
            "Right Click": ("#1e3a8a", "#60a5fa"),
            "Double Click": ("#581c87", "#c084fc"),
            "Middle Click": ("#78350f", "#fcd34d"),
            "DRAG START": ("#831843", "#f472b6"),
            "DRAG RELEASE": ("#374151", "#9ca3af"),
            "PAUSED": ("#7f1d1d", "#f87171"),
            "RESUMED": ("#14532d", "#4ade80"),
            "TRACKING ON": ("#14532d", "#4ade80"),
            "CALIBRATED": ("#065f46", "#34d399"),
            "MOUSE OVERRIDE": ("#78350f", "#fcd34d"),
            "GESTURE ONLY": ("#1e3a8a", "#93c5fd"),
            "POINTER ON": ("#14532d", "#4ade80"),
            "POINTER PAUSED": ("#78350f", "#fcd34d"),
            "CLICKS ON": ("#14532d", "#4ade80"),
            "CLICKS PAUSED": ("#78350f", "#fcd34d"),
            "DWELL CLICK": ("#581c87", "#c084fc"),
            "SMART SENS ON": ("#14532d", "#4ade80"),
            "SMART SENS OFF": ("#78350f", "#fcd34d"),
        }
        bg, fg = badge_colors.get(action_name, ("#27272a", "#e4e4e7"))
        self.badge_label.configure(text=f"[ {action_name.upper()} ]", fg_color=bg, text_color=fg)

    def _start_camera_stream(self):
        self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default API if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(self.config.camera_index)

        # Zero-Latency Ingestion: negotiate format first, then lock hardware buffer to 1
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.video_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.video_thread.start()

    def _camera_worker(self):
        """Dedicated thread for video capture, Face Mesh processing, and frame pre-rendering."""
        ui_w, ui_h = 480, 360
        consecutive_read_failures = 0
        while self.is_running:
            try:
                # CPU Throttling when master paused (saves 80%+ CPU during standby)
                if self.mouse.is_paused or not self.tracking_enabled:
                    time.sleep(0.03)

                if not self.cap or not self.cap.isOpened():
                    consecutive_read_failures += 1
                    if consecutive_read_failures == 15:
                        # Push diagnostic placeholder frame to GUI
                        offline_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
                        cv2.putText(
                            offline_frame,
                            "Camera Offline / Not Found",
                            (60, ui_h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 100, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            offline_frame,
                            f"Device index: {self.config.camera_index}",
                            (120, ui_h // 2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )
                        rgb = cv2.cvtColor(offline_frame, cv2.COLOR_BGR2RGB)
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put_nowait((Image.fromarray(rgb), {}))

                    # Universal Dwell fallback: continues operating even if camera is offline
                    if self.tracking_enabled and not self.mouse.is_paused:
                        try:
                            curr_pos = self.mouse.get_physical_cursor_pos()
                            self.mouse.check_dwell(cached_pos=curr_pos)
                        except Exception:
                            pass
                    time.sleep(0.03)
                    continue

                res = self.cap.read()
                if not isinstance(res, (tuple, list)) or len(res) != 2:
                    time.sleep(0.02)
                    continue
                ret, frame = res
                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures == 15:
                        offline_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
                        cv2.putText(
                            offline_frame,
                            "Camera Read Error / Disconnected",
                            (40, ui_h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 100, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        rgb = cv2.cvtColor(offline_frame, cv2.COLOR_BGR2RGB)
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put_nowait((Image.fromarray(rgb), {}))

                    if self.tracking_enabled and not self.mouse.is_paused:
                        try:
                            curr_pos = self.mouse.get_physical_cursor_pos()
                            self.mouse.check_dwell(cached_pos=curr_pos)
                        except Exception:
                            pass
                    time.sleep(0.02)
                    continue

                consecutive_read_failures = 0

                # Query OS physical cursor position ONCE per frame (zero redundant inquiries)
                try:
                    curr_pos = self.mouse.get_physical_cursor_pos()
                except Exception:
                    curr_pos = (int(self.mouse.screen_width / 2), int(self.mouse.screen_height / 2))

                # Check override status with cached physical cursor pos
                is_overridden, just_resumed = self.mouse.update_override_state(cached_pos=curr_pos)
                if just_resumed:
                    self.tracker.reset_filters(
                        curr_pos,
                        screen_w=self.mouse.screen_width,
                        screen_h=self.mouse.screen_height,
                    )
                    if hasattr(self, "hand_tracker"):
                        self.hand_tracker.reset_filters(curr_pos)

                target_pos, gesture, annotated_frame, telemetry = self.tracker.process_frame(
                    frame,
                    self.mouse.screen_width,
                    self.mouse.screen_height,
                    current_mouse_pos=curr_pos,
                    is_overridden=is_overridden,
                )

                # Process Hand Tracking if enabled
                hand_target_pos = None
                hand_gesture = None
                if getattr(self.config, "hand_tracking_enabled", True) and hasattr(self, "hand_tracker"):
                    hand_target_pos, hand_gesture, annotated_frame, hand_telemetry = self.hand_tracker.process_frame(
                        annotated_frame,
                        self.mouse.screen_width,
                        self.mouse.screen_height,
                        current_mouse_pos=curr_pos,
                        is_overridden=is_overridden,
                    )
                    telemetry.update(hand_telemetry)

                # Pointer Routing: Head vs Finger Tip
                pointer_src = getattr(self.config, "pointer_source", "head")
                effective_pos = None
                if pointer_src == "finger":
                    effective_pos = hand_target_pos
                else:
                    effective_pos = target_pos

                # Move mouse if active (pass cached_pos to avoid redundant Win32 inquiries)
                if self.tracking_enabled and not self.mouse.is_paused and effective_pos is not None:
                    self.mouse.move_to(effective_pos[0], effective_pos[1], cached_pos=curr_pos)

                # Trigger gestures (Face & Hand) - Must pass gesture even if None so release triggers fire cleanly
                if self.tracking_enabled and not self.mouse.is_paused:
                    self.mouse.process_gesture(gesture)
                    if hasattr(self.mouse, "process_hand_gesture"):
                        self.mouse.process_hand_gesture(hand_gesture)

                # Universal Dwell Control check (runs regardless of head pointer mode)
                dwell_action, dwell_progress = self.mouse.check_dwell(cached_pos=curr_pos)
                telemetry["dwell_progress"] = dwell_progress

                # Offload frame resize and RGB conversion to worker thread so Tkinter UI stays silky smooth
                resized = cv2.resize(annotated_frame, (ui_w, ui_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Push pre-rendered PIL image and telemetry to GUI queue (drop old frame if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait((pil_img, telemetry))
                except queue.Full:
                    pass

            except Exception as e:
                # Top-level thread error boundary prevents silent termination of camera capture
                print(f"[CameraWorker] Warning: Exception during frame processing: {e}")
                time.sleep(0.02)

    def _update_gui_loop(self):
        """Pulls latest frames and telemetry metrics from worker thread."""
        if not self.is_running:
            return

        try:
            # Drop-stale queue drain: take only the latest available frame
            latest_item = None
            while not self.frame_queue.empty():
                try:
                    latest_item = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_item is not None:
                pil_img, telemetry = latest_item

                # Update Advanced Gauges
                l_ear = telemetry.get("left_ear", 0.0)
                r_ear = telemetry.get("right_ear", 0.0)
                mar = telemetry.get("mar", 0.0)
                dwell_p = telemetry.get("dwell_progress", 0.0)

                if hasattr(self, "bar_left_ear"):
                    self.bar_left_ear.set(min(1.0, l_ear / 0.5))
                    self.lbl_left_ear.configure(text=f"{l_ear:.2f}")

                if hasattr(self, "bar_right_ear"):
                    self.bar_right_ear.set(min(1.0, r_ear / 0.5))
                    self.lbl_right_ear.configure(text=f"{r_ear:.2f}")

                if hasattr(self, "bar_mar"):
                    self.bar_mar.set(min(1.0, mar / 0.8))
                    self.lbl_mar.configure(text=f"{mar:.2f}")

                if hasattr(self, "bar_dwell"):
                    if abs(dwell_p - self._last_dwell_progress) >= 0.01 or (
                        dwell_p in (0.0, 1.0) and self._last_dwell_progress != dwell_p
                    ):
                        self.bar_dwell.set(dwell_p)
                        self.lbl_dwell.configure(text=f"{int(dwell_p * 100)}%")
                        self._last_dwell_progress = dwell_p

                # Update Hand Tracking Telemetry HUD
                hand_det = telemetry.get("hand_detected", False)
                hand_ign = telemetry.get("hand_ignored", False)
                handedness = telemetry.get("handedness", "Hand")
                h_gest = telemetry.get("hand_gesture", None)
                thumb_stat = telemetry.get("thumb_status", "TUCKED")

                # Advanced HUD text
                if hasattr(self, "lbl_hand_status"):
                    if hand_ign:
                        self.lbl_hand_status.configure(text=f"Hand: [ {handedness} ]", text_color="#f59e0b")
                        self.lbl_hand_gesture.configure(text="Gesture: [ IGNORED ]", text_color="#f59e0b")
                    elif hand_det:
                        thumb_str = f" | T:{thumb_stat}" if getattr(self.config, "hand_thumb_trigger_enabled", True) else ""
                        self.lbl_hand_status.configure(text=f"Hand: [ {handedness or 'DETECTED'}{thumb_str} ]", text_color="#4ade80")
                        self.lbl_hand_gesture.configure(text=f"Gesture: [ {h_gest or 'ACTIVE'} ]", text_color="#38bdf8")
                    else:
                        self.lbl_hand_status.configure(text="Hand: [ NO HAND ]", text_color="#71717a")
                        self.lbl_hand_gesture.configure(text="Gesture: [ NONE ]", text_color="#71717a")

                # Simple HUD text
                if hasattr(self, "lbl_simple_face"):
                    face_ok = (l_ear > 0.05 or r_ear > 0.05 or mar > 0.05)
                    if face_ok:
                        self.lbl_simple_face.configure(text="👤 Wajah: 🟢 Terdeteksi", text_color="#4ade80")
                    else:
                        self.lbl_simple_face.configure(text="👤 Wajah: ⚪ Mencari...", text_color="#a1a1aa")

                if hasattr(self, "lbl_simple_hand"):
                    if hand_ign:
                        self.lbl_simple_hand.configure(text=f"🖐️ Tangan: 🟡 Diabaikan ({handedness})", text_color="#f59e0b")
                    elif hand_det:
                        self.lbl_simple_hand.configure(text=f"🖐️ Tangan: 🟢 {handedness} Aktif", text_color="#4ade80")
                    else:
                        self.lbl_simple_hand.configure(text="🖐️ Tangan: ⚪ Tidak Terdeteksi", text_color="#71717a")

                if hasattr(self, "lbl_simple_ptr_src"):
                    cur_src = getattr(self.config, "pointer_source", "head")
                    src_str = "Jari Telunjuk (Hand)" if cur_src == "finger" else "Gerakan Kepala (Head)"
                    self.lbl_simple_ptr_src.configure(text=f"🎯 Pointer: {src_str}")

                # Reset Action Badge to Standby after 1.2 seconds of inactivity
                if time.time() - self.last_action_badge_time > 1.2:
                    if self.mouse.is_paused or not self.tracking_enabled:
                        self.badge_label.configure(
                            text="[ MASTER PAUSED ]", fg_color="#7f1d1d", text_color="#f87171"
                        )
                        self.lbl_stream_status.configure(text="○ PAUSED", fg_color="#450a0a", text_color="#f87171")
                    elif self.mouse.is_overridden:
                        self.badge_label.configure(
                            text="[ MOUSE OVERRIDE ]", fg_color="#78350f", text_color="#fcd34d"
                        )
                        self.lbl_stream_status.configure(text="● LIVE", fg_color="#064e3b", text_color="#4ade80")
                    elif not getattr(self.config, "pointer_enabled", True) and getattr(self.config, "clicking_enabled", True):
                        self.badge_label.configure(
                            text="[ CLICKS ONLY ]", fg_color="#1e3a8a", text_color="#93c5fd"
                        )
                        self.lbl_stream_status.configure(text="● LIVE", fg_color="#064e3b", text_color="#4ade80")
                    elif getattr(self.config, "pointer_enabled", True) and not getattr(self.config, "clicking_enabled", True):
                        self.badge_label.configure(
                            text="[ POINTER ONLY ]", fg_color="#065f46", text_color="#34d399"
                        )
                        self.lbl_stream_status.configure(text="● LIVE", fg_color="#064e3b", text_color="#4ade80")
                    else:
                        self.badge_label.configure(
                            text="[ TRACKING ]", fg_color="#14532d", text_color="#4ade80"
                        )
                        self.lbl_stream_status.configure(text="● LIVE", fg_color="#064e3b", text_color="#4ade80")

                # Render pre-built image to Tkinter Canvas in < 1ms
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(480, 360))
                self.video_canvas.configure(image=ctk_img, text="")
                self.video_canvas.image = ctk_img

        except Exception:
            pass

        self.after(20, self._update_gui_loop)

    def on_closing(self):
        """Clean shutdown handler."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if hasattr(self, "hand_tracker") and self.hand_tracker:
            self.hand_tracker.close()
        self.tracker.close()
        self.destroy()


# Backward-compatibility alias
ZeroTouchApp = FaceControlApp

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from config.settings import RECORDINGS_ROOT

from ui.styles import (
    PANEL_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    SUCCESS_COLOR,
    ERROR_COLOR,
    ACCENT_COLOR,
    BORDER_COLOR,
    FONT_SECTION,
    FONT_NORMAL,
    FONT_STATUS,
    FONT_BUTTON,
    PANEL_PADDING,
)


STROKE_TYPES = [
    "Forehand",
    "Backhand",
    "Serve",
    "Volley",
    "Dink",
    "Overhead",
]


SESSION_NUMBERS = [
    f"{number:02d}"
    for number in range(1, 3)
]


TRIAL_NUMBERS = [
    f"{number:02d}"
    for number in range(1, 31)
]


HANDEDNESS_OPTIONS = [
    "Right",
    "Left",
]


class RecordingPage(tk.Frame):
    def __init__(
        self,
        parent,
        on_start_recording,
        on_stop_recording,
    ):
        super().__init__(
            parent,
            bg=PANEL_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
        )

        self.configure(
            padx=PANEL_PADDING,
            pady=PANEL_PADDING,
        )

        self.on_start_recording = (
            on_start_recording
        )

        self.on_stop_recording = (
            on_stop_recording
        )

        # --------------------------------------------------
        # Variables
        # --------------------------------------------------

        self.participant_var = tk.StringVar()

        self.session_var = tk.StringVar(
            value="01"
        )

        self.stroke_var = tk.StringVar(
            value="Forehand"
        )

        self.trial_var = tk.StringVar(
            value="01"
        )

        self.handedness_var = tk.StringVar(
            value="Right"
        )

        self.recording = False

        self._build_widgets()

        # Update folder preview whenever
        # recording information changes.

        self.participant_var.trace_add(
            "write",
            self._recording_info_changed,
        )

        self.session_var.trace_add(
            "write",
            self._recording_info_changed,
        )

        self.stroke_var.trace_add(
            "write",
            self._recording_info_changed,
        )

        self.trial_var.trace_add(
            "write",
            self._recording_info_changed,
        )

        self._update_save_path()

    def _build_widgets(self):
        title = tk.Label(
            self,
            text="Recording Setup",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        title.grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 12),
        )

        # --------------------------------------------------
        # Participant ID
        # --------------------------------------------------

        participant_label = tk.Label(
            self,
            text="Participant ID:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        participant_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=4,
        )

        self.participant_entry = tk.Entry(
            self,
            textvariable=self.participant_var,
            font=FONT_NORMAL,
            width=15,
        )

        self.participant_entry.grid(
            row=1,
            column=1,
            sticky="w",
            padx=(8, 20),
            pady=4,
        )

        # --------------------------------------------------
        # Session
        # --------------------------------------------------

        session_label = tk.Label(
            self,
            text="Session:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        session_label.grid(
            row=1,
            column=2,
            sticky="w",
            pady=4,
        )

        self.session_dropdown = ttk.Combobox(
            self,
            textvariable=self.session_var,
            values=SESSION_NUMBERS,
            state="readonly",
            width=10,
        )

        self.session_dropdown.grid(
            row=1,
            column=3,
            sticky="w",
            padx=(8, 0),
            pady=4,
        )

        # --------------------------------------------------
        # Stroke
        # --------------------------------------------------

        stroke_label = tk.Label(
            self,
            text="Stroke:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        stroke_label.grid(
            row=2,
            column=0,
            sticky="w",
            pady=4,
        )

        self.stroke_dropdown = ttk.Combobox(
            self,
            textvariable=self.stroke_var,
            values=STROKE_TYPES,
            state="readonly",
            width=15,
        )

        self.stroke_dropdown.grid(
            row=2,
            column=1,
            sticky="w",
            padx=(8, 20),
            pady=4,
        )

        # --------------------------------------------------
        # Trial
        # --------------------------------------------------

        trial_label = tk.Label(
            self,
            text="Trial:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        trial_label.grid(
            row=2,
            column=2,
            sticky="w",
            pady=4,
        )

        self.trial_dropdown = ttk.Combobox(
            self,
            textvariable=self.trial_var,
            values=TRIAL_NUMBERS,
            state="readonly",
            width=10,
        )

        self.trial_dropdown.grid(
            row=2,
            column=3,
            sticky="w",
            padx=(8, 0),
            pady=4,
        )

        # --------------------------------------------------
        # Handedness
        # --------------------------------------------------

        handedness_label = tk.Label(
            self,
            text="Handedness:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        handedness_label.grid(
            row=3,
            column=0,
            sticky="w",
            pady=4,
        )

        self.handedness_dropdown = ttk.Combobox(
            self,
            textvariable=self.handedness_var,
            values=HANDEDNESS_OPTIONS,
            state="readonly",
            width=15,
        )

        self.handedness_dropdown.grid(
            row=3,
            column=1,
            sticky="w",
            padx=(8, 20),
            pady=4,
        )

        # --------------------------------------------------
        # Notes
        # --------------------------------------------------

        notes_label = tk.Label(
            self,
            text="Notes:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        notes_label.grid(
            row=4,
            column=0,
            sticky="nw",
            pady=(8, 4),
        )

        self.notes_text = tk.Text(
            self,
            width=50,
            height=3,
            font=FONT_NORMAL,
            wrap="word",
        )

        self.notes_text.grid(
            row=4,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 4),
        )

        # --------------------------------------------------
        # Save path
        # --------------------------------------------------

        save_label = tk.Label(
            self,
            text="Save path:",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        save_label.grid(
            row=5,
            column=0,
            sticky="nw",
            pady=(10, 4),
        )

        self.save_path_value = tk.Label(
            self,
            text="",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_NORMAL,
            anchor="w",
            justify="left",
            wraplength=500,
        )

        self.save_path_value.grid(
            row=5,
            column=1,
            columnspan=3,
            sticky="w",
            padx=(8, 0),
            pady=(10, 4),
        )

        # --------------------------------------------------
        # Record button
        # --------------------------------------------------

        self.record_button = tk.Button(
            self,
            text="START RECORDING",
            font=FONT_BUTTON,
            bg=ACCENT_COLOR,
            fg="white",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
            command=self._toggle_recording,
        )

        self.record_button.grid(
            row=6,
            column=0,
            columnspan=4,
            pady=(16, 12),
        )

        # --------------------------------------------------
        # Recording status
        # --------------------------------------------------

        status_title = tk.Label(
            self,
            text="Recording Status",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        status_title.grid(
            row=7,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(8, 8),
        )

        self.state_value = (
            self._add_status_row(
                row=8,
                label_text="State:",
            )
        )

        self.elapsed_value = (
            self._add_status_row(
                row=9,
                label_text="Elapsed:",
            )
        )

        self.imu_count_value = (
            self._add_status_row(
                row=10,
                label_text="IMU samples:",
            )
        )

        self.pose_count_value = (
            self._add_status_row(
                row=11,
                label_text="Pose frames:",
            )
        )

        self.state_value.config(
            text="READY",
            fg=SUCCESS_COLOR,
        )

        self.elapsed_value.config(
            text="00:00.000"
        )

        self.imu_count_value.config(
            text="0"
        )

        self.pose_count_value.config(
            text="0"
        )

    def _add_status_row(
        self,
        row,
        label_text,
    ):
        label = tk.Label(
            self,
            text=label_text,
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        label.grid(
            row=row,
            column=0,
            sticky="w",
            pady=2,
        )

        value = tk.Label(
            self,
            text="-",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_STATUS,
        )

        value.grid(
            row=row,
            column=1,
            columnspan=3,
            sticky="w",
            padx=(8, 0),
            pady=2,
        )

        return value

    def _recording_info_changed(
        self,
        *args,
    ):
        if not self.recording:
            self._update_save_path()

    def _build_names(self):
        participant = (
            self.participant_var.get().strip()
        )

        session = (
            self.session_var.get()
        )

        stroke = (
            self.stroke_var.get()
        )

        trial = (
            self.trial_var.get()
        )

        movement_name = (
            f"Participant{participant}_"
            f"Session{session}_"
            f"{stroke}"
        )

        trial_name = (
            f"Trial_{trial}"
        )

        return (
            movement_name,
            trial_name,
        )

    def _update_save_path(self):
        participant = (
            self.participant_var.get().strip()
        )

        if not participant:
            self.save_path_value.config(
                text="Enter Participant ID"
            )
            return

        movement_name, trial_name = (
            self._build_names()
        )

        path = (
            Path(RECORDINGS_ROOT)
            / movement_name
            / trial_name
        )

        self.save_path_value.config(
            text=str(path)
        )

    def _toggle_recording(self):
        if not self.recording:
            self._request_start()
        else:
            self._request_stop()

    def _request_start(self):
        participant = (
            self.participant_var.get().strip()
        )

        if not participant:
            self.state_value.config(
                text="ENTER PARTICIPANT ID",
                fg=ERROR_COLOR,
            )
            return

        movement_name, trial_name = (
            self._build_names()
        )

        metadata = {
            "participant_id": (
                f"P{participant}"
            ),

            "session_number": int(
                self.session_var.get()
            ),

            "stroke_type": (
                self.stroke_var.get()
            ),

            "trial_number": int(
                self.trial_var.get()
            ),

            "handedness": (
                self.handedness_var.get()
            ),

            "notes": (
                self.notes_text.get(
                    "1.0",
                    "end",
                ).strip()
            ),

            "movement_name": (
                movement_name
            ),

            "trial_name": (
                trial_name
            ),
        }

        success = self.on_start_recording(
            metadata
        )

        if success:
            self.set_recording_state(
                True
            )

    def _request_stop(self):
        success = (
            self.on_stop_recording()
        )

        if success:
            self.set_recording_state(
                False
            )

    def set_recording_state(
        self,
        recording: bool,
    ):
        self.recording = recording

        state = (
            "disabled"
            if recording
            else "normal"
        )

        dropdown_state = (
            "disabled"
            if recording
            else "readonly"
        )

        if recording:
            self.state_value.config(
                text="RECORDING",
                fg=ERROR_COLOR,
            )

            self.record_button.config(
                text="STOP RECORDING",
                bg=ERROR_COLOR,
            )

        else:
            self.state_value.config(
                text="READY",
                fg=SUCCESS_COLOR,
            )

            self.record_button.config(
                text="START RECORDING",
                bg=ACCENT_COLOR,
            )

        self.participant_entry.config(
            state=state
        )

        self.session_dropdown.config(
            state=dropdown_state
        )

        self.stroke_dropdown.config(
            state=dropdown_state
        )

        self.trial_dropdown.config(
            state=dropdown_state
        )

        self.handedness_dropdown.config(
            state=dropdown_state
        )

        self.notes_text.config(
            state=state
        )

    def update_recording_status(
        self,
        status,
    ):
        elapsed_seconds = status.get(
            "elapsed_seconds",
            0.0,
        )

        minutes = int(
            elapsed_seconds // 60
        )

        seconds = int(
            elapsed_seconds % 60
        )

        milliseconds = int(
            (
                elapsed_seconds
                - int(elapsed_seconds)
            )
            * 1000
        )

        self.elapsed_value.config(
            text=(
                f"{minutes:02d}:"
                f"{seconds:02d}."
                f"{milliseconds:03d}"
            )
        )

        self.imu_count_value.config(
            text=str(
                status.get(
                    "imu_samples_recorded",
                    0,
                )
            )
        )

        self.pose_count_value.config(
            text=str(
                status.get(
                    "pose_frames_recorded",
                    0,
                )
            )
        )
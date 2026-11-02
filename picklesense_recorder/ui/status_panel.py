import tkinter as tk

from ui.styles import (
    PANEL_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    ERROR_COLOR,
    BORDER_COLOR,
    FONT_SECTION,
    FONT_NORMAL,
    FONT_STATUS,
    PANEL_PADDING,
)


class StatusPanel(tk.Frame):
    def __init__(self, parent):
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

        self._build_widgets()

    def _build_widgets(self):
        title = tk.Label(
            self,
            text="System Status",
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
        # IMU / ESP32
        # --------------------------------------------------

        imu_title = tk.Label(
            self,
            text="ESP32 / MPU6050",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        imu_title.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
        )

        self.imu_state_label = tk.Label(
            self,
            text="NOT RECEIVING",
            bg=PANEL_COLOR,
            fg=ERROR_COLOR,
            font=FONT_STATUS,
        )

        self.imu_state_label.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 8),
        )

        self.imu_rate_value = self._add_status_row(
            row=3,
            column=0,
            label_text="Rate:",
        )

        self.imu_packets_value = self._add_status_row(
            row=4,
            column=0,
            label_text="Packets:",
        )

        self.imu_missing_value = self._add_status_row(
            row=5,
            column=0,
            label_text="Missing:",
        )

        self.imu_malformed_value = self._add_status_row(
            row=6,
            column=0,
            label_text="Malformed:",
        )

        self.imu_age_value = self._add_status_row(
            row=7,
            column=0,
            label_text="Last packet:",
        )

        # --------------------------------------------------
        # Camera / Pose
        # --------------------------------------------------

        camera_title = tk.Label(
            self,
            text="Camera / Pose",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        camera_title.grid(
            row=1,
            column=2,
            columnspan=2,
            sticky="w",
            padx=(40, 0),
        )

        self.camera_state_label = tk.Label(
            self,
            text="INACTIVE",
            bg=PANEL_COLOR,
            fg=ERROR_COLOR,
            font=FONT_STATUS,
        )

        self.camera_state_label.grid(
            row=2,
            column=2,
            columnspan=2,
            sticky="w",
            padx=(40, 0),
            pady=(4, 8),
        )

        self.camera_fps_value = self._add_status_row(
            row=3,
            column=2,
            label_text="FPS:",
            left_padding=40,
        )

        self.camera_frames_value = self._add_status_row(
            row=4,
            column=2,
            label_text="Frames:",
            left_padding=40,
        )

        self.pose_state_value = self._add_status_row(
            row=5,
            column=2,
            label_text="Pose:",
            left_padding=40,
        )

        self.pose_detected_value = self._add_status_row(
            row=6,
            column=2,
            label_text="With pose:",
            left_padding=40,
        )

        self.pose_missing_value = self._add_status_row(
            row=7,
            column=2,
            label_text="Without pose:",
            left_padding=40,
        )

    def _add_status_row(
        self,
        row,
        column,
        label_text,
        left_padding=0,
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
            column=column,
            sticky="w",
            padx=(left_padding, 8),
            pady=2,
        )

        value = tk.Label(
            self,
            text="-",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_NORMAL,
        )

        value.grid(
            row=row,
            column=column + 1,
            sticky="w",
            pady=2,
        )

        return value

    def update_imu_status(self, status):
        receiving = False

        age = status.get(
            "last_packet_age_seconds"
        )

        if (
            status.get("running")
            and age is not None
            and age < 2.0
        ):
            receiving = True

        if receiving:
            self.imu_state_label.config(
                text="RECEIVING",
                fg=SUCCESS_COLOR,
            )
        else:
            self.imu_state_label.config(
                text="NOT RECEIVING",
                fg=ERROR_COLOR,
            )

        self.imu_rate_value.config(
            text=(
                f"{status.get('current_rate_hz', 0.0):.2f} Hz"
            )
        )

        self.imu_packets_value.config(
            text=str(
                status.get(
                    "total_packets",
                    0,
                )
            )
        )

        missing = status.get(
            "missing_packets",
            0,
        )

        self.imu_missing_value.config(
            text=str(missing),
            fg=(
                SUCCESS_COLOR
                if missing == 0
                else WARNING_COLOR
            ),
        )

        malformed = status.get(
            "malformed_packets",
            0,
        )

        self.imu_malformed_value.config(
            text=str(malformed),
            fg=(
                SUCCESS_COLOR
                if malformed == 0
                else WARNING_COLOR
            ),
        )

        if age is None:
            age_text = "N/A"
        else:
            age_text = f"{age:.3f} s"

        self.imu_age_value.config(
            text=age_text
        )

    def update_camera_status(
        self,
        camera_status,
        pose_status,
    ):
        camera_running = camera_status.get(
            "running",
            False,
        )

        if camera_running:
            self.camera_state_label.config(
                text="ACTIVE",
                fg=SUCCESS_COLOR,
            )
        else:
            self.camera_state_label.config(
                text="INACTIVE",
                fg=ERROR_COLOR,
            )

        self.camera_fps_value.config(
            text=(
                f"{camera_status.get('current_fps', 0.0):.2f}"
            )
        )

        self.camera_frames_value.config(
            text=str(
                camera_status.get(
                    "frames_received",
                    0,
                )
            )
        )

        frames_with_pose = pose_status.get(
            "frames_with_pose",
            0,
        )

        frames_without_pose = pose_status.get(
            "frames_without_pose",
            0,
        )

        pose_detected = (
            frames_with_pose > 0
            and (
                frames_with_pose
                >= frames_without_pose
            )
        )

        if pose_detected:
            self.pose_state_value.config(
                text="DETECTED",
                fg=SUCCESS_COLOR,
            )
        else:
            self.pose_state_value.config(
                text="NOT DETECTED",
                fg=WARNING_COLOR,
            )

        self.pose_detected_value.config(
            text=str(frames_with_pose)
        )

        self.pose_missing_value.config(
            text=str(frames_without_pose)
        )
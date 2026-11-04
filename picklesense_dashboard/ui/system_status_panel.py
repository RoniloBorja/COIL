import tkinter as tk

from ui.styles import (
    PANEL_COLOR,
    BORDER_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    ERROR_COLOR,
    FONT_SECTION,
    FONT_SMALL,
    FONT_NORMAL,
    FONT_STATUS,
    PANEL_PADDING,
)


class SystemStatusPanel(tk.Frame):
    def __init__(
        self,
        parent,
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

        self.status_labels = {}

        self._build_widgets()

    def _build_widgets(
        self,
    ):
        title = tk.Label(
            self,
            text="System Status",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        title.pack(
            anchor="w",
        )

        subtitle = tk.Label(
            self,
            text="Live subsystem health",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        subtitle.pack(
            anchor="w",
            pady=(2, 12),
        )

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        self._add_status_row(
            key="camera",
            label="Camera",
        )

        self._add_detail_row(
            key="camera_fps",
            label="Camera FPS",
        )

        # --------------------------------------------------
        # Pose
        # --------------------------------------------------

        self._add_status_row(
            key="pose",
            label="Pose Tracking",
        )

        self._add_detail_row(
            key="pose_detection_rate",
            label="Pose detection",
        )

        # --------------------------------------------------
        # MAX78000
        # --------------------------------------------------

        self._add_status_row(
            key="max78000",
            label="MAX78000",
        )

        self._add_detail_row(
            key="packets",
            label="Packets",
        )

        self._add_detail_row(
            key="malformed",
            label="Malformed",
        )

        # --------------------------------------------------
        # MPU6500
        # --------------------------------------------------

        self._add_status_row(
            key="imu",
            label="MPU6500",
        )

        self._add_detail_row(
            key="imu_rate",
            label="IMU Rate",
        )

        self._add_detail_row(
            key="imu_samples",
            label="Window Samples",
        )

    def _add_status_row(
        self,
        key,
        label,
    ):
        row = tk.Frame(
            self,
            bg=PANEL_COLOR,
        )

        row.pack(
            fill="x",
            pady=3,
        )

        name_label = tk.Label(
            row,
            text=label,
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        name_label.pack(
            side="left",
        )

        value_label = tk.Label(
            row,
            text="Unknown",
            bg=PANEL_COLOR,
            fg=WARNING_COLOR,
            font=FONT_STATUS,
        )

        value_label.pack(
            side="right",
        )

        self.status_labels[
            key
        ] = value_label

    def _add_detail_row(
        self,
        key,
        label,
    ):
        row = tk.Frame(
            self,
            bg=PANEL_COLOR,
        )

        row.pack(
            fill="x",
            pady=1,
        )

        name_label = tk.Label(
            row,
            text=label,
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        name_label.pack(
            side="left",
        )

        value_label = tk.Label(
            row,
            text="--",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SMALL,
        )

        value_label.pack(
            side="right",
        )

        self.status_labels[
            key
        ] = value_label

    def update_camera_status(
        self,
        status,
    ):
        active = status.get(
            "active",
            False,
        )

        receiving = status.get(
            "receiving",
            False,
        )

        fps = float(
            status.get(
                "current_fps",
                0.0,
            )
        )

        if receiving:
            self.status_labels[
                "camera"
            ].config(
                text="Connected",
                fg=SUCCESS_COLOR,
            )

        elif active:
            self.status_labels[
                "camera"
            ].config(
                text="No Frames",
                fg=WARNING_COLOR,
            )

        else:
            self.status_labels[
                "camera"
            ].config(
                text="Disconnected",
                fg=ERROR_COLOR,
            )

        self.status_labels[
            "camera_fps"
        ].config(
            text=f"{fps:.1f} FPS"
        )

    def update_pose_status(
        self,
        status,
    ):
        running = status.get(
            "running",
            False,
        )

        frames_processed = int(
            status.get(
                "frames_processed",
                0,
            )
        )

        frames_with_pose = int(
            status.get(
                "frames_with_pose",
                0,
            )
        )

        detection_rate = float(
            status.get(
                "detection_rate",
                0.0,
            )
        )

        if (
            running
            and frames_with_pose > 0
        ):
            self.status_labels[
                "pose"
            ].config(
                text="Tracking",
                fg=SUCCESS_COLOR,
            )

        elif running:
            self.status_labels[
                "pose"
            ].config(
                text="Searching",
                fg=WARNING_COLOR,
            )

        else:
            self.status_labels[
                "pose"
            ].config(
                text="Stopped",
                fg=ERROR_COLOR,
            )

        if frames_processed > 0:
            self.status_labels[
                "pose_detection_rate"
            ].config(
                text=(
                    f"{detection_rate * 100:.1f}%"
                )
            )

        else:
            self.status_labels[
                "pose_detection_rate"
            ].config(
                text="--"
            )

    def update_max78000_status(
        self,
        status,
    ):
        connected = status.get(
            "connected",
            False,
        )

        receiving = status.get(
            "receiving",
            False,
        )

        total_packets = int(
            status.get(
                "total_packets",
                0,
            )
        )

        malformed_packets = int(
            status.get(
                "malformed_packets",
                0,
            )
        )

        if receiving:
            self.status_labels[
                "max78000"
            ].config(
                text="Receiving",
                fg=SUCCESS_COLOR,
            )

        elif connected:
            self.status_labels[
                "max78000"
            ].config(
                text="Connected",
                fg=WARNING_COLOR,
            )

        else:
            self.status_labels[
                "max78000"
            ].config(
                text="Disconnected",
                fg=ERROR_COLOR,
            )

        self.status_labels[
            "packets"
        ].config(
            text=str(
                total_packets
            )
        )

        self.status_labels[
            "malformed"
        ].config(
            text=str(
                malformed_packets
            )
        )

    def update_imu_status(
        self,
        statistics,
    ):
        if not statistics:
            self.status_labels[
                "imu"
            ].config(
                text="No Data",
                fg=ERROR_COLOR,
            )

            self.status_labels[
                "imu_rate"
            ].config(
                text="--"
            )

            self.status_labels[
                "imu_samples"
            ].config(
                text="0"
            )

            return

        sample_count = int(
            statistics.get(
                "sample_count",
                0,
            )
        )

        sampling_rate = float(
            statistics.get(
                "sampling_rate_hz",
                0.0,
            )
        )

        if sample_count > 0:
            self.status_labels[
                "imu"
            ].config(
                text="Active",
                fg=SUCCESS_COLOR,
            )

        else:
            self.status_labels[
                "imu"
            ].config(
                text="No Data",
                fg=ERROR_COLOR,
            )

        self.status_labels[
            "imu_rate"
        ].config(
            text=(
                f"{sampling_rate:.1f} Hz"
            )
        )

        self.status_labels[
            "imu_samples"
        ].config(
            text=str(
                sample_count
            )
        )

    def clear(
        self,
    ):
        self.status_labels[
            "camera"
        ].config(
            text="Unknown",
            fg=WARNING_COLOR,
        )

        self.status_labels[
            "camera_fps"
        ].config(
            text="--"
        )

        self.status_labels[
            "pose"
        ].config(
            text="Unknown",
            fg=WARNING_COLOR,
        )

        self.status_labels[
            "pose_detection_rate"
        ].config(
            text="--"
        )

        self.status_labels[
            "max78000"
        ].config(
            text="Unknown",
            fg=WARNING_COLOR,
        )

        self.status_labels[
            "packets"
        ].config(
            text="--"
        )

        self.status_labels[
            "malformed"
        ].config(
            text="--"
        )

        self.status_labels[
            "imu"
        ].config(
            text="Unknown",
            fg=WARNING_COLOR,
        )

        self.status_labels[
            "imu_rate"
        ].config(
            text="--"
        )

        self.status_labels[
            "imu_samples"
        ].config(
            text="--"
        )
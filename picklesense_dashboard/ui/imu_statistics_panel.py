import tkinter as tk

from ui.styles import (
    PANEL_COLOR,
    BORDER_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    STAT_ACCENT_COLOR,
    FONT_SECTION,
    FONT_SMALL,
    FONT_NORMAL,
    FONT_VALUE,
    PANEL_PADDING,
)


class ImuStatisticsPanel(tk.Frame):
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

        self.value_labels = {}

        self._build_widgets()

    def _build_widgets(
        self,
    ):
        title = tk.Label(
            self,
            text="IMU Measurements",
            bg=PANEL_COLOR,
            fg=STAT_ACCENT_COLOR,
            font=FONT_SECTION,
        )

        title.pack(
            anchor="w",
        )

        subtitle = tk.Label(
            self,
            text="Direct MPU6500 sensor statistics",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        subtitle.pack(
            anchor="w",
            pady=(2, 12),
        )

        # --------------------------------------------------
        # Primary statistics
        # --------------------------------------------------

        self._add_section_title(
            "Motion Magnitudes"
        )

        self._add_stat_row(
            key="peak_acceleration_g",
            label="Peak acceleration",
            unit="g",
        )

        self._add_stat_row(
            key="mean_acceleration_g",
            label="Mean acceleration",
            unit="g",
        )

        self._add_stat_row(
            key="acceleration_rms_g",
            label="Acceleration RMS",
            unit="g",
        )

        self._add_stat_row(
            key="peak_angular_velocity_dps",
            label="Peak angular velocity",
            unit="°/s",
        )

        self._add_stat_row(
            key="mean_angular_velocity_dps",
            label="Mean angular velocity",
            unit="°/s",
        )

        self._add_stat_row(
            key="gyroscope_rms_dps",
            label="Gyroscope RMS",
            unit="°/s",
        )

        # --------------------------------------------------
        # Axis maximums
        # --------------------------------------------------

        self._add_section_title(
            "Axis Maximums"
        )

        self._add_stat_row(
            key="max_ax_g",
            label="Max |Ax|",
            unit="g",
        )

        self._add_stat_row(
            key="max_ay_g",
            label="Max |Ay|",
            unit="g",
        )

        self._add_stat_row(
            key="max_az_g",
            label="Max |Az|",
            unit="g",
        )

        self._add_stat_row(
            key="max_gx_dps",
            label="Max |Gx|",
            unit="°/s",
        )

        self._add_stat_row(
            key="max_gy_dps",
            label="Max |Gy|",
            unit="°/s",
        )

        self._add_stat_row(
            key="max_gz_dps",
            label="Max |Gz|",
            unit="°/s",
        )

        # --------------------------------------------------
        # Stream information
        # --------------------------------------------------

        self._add_section_title(
            "Stream"
        )

        self._add_stat_row(
            key="sampling_rate_hz",
            label="Sampling rate",
            unit="Hz",
        )

        self._add_stat_row(
            key="sample_count",
            label="Samples in window",
            unit="",
            integer=True,
        )

    def _add_section_title(
        self,
        text,
    ):
        divider = tk.Frame(
            self,
            bg=BORDER_COLOR,
            height=1,
        )

        divider.pack(
            fill="x",
            pady=(8, 8),
        )

        label = tk.Label(
            self,
            text=text,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_NORMAL,
        )

        label.pack(
            anchor="w",
            pady=(0, 4),
        )

    def _add_stat_row(
        self,
        key,
        label,
        unit,
        integer=False,
    ):
        row = tk.Frame(
            self,
            bg=PANEL_COLOR,
        )

        row.pack(
            fill="x",
            pady=2,
        )

        name_label = tk.Label(
            row,
            text=label,
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
            anchor="w",
        )

        name_label.pack(
            side="left",
        )

        value_label = tk.Label(
            row,
            text="--",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_VALUE,
            anchor="e",
        )

        value_label.pack(
            side="right",
        )

        self.value_labels[
            key
        ] = {
            "label": value_label,
            "unit": unit,
            "integer": integer,
        }

    def update_statistics(
        self,
        statistics,
    ):
        if not statistics:
            self.clear()
            return

        for key, config in (
            self.value_labels.items()
        ):
            value = statistics.get(
                key
            )

            if value is None:
                config["label"].config(
                    text="--"
                )
                continue

            if config["integer"]:
                text = str(
                    int(value)
                )

            else:
                text = (
                    f"{float(value):.2f}"
                )

            if config["unit"]:
                text = (
                    f"{text} "
                    f"{config['unit']}"
                )

            config["label"].config(
                text=text
            )

    def clear(
        self,
    ):
        for config in (
            self.value_labels.values()
        ):
            config["label"].config(
                text="--"
            )
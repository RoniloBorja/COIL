import tkinter as tk

from ui.styles import (
    PANEL_COLOR,
    BORDER_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    IMU_AI_ACCENT_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    ERROR_COLOR,
    FONT_SECTION,
    FONT_NORMAL,
    FONT_SMALL,
    FONT_BIG_VALUE,
    FONT_VALUE,
    PANEL_PADDING,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)


class ImuAiPanel(tk.Frame):
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

        self._build_widgets()

    def _build_widgets(
        self,
    ):
        self.title_label = tk.Label(
            self,
            text="MAX78000 / MPU6500 AI",
            bg=PANEL_COLOR,
            fg=IMU_AI_ACCENT_COLOR,
            font=FONT_SECTION,
        )

        self.title_label.pack(
            anchor="w",
        )

        self.subtitle_label = tk.Label(
            self,
            text="Embedded IMU Stroke Classification",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        self.subtitle_label.pack(
            anchor="w",
            pady=(2, 12),
        )

        self.stroke_label = tk.Label(
            self,
            text="WAITING",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_BIG_VALUE,
        )

        self.stroke_label.pack(
            anchor="w",
        )

        self.confidence_label = tk.Label(
            self,
            text="Confidence: --",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_VALUE,
        )

        self.confidence_label.pack(
            anchor="w",
            pady=(4, 10),
        )

        divider = tk.Frame(
            self,
            bg=BORDER_COLOR,
            height=1,
        )

        divider.pack(
            fill="x",
            pady=(2, 10),
        )

        self.status_label = tk.Label(
            self,
            text="Waiting for MAX78000 prediction...",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_NORMAL,
        )

        self.status_label.pack(
            anchor="w",
        )

    def update_prediction(
        self,
        prediction,
    ):
        if prediction is None:
            self.clear()
            return

        class_name = prediction.get(
            "class_name",
            "WAITING",
        )

        confidence = float(
            prediction.get(
                "confidence",
                0.0,
            )
        )

        self.stroke_label.config(
            text=class_name.upper()
        )

        self.confidence_label.config(
            text=(
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            ),
            fg=self._confidence_color(
                confidence
            ),
        )

        self.status_label.config(
            text="Prediction received from MAX78000"
        )

    def set_connection_status(
        self,
        connected,
        receiving=False,
    ):
        if not connected:
            self.status_label.config(
                text="MAX78000 disconnected",
                fg=ERROR_COLOR,
            )

            return

        if connected and not receiving:
            self.status_label.config(
                text="MAX78000 connected — waiting for data",
                fg=WARNING_COLOR,
            )

            return

        self.status_label.config(
            text="MAX78000 receiving",
            fg=SUCCESS_COLOR,
        )

    def _confidence_color(
        self,
        confidence,
    ):
        if confidence >= CONFIDENCE_HIGH:
            return SUCCESS_COLOR

        if confidence >= CONFIDENCE_MEDIUM:
            return WARNING_COLOR

        return ERROR_COLOR

    def clear(
        self,
    ):
        self.stroke_label.config(
            text="WAITING"
        )

        self.confidence_label.config(
            text="Confidence: --",
            fg=SECONDARY_TEXT_COLOR,
        )

        self.status_label.config(
            text="Waiting for MAX78000 prediction...",
            fg=SECONDARY_TEXT_COLOR,
        )
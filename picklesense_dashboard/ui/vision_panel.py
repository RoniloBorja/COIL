import tkinter as tk

from ui.styles import (
    PANEL_COLOR,
    BORDER_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    VISION_ACCENT_COLOR,
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


class VisionPanel(tk.Frame):
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
            text="Vision AI",
            bg=PANEL_COLOR,
            fg=VISION_ACCENT_COLOR,
            font=FONT_SECTION,
        )

        self.title_label.pack(
            anchor="w",
        )

        self.subtitle_label = tk.Label(
            self,
            text="Pose-TCN Stroke Classification",
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

        probability_title = tk.Label(
            self,
            text="Class Probabilities",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_NORMAL,
        )

        probability_title.pack(
            anchor="w",
            pady=(0, 6),
        )

        self.probability_labels = {}

        for class_name in (
            "Forehand",
            "Backhand",
            "Serve",
            "Volley",
            "Dink",
            "Overhead",
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
                text=class_name,
                bg=PANEL_COLOR,
                fg=SECONDARY_TEXT_COLOR,
                font=FONT_SMALL,
                width=10,
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
                font=FONT_SMALL,
                anchor="e",
            )

            value_label.pack(
                side="right",
            )

            self.probability_labels[
                class_name
            ] = value_label

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

        probabilities = prediction.get(
            "probabilities",
            {},
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

        for class_name_key, label in (
            self.probability_labels.items()
        ):
            probability = probabilities.get(
                class_name_key
            )

            if probability is None:
                label.config(
                    text="--"
                )

            else:
                label.config(
                    text=(
                        f"{probability * 100:.1f}%"
                    )
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

        for label in (
            self.probability_labels.values()
        ):
            label.config(
                text="--"
            )
import tkinter as tk

import cv2
from PIL import (
    Image,
    ImageTk,
)

from ui.styles import (
    PANEL_COLOR,
    BORDER_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    FONT_SECTION,
    FONT_SMALL,
    PANEL_PADDING,
    CAMERA_BACKGROUND_COLOR,
)


POSE_CONNECTIONS = [
    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),
    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32),
]


class CameraPanel(tk.Frame):
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

        self.latest_frame = None
        self.latest_landmarks = None

        self.available_width = 640
        self.available_height = 480

        self.photo_image = None

        self._build_widgets()

    def _build_widgets(
        self,
    ):
        self.title_label = tk.Label(
            self,
            text="Live Pose",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        self.title_label.pack(
            anchor="w",
            pady=(0, 8),
        )

        self.video_frame = tk.Frame(
            self,
            bg=CAMERA_BACKGROUND_COLOR,
        )

        self.video_frame.pack(
            fill="both",
            expand=True,
        )

        self.image_label = tk.Label(
            self.video_frame,
            bg=CAMERA_BACKGROUND_COLOR,
        )

        self.image_label.pack(
            fill="both",
            expand=True,
        )

        self.image_label.bind(
            "<Configure>",
            self._on_resize,
        )

        self.status_label = tk.Label(
            self,
            text="Camera waiting...",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        self.status_label.pack(
            anchor="w",
            pady=(8, 0),
        )

    def _on_resize(
        self,
        event,
    ):
        if event.width <= 1:
            return

        if event.height <= 1:
            return

        self.available_width = (
            event.width
        )

        self.available_height = (
            event.height
        )

        self._render()

    def update_frame(
        self,
        frame,
        landmarks=None,
        pose_detected=False,
    ):
        if frame is None:
            return

        self.latest_frame = (
            frame.copy()
        )

        if landmarks is not None:
            self.latest_landmarks = list(
                landmarks
            )

        else:
            self.latest_landmarks = None

        if pose_detected:
            self.status_label.config(
                text="Pose tracking active"
            )

        else:
            self.status_label.config(
                text="Camera active — no pose detected"
            )

        self._render()

    def _render(
        self,
    ):
        if self.latest_frame is None:
            return

        frame = (
            self.latest_frame.copy()
        )

        if self.latest_landmarks:
            frame = self._draw_pose(
                frame,
                self.latest_landmarks,
            )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        available_width = max(
            1,
            self.available_width,
        )

        available_height = max(
            1,
            self.available_height,
        )

        frame_aspect = (
            frame_width
            / frame_height
        )

        area_aspect = (
            available_width
            / available_height
        )

        if frame_aspect > area_aspect:
            display_width = (
                available_width
            )

            display_height = int(
                display_width
                / frame_aspect
            )

        else:
            display_height = (
                available_height
            )

            display_width = int(
                display_height
                * frame_aspect
            )

        display_width = max(
            1,
            display_width,
        )

        display_height = max(
            1,
            display_height,
        )

        resized = cv2.resize(
            frame,
            (
                display_width,
                display_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        rgb_frame = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb_frame
        )

        self.photo_image = (
            ImageTk.PhotoImage(
                image=image
            )
        )

        self.image_label.config(
            image=self.photo_image
        )

    def _draw_pose(
        self,
        frame,
        landmarks,
    ):
        frame_height, frame_width = (
            frame.shape[:2]
        )

        points = []

        for landmark in landmarks:
            x = int(
                landmark[0]
                * frame_width
            )

            y = int(
                landmark[1]
                * frame_height
            )

            visibility = 1.0

            if len(landmark) >= 4:
                visibility = (
                    landmark[3]
                )

            points.append(
                (
                    x,
                    y,
                    visibility,
                )
            )

        for (
            start_index,
            end_index,
        ) in POSE_CONNECTIONS:

            if (
                start_index
                >= len(points)
                or end_index
                >= len(points)
            ):
                continue

            start = points[
                start_index
            ]

            end = points[
                end_index
            ]

            if (
                start[2] < 0.3
                or end[2] < 0.3
            ):
                continue

            cv2.line(
                frame,
                (
                    start[0],
                    start[1],
                ),
                (
                    end[0],
                    end[1],
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

        for (
            x,
            y,
            visibility,
        ) in points:

            if visibility < 0.3:
                continue

            cv2.circle(
                frame,
                (
                    x,
                    y,
                ),
                4,
                (
                    0,
                    255,
                    255,
                ),
                -1,
            )

        return frame

    def clear(
        self,
    ):
        self.latest_frame = None
        self.latest_landmarks = None
        self.photo_image = None

        self.image_label.config(
            image=""
        )

        self.status_label.config(
            text="Camera waiting..."
        )
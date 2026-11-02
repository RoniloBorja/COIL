import tkinter as tk

import cv2

from PIL import (
    Image,
    ImageTk,
)

from ui.styles import (
    PANEL_COLOR,
    TEXT_COLOR,
    BORDER_COLOR,
    FONT_SECTION,
    FONT_NORMAL,
    PANEL_PADDING,
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


class CameraPreview(tk.Frame):
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

        self.photo_image = None

        self.latest_frame = None
        self.latest_landmarks = None

        self._build_widgets()

    def _build_widgets(self):
        title = tk.Label(
            self,
            text="Camera Preview",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=FONT_SECTION,
        )

        title.pack(
            anchor="w",
            pady=(0, 10),
        )

        self.image_label = tk.Label(
            self,
            text="Waiting for camera...",
            bg="#111111",
            fg="white",
            font=FONT_NORMAL,
        )

        self.image_label.pack(
            fill="both",
            expand=True,
        )

        self.image_label.bind(
            "<Configure>",
            self._on_resize,
        )

    def update_frame(
        self,
        frame,
        landmarks=None,
    ):
        if frame is None:
            return

        self.latest_frame = frame.copy()

        if landmarks is not None:
            self.latest_landmarks = list(
                landmarks
            )
        else:
            self.latest_landmarks = None

        self._render()

    def _on_resize(
        self,
        event,
    ):
        self._render()

    def _render(self):
        if self.latest_frame is None:
            return

        display_frame = (
            self.latest_frame.copy()
        )

        if self.latest_landmarks:
            self._draw_pose(
                display_frame,
                self.latest_landmarks,
            )

        available_width = (
            self.image_label.winfo_width()
        )

        available_height = (
            self.image_label.winfo_height()
        )

        if (
            available_width <= 1
            or available_height <= 1
        ):
            return

        frame_height, frame_width = (
            display_frame.shape[:2]
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
            new_width = available_width

            new_height = int(
                available_width
                / frame_aspect
            )

        else:
            new_height = available_height

            new_width = int(
                available_height
                * frame_aspect
            )

        if (
            new_width <= 0
            or new_height <= 0
        ):
            return

        display_frame = cv2.resize(
            display_frame,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        rgb_frame = cv2.cvtColor(
            display_frame,
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
            image=self.photo_image,
            text="",
        )

    def _draw_pose(
        self,
        frame,
        landmarks,
    ):
        height, width, _ = (
            frame.shape
        )

        points = []

        for landmark in landmarks:
            (
                x,
                y,
                z,
                visibility,
            ) = landmark

            pixel_x = int(
                x * width
            )

            pixel_y = int(
                y * height
            )

            points.append(
                (
                    pixel_x,
                    pixel_y,
                    visibility,
                )
            )

        for (
            start_index,
            end_index,
        ) in POSE_CONNECTIONS:

            (
                start_x,
                start_y,
                start_visibility,
            ) = points[start_index]

            (
                end_x,
                end_y,
                end_visibility,
            ) = points[end_index]

            if (
                start_visibility < 0.5
                or end_visibility < 0.5
            ):
                continue

            cv2.line(
                frame,
                (
                    start_x,
                    start_y,
                ),
                (
                    end_x,
                    end_y,
                ),
                (0, 255, 0),
                3,
            )

        for (
            x,
            y,
            visibility,
        ) in points:

            if visibility < 0.5:
                continue

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 255),
                -1,
            )

    def clear(self):
        self.photo_image = None
        self.latest_frame = None
        self.latest_landmarks = None

        self.image_label.config(
            image="",
            text="Camera unavailable",
        )
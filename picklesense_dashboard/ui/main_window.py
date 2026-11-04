import threading
import time
import tkinter as tk
from tkinter import messagebox

from config.settings import (
    UI_UPDATE_INTERVAL_MS,
    STATUS_UPDATE_INTERVAL_MS,
)

from camera.camera_capture import (
    CameraCapture,
)

from camera.pose_detector import (
    PoseDetector,
)

from communication.max78000_receiver import (
    Max78000Receiver,
)

from inference.prediction_state import (
    PredictionState,
)

from processing.imu_buffer import (
    ImuBuffer,
)

from processing.imu_statistics import (
    calculate_imu_statistics,
)

from ui.camera_panel import (
    CameraPanel,
)

from ui.vision_panel import (
    VisionPanel,
)

from ui.imu_ai_panel import (
    ImuAiPanel,
)

from ui.imu_statistics_panel import (
    ImuStatisticsPanel,
)

from ui.system_status_panel import (
    SystemStatusPanel,
)

from ui.styles import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BACKGROUND_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    SECONDARY_TEXT_COLOR,
    FONT_TITLE,
    FONT_SMALL,
    PAGE_PADDING,
)


class MainWindow:
    def __init__(
        self,
    ):
        # --------------------------------------------------
        # Main Tk window
        # --------------------------------------------------

        self.root = tk.Tk()

        self.root.title(
            WINDOW_TITLE
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            1100,
            700,
        )

        self.root.configure(
            bg=BACKGROUND_COLOR
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        # --------------------------------------------------
        # Shared state
        # --------------------------------------------------

        self.prediction_state = (
            PredictionState()
        )

        self.imu_buffer = (
            ImuBuffer()
        )

        # --------------------------------------------------
        # Hardware / processing services
        # --------------------------------------------------

        self.camera = (
            CameraCapture()
        )

        self.pose_detector = (
            PoseDetector()
        )

        self.max78000_receiver = (
            Max78000Receiver(
                imu_buffer=(
                    self.imu_buffer
                ),
                on_prediction=(
                    self._on_max78000_prediction
                ),
            )
        )

        # --------------------------------------------------
        # Pose processing thread
        # --------------------------------------------------

        self.pose_thread = None
        self.pose_thread_running = False

        self.last_processed_camera_timestamp_ns = None

        self.latest_pose_result = None

        self.pose_lock = (
            threading.Lock()
        )

        # --------------------------------------------------
        # UI state
        # --------------------------------------------------

        self.last_ui_update_ns = 0
        self.last_status_update_ns = 0

        self._build_gui()

        self._start_services()

        self._schedule_updates()

    # ======================================================
    # GUI
    # ======================================================

    def _build_gui(
        self,
    ):
        # --------------------------------------------------
        # Header
        # --------------------------------------------------

        header = tk.Frame(
            self.root,
            bg=BACKGROUND_COLOR,
        )

        header.pack(
            fill="x",
            padx=PAGE_PADDING,
            pady=(
                PAGE_PADDING,
                10,
            ),
        )

        title = tk.Label(
            header,
            text="PickleSense",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
            font=FONT_TITLE,
        )

        title.pack(
            side="left"
        )

        subtitle = tk.Label(
            header,
            text=(
                "Multimodal Pickleball "
                "Stroke Recognition"
            ),
            bg=BACKGROUND_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        subtitle.pack(
            side="left",
            padx=(14, 0),
            pady=(8, 0),
        )

        # --------------------------------------------------
        # Main dashboard
        # --------------------------------------------------

        dashboard = tk.Frame(
            self.root,
            bg=BACKGROUND_COLOR,
        )

        dashboard.pack(
            fill="both",
            expand=True,
            padx=PAGE_PADDING,
            pady=(0, PAGE_PADDING),
        )

        dashboard.grid_rowconfigure(
            0,
            weight=1,
        )

        dashboard.grid_columnconfigure(
            0,
            weight=3,
        )

        dashboard.grid_columnconfigure(
            1,
            weight=2,
        )

        # --------------------------------------------------
        # Left column
        # --------------------------------------------------

        left_column = tk.Frame(
            dashboard,
            bg=BACKGROUND_COLOR,
        )

        left_column.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        left_column.grid_rowconfigure(
            0,
            weight=1,
        )

        left_column.grid_rowconfigure(
            1,
            weight=0,
        )

        left_column.grid_columnconfigure(
            0,
            weight=1,
        )

        self.camera_panel = (
            CameraPanel(
                left_column
            )
        )

        self.camera_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.system_status_panel = (
            SystemStatusPanel(
                left_column
            )
        )

        self.system_status_panel.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        # --------------------------------------------------
        # Right column
        # --------------------------------------------------

        right_column = tk.Frame(
            dashboard,
            bg=BACKGROUND_COLOR,
        )

        right_column.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        right_column.grid_columnconfigure(
            0,
            weight=1,
        )

        right_column.grid_rowconfigure(
            0,
            weight=0,
        )

        right_column.grid_rowconfigure(
            1,
            weight=0,
        )

        right_column.grid_rowconfigure(
            2,
            weight=1,
        )

        self.vision_panel = (
            VisionPanel(
                right_column
            )
        )

        self.vision_panel.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        self.imu_ai_panel = (
            ImuAiPanel(
                right_column
            )
        )

        self.imu_ai_panel.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        self.imu_statistics_panel = (
            ImuStatisticsPanel(
                right_column
            )
        )

        self.imu_statistics_panel.grid(
            row=2,
            column=0,
            sticky="nsew",
            pady=(8, 0),
        )

        # --------------------------------------------------
        # Footer
        # --------------------------------------------------

        footer = tk.Frame(
            self.root,
            bg=PANEL_COLOR,
        )

        footer.pack(
            fill="x",
        )

        self.footer_status = tk.Label(
            footer,
            text="Starting PickleSense...",
            bg=PANEL_COLOR,
            fg=SECONDARY_TEXT_COLOR,
            font=FONT_SMALL,
        )

        self.footer_status.pack(
            side="left",
            padx=PAGE_PADDING,
            pady=5,
        )

    # ======================================================
    # Startup
    # ======================================================

    def _start_services(
        self,
    ):
        camera_ok = False
        pose_ok = False
        max78000_ok = False

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        try:
            self.camera.start()

            camera_ok = True

        except Exception as error:
            print(
                "Camera start error:",
                error,
            )

        # --------------------------------------------------
        # MediaPipe Pose
        # --------------------------------------------------

        try:
            self.pose_detector.start()

            pose_ok = True

        except Exception as error:
            print(
                "Pose detector start error:",
                error,
            )

        # --------------------------------------------------
        # Pose processing worker
        # --------------------------------------------------

        if camera_ok and pose_ok:
            self.pose_thread_running = True

            self.pose_thread = (
                threading.Thread(
                    target=self._pose_loop,
                    daemon=True,
                )
            )

            self.pose_thread.start()

        # --------------------------------------------------
        # MAX78000
        # --------------------------------------------------

        try:
            self.max78000_receiver.start()

            max78000_ok = True

        except Exception as error:
            print(
                "MAX78000 connection error:",
                error,
            )

        # --------------------------------------------------
        # Footer
        # --------------------------------------------------

        states = []

        if camera_ok:
            states.append(
                "Camera ready"
            )

        else:
            states.append(
                "Camera unavailable"
            )

        if pose_ok:
            states.append(
                "Pose ready"
            )

        else:
            states.append(
                "Pose unavailable"
            )

        if max78000_ok:
            states.append(
                "MAX78000 connected"
            )

        else:
            states.append(
                "MAX78000 offline"
            )

        self.footer_status.config(
            text=" | ".join(
                states
            )
        )

    # ======================================================
    # Pose worker
    # ======================================================

    def _pose_loop(
        self,
    ):
        while self.pose_thread_running:
            frame_data = (
                self.camera.get_latest_frame()
            )

            if frame_data is None:
                time.sleep(
                    0.005
                )
                continue

            frame = frame_data[
                "frame"
            ]

            timestamp_ns = frame_data[
                "timestamp_ns"
            ]

            # --------------------------------------------------
            # Do not process the same camera frame twice.
            #
            # This is also important for MediaPipe VIDEO mode,
            # which expects increasing timestamps.
            # --------------------------------------------------

            if (
                timestamp_ns
                == self.last_processed_camera_timestamp_ns
            ):
                time.sleep(
                    0.002
                )
                continue

            self.last_processed_camera_timestamp_ns = (
                timestamp_ns
            )

            try:
                pose_result = (
                    self.pose_detector.process(
                        frame=frame,
                        timestamp_ns=timestamp_ns,
                    )
                )

            except Exception as error:
                print(
                    "Pose processing error:",
                    error,
                )

                time.sleep(
                    0.005
                )

                continue

            with self.pose_lock:
                self.latest_pose_result = (
                    pose_result
                )

    # ======================================================
    # MAX78000 callback
    # ======================================================

    def _on_max78000_prediction(
        self,
        prediction,
    ):
        self.prediction_state.update_max78000(
            class_name=(
                prediction.class_name
            ),
            confidence=(
                prediction.confidence
            ),
        )

    # ======================================================
    # UI update scheduling
    # ======================================================

    def _schedule_updates(
        self,
    ):
        self.root.after(
            UI_UPDATE_INTERVAL_MS,
            self._update_ui,
        )

    def _update_ui(
        self,
    ):
        try:
            self._update_camera_panel()

            self._update_predictions()

            now_ns = (
                time.monotonic_ns()
            )

            status_interval_ns = (
                STATUS_UPDATE_INTERVAL_MS
                * 1_000_000
            )

            if (
                now_ns
                - self.last_status_update_ns
                >= status_interval_ns
            ):
                self.last_status_update_ns = (
                    now_ns
                )

                self._update_statistics()

                self._update_system_status()

        except Exception as error:
            print(
                "Dashboard update error:",
                error,
            )

        finally:
            self.root.after(
                UI_UPDATE_INTERVAL_MS,
                self._update_ui,
            )

    # ======================================================
    # Camera UI
    # ======================================================

    def _update_camera_panel(
        self,
    ):
        frame_data = (
            self.camera.get_latest_frame()
        )

        if frame_data is None:
            return

        frame = frame_data[
            "frame"
        ]

        pose_result = None

        with self.pose_lock:
            if self.latest_pose_result is not None:
                pose_result = dict(
                    self.latest_pose_result
                )

        if pose_result is None:
            self.camera_panel.update_frame(
                frame=frame,
                landmarks=None,
                pose_detected=False,
            )

            return

        self.camera_panel.update_frame(
            frame=frame,
            landmarks=(
                pose_result.get(
                    "landmarks"
                )
            ),
            pose_detected=(
                pose_result.get(
                    "pose_detected",
                    False,
                )
            ),
        )

    # ======================================================
    # Prediction UI
    # ======================================================

    def _update_predictions(
        self,
    ):
        prediction_state = (
            self.prediction_state.get_all()
        )

        vision_prediction = (
            prediction_state.get(
                "vision"
            )
        )

        max78000_prediction = (
            prediction_state.get(
                "max78000"
            )
        )

        self.vision_panel.update_prediction(
            vision_prediction
        )

        self.imu_ai_panel.update_prediction(
            max78000_prediction
        )

        max_status = (
            self.max78000_receiver.get_status()
        )

        self.imu_ai_panel.set_connection_status(
            connected=(
                max_status.get(
                    "connected",
                    False,
                )
            ),
            receiving=(
                max_status.get(
                    "receiving",
                    False,
                )
            ),
        )

    # ======================================================
    # IMU statistics
    # ======================================================

    def _update_statistics(
        self,
    ):
        samples = (
            self.imu_buffer.get_samples()
        )

        statistics = (
            calculate_imu_statistics(
                samples
            )
        )

        self.imu_statistics_panel.update_statistics(
            statistics
        )

    # ======================================================
    # System status
    # ======================================================

    def _update_system_status(
        self,
    ):
        camera_status = (
            self.camera.get_status()
        )

        pose_status = (
            self.pose_detector.get_status()
        )

        max78000_status = (
            self.max78000_receiver.get_status()
        )

        samples = (
            self.imu_buffer.get_samples()
        )

        imu_statistics = (
            calculate_imu_statistics(
                samples
            )
        )

        self.system_status_panel.update_camera_status(
            camera_status
        )

        self.system_status_panel.update_pose_status(
            pose_status
        )

        self.system_status_panel.update_max78000_status(
            max78000_status
        )

        self.system_status_panel.update_imu_status(
            imu_statistics
        )

    # ======================================================
    # Shutdown
    # ======================================================

    def _on_close(
        self,
    ):
        self.pose_thread_running = False

        try:
            self.max78000_receiver.stop()

        except Exception:
            pass

        try:
            self.camera.stop()

        except Exception:
            pass

        try:
            self.pose_detector.stop()

        except Exception:
            pass

        if (
            self.pose_thread is not None
            and self.pose_thread.is_alive()
        ):
            self.pose_thread.join(
                timeout=1.0
            )

        try:
            self.camera_panel.clear()

        except Exception:
            pass

        self.root.destroy()

    # ======================================================
    # Run
    # ======================================================

    def run(
        self,
    ):
        self.root.mainloop()
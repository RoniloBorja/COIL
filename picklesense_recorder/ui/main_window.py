import threading
import tkinter as tk
from tkinter import messagebox

from communication.udp_receiver import UdpReceiver
from pose.camera_capture import (
    CameraCapture,
    scan_available_cameras,
)
from pose.pose_detector import PoseDetector
from recording.recording_manager import RecordingManager

from ui.recording_page import RecordingPage
from ui.status_panel import StatusPanel
from recording.session_manager import SessionManager
from ui.styles import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BACKGROUND_COLOR,
    TEXT_COLOR,
    FONT_TITLE,
    PAGE_PADDING,
)
from ui.camera_preview import CameraPreview

from ui.styles import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BACKGROUND_COLOR,
    TEXT_COLOR,
    FONT_TITLE,
    PAGE_PADDING,
)
from tkinter import ttk


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()

        self.root.title(
            WINDOW_TITLE
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.configure(
            bg=BACKGROUND_COLOR
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        # --------------------------------------------------
        # Core system objects
        # --------------------------------------------------

        self.recording_manager = (
            RecordingManager()
        )

        self.udp_receiver = (
            UdpReceiver(
                on_sample=(
                    self._handle_imu_sample
                )
            )
        )

        self.camera = CameraCapture()

        self.pose_detector = (
            PoseDetector()
        )

        # --------------------------------------------------
        # Camera / pose thread
        # --------------------------------------------------

        self.camera_thread = None

        self.camera_thread_running = False

        self.latest_pose_detected = False
        self.latest_preview_frame = None
        self.latest_preview_landmarks = None

        self.preview_lock = threading.Lock()

        # --------------------------------------------------
        # Build GUI
        # --------------------------------------------------

        self._build_gui()

        # --------------------------------------------------
        # Start hardware/data sources
        # --------------------------------------------------

        self._start_services()

        # --------------------------------------------------
        # Begin GUI status updates
        # --------------------------------------------------

        self._schedule_status_update()

    def _build_gui(self):
        title = tk.Label(
            self.root,
            text="PickleSense Recorder",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
            font=FONT_TITLE,
        )

        title.pack(
            anchor="w",
            padx=PAGE_PADDING,
            pady=(
                PAGE_PADDING,
                12,
            ),
        )

        # Main dashboard area
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

        dashboard.grid_columnconfigure(
            0,
            weight=1,
        )

        dashboard.grid_columnconfigure(
            1,
            weight=1,
        )

        dashboard.grid_rowconfigure(
            0,
            weight=1,
        )

        # --------------------------------------------------
        # LEFT SIDE - Camera
        # --------------------------------------------------

        self.camera_preview = CameraPreview(
            dashboard
        )

        self.camera_preview.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12),
        )
        camera_controls = tk.Frame(
            dashboard,
            bg=BACKGROUND_COLOR,
        )

        camera_controls.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 12),
            pady=(8, 0),
        )

        camera_label = tk.Label(
            camera_controls,
            text="Camera source:",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
        )

        camera_label.pack(
            side="left",
        )

        self.camera_source_var = tk.StringVar()

        self.camera_dropdown = ttk.Combobox(
            camera_controls,
            textvariable=self.camera_source_var,
            state="readonly",
            width=18,
        )

        self.camera_dropdown.pack(
            side="left",
            padx=(8, 8),
        )

        self.refresh_camera_button = tk.Button(
            camera_controls,
            text="Refresh Cameras",
            command=self._refresh_camera_list,
        )

        self.refresh_camera_button.pack(
            side="left",
            padx=(0, 8),
        )

        self.camera_source_button = tk.Button(
            camera_controls,
            text="Use Camera",
            command=self._change_camera_source,
        )

        self.camera_source_button.pack(
            side="left",
        )

        # --------------------------------------------------
        # RIGHT SIDE
        # --------------------------------------------------

        right_panel = tk.Frame(
            dashboard,
            bg=BACKGROUND_COLOR,
        )

        right_panel.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        self.status_panel = StatusPanel(
            right_panel
        )

        self.status_panel.pack(
            fill="x",
            pady=(0, 12),
        )

        self.recording_page = RecordingPage(
            right_panel,
            on_start_recording=(
                self._start_recording
            ),
            on_stop_recording=(
                self._stop_recording
            ),
        )

        self.recording_page.pack(
            fill="x",
        )
        self._refresh_camera_list()

    def _start_services(self):
        # --------------------------------------------------
        # UDP receiver
        # --------------------------------------------------

        try:
            self.udp_receiver.start()

        except Exception as error:
            messagebox.showerror(
                "UDP Error",
                (
                    "Could not start UDP receiver:\n\n"
                    f"{error}"
                ),
            )

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        try:
            self.camera.start()

        except Exception as error:
            messagebox.showerror(
                "Camera Error",
                (
                    "Could not start camera:\n\n"
                    f"{error}"
                ),
            )

            return

        # --------------------------------------------------
        # Pose detector
        # --------------------------------------------------

        try:
            self.pose_detector.start()

        except Exception as error:
            messagebox.showerror(
                "Pose Detector Error",
                (
                    "Could not start pose detector:\n\n"
                    f"{error}"
                ),
            )

            self.camera.stop()
            return

        # --------------------------------------------------
        # Start camera/pose worker
        # --------------------------------------------------

        self.camera_thread_running = True

        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
        )

        self.camera_thread.start()

    def _camera_loop(self):
        while self.camera_thread_running:
            camera_result = (
                self.camera.read()
            )

            if camera_result is None:
                continue

            frame = camera_result["frame"]

            frame_number = (
                camera_result[
                    "frame_number"
                ]
            )

            laptop_timestamp_ns = (
                camera_result[
                    "laptop_timestamp_ns"
                ]
            )

            try:
                pose_sample = (
                    self.pose_detector.process(
                        frame=frame,
                        frame_number=(
                            frame_number
                        ),
                        laptop_timestamp_ns=(
                            laptop_timestamp_ns
                        ),
                    )
                )

            except Exception:
                continue

            self.latest_pose_detected = (
                pose_sample.pose_detected
            )
            with self.preview_lock:
                self.latest_preview_frame = (
                    frame.copy()
                )

                if pose_sample.landmarks is not None:
                    self.latest_preview_landmarks = (
                        list(pose_sample.landmarks)
                    )
                else:
                    self.latest_preview_landmarks = None
            self.recording_manager.handle_pose_sample(
                pose_sample
            )

    def _handle_imu_sample(
        self,
        sample,
    ):
        self.recording_manager.handle_imu_sample(
            sample
        )

    def _start_recording(
        self,
        metadata,
    ):
        movement_name = (
            metadata.get(
                "movement_name",
                "",
            )
        )

        trial_name = (
            metadata.get(
                "trial_name",
                "",
            )
        )

        session_manager = SessionManager(
            movement_name=movement_name,
            trial_name=trial_name,
        )

        session_folder = (
            session_manager.session_folder
        )

        if session_folder.exists():
            confirm = messagebox.askyesno(
                "Overwrite Existing Trial?",
                (
                    "This recording folder already exists:\n\n"
                    f"{session_folder}\n\n"
                    "Starting a new recording will overwrite "
                    "the existing CSV and session files.\n\n"
                    "Do you want to continue?"
                ),
            )

            if not confirm:
                return False

        try:
            result = (
                self.recording_manager.start_recording(
                    metadata=metadata
                )
            )

        except Exception as error:
            messagebox.showerror(
                "Recording Error",
                (
                    "Could not start recording:\n\n"
                    f"{error}"
                ),
            )

            return False

        print()
        print("Recording started")
        print("-----------------")

        print(
            f"Participant: "
            f"{metadata.get('participant_id')}"
        )

        print(
            f"Session: "
            f"{metadata.get('session_number')}"
        )

        print(
            f"Stroke: "
            f"{metadata.get('stroke_type')}"
        )

        print(
            f"Trial: "
            f"{metadata.get('trial_number')}"
        )

        print(
            f"Session ID: "
            f"{result['session_id']}"
        )

        print(
            f"Save folder: "
            f"{result['session_folder']}"
        )

        return True

    def _stop_recording(self):
        try:
            result = (
                self.recording_manager.stop_recording()
            )

        except Exception as error:
            messagebox.showerror(
                "Recording Error",
                (
                    "Could not stop recording:\n\n"
                    f"{error}"
                ),
            )

            return False

        if result is None:
            return False

        print()
        print("Recording stopped")
        print("-----------------")
        print(
            f"IMU samples: "
            f"{result['imu_samples']}"
        )
        print(
            f"Pose frames: "
            f"{result['pose_frames']}"
        )
        print(
            f"Save folder: "
            f"{result['session_folder']}"
        )

        return True

    def _schedule_status_update(self):
        self._update_status()

        self.root.after(
            250,
            self._schedule_status_update,
        )

    def _update_status(self):
        # --------------------------------------------------
        # UDP / IMU
        # --------------------------------------------------

        imu_status = (
            self.udp_receiver.get_status()
        )

        self.status_panel.update_imu_status(
            imu_status
        )

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        camera_status = (
            self.camera.get_status()
        )

        pose_status = (
            self.pose_detector.get_status()
        )

        self.status_panel.update_camera_status(
            camera_status,
            pose_status,
        )
        with self.preview_lock:
            if self.latest_preview_frame is not None:
                preview_frame = (
                    self.latest_preview_frame.copy()
                )

                preview_landmarks = (
                    self.latest_preview_landmarks
                )

            else:
                preview_frame = None
                preview_landmarks = None


        if preview_frame is not None:
            self.camera_preview.update_frame(
                preview_frame,
                preview_landmarks,
            )
        # Override pose label with current frame state
        if self.latest_pose_detected:
            self.status_panel.pose_state_value.config(
                text="DETECTED",
            )
        else:
            self.status_panel.pose_state_value.config(
                text="NOT DETECTED",
            )

        # --------------------------------------------------
        # Recording
        # --------------------------------------------------

        recording_status = (
            self.recording_manager.get_status()
        )

        self.recording_page.update_recording_status(
            recording_status
        )

    def _on_close(self):
        # --------------------------------------------------
        # Stop recording if active
        # --------------------------------------------------

        if self.recording_manager.recording:
            try:
                self.recording_manager.stop_recording()
            except Exception:
                pass

        # --------------------------------------------------
        # Stop camera thread
        # --------------------------------------------------

        self.camera_thread_running = False

        # --------------------------------------------------
        # Stop services
        # --------------------------------------------------

        try:
            self.udp_receiver.stop()
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

        # --------------------------------------------------
        # Wait briefly for camera worker
        # --------------------------------------------------

        if (
            self.camera_thread is not None
            and self.camera_thread.is_alive()
        ):
            self.camera_thread.join(
                timeout=1.0
            )
        self.camera_preview.clear()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    def _change_camera_source(self):
        if self.recording_manager.recording:
            messagebox.showwarning(
                "Recording Active",
                (
                    "Stop the current recording before "
                    "changing the camera source."
                ),
            )
            return

        selected = (
            self.camera_source_var.get()
        )

        if not selected.startswith(
            "Camera "
        ):
            messagebox.showerror(
                "Camera Error",
                "Please select a valid camera.",
            )
            return

        try:
            new_index = int(
                selected.split()[-1]
            )

        except ValueError:
            messagebox.showerror(
                "Camera Error",
                "Invalid camera selection.",
            )
            return

        if new_index == self.camera.camera_index:
            return

        self.camera_thread_running = False

        if (
            self.camera_thread is not None
            and self.camera_thread.is_alive()
        ):
            self.camera_thread.join(
                timeout=1.0
            )

        try:
            self.camera.stop()

            self.camera.set_camera_index(
                new_index
            )

            self.camera.start()

        except Exception as error:
            messagebox.showerror(
                "Camera Error",
                (
                    f"Could not open "
                    f"Camera {new_index}:\n\n"
                    f"{error}"
                ),
            )
            return

        self.camera_thread_running = True

        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
        )

        self.camera_thread.start()

    def _refresh_camera_list(self):
        if self.recording_manager.recording:
            messagebox.showwarning(
                "Recording Active",
                (
                    "Stop the current recording before "
                    "scanning camera sources."
                ),
            )
            return

        available = scan_available_cameras(
            max_index=10
        )

        camera_names = [
            f"Camera {index}"
            for index in available
        ]

        self.camera_dropdown[
            "values"
        ] = camera_names

        if not available:
            self.camera_source_var.set(
                "No cameras found"
            )
            return

        current_name = (
            f"Camera {self.camera.camera_index}"
        )

        if current_name in camera_names:
            self.camera_source_var.set(
                current_name
            )
        else:
            self.camera_source_var.set(
                camera_names[0]
            )
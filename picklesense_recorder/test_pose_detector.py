import cv2

from pose.camera_capture import CameraCapture
from pose.pose_detector import PoseDetector


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


def draw_pose_landmarks(
    frame,
    landmarks,
):
    if not landmarks:
        return

    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x, y, z, visibility = landmark

        pixel_x = int(x * width)
        pixel_y = int(y * height)

        points.append(
            (
                pixel_x,
                pixel_y,
                visibility,
            )
        )

    for start_index, end_index in POSE_CONNECTIONS:
        start_x, start_y, start_visibility = (
            points[start_index]
        )

        end_x, end_y, end_visibility = (
            points[end_index]
        )

        if (
            start_visibility < 0.5
            or end_visibility < 0.5
        ):
            continue

        cv2.line(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (255, 255, 255),
            2,
        )

    for x, y, visibility in points:
        if visibility < 0.5:
            continue

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1,
        )

def main():
    camera = CameraCapture()
    pose_detector = PoseDetector()

    print("PickleSense pose detector test")
    print("------------------------------")

    try:
        camera.start()
        pose_detector.start()

        print("Camera started.")
        print("Pose detector started.")
        print("Press Q to stop.\n")

        while True:
            camera_result = camera.read()

            if camera_result is None:
                continue

            frame = camera_result["frame"]

            frame_number = (
                camera_result["frame_number"]
            )

            timestamp_ns = (
                camera_result[
                    "laptop_timestamp_ns"
                ]
            )

            pose_sample = pose_detector.process(
                frame=frame,
                frame_number=frame_number,
                laptop_timestamp_ns=timestamp_ns,
            )

            camera_status = (
                camera.get_status()
            )

            pose_status = (
                pose_detector.get_status()
            )

            if pose_sample.pose_detected:
                draw_pose_landmarks(
                    frame,
                    pose_sample.landmarks,
                )

                pose_text = "POSE: DETECTED"

            else:
                pose_text = "POSE: NOT DETECTED"

            cv2.putText(
                frame,
                pose_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Camera FPS: "
                    f"{camera_status['current_fps']:.2f}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Frames: "
                    f"{pose_status['frames_processed']}"
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"With pose: "
                    f"{pose_status['frames_with_pose']}"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Without pose: "
                    f"{pose_status['frames_without_pose']}"
                ),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "PickleSense Pose Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except Exception as error:
        print()
        print(f"Error: {error}")

    finally:
        camera.stop()
        pose_detector.stop()

        cv2.destroyAllWindows()

        camera_status = camera.get_status()
        pose_status = pose_detector.get_status()

        print()
        print("Test stopped.")
        print(
            f"Camera frames: "
            f"{camera_status['frames_received']}"
        )
        print(
            f"Pose frames processed: "
            f"{pose_status['frames_processed']}"
        )
        print(
            f"Frames with pose: "
            f"{pose_status['frames_with_pose']}"
        )
        print(
            f"Frames without pose: "
            f"{pose_status['frames_without_pose']}"
        )


if __name__ == "__main__":
    main()
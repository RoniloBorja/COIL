import cv2

from pose.camera_capture import CameraCapture


def main():
    camera = CameraCapture()

    print("PickleSense camera capture test")
    print("-------------------------------")

    try:
        camera.start()

        print("Camera started.")
        print("Press Q to stop.\n")

        while True:
            result = camera.read()

            if result is None:
                continue

            frame = result["frame"]
            frame_number = result["frame_number"]
            timestamp_ns = result["laptop_timestamp_ns"]

            status = camera.get_status()

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {status['current_fps']:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Timestamp ns: {timestamp_ns}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "PickleSense Camera Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except RuntimeError as error:
        print()
        print(f"Camera error: {error}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        status = camera.get_status()

        print()
        print("Camera stopped.")
        print(
            f"Frames received: "
            f"{status['frames_received']}"
        )
        print(
            f"Final FPS: "
            f"{status['current_fps']:.2f}"
        )


if __name__ == "__main__":
    main()
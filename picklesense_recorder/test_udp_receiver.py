import time

from communication.udp_receiver import UdpReceiver


def handle_sample(sample):
    print(
        f"seq={sample.sequence:6d} | "
        f"esp_us={sample.esp_timestamp_us:10d} | "
        f"A=({sample.ax_raw:6d},"
        f"{sample.ay_raw:6d},"
        f"{sample.az_raw:6d}) | "
        f"G=({sample.gx_raw:6d},"
        f"{sample.gy_raw:6d},"
        f"{sample.gz_raw:6d})"
    )


def main():
    receiver = UdpReceiver(
        on_sample=handle_sample
    )

    print("PickleSense permanent UDP receiver test")
    print("----------------------------------------")

    receiver.start()

    print("Receiver started.")
    print("Press Ctrl+C to stop.\n")

    last_status_time = time.monotonic()

    try:
        while True:
            time.sleep(0.1)

            now = time.monotonic()

            if now - last_status_time >= 1.0:
                status = receiver.get_status()

                print()
                print("----- UDP STATUS -----")
                print(
                    f"Packets received : "
                    f"{status['total_packets']}"
                )
                print(
                    f"Current rate     : "
                    f"{status['current_rate_hz']:.2f} Hz"
                )
                print(
                    f"Average rate     : "
                    f"{status['average_rate_hz']:.2f} Hz"
                )
                print(
                    f"Missing packets  : "
                    f"{status['missing_packets']}"
                )
                print(
                    f"Malformed packets: "
                    f"{status['malformed_packets']}"
                )

                age = status[
                    "last_packet_age_seconds"
                ]

                if age is None:
                    print(
                        "Last packet age  : N/A"
                    )
                else:
                    print(
                        f"Last packet age  : "
                        f"{age:.3f} s"
                    )

                print("----------------------")
                print()

                last_status_time = now

    except KeyboardInterrupt:
        print("\nStopping receiver...")

    finally:
        receiver.stop()

        status = receiver.get_status()

        print()
        print("Final results")
        print("-------------")
        print(
            f"Packets received : "
            f"{status['total_packets']}"
        )
        print(
            f"Missing packets  : "
            f"{status['missing_packets']}"
        )
        print(
            f"Malformed packets: "
            f"{status['malformed_packets']}"
        )


if __name__ == "__main__":
    main()
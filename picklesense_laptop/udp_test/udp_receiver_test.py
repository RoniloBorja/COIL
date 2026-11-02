import socket
import time

UDP_IP = "192.168.1.10"
UDP_PORT = 5005

EXPECTED_FIELDS = 8
EXPECTED_RATE_HZ = 100


def parse_packet(message: str):
    parts = message.strip().split(",")

    if len(parts) != EXPECTED_FIELDS:
        raise ValueError(
            f"Expected {EXPECTED_FIELDS} fields, got {len(parts)}"
        )

    sequence = int(parts[0])
    timestamp_us = int(parts[1])

    ax = int(parts[2])
    ay = int(parts[3])
    az = int(parts[4])

    gx = int(parts[5])
    gy = int(parts[6])
    gz = int(parts[7])

    return {
        "sequence": sequence,
        "timestamp_us": timestamp_us,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.bind((UDP_IP, UDP_PORT))

    print("PickleSense UDP Test Receiver")
    print("-----------------------------")
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print("Waiting for ESP32 packets...")
    print("Press Ctrl+C to stop.\n")

    total_packets = 0
    malformed_packets = 0
    missing_packets = 0

    last_sequence = None

    rate_window_start = time.monotonic()
    packets_in_window = 0

    first_packet_time = None
    last_packet_time = None

    try:
        while True:
            data, sender = sock.recvfrom(1024)

            receive_time = time.monotonic_ns()

            try:
                message = data.decode("utf-8")
                sample = parse_packet(message)

            except (UnicodeDecodeError, ValueError) as error:
                malformed_packets += 1

                print(
                    f"[INVALID] From {sender[0]}:{sender[1]} "
                    f"| {error}"
                )

                continue

            total_packets += 1
            packets_in_window += 1

            now = time.monotonic()

            if first_packet_time is None:
                first_packet_time = now

            last_packet_time = now

            sequence = sample["sequence"]

            if last_sequence is not None:
                expected_sequence = last_sequence + 1

                if sequence > expected_sequence:
                    missed = sequence - expected_sequence
                    missing_packets += missed

                    print(
                        f"[LOSS] Expected sequence "
                        f"{expected_sequence}, received {sequence} "
                        f"-> missing {missed}"
                    )

                elif sequence <= last_sequence:
                    print(
                        f"[WARN] Sequence reset/out-of-order: "
                        f"{last_sequence} -> {sequence}"
                    )

            last_sequence = sequence

            print(
                f"seq={sample['sequence']:6d} | "
                f"esp_us={sample['timestamp_us']:10d} | "
                f"A=({sample['ax']:6d},"
                f"{sample['ay']:6d},"
                f"{sample['az']:6d}) | "
                f"G=({sample['gx']:6d},"
                f"{sample['gy']:6d},"
                f"{sample['gz']:6d}) | "
                f"from={sender[0]}"
            )

            elapsed_window = now - rate_window_start

            if elapsed_window >= 1.0:
                receive_rate = packets_in_window / elapsed_window

                total_elapsed = (
                    now - first_packet_time
                    if first_packet_time is not None
                    else 0.0
                )

                average_rate = (
                    total_packets / total_elapsed
                    if total_elapsed > 0
                    else 0.0
                )

                print()
                print("----- UDP STATUS -----")
                print(f"Packets received : {total_packets}")
                print(f"Current rate     : {receive_rate:.2f} Hz")
                print(f"Average rate     : {average_rate:.2f} Hz")
                print(f"Expected rate    : {EXPECTED_RATE_HZ} Hz")
                print(f"Missing packets  : {missing_packets}")
                print(f"Malformed packets: {malformed_packets}")
                print("----------------------")
                print()

                packets_in_window = 0
                rate_window_start = now

    except KeyboardInterrupt:
        print("\nReceiver stopped.")

    finally:
        sock.close()

        print()
        print("Final results")
        print("-------------")
        print(f"Packets received : {total_packets}")
        print(f"Missing packets  : {missing_packets}")
        print(f"Malformed packets: {malformed_packets}")


if __name__ == "__main__":
    main()
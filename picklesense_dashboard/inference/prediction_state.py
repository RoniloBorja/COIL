import threading
import time


class PredictionState:
    def __init__(self):
        self._lock = threading.Lock()

        self.vision_prediction = None
        self.max78000_prediction = None

    def update_vision(
        self,
        class_name,
        confidence,
        probabilities=None,
    ):
        with self._lock:
            self.vision_prediction = {
                "class_name": class_name,
                "confidence": float(
                    confidence
                ),
                "probabilities": (
                    probabilities
                    if probabilities is not None
                    else {}
                ),
                "timestamp_ns": (
                    time.monotonic_ns()
                ),
            }

    def update_max78000(
        self,
        class_name,
        confidence,
        probabilities=None,
    ):
        with self._lock:
            self.max78000_prediction = {
                "class_name": class_name,
                "confidence": float(
                    confidence
                ),
                "probabilities": (
                    probabilities
                    if probabilities is not None
                    else {}
                ),
                "timestamp_ns": (
                    time.monotonic_ns()
                ),
            }

    def clear_vision(
        self,
    ):
        with self._lock:
            self.vision_prediction = None

    def clear_max78000(
        self,
    ):
        with self._lock:
            self.max78000_prediction = None

    def clear_all(
        self,
    ):
        with self._lock:
            self.vision_prediction = None
            self.max78000_prediction = None

    def get_vision(
        self,
    ):
        with self._lock:
            if self.vision_prediction is None:
                return None

            return dict(
                self.vision_prediction
            )

    def get_max78000(
        self,
    ):
        with self._lock:
            if self.max78000_prediction is None:
                return None

            return dict(
                self.max78000_prediction
            )

    def get_all(
        self,
    ):
        with self._lock:
            vision = None
            max78000 = None

            if self.vision_prediction is not None:
                vision = dict(
                    self.vision_prediction
                )

            if self.max78000_prediction is not None:
                max78000 = dict(
                    self.max78000_prediction
                )

            return {
                "vision": vision,
                "max78000": max78000,
            }

    def get_status(
        self,
    ):
        with self._lock:
            now_ns = (
                time.monotonic_ns()
            )

            vision_age_seconds = None
            max78000_age_seconds = None

            if self.vision_prediction is not None:
                vision_age_seconds = (
                    now_ns
                    - self.vision_prediction[
                        "timestamp_ns"
                    ]
                ) / 1_000_000_000

            if (
                self.max78000_prediction
                is not None
            ):
                max78000_age_seconds = (
                    now_ns
                    - self.max78000_prediction[
                        "timestamp_ns"
                    ]
                ) / 1_000_000_000

            return {
                "has_vision_prediction": (
                    self.vision_prediction
                    is not None
                ),

                "has_max78000_prediction": (
                    self.max78000_prediction
                    is not None
                ),

                "vision_age_seconds": (
                    vision_age_seconds
                ),

                "max78000_age_seconds": (
                    max78000_age_seconds
                ),
            }
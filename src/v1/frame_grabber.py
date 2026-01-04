from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class FrameGrabber:
    """
    Grabs frames continuously from your Camera and stores ONLY the latest frame.
    - No queue growth
    - Inference always runs on the freshest frame (drops old frames automatically)
    """

    def __init__(self, cam, target_fps: float = 30.0, copy_frame: bool = False) -> None:
        self.cam = cam
        self.target_fps = float(target_fps)
        self.copy_frame = bool(copy_frame)

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # stats
        self.frames_grabbed: int = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _loop(self) -> None:
        min_dt = 1.0 / max(self.target_fps, 0.1)
        next_t = time.time()

        while self._running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Optional copy if camera backend reuses buffers
            if self.copy_frame:
                frame = frame.copy()

            with self._lock:
                self._latest = frame
                self._latest_ts = time.time()
                self.frames_grabbed += 1

            # pacing
            next_t += min_dt
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        """
        Returns (latest_frame_rgb, timestamp)
        """
        with self._lock:
            return self._latest, self._latest_ts

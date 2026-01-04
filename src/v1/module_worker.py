import threading
import time
from typing import Any, Dict, Optional
import numpy as np

class ModuleWorker:
    def __init__(self, module, state: Dict[str, Any], lock: threading.Lock):
        self.module = module
        self.state = state
        self.lock = lock

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_infer_ms: float = 0.0
        self.infer_times = []
        self.infer_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def update_frame(self, frame: np.ndarray, ts: float):
        self._latest_frame = frame
        self._latest_ts = ts

    def _loop(self):
        last_ts = 0.0
        while self._running:
            frame = self._latest_frame
            ts = self._latest_ts

            # wait until we have a real frame and a new timestamp
            if frame is None or ts == 0.0 or ts == last_ts:
                time.sleep(0.005)
                continue

            last_ts = ts

            # run inference
            t0 = time.time()
            with self.lock:
                self.module.process(frame, self.state)
            dt = time.time() - t0

            # timings for FPS
            self.infer_times.append(dt)
            if len(self.infer_times) > 30:
                self.infer_times.pop(0)


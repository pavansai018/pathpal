from typing import Tuple, Optional
import time
import numpy as np
# -----------------------------
# Camera (auto backend)
# -----------------------------
class Camera:
    """
    One codebase:
    - Pi: uses Picamera2 if available
    - Laptop: falls back to OpenCV
    Returns RGB uint8 frames (H,W,3)
    """
    def __init__(self, size: Tuple[int, int] = (320, 240), fps: int = 15) -> None:
        self.size = size
        self.fps = fps
        self.backend = "unknown"

        # Try PiCamera2
        try:
            from picamera2 import Picamera2  # type: ignore
            self.backend = "picamera2"
            self._cam = Picamera2()
            cfg = self._cam.create_preview_configuration(
                main={"size": size, "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            self._cam.configure(cfg)
            self._cam.start()
            time.sleep(0.2)
            return
        except Exception:
            pass

        # Fallback to OpenCV
        import cv2  # type: ignore
        self.backend = "opencv"
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera via picamera2 or opencv.")

    def read(self) -> Optional[np.ndarray]:
        if self.backend == "picamera2":
            return self._cam.capture_array("main")
        ret, bgr = self._cap.read()
        if not ret:
            return None
        return bgr[:, :, ::-1].copy()  # BGR->RGB

    def close(self) -> None:
        if self.backend == "picamera2":
            self._cam.close()
        else:
            self._cap.release()
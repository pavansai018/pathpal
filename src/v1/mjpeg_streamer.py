from __future__ import annotations
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional, Tuple
import cv2
import variables
import numpy as np

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class MjpegStreamer:
    """
    Minimal MJPEG streamer
    - update_rgb(frame_rgb) stores latest frame
    - HTTP clients connect to /stream.mjpg (or /) and receives MJPEG
    - encodes JPEG only if at least one client is connected
    """

    def __init__(
        self,
        host: str = variables.STREAM_HOST,
        port: int = variables.STREAM_PORT,
        jpeg_quality: int = variables.STREAM_JPEG_QUALITY,
        stream_fps: float = variables.STREAM_FPS,
            ) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.stream_fps = float(stream_fps)

        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_ts: float = 0.0

        self._running: bool = False
        self._server: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        self._clients: int = 0

    def start(self) -> None:
        if self._running:
            return
        else:
            self._running = True
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/view", "/"):
                    html = f"""
                    <html>
                    <head>
                        <title>PathPal Stream</title>
                        <style>
                        html, body {{
                            margin: 0; padding: 0; background: #000; height: 100%;
                            display: flex; align-items: center; justify-content: center;
                        }}
                        img {{
                            width: 100vw;
                            height: 100vh;
                            object-fit: contain;
                        }}
                        </style>
                    </head>
                    <body>
                        <img src="/stream.mjpg" />
                    </body>
                    </html>
                    """.encode("utf-8")

                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                if self.path != "/stream.mjpg":
                    self.send_response(404)
                    self.end_headers()
                    return

                streamer._clients += 1
                try:
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()

                    frame_interval = 1.0 / max(streamer.stream_fps, 0.1)

                    while streamer._running:
                        jpg = streamer._get_latest_jpeg()
                        if jpg is None:
                            time.sleep(0.02)
                            continue

                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            time.sleep(frame_interval)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                finally:
                    streamer._clients -= 1
            def log_message(self, format, *args) -> None:
                return
        self._server = _ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def has_clients(self) -> bool:
        return self._clients > 0

    def update_rgb(self, frame_rgb: np.ndarray) -> None:
        """
        Accept RGB frame and store internally as BGR for OpenCV encode.
        """
        if frame_rgb is None:
            return
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest_bgr = frame_bgr

    def update_bgr(self, frame_bgr: np.ndarray) -> None:
        """
        Accept BGR frame and store internally as BGR.
        """
        if frame_bgr is None:
            return
        with self._lock:
            self._latest_bgr = frame_bgr
    def _encode_latest(self, bgr: np.ndarray) -> Optional[bytes]:
        """
        Encode BGR frame to JPEG.
        """
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return buf.tobytes()

    def _get_latest_jpeg(self) -> Optional[bytes]:
        """
        Returns cached JPEG; re-encodes at most at stream_fps.
        Encodes only if at least one client is connected.
        """
        if not self.has_clients():
            return None

        now = time.time()
        min_dt = 1.0 / max(self.stream_fps, 0.1)

        with self._lock:
            bgr = self._latest_bgr
            jpg = self._latest_jpeg
            ts = self._latest_jpeg_ts

        if bgr is None:
            return None

        # throttle encoding
        if jpg is not None and (now - ts) < min_dt:
            return jpg

        new_jpg = self._encode_latest(bgr)
        if new_jpg is None:
            return None

        with self._lock:
            self._latest_jpeg = new_jpg
            self._latest_jpeg_ts = now

        return new_jpg
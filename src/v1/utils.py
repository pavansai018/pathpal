from typing import Tuple, List
import cv2
import numpy as np
from det import Det

def side_from_bbox(b: Tuple[int, int, int, int], w: int) -> str:
    x1, _, x2, _ = b
    cx = (x1 + x2) / 2.0
    if cx < w / 3:
        return "left"
    if cx > 2 * w / 3:
        return "right"
    return "center"


def draw_box(frame_bgr: np.ndarray, det: Det, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame_bgr,
        f"{det.label} {det.score:.2f}",
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

def overlay_status(frame_bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        frame_bgr,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )



def render_display(
    frame_rgb: np.ndarray,
    persons: List[Det],
    faces: List[Det],
    range_cm: float | None = None,
    fps_loop: float | None = None,
    fps_det: float | None = None,
    window_name: str = "PathPal Live",
) -> bool:
    """
    Renders live debug display with readable overlays.
    Returns False if user pressed 'q' to quit.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    h, w = frame_bgr.shape[:2]

    # Scale UI based on resolution
    scale = max(0.6, min(1.2, w / 800.0))
    thickness = max(1, int(2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(3, int(6 * scale))

    def draw_labeled_box(d: Det, color: Tuple[int, int, int], label_prefix: str = "") -> None:
        x1, y1, x2, y2 = d.bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return

        # Box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

        # Label with score (YES for face too)
        label = f"{label_prefix}{d.label} {d.score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, font, 0.6 * scale, thickness)
        th_total = th + baseline

        # Prefer label inside top of bbox; if not enough space, put above bbox
        label_y2 = y1 + th_total + 2 * pad
        place_inside = label_y2 < y2  # enough vertical space inside

        if place_inside:
            ry1 = y1
            ry2 = y1 + th_total + 2 * pad
        else:
            ry2 = y1
            ry1 = max(0, y1 - (th_total + 2 * pad))

        rx1 = x1
        rx2 = min(w - 1, x1 + tw + 2 * pad)

        # Filled background bar for readability
        cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), color, -1)

        # Text (black on colored bar)
        tx = rx1 + pad
        ty = ry2 - pad - baseline
        cv2.putText(frame_bgr, label, (tx, ty), font, 0.6 * scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Draw persons (yellow) then faces (cyan)
    for d in persons:
        draw_labeled_box(d, (0, 255, 255))
    for d in faces:
        draw_labeled_box(d, (255, 255, 0))

    # Status header with background
    # status = "PERSON: YES" if persons else "PERSON: NO"
    # header = f"{status} | persons={len(persons)} faces={len(faces)}"
    # (tw, th), baseline = cv2.getTextSize(header, font, 0.8 * scale, thickness)
    # bar_h = th + baseline + 2 * pad
    # cv2.rectangle(frame_bgr, (0, 0), (min(w - 1, tw + 2 * pad), bar_h), (0, 255, 0), -1)
    # cv2.putText(frame_bgr, header, (pad, bar_h - pad - baseline), font, 0.8 * scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if range_cm is not None:
        txt = f"DIST: {range_cm:.2f} cm"
        cv2.putText(
            frame_bgr,
            txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    fps_loop_s = f"{fps_loop:.1f}" if fps_loop is not None else '--'
    fps_det_s  = f"{fps_det:.1f}"  if fps_det  is not None else '--'
    cv2.putText(
            frame_bgr,
            f'FPS: {fps_loop_s}  DET: {fps_det_s}',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    cv2.imshow(window_name, frame_bgr)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def _clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def annotate_bgr(
    frame_rgb: np.ndarray,
    persons: List[Det],
    faces: List[Det],
    range_cm: float | None = None,
    fps_loop: float | None = None,
    fps_det: float | None = None,
) -> np.ndarray:
    """
    Returns BGR image with boxes + labels drawn.
    """
    h, w, _ = frame_rgb.shape
    frame_bgr = frame_rgb #cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # PERSON boxes (yellow)
    for d in persons:
        x1, y1, x2, y2 = _clamp_bbox(d.bbox, w, h)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame_bgr,
            f"{d.label} {d.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # FACE boxes (cyan)
    for d in faces:
        x1, y1, x2, y2 = _clamp_bbox(d.bbox, w, h)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"face {d.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Status line
    # status = "PERSON: YES" if persons else "PERSON: NO"
    # cv2.putText(
    #     frame_bgr,
    #     status,
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.9,
    #     (0, 255, 0),
    #     2,
    #     cv2.LINE_AA,
    # )
    # ultrasonic distance overlay 
    if range_cm is not None:
        txt = f'DIST: {range_cm:.2f} cm'
        cv2.putText(
            frame_bgr,
            txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    fps_loop_s = f"{fps_loop:.1f}" if fps_loop is not None else '--'
    fps_det_s  = f"{fps_det:.1f}"  if fps_det  is not None else '--'
    cv2.putText(
            frame_bgr,
            f'FPS: {fps_loop_s}  DET: {fps_det_s}',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    return frame_bgr

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
from module import Module
from camera import Camera
from coco_detector import CocoModule
from utils import side_from_bbox, render_display
from det import Det
from face_module import FaceModule
from event_policy import EventPolicy
import variables



def main() -> None:
    ENABLE_DISPLAY = variables.ENABLE_DISPLAY
    WINDOW_NAME = variables.WINDOW_NAME

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width=variables.DISPLAY_WIDTH, height=variables.DISPLAY_HEIGHT) 
    
    COCO_MODEL = variables.COCO_SSD_MOBILENET_V1_PATH
    COCO_LABELS = variables.COCO_LABELS_PATH

    cam = Camera(size=(variables.CAM_WIDTH, variables.CAM_HEIGHT), fps=variables.FPS)
    print(f"[INFO] Camera backend: {cam.backend}")

    state: Dict[str, Any] = {
        "coco_dets": [],
        "faces": [],
        "person_present": False,
        "persons": [],
    }

    modules: List[Module] = [
        CocoModule(COCO_MODEL, COCO_LABELS),
        FaceModule(),
        # Later: add new modules here (OCRModule, MotionHazardModule, etc.)
    ]

    events = EventPolicy(cooldown_s=2.0)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            now = time.time()
            for m in modules:
                if m.should_run(now):
                    m.process(frame, state)
                    m.mark_ran(now)

            # Simple demo events
            h, w, _ = frame.shape
            persons: List[Det] = state.get("persons", [])
            faces: List[Det] = state.get("faces", [])

            if persons:
                # pick largest person
                p = max(persons, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
                events.emit(f"[EVENT] person on {side_from_bbox(p.bbox, w)}")
            else:
                events.emit("[EVENT] no person")

            if faces:
                f = max(faces, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
                events.emit(f"[EVENT] face on {side_from_bbox(f.bbox, w)}")

            # ===== LIVE PREVIEW (laptop) =====
            # Your Camera.read() returns RGB. OpenCV expects BGR for display.
            if ENABLE_DISPLAY:
                if not render_display(frame, persons, faces, WINDOW_NAME):
                    break

    finally:
        cam.close()
        if ENABLE_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

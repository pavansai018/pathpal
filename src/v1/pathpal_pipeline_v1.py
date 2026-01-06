from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import cv2
from module import Module
from camera import Camera
from coco_detector import CocoModule
from utils import side_from_bbox, render_display, annotate_bgr
from det import Det
from face_module import FaceModule
from event_policy import EventPolicy
import variables
from frame_grabber import FrameGrabber
import threading
state_lock = threading.Lock()

if variables.ENABLE_FPS:
    from collections import deque
if variables.ENABLE_ULTRASONIC:
    from ultrasonic_module import UltrasonicModule

def fps_from_times(times):
    if not times:
        return 0.0
    return len(times) / sum(times)

def main() -> None:
    ENABLE_DISPLAY = variables.ENABLE_DISPLAY
    WINDOW_NAME = variables.WINDOW_NAME
    if ENABLE_DISPLAY:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, width=variables.DISPLAY_WIDTH, height=variables.DISPLAY_HEIGHT) 
    
    COCO_MODEL = variables.EFFICIENTDET_V0_PATH
    COCO_LABELS = variables.COCO_LABELS_PATH

    cam = Camera(size=(variables.CAM_WIDTH, variables.CAM_HEIGHT), fps=variables.FPS)
    if variables.DEBUG:
        print(f"[INFO] Camera backend: {cam.backend}")
    grabber = FrameGrabber(cam, target_fps=variables.TARGET_FRAME_GRABBER_FPS, copy_frame=variables.GRABBER_COPY_FRAME)
    grabber.start()
    state: Dict[str, Any] = {
        'coco_dets': [],
        'faces': [],
        'person_present': False,
        'persons': []
    }
    modules: List[Module] = []
    workers = []

    if variables.ENABLE_ULTRASONIC:
        modules.append(UltrasonicModule())

    coco = CocoModule(COCO_MODEL, COCO_LABELS)
    face = FaceModule()

    modules.extend([coco, face])
    # modules: List[Module] = [
    #     CocoModule(COCO_MODEL, COCO_LABELS),
    #     FaceModule(),
    #     # Later: add new modules here (OCRModule, MotionHazardModule, etc.)
    # ]
    from module_worker import ModuleWorker

    coco_worker = ModuleWorker(coco, state, state_lock)
    face_worker = ModuleWorker(face, state, state_lock)

    coco_worker.start()
    face_worker.start()

    events = EventPolicy(cooldown_s=2.0)
    streamer = None
    if variables.ENABLE_STREAM:
        from mjpeg_streamer import MjpegStreamer
        streamer = MjpegStreamer(
            host=variables.STREAM_HOST,
            port=variables.STREAM_PORT,
            jpeg_quality=variables.STREAM_JPEG_QUALITY,
            stream_fps=variables.STREAM_FPS,
        )
        streamer.start()
        if variables.DEBUG:
            print(f"[INFO] MJPEG: http://10.32.30.165:{variables.STREAM_PORT}/view")
    try:
        if variables.ENABLE_FPS:
            loop_times = deque(maxlen=30)
        last_ts = 0.0
        while True:
            loop_fps = None
            det_fps = None
            if variables.ENABLE_FPS:
                loop_start = time.time()
            frame, ts = grabber.get_latest()
            # No frame yet
            if frame is None:
                time.sleep(0.01)
                continue
            # if we already processed this frame, skip it
            if ts == last_ts:
                time.sleep(0.002)
                continue
            last_ts = ts

            now = time.time()
            state['now_ts'] = now  # useful for sensor modules

            # for m in modules:
            #     if m.should_run(now):
            #         t0 = time.time()
            #         m.process(frame, state)
            #         m.mark_ran(now)
            #         if m.name == 'coco':
            #             det_times.append(time.time() - t0)
            coco_worker.update_frame(frame, ts)
            face_worker.update_frame(frame, ts)

            # Simple demo events
            h, w, _ = frame.shape
            # persons: List[Det] = state.get('persons', [])
            # faces: List[Det] = state.get('faces', [])
            with state_lock:
                persons = list(state.get('persons', []))
                faces   = list(state.get('faces', []))
                dets    = list(state.get('coco_dets', []))
                range_cm = state.get('range_cm')

            # ---- existing person/face events (keep as-is if you want) ----
            if persons:
                p = max(persons, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
                events.emit(f"[EVENT] person on {side_from_bbox(p.bbox, w)}")
            else:
                events.emit("[EVENT] no person")

            if faces:
                f = max(faces, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
                events.emit(f"[EVENT] face on {side_from_bbox(f.bbox, w)}")

            # ---- NEW: ultrasonic + vision fusion ----
            # range_cm = state.get('range_cm', None)
            # dets: List[Det] = state.get('coco_dets', [])

            # choose which detection drives direction (controlled via variables.py)
            direction = variables.TARGET_DIRECTION
            target: Optional[Det] = None

            if dets:
                # 1) try priority labels in order
                for lbl in getattr(variables, 'RANGE_DIR_PRIORITY', ['person']):
                    candidates = [d for d in dets if d.label == lbl]
                    if candidates:
                        target = max(
                            candidates,
                            key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1])
                        )
                        break

                # 2) fallback logic
                if target is None:
                    fb = getattr(variables, 'RANGE_DIR_FALLBACK', 'largest_any')
                    if fb == 'largest_any':
                        target = max(
                            dets,
                            key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1])
                        )
                    elif fb == 'center_only':
                        target = None
                        direction = 'center'
                    elif fb == 'none':
                        target = None

            if target is not None:
                direction = side_from_bbox(target.bbox, w)

            if range_cm is not None:
                if range_cm <= variables.OBSTACLE_NEAR_CM:
                    events.emit(f"[EVENT] obstacle {range_cm:.3f} cm {direction}")
                elif range_cm <= variables.OBSTACLE_FAR_CM:
                    events.emit(f"[EVENT] obstacle {range_cm:.1f} cm ahead")
            if variables.ENABLE_FPS:
                loop_times.append(time.time() - loop_start)
                loop_fps = fps_from_times(loop_times)
                det_fps = None
                if coco_worker.infer_times:
                    det_fps = len(coco_worker.infer_times) / sum(coco_worker.infer_times)
            # ===== LIVE PREVIEW (laptop) =====
            if ENABLE_DISPLAY:
                if not render_display(
                    frame_rgb=frame,
                    persons=dets,
                    faces=faces,
                    fps_det=det_fps,
                    fps_loop=loop_fps,
                    range_cm=state.get('range_cm'),
                    window_name=WINDOW_NAME
                ):
                    break

            # Stream to laptop
            if streamer is not None and streamer.has_clients():
                bgr_annot = annotate_bgr(
                    frame_rgb=frame,
                    persons=dets,
                    faces=faces,
                    fps_det=det_fps,
                    fps_loop=loop_fps,
                    range_cm=state.get('range_cm'),
                )
                streamer.update_bgr(bgr_annot)
            
            try:
                if int(time.time()) % 2 == 0:   # print every ~2 seconds
                    print(f"[FPS] loop={loop_fps:.1f} det={det_fps:.1f}")
            except Exception as e:
                pass

    finally:
        grabber.stop()
        cam.close()
        coco_worker.stop()
        face_worker.stop()
        if ENABLE_DISPLAY:
            cv2.destroyAllWindows()
        if streamer is not None:
            streamer.stop()

if __name__ == "__main__":
    main()

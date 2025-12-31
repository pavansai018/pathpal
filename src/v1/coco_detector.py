from interpreter_backend import make_interpreter
from typing import List, Dict, Any
from PIL import Image
from module import Module
from labels import load_labels
import numpy as np
from det import Det
import variables


class CocoModule(Module):
    name = "coco"
    hz = 1.5  # ~1–2 FPS

    def __init__(self, model_path: str, labels_path: str) -> None:
        super().__init__()
        labels = load_labels(labels_path)
        self.det = CocoDetector(model_path=model_path, labels=labels, score_thresh=0.45)

    def process(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        dets = self.det.infer(frame)
        print("[DEBUG] top:", [(d.label, round(d.score, 2)) for d in sorted(dets, key=lambda x: x.score, reverse=True)[:5]])

        # Keep only relevant classes for now; later you can widen this list
        wanted = {"person", "car", "bus", "truck", "bicycle", "motorcycle"}
        dets = [d for d in dets if d.label in wanted]
        state["coco_dets"] = dets

        # “person present” summary for gating
        persons = [d for d in dets if d.label == "person"]
        state["person_present"] = len(persons) > 0
        state["persons"] = persons
# -----------------------------
# TFLite COCO detector wrapper
# -----------------------------
class CocoDetector:
    """
    Minimal SSD-style TFLite detector:
    expects common TFLite OD outputs: boxes, classes, scores, num_detections.
    Works for typical SSD MobileNet COCO models.
    """
    def __init__(self, model_path: str, labels: List[str], score_thresh: float = 0.4) -> None:
        self.labels = labels
        self.score_thresh = score_thresh
        self.interp = make_interpreter(model_path, num_threads=2, force=variables.INTERPRETER_MODE)  # or "tf" or "runtime"
        self.interp.allocate_tensors()

        self.in_details = self.interp.get_input_details()[0]
        self.out_details = self.interp.get_output_details()

        # input shape: [1, H, W, 3]
        _, self.in_h, self.in_w, _ = self.in_details["shape"]

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(rgb).resize((self.in_w, self.in_h))
        x = np.asarray(img, dtype=np.uint8)

        # Handle float inputs if model expects float32
        if self.in_details["dtype"] == np.float32:
            x = x.astype(np.float32) / 255.0

        x = np.expand_dims(x, axis=0)
        return x

    def infer(self, frame_rgb: np.ndarray) -> List[Det]:
        h, w, _ = frame_rgb.shape
        x = self._preprocess(frame_rgb)

        self.interp.set_tensor(self.in_details["index"], x)
        self.interp.invoke()

        # Typical order: boxes, classes, scores, num
        outs = [self.interp.get_tensor(d["index"]) for d in self.out_details]
        # Make robust to minor variations
        outs = [o.squeeze() for o in outs]

        # Heuristic mapping
        boxes = None
        classes = None
        scores = None
        num = None

        for o in outs:
            if o.ndim == 2 and o.shape[1] == 4:  # Nx4
                boxes = o
            elif o.ndim == 1 and o.dtype in (np.float32, np.float16) and o.size <= 200:
                # could be scores OR classes; disambiguate later
                pass

        # Many SSD models output in fixed order; try that first:
        if len(outs) >= 4 and outs[0].ndim == 2 and outs[0].shape[1] == 4:
            boxes = outs[0]
            classes = outs[1].astype(np.int32)
            scores = outs[2].astype(np.float32)
            num = int(outs[3]) if np.isscalar(outs[3]) else int(outs[3].item())
        else:
            # Fallback: assume standard names by position (still best-effort)
            boxes = outs[0]
            classes = outs[1].astype(np.int32)
            scores = outs[2].astype(np.float32)
            num = int(outs[3]) if np.isscalar(outs[3]) else int(outs[3].item())

        dets: List[Det] = []
        n = min(num, len(scores))
        for i in range(n):
            if float(scores[i]) < self.score_thresh:
                continue
            cls = int(classes[i])
            label = self.labels[cls] if 0 <= cls < len(self.labels) else f"class_{cls}"

            # FIX: common SSD label offset where labels[0] is '???'/'background'
            if label in {"???", "background"} and (cls + 1) < len(self.labels):
                label = self.labels[cls + 1]

            # boxes are usually [ymin, xmin, ymax, xmax] normalized
            ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            dets.append(Det(label=label, score=float(scores[i]), bbox=(x1, y1, x2, y2)))
        return dets
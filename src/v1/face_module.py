from module import Module
from typing import Dict, Any, List
from PIL import Image
from det import Det
import math
import numpy as np
try:
    import numpy
    if not hasattr(numpy, 'math'):
        # Create dummy math attribute if missing
        numpy.math = math
except Exception as e:
    print(e)
# Face detector (BlazeFace style via face-detection-tflite)
from vendor_fdlite import FaceDetection, FaceDetectionModel
class FaceModule(Module):
    name = "face"
    hz = 2.5  # 2–3 FPS, but gated

    def __init__(self) -> None:
        super().__init__()
        self.fd = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)

    def process(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        # Gate: only run face detection if person exists
        if not state.get("person_present", False):
            state["faces"] = []
            return

        # Optional ROI: run only in top 75% for chest pendant (cuts false positives)
        h, w, _ = frame.shape
        roi = frame[: int(h * 0.75), :, :]
        img = Image.fromarray(roi)
        dets = self.fd(img)

        faces: List[Det] = []
        for d in dets:
            bb = d.bbox  # normalized xmin,ymin,width,height relative to ROI
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * roi.shape[0])
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * roi.shape[0])
            faces.append(Det(label="face", score=float(d.score), bbox=(x1, y1, x2, y2)))
        state["faces"] = faces
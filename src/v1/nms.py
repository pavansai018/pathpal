from __future__ import annotations
from typing import List
from det import Det
import variables

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def nms_dets(dets: List[Det], iou_thresh: float = variables.COCO_NMS_THRESH) -> List[Det]:
    """
    Class-wise NMS on pixel bboxes. Keeps best score per overlapping region.
    """
    if not dets:
        return dets
    
    out: List[Det] = []
    # group by label to avoid suppressing different classes
    by_label = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)

    for label, group in by_label.items():
        group = sorted(group, key=lambda d: d.score, reverse=True)
        kept: List["Det"] = []
        for d in group:
            if all(_iou(d.bbox, k.bbox) < iou_thresh for k in kept):
                kept.append(d)
        out.extend(kept)

    # keep overall ordering by score
    out.sort(key=lambda d: d.score, reverse=True)
    if variables.DEBUG:
        print(f'Before NMS: {len(dets)} objects. After NMS: {len(out)} objects.')
    return out

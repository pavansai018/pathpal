from dataclasses import dataclass
from typing import Tuple
# -----------------------------
# Shared types
# -----------------------------
@dataclass(frozen=True)
class Det:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 in pixels
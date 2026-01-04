from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional
from gpiozero import DistanceSensor
import variables
from module import Module


@dataclass
class RangeReading:
    cm: float
    ts: float


class UltrasonicModule(Module):
    """
    Uses gpiozero.DistanceSensor and publishes:
      state["range_cm"]         : Optional[float]  (smoothed)
      state["range_raw_cm"]     : Optional[float]  (instant)
      state["range_ts"]         : float            (timestamp)
      state["obstacle_near"]    : bool             (based on variables)
    """
    name = "ultrasonic"
    hz = 15.0  # sensor read rate

    def __init__(self) -> None:
        super().__init__()
        if not variables.ENABLE_ULTRASONIC:
            self.enabled = False
            return
        self.enabled = True
        self.sensor = DistanceSensor(
            echo=variables.ECHO,
            trigger=variables.TRIGGER,
            max_distance=variables.MAX_DISTANCE,  # meters in gpiozero
        )

        self._hist: Deque[float] = deque(maxlen=getattr(variables, 'RANGE_SMOOTH_N', 5))

    def _read_cm(self) -> Optional[float]:
        # gpiozero returns distance in meters normalized to max_distance
        # distance_sensor.distance is 0.0..1.0 of max_distance
        # BUT gpiozero DistanceSensor.distance is documented as "distance as a proportion of max_distance"
        # so distance_m = distance * max_distance
        try:
            d_prop = float(self.sensor.distance)
        except Exception:
            return None

        if d_prop < 0.0:
            return None
        d_cm = d_prop * 100.0

        # clamp obviously bad readings
        if d_cm <= 0.0 or d_cm > float(variables.MAX_DISTANCE) * 100.0:
            return None
        return d_cm

    def process(self, frame, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        cm = self._read_cm()
        if cm is None:
            state['range_raw_cm'] = None
            return

        state['range_raw_cm'] = cm
        self._hist.append(cm)

        # median smoothing (cheap + robust)
        med = sorted(self._hist)[len(self._hist) // 2]
        state['range_cm'] = float(med)
        state['range_ts'] = state.get('now_ts', 0.0)

        near_thr = getattr(variables, 'OBSTACLE_NEAR_CM', 80.0)
        state['obstacle_near'] = (med <= near_thr)

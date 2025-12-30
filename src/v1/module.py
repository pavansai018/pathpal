from typing import Dict, Any
import numpy as np

class Module:
    name: str = "module"
    hz: float = 1.0  # run frequency

    def __init__(self) -> None:
        self._next_t = 0.0

    def should_run(self, now: float) -> bool:
        return now >= self._next_t

    def mark_ran(self, now: float) -> None:
        self._next_t = now + (1.0 / max(self.hz, 1e-6))

    def process(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        raise NotImplementedError
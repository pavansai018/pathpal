from typing import Optional
import time
# -----------------------------
# Event policy (no spam)
# -----------------------------
class EventPolicy:
    def __init__(self, cooldown_s: float = 2.0) -> None:
        self.cooldown_s = cooldown_s
        self.last_msg: Optional[str] = None
        self.last_t = 0.0

    def emit(self, msg: str) -> None:
        now = time.time()
        if msg == self.last_msg:
            return
        if (now - self.last_t) < self.cooldown_s:
            return
        print(msg)
        self.last_msg = msg
        self.last_t = now
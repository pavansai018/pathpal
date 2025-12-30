from __future__ import annotations
from typing import Optional

def make_interpreter(model_path: str, num_threads: int = 2, force: Optional[str] = None):
    """
    force:
      - "runtime": use tflite-runtime only
      - "tf": use tensorflow tflite only
      - None: try runtime, fall back to tf
    """
    if force == "runtime":
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)

    if force == "tf":
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)

    # auto
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)

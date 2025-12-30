from typing import List

def load_labels(path: str) -> List[str]:
    # Expect one label per line, index = line number
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
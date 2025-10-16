import os
from pathlib import Path

import wget

LINK = ""


def download_best_model():
    dir = Path("models")
    dir.mkdir(exist_ok=True)
    model_path = dir / "model_best.pth"
    if not model_path.exists():
        wget.download(LINK, str(model_path))


if __name__ == "__main__":
    download_best_model()

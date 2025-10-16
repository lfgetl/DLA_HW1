import os
from pathlib import Path

import wget

LINK = (
    "https://drive.google.com/uc?export=download&id=1MP1Gt6OMT-PIvkPFDuYUGkGdXUM_n-tv"
)


def download_best_model():
    dir = Path("models")
    dir.mkdir(exist_ok=True)
    model_path = dir / "model_best.pth"
    if not model_path.exists():
        wget.download(LINK, str(model_path))


if __name__ == "__main__":
    download_best_model()

import os
from pathlib import Path

import wget

LINK = (
    "https://drive.google.com/uc?export=download&id=1-gZIc9EsM5EiOTaz96iIWJeRvYU7_1vq"
)


def download_best_model():
    dir = Path("models")
    dir.mkdir(exist_ok=True)
    model_path = dir / "model_best.pth"
    if not model_path.exists():
        wget.download(LINK, str(model_path))


if __name__ == "__main__":
    download_best_model()

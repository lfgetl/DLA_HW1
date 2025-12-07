from pathlib import Path

import gdown

LINK = (
    "https://drive.google.com/uc?export=download&id=1dF-xLWhXUKCkV1PqrChAAXPvyrcyBTfs"
)


def download_best_model():
    dir = Path("models")
    dir.mkdir(exist_ok=True)
    model_path = dir / "model_best.pth"
    if not model_path.exists():
        gdown.download(LINK, str(model_path))


if __name__ == "__main__":
    download_best_model()

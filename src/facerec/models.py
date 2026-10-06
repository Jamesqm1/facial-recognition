"""Locations and download helpers for the ONNX models from the OpenCV model zoo."""

from pathlib import Path
from urllib.request import urlretrieve

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    DETECTOR_PATH: f"{_ZOO}/face_detection_yunet/{DETECTOR_PATH.name}",
    RECOGNIZER_PATH: f"{_ZOO}/face_recognition_sface/{RECOGNIZER_PATH.name}",
}


def download_models(force: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for dest, url in MODEL_URLS.items():
        if dest.exists() and not force:
            print(f"already have {dest.name}")
            continue
        print(f"downloading {dest.name} ...")
        urlretrieve(url, dest)


def ensure_models() -> None:
    missing = [p.name for p in MODEL_URLS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing models: {', '.join(missing)}. Run `facerec download-models` first."
        )

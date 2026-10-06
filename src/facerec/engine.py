"""Face detection (YuNet) and recognition (SFace) wrappers."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .models import DETECTOR_PATH, RECOGNIZER_PATH, ensure_models

# Recommended cosine-similarity threshold for SFace (higher = more similar).
COSINE_THRESHOLD = 0.363


class FaceEngine:
    def __init__(self, score_threshold: float = 0.9):
        ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return an (N, 15) array: x, y, w, h, 5 landmark points, score."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face)
        return self.recognizer.feature(aligned).flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class Gallery:
    """Known people and their face embeddings."""

    names: list[str] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.names.append(name)
        self.embeddings.append(embedding)

    def identify(
        self, embedding: np.ndarray, threshold: float = COSINE_THRESHOLD
    ) -> tuple[str | None, float]:
        """Return (name, score) of the best match, or (None, score) if below threshold."""
        best_name, best_score = None, -1.0
        for name, known in zip(self.names, self.embeddings):
            score = cosine_similarity(embedding, known)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < threshold:
            return None, best_score
        return best_name, best_score

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names), embeddings=np.array(self.embeddings))

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        data = np.load(path)
        return cls([str(n) for n in data["names"]], list(data["embeddings"]))

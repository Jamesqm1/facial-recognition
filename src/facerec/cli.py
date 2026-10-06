"""Command-line entry point: `facerec <command>`."""

import argparse
from pathlib import Path

import cv2

from .engine import COSINE_THRESHOLD, FaceEngine, Gallery
from .models import download_models

ROOT = Path(__file__).resolve().parents[2]
KNOWN_DIR = ROOT / "data" / "known"
GALLERY_PATH = ROOT / "data" / "gallery.npz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cmd_enroll(args: argparse.Namespace) -> None:
    """Build the gallery from data/known/<person>/*.jpg."""
    engine = FaceEngine()
    gallery = Gallery()
    for person_dir in sorted(p for p in args.known_dir.iterdir() if p.is_dir()):
        count = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  skip {img_path.name}: unreadable")
                continue
            faces = engine.detect(image)
            if len(faces) != 1:
                print(f"  skip {img_path.name}: found {len(faces)} faces (need exactly 1)")
                continue
            gallery.add(person_dir.name, engine.embed(image, faces[0]))
            count += 1
        print(f"{person_dir.name}: {count} image(s) enrolled")
    if not gallery.names:
        raise SystemExit(f"No faces enrolled. Add photos under {args.known_dir}/<name>/")
    gallery.save(args.gallery)
    print(f"Saved {len(gallery.names)} embedding(s) to {args.gallery}")


def cmd_run(args: argparse.Namespace) -> None:
    """Detect (and, if a gallery exists, recognize) faces on a webcam or image."""
    engine = FaceEngine()
    gallery = Gallery.load(args.gallery) if args.gallery.exists() else None
    if gallery is None:
        print("No gallery found - detection only. Run `facerec enroll` to enable recognition.")

    def annotate(frame):
        for face in engine.detect(frame):
            x, y, w, h = face[:4].astype(int)
            label, color = "face", (255, 200, 0)
            if gallery is not None:
                name, score = gallery.identify(engine.embed(frame, face), args.threshold)
                label = f"{name or 'unknown'} {score:.2f}"
                color = (0, 200, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        cv2.imshow("facerec", annotate(frame))
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")
        print("Press q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("facerec", annotate(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(prog="facerec")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-models", help="download YuNet + SFace ONNX models")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: download_models(a.force))

    p = sub.add_parser("enroll", help="build gallery from data/known/<name>/ images")
    p.add_argument("--known-dir", type=Path, default=KNOWN_DIR)
    p.add_argument("--gallery", type=Path, default=GALLERY_PATH)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("run", help="detect/recognize faces from webcam or an image")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", type=Path)
    p.add_argument("--gallery", type=Path, default=GALLERY_PATH)
    p.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

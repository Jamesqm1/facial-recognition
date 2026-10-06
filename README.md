# facial-recognition

Face detection and recognition with OpenCV. It uses two models from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo):

- **YuNet** finds faces in an image.
- **SFace** turns each face into a 128-d embedding. Faces are matched by cosine similarity.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
facerec download-models
```

## Usage

1. **Detection only (webcam):** `facerec run`
2. **Enroll known people:** add photos to `data/known/<name>/`, one face per photo. A few photos per person work best. Then run:
   ```powershell
   facerec enroll
   ```
3. **Recognize:** `facerec run` (webcam) or `facerec run --image path\to\photo.jpg`

Press `q` to quit the webcam view. Adjust match strictness with `--threshold` (default 0.363; higher is stricter).

## Layout

```
src/facerec/
  cli.py      # `facerec` command: download-models, enroll, run
  engine.py   # FaceEngine (detect/embed) and Gallery (match/save/load)
  models.py   # model paths and download
data/known/   # your enrollment photos (git-ignored)
models/       # downloaded ONNX models (git-ignored)
tests/
```

## Tests

```powershell
pytest
```

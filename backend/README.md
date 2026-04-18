# Backend

Entry scripts live in `backend/scripts/`:

- `check_dataset.py` — validate YOLO dataset structure/labels
- `train_model.py` — train YOLOv8 on `backend/data/datasets/gun_ir/`
- `gunDetect.py` — run inference on a single image
- `build_poi_db.py` — build DeepFace embeddings JSON from `backend/data/faces/poi/`
- `recognize_poi.py` — match a suspect image against POI images via DeepFace

## Runtime Services

For the `rgb/integrationTest` flow, use the repo-root server files:

- `pc_server.py` — run this on the PC or Mac that hosts the frontend
- `pi_camera_server.py` — run this on the Raspberry Pi

The copies in `backend/pc_server.py`, `backend/pi_camera_server.py`, and `backend/api/main.py` are compatibility wrappers that forward to the repo-root modules.
Older duplicate API implementations and v2 server files are archived under `backend/archive/web_stack/`.

## Install

From the `PeekIR` repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r CapstoneProject/requirements.txt
```

## Run The PC Backend

```bash
source .venv/bin/activate
export PI_CAMERA_BASE_URL=http://<pi-ip>:9000
python -m uvicorn pc_server:app --host 0.0.0.0 --port 8000
```

## Run The Pi Camera Service

```bash
sudo apt-get install -y python3-picamera2
source .venv/bin/activate
python -m pip install -r CapstoneProject/backend/requirements-pi.txt
python -m uvicorn pi_camera_server:app --host 0.0.0.0 --port 9000
```

The Pi service exposes both feeds from one process:

- `GET /api/camera/stream` for RGB
- `GET /api/ir/stream` for IR

If the IR camera enumerates as a different Linux video device, set it before starting the server:

```bash
IR_CAMERA_DEVICE=/dev/video2 python -m uvicorn pi_camera_server:app --host 0.0.0.0 --port 9000
```

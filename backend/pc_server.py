"""
PC Backend Server (FastAPI)

Run:
  python3 -m uvicorn pc_server:app --host 0.0.0.0 --port 8000

Env:
  PI_CAMERA_BASE_URL=http://<pi-ip>:9000

Endpoints:
  GET /api/health
  GET /api/camera/frame
  GET /api/camera/stream
  GET /api/ir/frame
  GET /api/ir/stream
  GET /api/ir/detect
  POST /api/detect-upload   (legacy multipart/form-data, field name: image)
  POST /api/detect-person  (multipart/form-data, field name: file)
  POST /api/detect-person-path (JSON payload)
  POST /api/poi/match       (multipart/form-data, field name: file)
  POST /api/poi/match-base64 (JSON payload)
"""
from __future__ import annotations

import base64
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import cv2
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

def _cors_allowlist() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["*"]

REPO_ROOT = Path(__file__).resolve().parent
PRIMARY_BACKEND = REPO_ROOT / "backend"
LEGACY_CAPSTONE_BACKEND = REPO_ROOT / "CapstoneProject" / "backend"
BACKEND_ROOT = PRIMARY_BACKEND if PRIMARY_BACKEND.exists() else LEGACY_CAPSTONE_BACKEND
BACKEND_PACKAGE_ROOT = REPO_ROOT if PRIMARY_BACKEND.exists() else REPO_ROOT / "CapstoneProject"

POI_DB_PATH = BACKEND_ROOT / "data" / "poi_db" / "poi_embeddings.json"
POI_METADATA_PATH = BACKEND_ROOT / "data" / "poi_db" / "poi_metadata.json"
POI_DIR = BACKEND_ROOT / "data" / "faces" / "poi"
POI_MODEL_NAME = "ArcFace"
POI_DETECTOR_BACKEND = "retinaface"
POI_DISTANCE_METRIC = "cosine"
POI_DEFAULT_THRESHOLD = 0.68
KNIFE_WEIGHTS_PATH = BACKEND_ROOT / "models" / "yolov8n.pt"
KNIFE_LABELS = {"knife"}
GUN_LABELS = {"gun", "weapon", "firearm", "handgun", "pistol", "rifle"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_DIR = BACKEND_ROOT / "logs"
LOG_FILE = LOG_DIR / "backend.log"
if BACKEND_PACKAGE_ROOT.exists():
    sys.path.insert(0, str(BACKEND_PACKAGE_ROOT))

from backend.scripts.personDetect import (  # type: ignore  # noqa: E402
    detect_persons,
    DEFAULT_PERSON_WEIGHTS_PATH,
    CONF_THRESHOLD as DEFAULT_PERSON_CONF,
)
from backend.scripts.gunDetect import (  # type: ignore  # noqa: E402
    detect_weapons,
    Detection as WeaponDetection,
    DEFAULT_WEIGHTS_PATH,
    CONF_THRESHOLD as DEFAULT_WEAPON_CONF,
    load_model,
)

try:
    from backend import metal_detector as metal_detector_module  # type: ignore  # noqa: E402
except Exception as e:  # pragma: no cover - runtime BLE dependency
    metal_detector_module = None
    _METAL_DETECTOR_IMPORT_ERROR = e
else:
    _METAL_DETECTOR_IMPORT_ERROR = None


class POIMatchBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 or data URL for the captured suspect image.")
    filename: str | None = Field(default=None, description="Optional filename hint for the uploaded image.")


class DetectPathRequest(BaseModel):
    path: str = Field(..., description="Image path on the machine running the backend server.")
    conf_threshold: float = Field(default=DEFAULT_WEAPON_CONF, ge=0.0, le=1.0)
    weights_path: str | None = Field(default=None, description="Optional weights path override.")


class DetectPersonPathRequest(BaseModel):
    path: str = Field(..., description="Image path on the machine running the backend server.")
    conf_threshold: float = Field(default=DEFAULT_PERSON_CONF, ge=0.0, le=1.0)
    weights_path: str | None = Field(default=None, description="Optional person detector weights path override.")


def _configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("peekir.pc_backend")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).name == LOG_FILE.name
        for h in logger.handlers
    ):
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


logger = _configure_logging()

app = FastAPI(title="PC Backend Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowlist(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_metal_lock = threading.Lock()
_metal_stop_event = threading.Event()
_metal_thread: threading.Thread | None = None
_metal_state: dict[str, Any] = {
    "enabled": False,
    "connected": False,
    "device_name": None,
    "device_address": None,
    "metal_detected": None,
    "pin_state": None,
    "battery_status": None,
    "last_message": None,
    "last_update_at": None,
    "error": None,
}

FILE_TRACE_MAX = 500
_file_trace_lock = threading.Lock()
_file_trace_events: deque[dict[str, Any]] = deque(maxlen=FILE_TRACE_MAX)


def _metal_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metal_detector_enabled() -> bool:
    raw = os.getenv("METAL_DETECTOR_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _set_metal_state(**updates: Any) -> None:
    with _metal_lock:
        _metal_state.update(updates)


def _get_metal_state() -> dict[str, Any]:
    with _metal_lock:
        return dict(_metal_state)


def _trace_path(path: str | Path | None) -> str | None:
    if path is None:
        return None

    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return path

    raw = Path(path)
    try:
        resolved = raw.resolve()
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def _record_file_event(
    operation: str,
    path: str | Path | None,
    location: str,
    *,
    status: str = "ok",
    detail: str | None = None,
) -> None:
    display_path = _trace_path(path)
    exists = None
    if path is not None:
        try:
            exists = Path(path).exists()
        except Exception:
            exists = None

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "location": location,
        "path": display_path,
        "exists": exists,
        "status": status,
        "detail": detail,
    }
    with _file_trace_lock:
        _file_trace_events.append(event)

    logger.info(
        "file_event op=%s location=%s path=%s exists=%s status=%s detail=%s",
        operation,
        location,
        display_path,
        exists,
        status,
        detail,
    )


def _get_file_events(tail: int) -> list[dict[str, Any]]:
    with _file_trace_lock:
        return list(_file_trace_events)[-tail:]


def _handle_metal_message(message: str) -> None:
    clean = message.strip()
    updates: dict[str, Any] = {
        "last_message": clean,
        "last_update_at": _metal_now(),
        "error": None,
    }
    if clean.startswith("PIN: HIGH"):
        updates["metal_detected"] = True
        updates["pin_state"] = "HIGH"
    elif clean.startswith("PIN: LOW"):
        updates["metal_detected"] = False
        updates["pin_state"] = "LOW"
    elif clean.startswith("BAT:"):
        updates["battery_status"] = clean[4:].strip()
    _set_metal_state(**updates)


async def _metal_monitor_async() -> None:
    if metal_detector_module is None:
        _set_metal_state(
            enabled=False,
            connected=False,
            error=f"Metal detector unavailable: {_METAL_DETECTOR_IMPORT_ERROR}",
            last_update_at=_metal_now(),
        )
        return

    device_name = os.getenv("METAL_DETECTOR_NAME", metal_detector_module.DEFAULT_DEVICE_NAME)
    address = os.getenv("METAL_DETECTOR_ADDRESS", "").strip() or None
    reconnect_delay = float(os.getenv("METAL_DETECTOR_RECONNECT_DELAY", "3.0"))
    _set_metal_state(enabled=True, error=None)

    while not _metal_stop_event.is_set():
        try:
            _set_metal_state(error="Scanning for metal detector", connected=False)
            device = await metal_detector_module.find_device(device_name, address)
            disconnected_event = asyncio.Event()

            def handle_disconnect(_client: Any) -> None:
                _set_metal_state(connected=False, error="Metal detector disconnected", last_update_at=_metal_now())
                disconnected_event.set()

            async with metal_detector_module.BleakClient(device, disconnected_callback=handle_disconnect) as client:
                if not client.is_connected:
                    raise RuntimeError("Metal detector connection failed")

                _set_metal_state(
                    connected=True,
                    device_name=device.name,
                    device_address=device.address,
                    error=None,
                    last_update_at=_metal_now(),
                )

                def notification_handler(_sender: Any, data: bytearray) -> None:
                    message = data.decode("utf-8", errors="replace")
                    _handle_metal_message(message)

                await client.start_notify(metal_detector_module.CHARACTERISTIC_UUID_TX, notification_handler)
                try:
                    while not _metal_stop_event.is_set() and not disconnected_event.is_set():
                        await asyncio.sleep(0.2)
                finally:
                    with contextlib.suppress(Exception):
                        await client.stop_notify(metal_detector_module.CHARACTERISTIC_UUID_TX)

        except Exception as e:
            if _metal_stop_event.is_set():
                break
            _set_metal_state(connected=False, error=str(e), last_update_at=_metal_now())
            logger.warning("metal detector monitor error: %s", e)

        slept = 0.0
        while slept < reconnect_delay and not _metal_stop_event.is_set():
            await asyncio.sleep(0.2)
            slept += 0.2


def _metal_monitor_worker() -> None:
    try:
        asyncio.run(_metal_monitor_async())
    except Exception as e:
        _set_metal_state(connected=False, error=str(e), last_update_at=_metal_now())
        logger.exception("metal detector monitor stopped: %s", e)


@app.on_event("startup")
def start_metal_detector_monitor() -> None:
    global _metal_thread
    if not _metal_detector_enabled():
        _set_metal_state(enabled=False, connected=False, error="Metal detector monitor disabled")
        return
    if _metal_thread and _metal_thread.is_alive():
        return
    _metal_stop_event.clear()
    _metal_thread = threading.Thread(target=_metal_monitor_worker, name="metal-detector-monitor", daemon=True)
    _metal_thread.start()


@app.on_event("shutdown")
def stop_metal_detector_monitor() -> None:
    _metal_stop_event.set()


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("%s %s -> EXCEPTION (%.1fms): %s", request.method, request.url.path, elapsed_ms, e)
        raise


def _pi_base_url() -> str:
    base = os.getenv("PI_CAMERA_BASE_URL", "").strip()
    if not base:
        raise HTTPException(status_code=500, detail="PI_CAMERA_BASE_URL not set")
    return base.rstrip("/")


def _resolve_user_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        # Try repo root first
        repo_candidate = (REPO_ROOT / candidate).resolve()
        if repo_candidate.exists():
            candidate = repo_candidate
        else:
            # Try CapstoneProject backend relative paths
            cap_backend_candidate = (BACKEND_ROOT / candidate).resolve()
            if cap_backend_candidate.exists():
                candidate = cap_backend_candidate
            else:
                cap_candidate = (REPO_ROOT / "CapstoneProject" / candidate).resolve()
                candidate = cap_candidate
    else:
        candidate = candidate.resolve()

    repo = REPO_ROOT.resolve()
    try:
        candidate.relative_to(repo)
    except Exception:
        raise HTTPException(status_code=400, detail="Path must be within the project folder.")
    return candidate


def _normalize_repo_rel_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    raw = Path(path_value)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(REPO_ROOT).as_posix()
        except Exception:
            return raw.as_posix()

    # Try to resolve against repo root and CapstoneProject locations.
    for base in (REPO_ROOT, BACKEND_ROOT, REPO_ROOT / "CapstoneProject"):
        candidate = (base / raw).resolve()
        if candidate.exists():
            try:
                return candidate.relative_to(REPO_ROOT).as_posix()
            except Exception:
                return candidate.as_posix()

    return raw.as_posix()


def _image_suffix_from_mime(mime: str | None) -> str | None:
    if not mime:
        return None
    m = mime.lower()
    if m in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if m == "image/png":
        return ".png"
    if m == "image/webp":
        return ".webp"
    if m == "image/bmp":
        return ".bmp"
    return None


def _decode_image_base64(payload: str) -> tuple[bytes, str | None]:
    if not payload:
        raise HTTPException(status_code=400, detail="Empty base64 payload.")

    mime = None
    data = payload.strip()
    if data.startswith("data:"):
        header, _, rest = data.partition(",")
        if not rest:
            raise HTTPException(status_code=400, detail="Invalid data URL payload.")
        data = rest
        mime = header[5:].split(";")[0] if ";" in header else header[5:]

    try:
        decoded = base64.b64decode(data, validate=True)
    except Exception:
        try:
            decoded = base64.b64decode(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.") from e

    if not decoded:
        raise HTTPException(status_code=400, detail="Empty decoded image data.")
    return decoded, mime


def _cosine_distance(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / denom)


def _thumbnail_base64(path: Path, size: int = 96) -> str | None:
    _record_file_event("read", path, "gallery thumbnail")
    try:
        from PIL import Image

        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _iter_images(root: Path) -> list[Path]:
    _record_file_event("scan", root, "image iterator")
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _pick_face_representation(reps: Any) -> dict:
    if isinstance(reps, list):
        if not reps:
            raise ValueError("DeepFace.represent() returned an empty list.")
        if len(reps) == 1:
            return reps[0]

        def area(rep: dict) -> float:
            fa = rep.get("facial_area") or {}
            w = fa.get("w") or fa.get("width") or 0
            h = fa.get("h") or fa.get("height") or 0
            return float(w) * float(h)

        return max(reps, key=area)

    if isinstance(reps, dict):
        return reps

    raise TypeError(f"Unexpected represent() return type: {type(reps).__name__}")


def _poi_threshold() -> float:
    try:
        from deepface.commons import distance as dist

        if hasattr(dist, "find_threshold"):
            return float(dist.find_threshold(POI_MODEL_NAME, POI_DISTANCE_METRIC))
        if hasattr(dist, "findThreshold"):
            return float(dist.findThreshold(POI_MODEL_NAME, POI_DISTANCE_METRIC))
    except Exception:
        pass
    return POI_DEFAULT_THRESHOLD


def _result_plot_png_base64(result: Any) -> str | None:
    try:
        annotated = result.plot()
    except Exception:
        return None

    try:
        rgb = annotated[..., ::-1]
        ok, encoded = cv2.imencode(".png", rgb)
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    except Exception:
        return None


def _should_draw_weapon_box(detection: WeaponDetection) -> bool:
    label = str(detection.label).strip().lower()
    return label in GUN_LABELS or label in KNIFE_LABELS


def _weapon_boxes_png_base64(image_path: str | Path, detections: list[WeaponDetection]) -> str | None:
    drawable = [d for d in detections if _should_draw_weapon_box(d)]
    if not drawable:
        return None

    _record_file_event("read", image_path, "annotated weapon preview")
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    for detection in drawable:
        if not detection.xyxy or len(detection.xyxy) < 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in detection.xyxy[:4]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.label} {detection.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = min(height - 1, label_y1 + text_h + baseline + 6)
        label_x2 = min(width - 1, x1 + text_w + 8)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return -1


def _detect_knives(
    image_path: str | Path,
    *,
    weights_path: str | Path = KNIFE_WEIGHTS_PATH,
    conf_threshold: float = DEFAULT_WEAPON_CONF,
) -> tuple[bool, list[WeaponDetection], Any]:
    image_path = Path(image_path)
    _record_file_event("read", image_path, "knife detector input")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    _record_file_event("read", weights_path, "knife detector weights")
    if not Path(weights_path).exists():
        return False, [], None

    model = load_model(weights_path)
    names = getattr(model, "names", {}) or {}
    knife_class_ids = [
        int(cls_id)
        for cls_id, label in names.items()
        if str(label).strip().lower() in KNIFE_LABELS
    ]
    if not knife_class_ids:
        return False, [], None

    def _run_predict(conf: float, imgsz: int):
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            classes=knife_class_ids,
            verbose=False,
        )
        return results[0], conf, imgsz

    result, used_conf, used_imgsz = _run_predict(conf_threshold, 640)

    def _extract_detections(res: Any) -> list[WeaponDetection]:
        res_names = res.names
        dets: list[WeaponDetection] = []
        for box in res.boxes:
            cls_id = _safe_int(box.cls[0])
            label = str(res_names.get(cls_id, str(cls_id)))
            conf = _safe_float(box.conf[0])
            xyxy = None
            try:
                xyxy_raw = box.xyxy[0].tolist()
                xyxy = [float(v) for v in xyxy_raw]
            except Exception:
                xyxy = None
            dets.append(WeaponDetection(class_id=cls_id, label=label, confidence=conf, xyxy=xyxy))
        return dets

    detections = _extract_detections(result)
    if not detections:
        for conf_try, imgsz_try in [
            (min(conf_threshold, 0.2), 960),
            (0.1, 960),
        ]:
            try:
                res_try, used_conf, used_imgsz = _run_predict(conf_try, imgsz_try)
                dets_try = _extract_detections(res_try)
                if dets_try:
                    result = res_try
                    detections = dets_try
                    break
            except Exception:
                continue

    has_knife = len(detections) > 0
    try:
        result._meta = {
            "used_conf": used_conf,
            "used_imgsz": used_imgsz,
            "source": "yolov8n-knife",
        }
    except Exception:
        pass
    return has_knife, detections, result


def _load_or_build_poi_db(enforce_detection: bool = False) -> dict[str, Any]:
    if POI_DB_PATH.exists():
        try:
            _record_file_event("read", POI_DB_PATH, "POI embeddings database")
            return json.loads(POI_DB_PATH.read_text())
        except Exception:
            _record_file_event("read", POI_DB_PATH, "POI embeddings database", status="error", detail="invalid JSON")
            pass

    try:
        from deepface import DeepFace
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DeepFace unavailable: {e}")

    images = _iter_images(POI_DIR)
    if not images:
        raise HTTPException(status_code=400, detail=f"No POI images found under: {POI_DIR}")

    created_at = datetime.now(timezone.utc).isoformat()
    entries: list[dict[str, Any]] = []
    skipped = 0

    for img_path in images:
        try:
            _record_file_event("read", img_path, "POI database build image")
            reps = DeepFace.represent(
                img_path=str(img_path),
                model_name=POI_MODEL_NAME,
                detector_backend=POI_DETECTOR_BACKEND,
                enforce_detection=enforce_detection,
            )
            rep = _pick_face_representation(reps)
            embedding = rep.get("embedding")
            if embedding is None:
                raise KeyError("Missing 'embedding' in DeepFace representation.")

            rel = _normalize_repo_rel_path(img_path.relative_to(REPO_ROOT).as_posix())
            entries.append(
                {
                    "name": img_path.stem,
                    "image_path": rel,
                    "embedding": embedding,
                }
            )
        except Exception:
            skipped += 1

    payload = {
        "schema_version": 1,
        "model_name": POI_MODEL_NAME,
        "detector_backend": POI_DETECTOR_BACKEND,
        "created_at_utc": created_at,
        "count": len(entries),
        "skipped": skipped,
        "entries": entries,
    }
    POI_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _record_file_event("write", POI_DB_PATH, "POI embeddings database")
    POI_DB_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _load_poi_metadata() -> dict[str, dict[str, Any]]:
    if not POI_METADATA_PATH.exists():
        _record_file_event("read", POI_METADATA_PATH, "POI metadata", status="missing")
        return {}

    try:
        _record_file_event("read", POI_METADATA_PATH, "POI metadata")
        payload = json.loads(POI_METADATA_PATH.read_text())
    except Exception:
        _record_file_event("read", POI_METADATA_PATH, "POI metadata", status="error", detail="invalid JSON")
        return {}

    records = payload.get("records")
    if not isinstance(records, list):
        return {}

    metadata_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue

        keys = [
            record.get("image_filename"),
            record.get("name"),
            record.get("full_name"),
        ]
        for key in keys:
            if isinstance(key, str) and key.strip():
                metadata_by_key[key.strip().lower()] = record
    return metadata_by_key


def _poi_metadata_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = _load_poi_metadata()
    image_path = entry.get("image_path")
    image_name = Path(str(image_path)).name.lower() if image_path else None
    keys = [
        image_name,
        str(entry.get("name") or "").strip().lower() or None,
        str(entry.get("full_name") or "").strip().lower() or None,
    ]
    for key in keys:
        if key and key in metadata:
            return metadata[key]
    return {}


def _poi_details_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    metadata = _poi_metadata_for_entry(entry)
    details = {
        "name": metadata.get("full_name") or entry.get("full_name") or metadata.get("name") or entry.get("name"),
        "age": metadata.get("age", entry.get("age")),
        "dob": metadata.get("dob", entry.get("dob")),
        "crime": metadata.get("crime", entry.get("crime")),
        "wanted": metadata.get("wanted", entry.get("wanted")),
        "extra_info": metadata.get("extra_info", entry.get("extra_info")),
    }
    if any(v is not None for v in details.values()):
        return details
    return None


def _run_poi_match_bytes(data: bytes, suffix: str) -> dict[str, Any]:
    try:
        from deepface import DeepFace
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DeepFace unavailable: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.write_bytes(data)
        _record_file_event("write", tmp_path, "POI suspect temp upload", detail=f"{len(data)} bytes")

        db = _load_or_build_poi_db(enforce_detection=False)
        entries = db.get("entries") or []
        if not entries:
            raise HTTPException(status_code=400, detail="POI database is empty.")

        _record_file_event("read", tmp_path, "POI suspect representation")
        reps = DeepFace.represent(
            img_path=str(tmp_path),
            model_name=POI_MODEL_NAME,
            detector_backend=POI_DETECTOR_BACKEND,
            enforce_detection=False,
        )
        rep = _pick_face_representation(reps)
        suspect_emb = rep.get("embedding")
        if suspect_emb is None:
            raise HTTPException(status_code=400, detail="No face embedding could be computed for this image.")

        best_entry = None
        best_distance = None
        for entry in entries:
            emb = entry.get("embedding")
            if not emb:
                continue
            dist = _cosine_distance(suspect_emb, emb)
            if best_distance is None or dist < best_distance:
                best_entry = entry
                best_distance = dist

        if best_entry is None or best_distance is None:
            raise HTTPException(status_code=400, detail="No valid embeddings in POI database.")

        threshold = _poi_threshold()
        match = float(best_distance) <= float(threshold)
        poi_details = _poi_details_from_entry(best_entry)

        return {
            "match": bool(match),
            "distance": float(best_distance),
            "threshold": float(threshold),
            "model_name": POI_MODEL_NAME,
            "detector_backend": POI_DETECTOR_BACKEND,
            "distance_metric": POI_DISTANCE_METRIC,
            "poi_name": best_entry.get("name"),
            "poi_image_path": _normalize_repo_rel_path(best_entry.get("image_path")),
            "poi_details": poi_details,
        }
    finally:
        if tmp_path and tmp_path.exists():
            try:
                _record_file_event("delete", tmp_path, "POI suspect temp upload")
                tmp_path.unlink()
            except Exception:
                pass


def _run_weapon_detection_bytes(data: bytes, suffix: str, conf_threshold: float = DEFAULT_WEAPON_CONF) -> dict[str, Any]:
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.write_bytes(data)
        _record_file_event("write", tmp_path, "weapon detection temp upload", detail=f"{len(data)} bytes")

        _record_file_event("read", tmp_path, "gun detector input")
        _record_file_event("read", DEFAULT_WEIGHTS_PATH, "gun detector weights")
        has_gun, gun_detections, gun_result = detect_weapons(
            tmp_path,
            weights_path=DEFAULT_WEIGHTS_PATH,
            conf_threshold=conf_threshold,
        )
        has_knife, knife_detections, knife_result = _detect_knives(
            tmp_path,
            weights_path=KNIFE_WEIGHTS_PATH,
            conf_threshold=conf_threshold,
        )
        detections = [*gun_detections, *knife_detections]
        has_weapon = bool(has_gun) or bool(has_knife)
        annotated_png_b64 = _weapon_boxes_png_base64(tmp_path, detections)
        meta = {
            "gun": getattr(gun_result, "_meta", None) if gun_result is not None else None,
            "knife": getattr(knife_result, "_meta", None) if knife_result is not None else None,
        }

        return {
            "has_gun": bool(has_gun),
            "has_weapon": bool(has_weapon),
            "has_knife": bool(has_knife),
            "confidence_threshold": float(conf_threshold),
            "weights_path": str(Path(DEFAULT_WEIGHTS_PATH)),
            "detections": [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": d.confidence,
                    "xyxy": d.xyxy,
                }
                for d in detections
            ],
            "annotated_png_base64": annotated_png_b64,
            "warning": None,
            "debug": meta,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IR detection failed: {type(e).__name__}: {e}")
    finally:
        if tmp_path and tmp_path.exists():
            try:
                _record_file_event("delete", tmp_path, "weapon detection temp upload")
                tmp_path.unlink()
            except Exception:
                pass


def _proxy_pi_frame(route: str) -> Response:
    base = _pi_base_url()
    url = f"{base}{route}"
    _record_file_event("proxy", url, "Pi camera frame")
    try:
        r = httpx.get(url, timeout=httpx.Timeout(5.0, connect=3.0))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pi unreachable: {e}")
    if r.status_code != 200:
        detail = r.text.strip()[:500] if r.text else ""
        raise HTTPException(status_code=502, detail=f"Pi error: {r.status_code} {detail}".strip())
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "image/jpeg"),
        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
    )


async def _proxy_pi_stream(route: str) -> StreamingResponse:
    base = _pi_base_url()
    url = f"{base}{route}"
    _record_file_event("proxy", url, "Pi camera stream")

    timeout = httpx.Timeout(connect=3.0, read=None)
    client = httpx.AsyncClient(timeout=timeout)
    try:
        request = client.build_request("GET", url)
        stream = await client.send(request, stream=True)
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Pi unreachable: {e}")

    if stream.status_code != 200:
        try:
            detail_bytes = await stream.aread()
            detail = detail_bytes.decode("utf-8", errors="replace").strip()[:500]
        except Exception:
            detail = ""
        await stream.aclose()
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Pi error: {stream.status_code} {detail}".strip())

    content_type = stream.headers.get("content-type", "multipart/x-mixed-replace; boundary=frame")

    async def _gen():
        try:
            async for chunk in stream.aiter_bytes():
                yield chunk
        finally:
            await stream.aclose()
            await client.aclose()

    return StreamingResponse(
        _gen(),
        media_type=content_type,
        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/metal/status")
def metal_status() -> dict[str, Any]:
    return _get_metal_state()


@app.get("/api/logs")
def logs(tail: int = 200):
    if tail < 1 or tail > 5000:
        raise HTTPException(status_code=400, detail="tail must be between 1 and 5000.")
    if not LOG_FILE.exists():
        _record_file_event("read", LOG_FILE, "backend log tail", status="missing")
        return {"path": str(LOG_FILE), "lines": []}

    _record_file_event("read", LOG_FILE, "backend log tail")
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[-tail:]
    return {"path": str(LOG_FILE), "lines": lines}


@app.get("/api/file-events")
def file_events(tail: int = 200):
    if tail < 1 or tail > FILE_TRACE_MAX:
        raise HTTPException(status_code=400, detail=f"tail must be between 1 and {FILE_TRACE_MAX}.")
    return {"count": len(_file_trace_events), "events": _get_file_events(tail)}


@app.get("/api/images")
def list_images(scope: str = "repo", limit: int = 200):
    if limit < 1 or limit > 2000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 2000.")

    root = BACKEND_ROOT if scope == "backend" else REPO_ROOT if scope == "repo" else None
    if root is None:
        raise HTTPException(status_code=400, detail="scope must be 'backend' or 'repo'.")

    _record_file_event("scan", root, f"gallery list scope={scope}", detail=f"limit={limit}")
    items: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        items.append(
            {
                "path": rel,
                "name": p.name,
                "thumb_jpeg_base64": _thumbnail_base64(p),
            }
        )
        if len(items) >= limit:
            break

    return {"scope": scope, "count": len(items), "items": items}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = DEFAULT_WEAPON_CONF) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    data = await file.read()
    _record_file_event("upload", file.filename or "upload.jpg", "weapon detection upload", detail=f"{len(data)} bytes")
    return _run_weapon_detection_bytes(data, suffix, conf_threshold=conf_threshold)


@app.post("/api/detect-upload")
async def detect_upload(image: UploadFile = File(...), conf: float = Form(DEFAULT_WEAPON_CONF)) -> dict[str, Any]:
    """Legacy upload endpoint from the archived Node/Express API.

    The React frontend uses /api/detect with a ``file`` field. This endpoint is
    kept for old scripts and docs that still post an ``image`` field with a
    ``conf`` form value.
    """
    if conf < 0.0 or conf > 1.0:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1.")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    data = await image.read()
    _record_file_event("upload", image.filename or "upload.jpg", "legacy weapon detection upload", detail=f"{len(data)} bytes")
    return _run_weapon_detection_bytes(data, suffix, conf_threshold=conf)


@app.post("/api/detect-path")
def detect_path(req: DetectPathRequest) -> dict[str, Any]:
    image_path = _resolve_user_path(req.path)
    weights_path = Path(req.weights_path) if req.weights_path else DEFAULT_WEIGHTS_PATH
    _record_file_event("resolve", image_path, "weapon detection path input", detail=f"requested={req.path}")
    _record_file_event("read", weights_path, "weapon detection path weights")
    _record_file_event("read", KNIFE_WEIGHTS_PATH, "knife detection path weights")

    if not image_path.exists():
        _record_file_event("read", image_path, "weapon detection path input", status="missing")
        raise HTTPException(status_code=400, detail=f"Image path does not exist: {image_path}")
    if not weights_path.exists():
        _record_file_event("read", weights_path, "weapon detection path weights", status="missing")
        raise HTTPException(status_code=400, detail=f"Weights path does not exist: {weights_path}")

    try:
        _record_file_event("read", image_path, "gun detector path input")
        has_gun, gun_detections, gun_result = detect_weapons(
            image_path,
            weights_path=weights_path,
            conf_threshold=req.conf_threshold,
        )
        has_knife, knife_detections, knife_result = _detect_knives(
            image_path,
            weights_path=KNIFE_WEIGHTS_PATH,
            conf_threshold=req.conf_threshold,
        )
        detections = [*gun_detections, *knife_detections]
        has_weapon = bool(has_gun) or bool(has_knife)
        annotated_png_b64 = _weapon_boxes_png_base64(image_path, detections)
        meta = {
            "gun": getattr(gun_result, "_meta", None) if gun_result is not None else None,
            "knife": getattr(knife_result, "_meta", None) if knife_result is not None else None,
        }

        return {
            "has_gun": bool(has_gun),
            "has_weapon": bool(has_weapon),
            "has_knife": bool(has_knife),
            "confidence_threshold": float(req.conf_threshold),
            "weights_path": str(weights_path),
            "image_path": str(image_path),
            "detections": [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": d.confidence,
                    "xyxy": d.xyxy,
                }
                for d in detections
            ],
            "annotated_png_base64": annotated_png_b64,
            "warning": None,
            "debug": meta,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("detect path failed (path=%s): %s", image_path, e)
        raise HTTPException(status_code=500, detail=f"Detection failed: {type(e).__name__}: {e}. See backend logs at /api/logs")


@app.get("/api/camera/frame")
def camera_frame():
    return _proxy_pi_frame("/api/camera/frame")


@app.get("/api/camera/stream")
async def camera_stream():
    return await _proxy_pi_stream("/api/camera/stream")


@app.get("/api/ir/frame")
def ir_frame():
    return _proxy_pi_frame("/api/ir/frame")


@app.get("/api/ir/stream")
async def ir_stream():
    return await _proxy_pi_stream("/api/ir/stream")


@app.get("/api/ir/detect")
def ir_detect(conf_threshold: float = DEFAULT_WEAPON_CONF) -> dict[str, Any]:
    frame = _proxy_pi_frame("/api/ir/frame")
    return _run_weapon_detection_bytes(frame.body, ".jpg", conf_threshold=conf_threshold)


@app.get("/api/image")
def get_image(path: str):
    resolved = _resolve_user_path(path)
    _record_file_event("resolve", resolved, "image preview", detail=f"requested={path}")
    if not resolved.exists():
        _record_file_event("read", resolved, "image preview", status="missing")
        raise HTTPException(status_code=404, detail="Image not found.")
    if resolved.suffix.lower() not in IMAGE_EXTS:
        _record_file_event("read", resolved, "image preview", status="unsupported")
        raise HTTPException(status_code=400, detail="Not a supported image type.")
    _record_file_event("read", resolved, "image preview")
    return FileResponse(resolved)


@app.post("/api/detect-person")
async def detect_person(file: UploadFile = File(...), conf_threshold: float = DEFAULT_PERSON_CONF):
    data = await file.read()
    _record_file_event("upload", file.filename or "upload.jpg", "person detection upload", detail=f"{len(data)} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Decode using OpenCV (as required)
    image_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Run existing detection logic (expects a path) by writing temp file
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
        _record_file_event("write", tmp_path, "person detection temp upload", detail=f"{len(data)} bytes")

        _record_file_event("read", tmp_path, "person detector input")
        _record_file_event("read", DEFAULT_PERSON_WEIGHTS_PATH, "person detector weights")
        _has_person, detections, _ = detect_persons(
            tmp_path,
            weights_path=DEFAULT_PERSON_WEIGHTS_PATH,
            conf_threshold=conf_threshold,
        )

        payload = [
            {
                "bbox": d.xyxy,
                "confidence": d.confidence,
            }
            for d in detections
        ]
        return payload
    finally:
        if tmp_path:
            try:
                _record_file_event("delete", tmp_path, "person detection temp upload")
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/api/detect-person-path")
def detect_person_path(req: DetectPersonPathRequest) -> dict[str, Any]:
    image_path = _resolve_user_path(req.path)
    weights_path = Path(req.weights_path) if req.weights_path else DEFAULT_PERSON_WEIGHTS_PATH
    _record_file_event("resolve", image_path, "person detection path input", detail=f"requested={req.path}")
    _record_file_event("read", weights_path, "person detection path weights")

    if not image_path.exists():
        _record_file_event("read", image_path, "person detection path input", status="missing")
        raise HTTPException(status_code=400, detail=f"Image path does not exist: {image_path}")
    if not weights_path.exists():
        _record_file_event("read", weights_path, "person detection path weights", status="missing")
        raise HTTPException(status_code=400, detail=f"Weights path does not exist: {weights_path}")

    try:
        _record_file_event("read", image_path, "person detector path input")
        has_person, detections, result = detect_persons(
            image_path,
            weights_path=weights_path,
            conf_threshold=req.conf_threshold,
        )
        return {
            "has_person": bool(has_person),
            "confidence_threshold": float(req.conf_threshold),
            "weights_path": str(weights_path),
            "image_path": str(image_path),
            "detections": [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": d.confidence,
                    "xyxy": d.xyxy,
                }
                for d in detections
            ],
            "annotated_png_base64": _result_plot_png_base64(result),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("detect person path failed (path=%s): %s", image_path, e)
        raise HTTPException(
            status_code=500,
            detail=f"Person detection failed: {type(e).__name__}: {e}. See backend logs at /api/logs",
        )


@app.post("/api/poi/match")
async def poi_match(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "suspect.jpg").suffix or ".jpg"
    data = await file.read()
    _record_file_event("upload", file.filename or "suspect.jpg", "POI match upload", detail=f"{len(data)} bytes")
    return _run_poi_match_bytes(data, suffix)


@app.post("/api/poi/match-base64")
async def poi_match_base64(req: POIMatchBase64Request) -> dict[str, Any]:
    data, mime = _decode_image_base64(req.image_base64)
    filename = req.filename or "capture"
    suffix = Path(filename).suffix or _image_suffix_from_mime(mime) or ".jpg"
    _record_file_event("capture", filename, "POI match camera capture", detail=f"{len(data)} bytes")
    return _run_poi_match_bytes(data, suffix)

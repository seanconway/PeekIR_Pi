"""
Pi Camera Service (FastAPI)

Run:
  python3 -m uvicorn pi_camera_server:app --host 0.0.0.0 --port 9000

Endpoints:
  GET /api/health
  GET /api/camera/frame
  GET /api/camera/stream
  GET /api/ir/frame
  GET /api/ir/stream

Notes:
- Captures frames in a background thread (~15 FPS).
- Stores latest JPEG bytes in memory (thread-safe).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

try:
    import cv2
except Exception as e:  # pragma: no cover - runtime dependency
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from picamera2 import Picamera2
except Exception as e:  # pragma: no cover - runtime dependency
    Picamera2 = None
    _PICAMERA_IMPORT_ERROR = e
else:
    _PICAMERA_IMPORT_ERROR = None

app = FastAPI(title="Pi Camera Service", version="0.1.0")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Settings. Set this on the Pi to the IR camera's V4L2 device:
#   IR_CAMERA_DEVICE=/dev/video2 python -m uvicorn pi_camera_server:app --host 0.0.0.0 --port 9000
FPS = _env_float("RGB_FPS", 15.0)
JPEG_QUALITY = _env_int("JPEG_QUALITY", 80)
FRAME_WIDTH = _env_int("RGB_FRAME_WIDTH", 640)
FRAME_HEIGHT = _env_int("RGB_FRAME_HEIGHT", 480)
IR_CAMERA_DEVICE = os.getenv("IR_CAMERA_DEVICE", "").strip()
IR_FRAME_WIDTH = _env_int("IR_FRAME_WIDTH", 160)
IR_FRAME_HEIGHT = _env_int("IR_FRAME_HEIGHT", 120)
IR_FPS = _env_float("IR_FPS", 9.0)
IR_RECONNECT_SECONDS = _env_float("IR_RECONNECT_SECONDS", 2.0)
IR_COLORIZE = _env_bool("IR_COLORIZE", True)

# Shared state
_latest_rgb_jpeg: Optional[bytes] = None
_latest_ir_jpeg: Optional[bytes] = None
_rgb_lock = threading.Lock()
_ir_lock = threading.Lock()
_stop_event = threading.Event()
_rgb_thread: Optional[threading.Thread] = None
_ir_thread: Optional[threading.Thread] = None
_rgb_camera: Optional[Picamera2] = None
_ir_camera = None
_rgb_error: Optional[str] = None
_ir_error: Optional[str] = None
_last_rgb_frame_at: Optional[float] = None
_last_ir_frame_at: Optional[float] = None


def _init_camera() -> Picamera2:
    if Picamera2 is None:
        raise RuntimeError(f"picamera2 unavailable: {_PICAMERA_IMPORT_ERROR}")

    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.2)
    return cam


def _init_ir_camera():
    if cv2 is None:
        raise RuntimeError(f"opencv unavailable: {_CV2_IMPORT_ERROR}")
    if not IR_CAMERA_DEVICE:
        raise RuntimeError("IR_CAMERA_DEVICE not set")

    cap = cv2.VideoCapture(IR_CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(IR_CAMERA_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open IR camera device {IR_CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IR_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IR_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, IR_FPS)
    return cap


def _release_ir_camera() -> None:
    global _ir_camera
    if _ir_camera is not None:
        try:
            _ir_camera.release()
        except Exception:
            pass
    _ir_camera = None


def _encode_rgb_jpeg(frame) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV not available for JPEG encoding.")
    # frame from Picamera2 is RGB; OpenCV expects BGR
    bgr = frame[..., ::-1]
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()


def _normalize_ir_frame(frame):
    if cv2 is None or np is None:
        return frame

    if frame is None:
        return frame

    # Many UVC thermal cameras already return BGR. Raw Lepton-style paths can
    # return a single-channel or 16-bit frame; normalize those for browser JPEGs.
    if getattr(frame, "dtype", None) == np.uint8 and getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
        return frame

    data = frame
    if getattr(data, "ndim", 0) == 3:
        data = data[:, :, 0]

    data = data.astype(np.float32, copy=False)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        scaled = np.zeros(data.shape[:2], dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, (1, 99))
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            scaled = np.zeros(data.shape[:2], dtype=np.uint8)
        else:
            scaled = np.clip((data - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    if IR_COLORIZE:
        return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return scaled


def _encode_ir_jpeg(frame) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV not available for JPEG encoding.")
    frame = _normalize_ir_frame(frame)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()


def _rgb_capture_loop():
    global _latest_rgb_jpeg, _rgb_error, _last_rgb_frame_at
    interval = 1.0 / FPS
    while not _stop_event.is_set():
        try:
            frame = _rgb_camera.capture_array()
            jpg = _encode_rgb_jpeg(frame)
            with _rgb_lock:
                _latest_rgb_jpeg = jpg
                _last_rgb_frame_at = time.time()
                _rgb_error = None
        except Exception as e:
            _rgb_error = str(e)
            # On capture error, sleep briefly and retry
            time.sleep(0.1)
        time.sleep(interval)


def _ir_capture_loop():
    global _latest_ir_jpeg, _ir_camera, _ir_error, _last_ir_frame_at
    interval = 1.0 / IR_FPS if IR_FPS > 0 else 0.1
    while not _stop_event.is_set():
        if _ir_camera is None:
            try:
                _ir_camera = _init_ir_camera()
                _ir_error = None
            except Exception as e:
                _ir_error = str(e)
                time.sleep(IR_RECONNECT_SECONDS)
                continue

        try:
            ok, frame = _ir_camera.read()
            if not ok or frame is None:
                _ir_error = f"no frame from {IR_CAMERA_DEVICE}"
                _release_ir_camera()
                time.sleep(0.1)
                continue
            jpg = _encode_ir_jpeg(frame)
            with _ir_lock:
                _latest_ir_jpeg = jpg
                _last_ir_frame_at = time.time()
                _ir_error = None
        except Exception as e:
            _ir_error = str(e)
            _release_ir_camera()
            time.sleep(0.1)
        time.sleep(interval)


@app.on_event("startup")
def _startup() -> None:
    global _rgb_camera, _rgb_thread, _ir_thread, _rgb_error, _ir_error
    _stop_event.clear()

    if _PICAMERA_IMPORT_ERROR is None:
        try:
            _rgb_camera = _init_camera()
            _rgb_thread = threading.Thread(target=_rgb_capture_loop, daemon=True)
            _rgb_thread.start()
        except Exception as e:
            _rgb_error = str(e)
            _rgb_camera = None
            _rgb_thread = None
    else:
        _rgb_error = str(_PICAMERA_IMPORT_ERROR)

    if cv2 is not None:
        _ir_thread = threading.Thread(target=_ir_capture_loop, daemon=True)
        _ir_thread.start()
    else:
        _ir_error = str(_CV2_IMPORT_ERROR)


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop_event.set()
    if _rgb_thread and _rgb_thread.is_alive():
        _rgb_thread.join(timeout=2)
    if _ir_thread and _ir_thread.is_alive():
        _ir_thread.join(timeout=2)
    if _rgb_camera is not None:
        try:
            _rgb_camera.stop()
        except Exception:
            pass
    _release_ir_camera()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "rgb_available": _rgb_camera is not None,
        "rgb_last_frame_at": _last_rgb_frame_at,
        "rgb_error": _rgb_error,
        "ir_available": _ir_camera is not None,
        "ir_device": IR_CAMERA_DEVICE,
        "ir_last_frame_at": _last_ir_frame_at,
        "ir_error": _ir_error,
    }


@app.get("/api/camera/frame")
def camera_frame():
    if _rgb_camera is None:
        raise HTTPException(status_code=503, detail=f"RGB camera unavailable: {_PICAMERA_IMPORT_ERROR}")

    with _rgb_lock:
        jpg = _latest_rgb_jpeg
    if jpg is None:
        raise HTTPException(status_code=503, detail="No RGB frame available yet")
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/camera/stream")
def camera_stream():
    if _rgb_camera is None:
        raise HTTPException(status_code=503, detail=f"RGB camera unavailable: {_PICAMERA_IMPORT_ERROR}")

    boundary = b"--frame\r\n"

    def _gen():
        while True:
            with _rgb_lock:
                jpg = _latest_rgb_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            yield jpg
            yield b"\r\n"
            time.sleep(1.0 / FPS)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/ir/frame")
def ir_frame():
    with _ir_lock:
        jpg = _latest_ir_jpeg
    if jpg is None:
        detail = _ir_error or f"No IR frame available yet from {IR_CAMERA_DEVICE}"
        raise HTTPException(status_code=503, detail=detail)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/ir/stream")
def ir_stream():
    boundary = b"--frame\r\n"

    def _gen():
        while True:
            with _ir_lock:
                jpg = _latest_ir_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            yield jpg
            yield b"\r\n"
            time.sleep(1.0 / IR_FPS if IR_FPS > 0 else 0.1)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )

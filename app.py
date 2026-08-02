import asyncio
import io
import os
import secrets
import threading

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageFilter, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
MODEL_NAME = os.environ.get("REMBG_MODEL", "u2netp") or "u2netp"
MODEL_PATH = os.environ.get("REMBG_MODEL_PATH", "/app/.u2net/u2netp.onnx")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
process_lock = asyncio.Lock()
model_session = None
model_lock = threading.Lock()


def get_model_session():
    global model_session
    if model_session is not None:
        return model_session
    with model_lock:
        if model_session is not None:
            return model_session
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        model_session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return model_session


@app.on_event("startup")
def preload_model():
    threading.Thread(target=get_model_session, name="model-preload", daemon=True).start()


def validate_image(payload: bytes) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width < 32 or height < 32 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Unsupported image dimensions")
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=415, detail="Invalid image") from None


def remove_background(payload: bytes) -> bytes:
    source = Image.open(io.BytesIO(payload)).convert("RGBA")
    rgb = source.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    tensor = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    session = get_model_session()
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: tensor})[0]
    mask = np.squeeze(prediction).astype(np.float32)
    mask -= float(mask.min())
    peak = float(mask.max())
    if peak > 0:
        mask /= peak
    alpha = Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255), mode="L")
    alpha = alpha.resize(source.size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(0.55))

    original_alpha = source.getchannel("A")
    alpha_values = np.asarray(alpha, dtype=np.uint16)
    original_values = np.asarray(original_alpha, dtype=np.uint16)
    source.putalpha(Image.fromarray(np.uint8(alpha_values * original_values / 255), mode="L"))
    output = io.BytesIO()
    source.save(output, format="PNG", optimize=True)
    return output.getvalue()


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/remove")
async def remove_endpoint(request: Request):
    authorization = request.headers.get("authorization", "")
    expected = f"Bearer {SERVICE_TOKEN}"
    if not SERVICE_TOKEN or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="JPEG, PNG or WebP required")
    content_length = int(request.headers.get("content-length", "0") or 0)
    if content_length > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    payload = await request.body()
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    validate_image(payload)

    async with process_lock:
        result = await run_in_threadpool(remove_background, payload)
    return Response(
        content=result,
        media_type="image/png",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )

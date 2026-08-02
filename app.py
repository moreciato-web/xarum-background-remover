import asyncio
import io
import os
import secrets

from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, UnidentifiedImageError
from rembg import new_session, remove
from starlette.concurrency import run_in_threadpool


MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
MODEL_NAME = os.environ.get("REMBG_MODEL", "u2netp")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
process_lock = asyncio.Lock()
model_session = None


def get_model_session():
    global model_session
    if model_session is None:
        model_session = new_session(MODEL_NAME)
    return model_session


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
    return remove(
        payload,
        session=get_model_session(),
        post_process_mask=True,
        force_return_bytes=True,
    )


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

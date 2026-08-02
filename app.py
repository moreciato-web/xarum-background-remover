import asyncio
import io
import os
import secrets
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
MODEL_NAME = os.environ.get("REMBG_MODEL", "u2netp") or "u2netp"
MODEL_PATH = os.environ.get("REMBG_MODEL_PATH", "/app/.u2net/u2netp.onnx")
ASSET_ROOT = Path(__file__).parent / "assets"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

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


def fit_title_lines(draw, title: str, font, max_width: int) -> list[str]:
    words = title.strip().split() or ["new item"]
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:4]


def compose_cover(payload: bytes, title: str, date_label: str, mascot_name: str) -> bytes:
    background = Image.open(ASSET_ROOT / "channel-cover-bg-v1.png").convert("RGBA")
    background = ImageOps.fit(background, (1024, 1024), method=Image.Resampling.LANCZOS)

    allowed_mascot = Path(mascot_name).name
    mascot_path = ASSET_ROOT / "mascots" / allowed_mascot
    if not mascot_path.is_file() or mascot_path.suffix.lower() != ".png":
        mascot_path = ASSET_ROOT / "mascots" / "xarum-bow-v1.png"
    mascot = Image.open(mascot_path).convert("RGBA")
    mascot.thumbnail((285, 375), Image.Resampling.LANCZOS)
    background.alpha_composite(mascot, (1024 - mascot.width - 48, 235))

    cutout = Image.open(io.BytesIO(remove_background(payload))).convert("RGBA")
    cutout.thumbnail((920, 650), Image.Resampling.LANCZOS)
    product_x = (1024 - cutout.width) // 2
    product_y = 955 - cutout.height
    shadow_alpha = cutout.getchannel("A").filter(ImageFilter.GaussianBlur(18))
    shadow = Image.new("RGBA", cutout.size, (10, 22, 38, 0))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.28)))
    background.alpha_composite(shadow, (product_x + 12, product_y + 18))
    background.alpha_composite(cutout, (product_x, product_y))

    overlay = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font_size = 72
    title_font = ImageFont.truetype(FONT_BOLD, title_font_size)
    lines = fit_title_lines(draw, title, title_font, 600)
    while any(draw.textbbox((0, 0), line, font=title_font)[2] > 600 for line in lines) and title_font_size > 48:
        title_font_size -= 4
        title_font = ImageFont.truetype(FONT_BOLD, title_font_size)
        lines = fit_title_lines(draw, title, title_font, 600)
    y = 148
    for line in lines:
        box = draw.textbbox((0, 0), line, font=title_font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        draw.rectangle((66, y - 5, 96 + width, y + height + 5), fill="#ea580c")
        draw.text((80, y - box[1]), line, font=title_font, fill="white")
        y += title_font_size + 8

    date_font = ImageFont.truetype(FONT_BOLD, 27)
    date_box = draw.textbbox((0, 0), date_label, font=date_font)
    date_width = date_box[2] - date_box[0]
    badge = (512 - date_width // 2 - 22, 28, 512 + date_width // 2 + 22, 88)
    draw.rounded_rectangle(badge, radius=30, fill="#1e293b")
    draw.text((512, 58), date_label, font=date_font, fill="white", anchor="mm")
    background = Image.alpha_composite(background, overlay)

    output = io.BytesIO()
    background.convert("RGB").save(output, format="JPEG", quality=91, optimize=True)
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


@app.post("/cover")
async def cover_endpoint(request: Request):
    authorization = request.headers.get("authorization", "")
    expected = f"Bearer {SERVICE_TOKEN}"
    if not SERVICE_TOKEN or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.body()
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    validate_image(payload)
    title = request.query_params.get("title", "new item")[:180]
    date_label = request.query_params.get("date", "new drop")[:40]
    mascot = request.query_params.get("mascot", "xarum-bow-v1.png")[:100]

    async with process_lock:
        result = await run_in_threadpool(compose_cover, payload, title, date_label, mascot)
    return Response(
        content=result,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )

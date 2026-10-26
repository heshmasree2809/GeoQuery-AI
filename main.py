"""SatQuery AI — FastAPI service. Fully offline: no external API calls, no keys."""
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import engine
from .imaging import ImageError, decode, decode_gray_like

VERSION = "2.0.0"
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SatQuery AI", version=VERSION)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


async def _read(f: UploadFile | None):
    if f is None or not f.filename:
        return None
    data = await f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_BYTES // 1024 // 1024} MB limit.")
    return data


def _analyse(question, a_bytes, b_bytes, nir_bytes, gsd):
    img_a, scale = decode(a_bytes)
    gsd_eff = gsd / scale  # downscaling makes each pixel cover more ground
    img_b = decode(b_bytes)[0] if b_bytes else None
    nir = decode_gray_like(nir_bytes, img_a.shape[:2]) if nir_bytes else None
    return engine.run(question, img_a, img_b, nir, gsd_eff)


@app.post("/api/query")
async def query(
    question: str = Form(..., max_length=500),
    image: UploadFile = File(...),
    image_b: UploadFile | None = File(None),
    nir: UploadFile | None = File(None),
    gsd: float = Form(10.0, gt=0, le=10000, description="Ground sample distance in metres/pixel"),
):
    a, b, n = await _read(image), await _read(image_b), await _read(nir)
    if a is None:
        raise HTTPException(400, "An image is required.")
    try:
        result = await run_in_threadpool(_analyse, question, a, b, n, gsd)
    except ImageError as e:
        raise HTTPException(422, str(e))
    return JSONResponse(result)

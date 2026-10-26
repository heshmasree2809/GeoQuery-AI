"""Image decoding and encoding helpers (pure OpenCV/NumPy)."""
import base64
import cv2
import numpy as np

MAX_SIDE = 2048


class ImageError(ValueError):
    pass


def decode(data: bytes, max_side: int = MAX_SIDE):
    """Decode bytes to an 8-bit BGR image. Returns (img, scale); scale < 1 if downsized."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError("Could not decode the image. Use PNG, JPEG or TIFF.")
    if img.dtype != np.uint8:
        peak = float(img.max()) or 1.0
        img = (img.astype(np.float32) / peak * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[..., :3]
    elif img.shape[2] != 3:
        raise ImageError("Unsupported number of channels.")
    h, w = img.shape[:2]
    if min(h, w) < 32:
        raise ImageError("Image is too small (minimum 32x32 pixels).")
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img), scale


def decode_gray_like(data: bytes, shape):
    """Decode a single-band image (e.g. NIR) and resize to `shape` (h, w)."""
    img, _ = decode(data, max_side=10**6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape != tuple(shape):
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return gray


def to_data_uri(img: np.ndarray, max_side: int = 900) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

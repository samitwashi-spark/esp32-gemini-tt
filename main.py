import io
import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, Query, Response
from PIL import Image, ImageDraw, ImageFont, features

app = FastAPI()
HERE = os.path.dirname(os.path.abspath(__file__))

# Put these font files in a "fonts" folder next to main.py
FONT_FILES = {
    "bengali": "NotoSansBengali-Regular.ttf",
    "devanagari": "NotoSansDevanagari-Regular.ttf",
    "arabic": "NotoSansArabic-Regular.ttf",
    "latin": "NotoSans-Regular.ttf",  # English, Spanish, Russian, etc.
}

# RAQM (HarfBuzz) is required for correct Bangla/Hindi/Arabic shaping
HAS_RAQM = features.check("raqm")
ENGINE = ImageFont.Layout.RAQM if HAS_RAQM else ImageFont.Layout.BASIC


def detect_script(text: str) -> str:
    for ch in text:
        c = ord(ch)
        if 0x0980 <= c <= 0x09FF:
            return "bengali"
        if 0x0900 <= c <= 0x097F:
            return "devanagari"
        if (0x0600 <= c <= 0x06FF or 0x0750 <= c <= 0x077F
                or 0xFB50 <= c <= 0xFDFF or 0xFE70 <= c <= 0xFEFF):
            return "arabic"
    return "latin"


@lru_cache(maxsize=64)
def load_font(script: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(HERE, "fonts", FONT_FILES[script])
    return ImageFont.truetype(path, size, layout_engine=ENGINE)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            test = (cur + " " + word).strip()
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def render(text: str, w: int, h: int) -> Image.Image:
    script = detect_script(text)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 3

    # Try big fonts first, shrink until the text fits the screen
    for size in range(24, 9, -1):
        font = load_font(script, size)
        asc, desc = font.getmetrics()
        line_h = asc + desc + 3
        lines = wrap(d, text, font, w - 2 * pad)
        fits_h = len(lines) * line_h <= h - 2 * pad
        fits_w = all(d.textlength(l, font=font) <= w - 2 * pad for l in lines)
        if fits_h and fits_w:
            break

    y = pad
    for line in lines:
        tw = d.textlength(line, font=font)
        x = w - pad - tw if script == "arabic" else pad  # right-align Arabic
        d.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    return img


def to_rgb565_le(img: Image.Image) -> bytes:
    a = np.asarray(img, dtype=np.uint16)
    v = ((a[..., 0] & 0xF8) << 8) | ((a[..., 1] & 0xFC) << 3) | (a[..., 2] >> 3)
    return v.astype("<u2").tobytes()  # little-endian, matches ESP32 uint16_t


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@app.get("/")
def root():
    return {"ok": True, "raqm": HAS_RAQM}


@app.get("/render")
def render_raw(text: str = Query(..., max_length=400), w: int = 128, h: int = 160):
    img = render(text, clamp(w, 16, 320), clamp(h, 16, 320))
    return Response(content=to_rgb565_le(img), media_type="application/octet-stream")


@app.get("/preview")
def preview(text: str = Query(..., max_length=400), w: int = 128, h: int = 160):
    """Open in a browser to check the text before testing on the ESP32."""
    img = render(text, clamp(w, 16, 320), clamp(h, 16, 320))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

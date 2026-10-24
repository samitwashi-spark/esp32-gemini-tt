import sys

# Compatibility patch for Python 3.13+ (audioop was removed from the stdlib)
try:
    import audioop  # noqa: F401
except ImportError:
    try:
        import audioop_lts as _audioop
        sys.modules["audioop"] = _audioop
    except ImportError:
        pass

import io
import os
import re
import time
import struct
import asyncio
from array import array
from contextlib import asynccontextmanager

import edge_tts
from fastapi import FastAPI, WebSocket

try:
    import miniaudio          # fast MP3 decoding (recommended)
except Exception:
    miniaudio = None

try:
    from langdetect import detect as _ld_detect, DetectorFactory
    DetectorFactory.seed = 0  # same text -> same result
except Exception:
    _ld_detect = None

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DEVICE_KEY = os.environ.get("DEVICE_KEY", "")      # optional shared secret
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en")
MAX_TEXT_CHARS = 1500
TTS_PARALLEL = 3

SAMPLE_RATE = 24000          # speech sent to the ESP32
FRAME_BYTES = 4096
FIRST_CHUNK_CHARS = 40
LONG_CHUNK_CHARS = 120
MIN_CHUNK_CHARS = 25         # shorter sentences are joined with the next one
MIN_PCM_BYTES = 2400         # speech shorter than 50 ms counts as a failure

# ---------------------------------------------------------------
# Voices
# ---------------------------------------------------------------
PREFERRED_VOICES = {
    "en": "en-US-EmmaNeural", "bn": "bn-BD-NabanitaNeural", "hi": "hi-IN-SwaraNeural",
    "ar": "ar-AE-FatimaNeural", "ur": "ur-PK-UzmaNeural", "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural", "ml": "ml-IN-SobhanaNeural", "kn": "kn-IN-SapnaNeural",
    "mr": "mr-IN-AarohiNeural", "gu": "gu-IN-DhwaniNeural", "ne": "ne-NP-HemkalaNeural",
    "si": "si-LK-ThiliniNeural", "id": "id-ID-GadisNeural", "ms": "ms-MY-YasminNeural",
    "th": "th-TH-PremwadeeNeural", "vi": "vi-VN-HoaiMyNeural", "tr": "tr-TR-EmelNeural",
    "ru": "ru-RU-SvetlanaNeural", "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural", "it": "it-IT-ElsaNeural", "pt": "pt-BR-FranciscaNeural",
    "fa": "fa-IR-DilaraNeural", "he": "he-IL-HilaNeural", "el": "el-GR-AthinaNeural",
    "pa": "pa-IN-OjasNeural", "uk": "uk-UA-PolinaNeural", "pl": "pl-PL-ZofiaNeural",
    "nl": "nl-NL-ColetteNeural",
}
VOICE_MAP = dict(PREFERRED_VOICES)
LANG_ALIASES = {"iw": "he", "no": "nb", "tl": "fil", "jw": "jv"}


async def load_voices():
    """Build the language -> voice table from Edge's real voice list."""
    voices = await edge_tts.list_voices()
    names = {v["ShortName"] for v in voices}
    by_lang = {}
    for v in voices:
        lang = v["Locale"].split("-")[0].lower()
        by_lang.setdefault(lang, []).append(v)
    for lang, items in by_lang.items():
        preferred = PREFERRED_VOICES.get(lang)
        if preferred in names:
            VOICE_MAP[lang] = preferred
        else:
            female = [v for v in items if v.get("Gender") == "Female"] or items
            VOICE_MAP[lang] = female[0]["ShortName"]
    print(f"Voice table ready: {len(VOICE_MAP)} languages")


# ---------------------------------------------------------------
# Language detection (from the sentence itself)
# ---------------------------------------------------------------
# (start, end, language)
SCRIPT_RANGES = [
    (0x0980, 0x09FF, "bn"), (0x0900, 0x097F, "hi"), (0x0600, 0x06FF, "ar"),
    (0x0750, 0x077F, "ar"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"), (0x0D80, 0x0DFF, "si"), (0x0E00, 0x0E7F, "th"),
    (0x0590, 0x05FF, "he"), (0x0370, 0x03FF, "el"), (0x0400, 0x04FF, "ru"),
    (0x3040, 0x30FF, "ja"), (0xAC00, 0xD7AF, "ko"), (0x1100, 0x11FF, "ko"),
    (0x4E00, 0x9FFF, "zh"),
]
# Scripts shared by several languages: langdetect may refine within these
SHARED = {
    "hi": {"hi", "mr", "ne"},
    "ar": {"ar", "ur", "fa"},
    "ru": {"ru", "uk", "bg", "sr", "mk", "be"},
}


def detect_lang(text: str) -> str:
    counts = {}
    latin = 0
    for ch in text:
        o = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                latin += 1
            continue
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
        else:
            if ch.isalpha():
                latin += 1            # accented Latin letters (é, ü, ñ ...)

    # Japanese text usually mixes kana with CJK characters
    if counts.get("ja") and counts.get("zh"):
        counts["ja"] += counts.pop("zh")

    best_lang, best = (max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0))

    if best_lang and best >= latin and best >= 1:
        group = SHARED.get(best_lang)
        if group and _ld_detect:
            try:
                guess = _ld_detect(text).split("-")[0]
                if guess in group:
                    return guess
            except Exception:
                pass
        return best_lang

    if _ld_detect and latin >= 3:
        try:
            return _ld_detect(text).split("-")[0].lower()
        except Exception:
            pass
    return DEFAULT_LANG


def voice_for(lang: str) -> str:
    primary = (lang or "").strip().lower().split("-")[0]
    primary = LANG_ALIASES.get(primary, primary)
    return (VOICE_MAP.get(primary) or VOICE_MAP.get(DEFAULT_LANG)
            or VOICE_MAP.get("en", "en-US-EmmaNeural"))


# ---------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------
def clean_for_speech(text: str) -> str:
    """Remove markdown symbols. Asterisks between digits (55*453) are kept."""
    text = re.sub(r"(?<!\d)\*+|\*+(?!\d)", "", text)
    text = re.sub(r"[_#`>~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


SENT_RE = re.compile(r'.*?[.!?।؟…]+["\')\]]*(?=\s)|.*?[。！？]+|.*?\n', re.S)
TAG_RE = re.compile(r"^\s*\[([A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?)\]\s*")


def pop_chunk(buf: str, first: bool):
    """Return (chunk, rest). chunk is '' if nothing more can be split off."""
    m = SENT_RE.match(buf)
    if m and m.end() > 0:
        chunk, rest = buf[:m.end()], buf[m.end():]
        ok = True
        while len(chunk.strip()) < MIN_CHUNK_CHARS:
            stripped = rest.lstrip()
            m2 = SENT_RE.match(stripped)
            if not m2 or m2.end() == 0:
                ok = False
                break
            lead = len(rest) - len(stripped)
            chunk += rest[:lead + m2.end()]
            rest = rest[lead + m2.end():]
        if ok:
            return chunk, rest

    limit = FIRST_CHUNK_CHARS if first else LONG_CHUNK_CHARS
    if len(buf) >= limit:
        cut = max(buf.rfind(","), buf.rfind("،"))
        if cut < 15:
            cut = buf.rfind(" ")
        else:
            cut += 1
        if cut > 10:
            return buf[:cut], buf[cut:]
    return "", buf


def split_text(text: str):
    chunks, first = [], True
    buf = text
    while True:
        chunk, buf = pop_chunk(buf, first)
        if not chunk:
            break
        if chunk.strip():
            chunks.append(chunk.strip())
            first = False
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def peak_level(pcm: bytes) -> int:
    """Loudest sample (0 = silence). -1 if it cannot be measured."""
    try:
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        if not samples:
            return 0
        return max(max(samples), -min(samples))
    except Exception:
        return -1


# ---------------------------------------------------------------
# Speech
# ---------------------------------------------------------------
def mp3_to_pcm(mp3: bytes) -> bytes:
    """MP3 -> raw 24 kHz mono 16-bit PCM"""
    if miniaudio is not None:
        decoded = miniaudio.decode(
            mp3,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
        )
        return decoded.samples.tobytes()

    from pydub import AudioSegment    # only used when miniaudio is missing
    sound = AudioSegment.from_mp3(io.BytesIO(mp3))
    sound = sound.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return sound.raw_data


async def synth_pcm(text: str, voice: str) -> bytes:
    """Text -> speech PCM. Retries when Edge TTS returns nothing, too little, or silence."""
    last = None
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text, voice)
            mp3 = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
            if mp3:
                pcm = await asyncio.to_thread(mp3_to_pcm, bytes(mp3))
                peak = peak_level(pcm)
                if len(pcm) >= MIN_PCM_BYTES and (peak < 0 or peak > 50):
                    return pcm
                last = RuntimeError(f"speech too short or silent ({len(pcm)} bytes, peak {peak})")
            else:
                last = RuntimeError("no audio received")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last = e
        print(f"TTS attempt {attempt + 1} failed for '{text[:30]}': {last}")
        await asyncio.sleep(0.3)
    raise last


async def send_pcm(ws: WebSocket, pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        await ws.send_bytes(pcm[i:i + FRAME_BYTES])


# ---------------------------------------------------------------
# One turn: text -> sentences -> (language, voice) -> speech -> ESP32
# ---------------------------------------------------------------
async def run_turn(ws: WebSocket, tid: str, raw_text: str):
    t0 = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    tasks = []
    try:
        text = clean_for_speech(raw_text)[:MAX_TEXT_CHARS]

        # Optional forced language, e.g. "[bn] ..."
        forced = None
        m = TAG_RE.match(text)
        if m:
            forced = m.group(1)
            text = text[m.end():]

        if not any(c.isalnum() for c in text):
            await ws.send_text(f"end:{tid}")
            return

        print(f"[0 ms] Turn {tid}: {text[:60]}")

        sem = asyncio.Semaphore(TTS_PARALLEL)

        async def synth(chunk: str, voice: str) -> bytes:
            async with sem:
                return await synth_pcm(chunk, voice)

        for chunk in split_text(text):
            lang = forced or detect_lang(chunk)
            voice = voice_for(lang)
            print(f"[{ms()} ms] chunk lang={lang} voice={voice}: {chunk[:50]}")
            tasks.append(asyncio.create_task(synth(chunk, voice)))

        started = False
        for task in tasks:                 # play in order
            try:
                pcm = await task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[{ms()} ms] speech error: {e}")
                continue
            if not pcm:
                continue
            if not started:
                await ws.send_text(f"start:{tid}")
                started = True
                print(f"[{ms()} ms] first audio sent")
            await send_pcm(ws, pcm)

        if not started:
            await ws.send_text(f"error:{tid}")
        else:
            await ws.send_text(f"end:{tid}")
        print(f"[{ms()} ms] done")

    except asyncio.CancelledError:
        print(f"[{ms()} ms] cancelled")
        raise
    except Exception as e:
        print(f"[{ms()} ms] error: {e}")
        try:
            await ws.send_text(f"error:{tid}")
        except Exception:
            pass
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


# ---------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------
async def _warm():
    try:
        await load_voices()
    except Exception as e:
        print(f"Voice list not loaded, using built-in table: {e}")
    print(f"Language detection: {'langdetect + scripts' if _ld_detect else 'scripts only'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm = asyncio.create_task(_warm())
    yield
    warm.cancel()


app = FastAPI(lifespan=lifespan)


@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if DEVICE_KEY and ws.query_params.get("key") != DEVICE_KEY:
        await ws.close(code=1008)
        return

    await ws.accept()
    print("ESP32 connected")
    await ws.send_text("ready")

    tid = "0"
    current = None

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text is None:
                continue                   # binary data is ignored (no mic/camera)

            if text.startswith("BEGIN:"):
                if current and not current.done():
                    current.cancel()
                tid = text[6:].strip() or "0"

            elif text.startswith("CANCEL"):
                if current and not current.done():
                    current.cancel()

            elif text.startswith("TEXT:"):
                sentence = text[5:].strip()
                if sentence:
                    if current and not current.done():
                        current.cancel()
                    current = asyncio.create_task(run_turn(ws, tid, sentence))
                else:
                    await ws.send_text(f"end:{tid}")

            elif text == "AUDIO_END":
                # No speech recognition on this server any more
                await ws.send_text(f"end:{tid}")

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        print("ESP32 disconnected")
        if current and not current.done():
            current.cancel()

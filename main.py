import sys

# Drop-in compatibility patch for Python 3.13+ on Render
try:
    import audioop
except ImportError:
    import audioop_lts
    sys.modules['audioop'] = audioop_lts

import io
import os
import re
import time
import struct
import asyncio

from fastapi import FastAPI, WebSocket
import edge_tts
from pydub import AudioSegment
from google import genai
from google.genai import types

try:
    import miniaudio          # optional: much faster MP3 decoding
except Exception:
    miniaudio = None

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEVICE_KEY = os.environ.get("DEVICE_KEY", "")      # optional shared secret

# Tried in order. Add a fallback in Render, e.g. GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.5-flash
GEMINI_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_MODELS", "gemini-3.5-flash-lite"
    ).split(",") if m.strip()
]

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SAMPLE_RATE = 24000          # speech sent to the ESP32
MIC_RATE = 16000             # microphone audio received from the ESP32
FRAME_BYTES = 4096
FIRST_CHUNK_CHARS = 40
LONG_CHUNK_CHARS = 120
MIN_AUDIO_BYTES = 12800      # about 0.4 s
MAX_AUDIO_BYTES = 2_000_000  # about 60 s
MAX_PHOTO_BYTES = 600_000

SYSTEM_PROMPT = (
    "You are a fast voice assistant speaking through a small speaker, and you have a camera. "
    "You receive the user's question either as audio or as typed text. "
    "Understand it in ANY language. "
    "A photo from the device camera, taken at the moment of the question, may be attached. "
    "Use the photo ONLY when the question is about what the camera can see, for example "
    "'what is this', 'what do you see', 'read this', 'describe', 'what color', 'how many', "
    "'is this ripe'. Otherwise ignore the photo completely and never mention it. "
    "Reply in the same language the user spoke or wrote. "
    "Start your reply with the language code in square brackets, then a space, then the answer. "
    "Examples: [en] Hello! [bn] ... [hi] ... [ar] ... [ur] ... [es] ... "
    "Start with the answer immediately. Use 1 to 3 short sentences unless asked for more. "
    "Use plain text only: no markdown, no bullet points, no emojis, no asterisks. "
    "Write math operators as words in the user's language, never as symbols like * or +. "
    "If the audio has no understandable speech, reply exactly: [en] NO_SPEECH"
)

GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=600,
)

BUSY_MESSAGE = "Sorry, I am busy right now. Please try again in a moment."
QUOTA_MESSAGE = "Sorry, my daily limit has been reached. Please try again later."
NOT_HEARD_MESSAGE = "Sorry, I could not hear you clearly. Please try again."

# ---------------------------------------------------------------
# Voices (any language)
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
    "fa": "fa-IR-DilaraNeural", "he": "he-IL-HilaNeural",
}
VOICE_MAP = dict(PREFERRED_VOICES)
LANG_ALIASES = {"iw": "he", "no": "nb", "tl": "fil", "jw": "jv"}

async def load_voices():
    """Build the language -> voice table from Edge's real voice list (validates the names)"""
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

def script_lang(text: str) -> str:
    """Fallback when the model forgets the language tag: guess from the writing script"""
    bn = hi = ar = 0
    for ch in text:
        o = ord(ch)
        if 0x0980 <= o <= 0x09FF:
            bn += 1
        elif 0x0900 <= o <= 0x097F:
            hi += 1
        elif 0x0600 <= o <= 0x06FF:
            ar += 1
    best = max(bn, hi, ar)
    if best >= 2:
        return "bn" if best == bn else ("hi" if best == hi else "ar")
    return "en"

def voice_for(lang, text: str) -> str:
    code = (lang or "").strip().lower() or script_lang(text)
    primary = code.split("-")[0]
    primary = LANG_ALIASES.get(primary, primary)
    return VOICE_MAP.get(primary) or VOICE_MAP.get("en", "en-US-EmmaNeural")

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
    """Return (chunk, rest). chunk is '' if nothing is ready to speak yet."""
    m = SENT_RE.match(buf)
    if m and m.end() > 0:
        return buf[:m.end()], buf[m.end():]

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

def pcm_to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
        struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) +
        b"data" + struct.pack("<I", len(pcm))
    )
    return header + pcm

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

    sound = AudioSegment.from_mp3(io.BytesIO(mp3))
    sound = sound.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return sound.raw_data

async def synth_pcm(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    mp3 = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    if not mp3:
        return b""
    return await asyncio.to_thread(mp3_to_pcm, bytes(mp3))

# ---------------------------------------------------------------
# Gemini (audio + photo + text, streamed)
# ---------------------------------------------------------------
def build_contents(question: str, wav, jpeg):
    contents = []
    if jpeg:
        contents.append("Camera photo taken at the moment of the question:")
        contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    if wav:
        contents.append("The user's spoken question (audio):")
        contents.append(types.Part.from_bytes(data=wav, mime_type="audio/wav"))
    else:
        contents.append(f"The user's typed question: {question}")
    return contents

async def stream_gemini(contents):
    """Yields text pieces as Gemini writes them. Retries once on 503, then tries the next model."""
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")

    last_error = None
    for model in GEMINI_MODELS:
        for attempt in range(2):
            got_text = False
            try:
                stream = await gemini_client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=GEN_CONFIG
                )
                async for chunk in stream:
                    text = chunk.text
                    if text:
                        got_text = True
                        yield text
                if got_text:
                    return
                last_error = RuntimeError("Gemini returned an empty answer")
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[{model}] attempt {attempt + 1} failed: {msg[:200]}")
                if got_text:
                    return
                if "503" in msg or "UNAVAILABLE" in msg:
                    await asyncio.sleep(1)
                    continue          # retry the same model once
                break                 # quota or other error: next model
    raise last_error

# ---------------------------------------------------------------
# One turn: Gemini stream -> sentences -> speech -> ESP32
# ---------------------------------------------------------------
async def send_pcm(ws: WebSocket, pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        await ws.send_bytes(pcm[i:i + FRAME_BYTES])

async def run_turn(ws: WebSocket, tid: str, question: str, wav, jpeg):
    t0 = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    kind = "voice" if wav else "text"
    print(f"[0 ms] Turn {tid}: {kind}, photo={'yes' if jpeg else 'no'}, "
          f"audio={len(wav) if wav else 0} B, photo={len(jpeg) if jpeg else 0} B {question[:60]}")

    queue: asyncio.Queue = asyncio.Queue()
    tasks = []
    state = {"voice": None, "lang": None, "error": None, "no_speech": False}

    def submit(raw: str) -> bool:
        text = clean_for_speech(raw)
        if text.upper().startswith("NO_SPEECH"):
            state["no_speech"] = True
            return False
        if not any(c.isalnum() for c in text):
            return False
        if state["voice"] is None:
            state["voice"] = voice_for(state["lang"], text)
            print(f"[{ms()} ms] language={state['lang']} voice={state['voice']}")
        task = asyncio.create_task(synth_pcm(text, state["voice"]))
        tasks.append(task)
        queue.put_nowait(task)
        print(f"[{ms()} ms] speech chunk: {text[:60]}")
        return True

    async def producer():
        buf = ""
        first = True
        resolved = False
        first_token = False
        try:
            async for piece in stream_gemini(build_contents(question, wav, jpeg)):
                if not first_token:
                    print(f"[{ms()} ms] first Gemini text")
                    first_token = True
                buf += piece

                # Read the language tag at the start, e.g. "[bn] ..."
                if not resolved:
                    stripped = buf.lstrip()
                    if not stripped:
                        continue
                    if stripped.startswith("["):
                        if "]" not in stripped and len(stripped) < 12:
                            continue                  # wait for the rest of the tag
                        m = TAG_RE.match(buf)
                        if m:
                            state["lang"] = m.group(1)
                            buf = buf[m.end():]
                    resolved = True

                while True:
                    chunk, buf = pop_chunk(buf, first)
                    if not chunk:
                        break
                    if submit(chunk):
                        first = False

            if not resolved:
                m = TAG_RE.match(buf)
                if m:
                    state["lang"] = m.group(1)
                    buf = buf[m.end():]
            if buf.strip():
                submit(buf)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[{ms()} ms] Gemini error: {e}")
            state["error"] = str(e)
        finally:
            queue.put_nowait(None)

    prod = asyncio.create_task(producer())
    started = False
    try:
        while True:
            task = await queue.get()
            if task is None:
                break
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

        await prod

        if not started:
            if state["no_speech"]:
                message = NOT_HEARD_MESSAGE
            elif state["error"] and ("429" in state["error"] or "RESOURCE_EXHAUSTED" in state["error"]):
                message = QUOTA_MESSAGE
            else:
                message = BUSY_MESSAGE
            pcm = await synth_pcm(message, VOICE_MAP.get("en", "en-US-EmmaNeural"))
            await ws.send_text(f"start:{tid}")
            await send_pcm(ws, pcm)

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
        prod.cancel()
        for t in tasks:
            if not t.done():
                t.cancel()

# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.on_event("startup")
async def warm_up():
    async def _warm():
        try:
            await load_voices()
        except Exception as e:
            print(f"Voice list not loaded, using built-in table: {e}")
        try:
            await synth_pcm("hi", VOICE_MAP.get("en", "en-US-EmmaNeural"))
            print("Warm-up done")
        except Exception as e:
            print(f"Warm-up skipped: {e}")
    asyncio.create_task(_warm())

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
    photo = bytearray()
    audio = bytearray()
    current = None

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            text = message.get("text")

            # Binary: first byte says what it is. 1 = microphone audio, 2 = camera photo
            if data is not None:
                if len(data) < 2:
                    continue
                kind, body = data[0], data[1:]
                if kind == 2 and len(photo) < MAX_PHOTO_BYTES:
                    photo.extend(body)
                elif kind == 1 and len(audio) < MAX_AUDIO_BYTES:
                    audio.extend(body)
                continue

            if text is None:
                continue

            if text.startswith("BEGIN:"):
                if current and not current.done():
                    current.cancel()
                tid = text[6:].strip() or "0"
                photo = bytearray()
                audio = bytearray()

            elif text.startswith("CANCEL"):
                if current and not current.done():
                    current.cancel()
                photo = bytearray()
                audio = bytearray()

            elif text.startswith("TEXT:"):
                question = text[5:].strip()
                if question:
                    jpeg = bytes(photo) if photo else None
                    current = asyncio.create_task(run_turn(ws, tid, question, None, jpeg))
                photo = bytearray()
                audio = bytearray()

            elif text == "AUDIO_END":
                pcm = bytes(audio)
                jpeg = bytes(photo) if photo else None
                photo = bytearray()
                audio = bytearray()
                if len(pcm) < MIN_AUDIO_BYTES:
                    await ws.send_text(f"end:{tid}")
                else:
                    current = asyncio.create_task(
                        run_turn(ws, tid, "", pcm_to_wav(pcm), jpeg))

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        print("ESP32 disconnected")
        if current and not current.done():
            current.cancel()

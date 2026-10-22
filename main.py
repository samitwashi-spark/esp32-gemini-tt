import sys

# Drop-in compatibility patch for Python 3.13+ (audioop was removed from the stdlib)
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

import httpx
import edge_tts
from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types

try:
    import miniaudio          # optional: much faster MP3 decoding
except Exception:
    miniaudio = None

# ---------------------------------------------------------------
# Settings (all can be overridden with Render environment variables)
# ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
DEVICE_KEY = os.environ.get("DEVICE_KEY", "")      # optional shared secret

# Tried in order. Example: GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.5-flash
GEMINI_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_MODELS", "gemini-3.5-flash-lite"
    ).split(",") if m.strip()
]

# Optional speed boost: set THINKING_LEVEL=minimal (or low) on Render if your model supports it.
THINKING_LEVEL = os.environ.get("THINKING_LEVEL", "").strip()

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "3"))
MAX_TOOL_ROUNDS = 2          # at most 2 search rounds per question

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=30_000))
    if GEMINI_API_KEY else None
)

SAMPLE_RATE = 24000          # speech sent to the ESP32
MIC_RATE = 16000             # microphone audio received from the ESP32
FRAME_BYTES = 4096
FIRST_CHUNK_CHARS = 40
LONG_CHUNK_CHARS = 120
MIN_CHUNK_CHARS = 25         # shorter sentences are joined with the next one
MIN_PCM_BYTES = 2400         # speech shorter than 50 ms counts as a failure
MIN_AUDIO_BYTES = 12800      # about 0.4 s of microphone audio
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
    "You also have a web_search tool. Use it ONLY when the answer depends on current or "
    "recent information you cannot know for sure: news, today's events, weather, sports "
    "scores, prices, exchange rates, stock or crypto values, current office holders, recent "
    "releases, or anything after your training data. NEVER use it for general knowledge, "
    "math, translations, jokes, chit-chat, advice, or questions about the camera photo. "
    "When you search, call the tool immediately WITHOUT writing any text first, use a short "
    "query (in English unless the topic is local), and then answer from the results in the "
    "user's language. Never mention the tool, the search, or the sources. "
    "Reply in the same language the user spoke or wrote. "
    "Start your reply with the language code in square brackets, then a space, then the answer. "
    "Examples: [en] Hello! [bn] ... [hi] ... [ar] ... [ur] ... [es] ... "
    "Start with the answer immediately. Use 1 to 3 short sentences unless asked for more. "
    "Use plain text only: no markdown, no bullet points, no emojis, no asterisks. "
    "Write math operators as words in the user's language, never as symbols like * or +. "
    "If the audio has no understandable speech, reply exactly: [en] NO_SPEECH"
)

SEARCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the internet for current or recent facts (news, weather, scores, prices, "
            "latest events). Returns a short summary and top results."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Short, specific search query.",
                ),
                "topic": types.Schema(
                    type=types.Type.STRING,
                    enum=["general", "news"],
                    description="Use 'news' for breaking or recent news, otherwise 'general'.",
                ),
            },
            required=["query"],
        ),
    )
])


def make_config(with_tools: bool) -> types.GenerateContentConfig:
    kwargs = dict(system_instruction=SYSTEM_PROMPT, max_output_tokens=600)
    if THINKING_LEVEL:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        except Exception as e:
            print(f"THINKING_LEVEL ignored: {e}")
    if with_tools:
        kwargs["tools"] = [SEARCH_TOOL]
        kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    return types.GenerateContentConfig(**kwargs)


TOOLS_ENABLED = bool(TAVILY_API_KEY)
CONFIG_WITH_TOOLS = make_config(True) if TOOLS_ENABLED else None
CONFIG_PLAIN = make_config(False)

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
    """Return (chunk, rest). chunk is '' if nothing is ready to speak yet.
    Very short sentences ("Hello!") are joined with the next one, because tiny
    speech requests fail more often."""
    m = SENT_RE.match(buf)
    if m and m.end() > 0:
        chunk, rest = buf[:m.end()], buf[m.end():]
        ok = True
        while len(chunk.strip()) < MIN_CHUNK_CHARS:
            stripped = rest.lstrip()
            m2 = SENT_RE.match(stripped)
            if not m2 or m2.end() == 0:
                ok = False            # next sentence is not finished yet
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


def pcm_to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
        struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) +
        b"data" + struct.pack("<I", len(pcm))
    )
    return header + pcm


def peak_level(pcm: bytes) -> int:
    """Loudest sample (0 = pure silence, about 30000 = very loud). -1 if it cannot be measured.
    Pure Python (array module), so it does not depend on audioop."""
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

    from pydub import AudioSegment    # imported only when miniaudio is missing
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


# Fixed spoken messages are generated once and then reused (instant, and works
# even when Edge TTS is having a bad moment).
_fallback_cache: dict = {}


async def fallback_pcm(message: str) -> bytes:
    pcm = _fallback_cache.get(message)
    if pcm is None:
        pcm = await synth_pcm(message, VOICE_MAP.get("en", "en-US-EmmaNeural"))
        _fallback_cache[message] = pcm
    return pcm


# ---------------------------------------------------------------
# Tavily web search (called by Gemini only when it decides it needs it)
# ---------------------------------------------------------------
_http: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(7.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=2, keepalive_expiry=60),
        )
    return _http


async def tavily_search(query: str, topic: str = "general") -> str:
    """Returns a compact text block for Gemini. Never raises."""
    if not TAVILY_API_KEY:
        return "Search is not available."
    topic = topic if topic in ("general", "news") else "general"
    payload = {
        "query": query[:300],
        "topic": topic,
        "search_depth": "basic",
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    if topic == "news":
        payload["days"] = 7
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}

    t0 = time.perf_counter()
    data, last = None, "unknown error"
    for attempt in range(2):
        try:
            r = await get_http().post(TAVILY_URL, json=payload, headers=headers)
            if r.status_code == 200:
                data = r.json()
                break
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 429, 432, 433):
                break                      # retrying will not help
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last = str(e)
        await asyncio.sleep(0.2)

    took = int((time.perf_counter() - t0) * 1000)
    if data is None:
        print(f"Tavily failed ({took} ms): {last}")
        return "Search failed. Answer from your own knowledge and say you are not sure it is up to date."

    print(f"Tavily '{query[:50]}' ({topic}) done in {took} ms")
    lines = []
    if data.get("answer"):
        lines.append(f"Summary: {data['answer']}")
    for item in (data.get("results") or [])[:TAVILY_MAX_RESULTS]:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()[:300]
        lines.append(f"- {title}: {content}")
    return "\n".join(lines)[:1800] or "No results found."


async def run_tool(call) -> str:
    args = dict(call.args or {})
    if call.name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "No search query was given."
        return await tavily_search(query, str(args.get("topic") or "general"))
    return "Unknown tool."


# ---------------------------------------------------------------
# Gemini (audio + photo + text, streamed, with optional search tool)
# ---------------------------------------------------------------
def build_contents(question: str, wav, jpeg):
    today = time.strftime("%A, %B %d, %Y", time.gmtime())
    parts = [types.Part.from_text(text=f"Current date (UTC): {today}")]
    if jpeg:
        parts.append(types.Part.from_text(text="Camera photo taken at the moment of the question:"))
        parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    if wav:
        parts.append(types.Part.from_text(text="The user's spoken question (audio):"))
        parts.append(types.Part.from_bytes(data=wav, mime_type="audio/wav"))
    else:
        parts.append(types.Part.from_text(text=f"The user's typed question: {question}"))
    return [types.Content(role="user", parts=parts)]


async def _generate(model: str, contents):
    """One model: streams text; if Gemini asks for a search, runs it and continues."""
    history = list(contents)
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        use_tools = TOOLS_ENABLED and round_no < MAX_TOOL_ROUNDS
        config = CONFIG_WITH_TOOLS if use_tools else CONFIG_PLAIN

        model_parts, calls, got_text = [], [], False
        stream = await gemini_client.aio.models.generate_content_stream(
            model=model, contents=history, config=config
        )
        async for chunk in stream:
            cands = chunk.candidates
            if not cands or not cands[0].content or not cands[0].content.parts:
                continue
            for part in cands[0].content.parts:
                model_parts.append(part)       # keep parts as-is (thought signatures)
                if part.function_call:
                    calls.append(part.function_call)
                elif part.text and not part.thought:
                    got_text = True
                    yield part.text

        if not calls:
            if not got_text:
                raise RuntimeError("Gemini returned an empty answer")
            return

        names = ", ".join(str(dict(c.args or {}).get("query", "")) for c in calls)
        print(f"Gemini wants to search: {names[:100]}")
        results = await asyncio.gather(*(run_tool(c) for c in calls))
        history.append(types.Content(role="model", parts=model_parts))
        history.append(types.Content(role="user", parts=[
            types.Part.from_function_response(name=c.name, response={"result": r})
            for c, r in zip(calls, results)
        ]))

    raise RuntimeError("Gemini kept asking for searches")


async def stream_gemini(contents):
    """Yields text pieces as Gemini writes them. Retries once on 503, then tries the next model."""
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")

    last_error = RuntimeError("no Gemini model configured")
    for model in GEMINI_MODELS:
        for attempt in range(2):
            emitted = False
            try:
                async for text in _generate(model, contents):
                    emitted = True
                    yield text
                return
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[{model}] attempt {attempt + 1} failed: {msg[:200]}")
                if emitted:
                    return                # already speaking: do not start over
                if "503" in msg or "UNAVAILABLE" in msg:
                    await asyncio.sleep(1)
                    continue              # retry the same model once
                break                     # quota or other error: next model
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
    print(f"[0 ms] Turn {tid}: {kind}, audio={len(wav) if wav else 0} B, "
          f"photo={len(jpeg) if jpeg else 0} B {question[:60]}")

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
            print(f"[{ms()} ms] speech ready: {len(pcm)} bytes, "
                  f"{len(pcm) / (SAMPLE_RATE * 2):.1f} s")
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
            pcm = await fallback_pcm(message)
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
# App + routes
# ---------------------------------------------------------------
async def _warm():
    try:
        await load_voices()
    except Exception as e:
        print(f"Voice list not loaded, using built-in table: {e}")
    results = await asyncio.gather(
        fallback_pcm(BUSY_MESSAGE),
        fallback_pcm(QUOTA_MESSAGE),
        fallback_pcm(NOT_HEARD_MESSAGE),
        return_exceptions=True,
    )
    failed = [r for r in results if isinstance(r, Exception)]
    print("Warm-up done" if not failed else f"Warm-up partly skipped: {failed[0]}")
    print(f"Web search: {'ON (Tavily)' if TOOLS_ENABLED else 'OFF (TAVILY_API_KEY not set)'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm = asyncio.create_task(_warm())
    yield
    warm.cancel()
    if _http is not None:
        await _http.aclose()


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

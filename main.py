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
import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import edge_tts
from pydub import AudioSegment
from langdetect import detect
from google import genai
from google.genai import types

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Tried in order. If one hits its quota or is overloaded, the next is used.
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

# Speech-to-text. whisper-large-v3 is a bit more accurate (good for Bangla).
# whisper-large-v3-turbo is faster and cheaper. Change it here or in Render's Environment tab.
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Recordings smaller than this are treated as "too short" (about half a second)
MIN_AUDIO_BYTES = 16000

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = (
    "You are a voice assistant speaking through a small speaker. "
    "Answer in 2 to 4 short sentences. "
    "Reply in the same language the user asked in. "
    "Use plain text only: no markdown, no bullet points, no emojis, no asterisks. "
    "Write math operators as words in the user's language, never as symbols like * or +. "
    "You have a web_search tool. Use it for news, current events, anything recent or "
    "'latest', prices, scores, weather, and general-knowledge or factual lookups "
    "(who, what, when, where questions about real people, places, organizations or events). "
    "Do NOT use it for greetings, small talk, jokes, opinions, advice, math, translation, "
    "writing, or explaining how things work. Just answer those directly. "
    "When web search results are provided, treat them as more up to date than your own "
    "knowledge and base your answer on them. If the results do not contain the answer, "
    "say you could not find it instead of guessing."
)

SEARCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current news, recent events, or factual information.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING",
                    description="A short, specific web search query.",
                )
            },
            required=["query"],
        ),
    )
])

BUSY_MESSAGE = "Sorry, I am busy right now. Please try again in a moment."
TOO_SHORT_MESSAGE = "The recording was too short. Please hold the button and try again."
NOT_HEARD_MESSAGE = "Sorry, I could not hear you clearly. Please try again."

def get_voice_by_langdetect(text: str) -> str:
    """Detects language using langdetect and assigns the neural voice"""
    try:
        lang = detect(text)
        print(f"langdetect identified language code: {lang}")

        if lang == 'bn':
            return "bn-BD-NabanitaNeural"
        elif lang == 'hi':
            return "hi-IN-SwaraNeural"
        elif lang == 'ar':
            return "ar-AE-FatimaNeural"

    except Exception as e:
        print(f"Language detection fallback to English: {e}")

    return "en-US-EmmaNeural"

def clean_for_speech(text: str) -> str:
    """Remove markdown symbols that would be read aloud or break TTS.
    Asterisks between digits (like 55*453) are kept so math is not damaged."""
    text = re.sub(r"(?<!\d)\*+|\*+(?!\d)", "", text)
    text = re.sub(r"[_#`>~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def generate_true_wav_bytes(text: str, voice_model: str) -> bytes:
    """Generates speech in memory and converts it into 24kHz mono 16-bit PCM WAV"""
    communicate = edge_tts.Communicate(text, voice_model)

    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])

    mp3_buffer.seek(0)

    sound = AudioSegment.from_mp3(mp3_buffer)
    sound = sound.set_frame_rate(24000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)

    wav_buffer = io.BytesIO()
    sound.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()

def wav_response(wav_bytes: bytes) -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=response.wav",
            "Content-Length": str(len(wav_bytes))
        }
    )

async def speak_message(text: str) -> Response:
    """Speak a fixed English message (errors, hints) so the speaker is never silent"""
    try:
        return wav_response(await generate_true_wav_bytes(text, "en-US-EmmaNeural"))
    except Exception as e:
        print(f"Could not generate message audio: {e}")
        raise HTTPException(status_code=500, detail=text)

async def tavily_search(query: str) -> str:
    """Search the web with Tavily and return a compact text summary.
    Returns an empty string if search is unavailable, so Gemini still answers."""
    if not TAVILY_API_KEY:
        print("TAVILY_API_KEY not set, skipping web search")
        return ""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3,
                    "include_answer": True,
                    "include_images": True,
                    "include_image_descriptions": True,
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"Tavily search failed, continuing without it: {e}")
        return ""

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")
    for item in data.get("results", []):
        title = item.get("title", "")
        content = (item.get("content") or "")[:500]
        parts.append(f"- {title}: {content}")
    for img in data.get("images", [])[:3]:
        if isinstance(img, dict) and img.get("description"):
            parts.append(f"- Image: {img['description']}")
    return "\n".join(parts)

async def transcribe_with_groq(wav_bytes: bytes, language: str = "") -> str:
    """Send the recorded WAV to Groq Whisper and return the transcript text.
    Leave language empty for auto-detect, or pass a code like 'bn', 'hi', 'en'."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server")

    form = {
        "model": GROQ_STT_MODEL,
        "response_format": "json",
        "temperature": "0",
    }
    if language:
        form["language"] = language

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("question.wav", wav_bytes, "audio/wav")},
            data=form,
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

async def generate_with_fallback(contents, config):
    """Try each model in order. Retry once on 503 (overloaded).
    On quota errors (429) or any other error, move to the next model."""
    last_error = None
    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                return await gemini_client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[{model}] attempt {attempt + 1} failed: {msg[:200]}")
                if "503" in msg or "UNAVAILABLE" in msg:
                    await asyncio.sleep(1)
                    continue          # retry the same model once
                break                 # quota or other error: go to next model
    raise last_error

async def ask_gemini(question: str) -> str:
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")

    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    # Step 1: Gemini decides. It either answers directly or asks for a web search.
    first = await generate_with_fallback(
        contents=f"Today's date: {today}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            tools=[SEARCH_TOOL],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    calls = first.function_calls
    if not calls:
        print("No web search needed, answering directly")
        return first.text or "Sorry, I could not think of an answer."

    # Step 2: Gemini asked for a search, so run Tavily and answer from the results.
    query = (calls[0].args or {}).get("query") or question
    print(f"Searching Tavily for: {query}")
    search_context = await tavily_search(query)
    print(f"Search context:\n{search_context or '(none)'}")

    prompt = f"Today's date: {today}\n\n"
    if search_context:
        prompt += f"Web search results:\n{search_context}\n\n"
    prompt += f"Question: {question}"

    second = await generate_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return second.text or "Sorry, I could not think of an answer."

async def answer_to_wav(question: str) -> bytes:
    """Shared by /ask (typed) and /voice (spoken): question -> Gemini -> speech WAV"""
    answer = clean_for_speech(await ask_gemini(question))
    print(f"Question: {question}\nGemini answer: {answer}")
    voice = get_voice_by_langdetect(answer)
    return await generate_true_wav_bytes(answer, voice)

@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}

# Speaks exactly the text you send
@app.get("/tts")
async def text_to_speech_wav(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")

    try:
        selected_voice = get_voice_by_langdetect(text)
        wav_bytes = await generate_true_wav_bytes(text, selected_voice)
        return wav_response(wav_bytes)

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Typed question (Serial Monitor): Gemini decides whether to search, then speaks
@app.get("/ask")
async def ask_and_speak(q: str):
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return wav_response(await answer_to_wav(q))
    except Exception as e:
        print(f"Server Error: {e}")
        return await speak_message(BUSY_MESSAGE)

# Spoken question (microphone): the ESP32 POSTs the recorded WAV as the raw body.
# Optional: /voice?lang=bn to force a language (default is auto-detect).
@app.post("/voice")
async def voice_question(request: Request, lang: str = ""):
    body = await request.body()
    print(f"Received audio: {len(body)} bytes")

    if len(body) < MIN_AUDIO_BYTES or body[:4] != b"RIFF":
        print("Audio too short or not a WAV file")
        return await speak_message(TOO_SHORT_MESSAGE)

    try:
        question = await transcribe_with_groq(body, lang)
        print(f"Transcript: {question}")

        if not question:
            return await speak_message(NOT_HEARD_MESSAGE)

        return wav_response(await answer_to_wav(question))

    except Exception as e:
        print(f"Server Error: {e}")
        return await speak_message(BUSY_MESSAGE)

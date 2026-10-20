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
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import edge_tts
from pydub import AudioSegment
from langdetect import detect
from google import genai
from google.genai import types

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"   # check AI Studio for the latest model name
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = (
    "You are a voice assistant speaking through a small speaker. "
    "Answer in 2 to 4 short sentences. "
    "Reply in the same language the user asked in. "
    "Use plain text only: no markdown, no bullet points, no emojis, no asterisks."
)

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
    """Remove markdown symbols that would be read aloud or break TTS"""
    text = re.sub(r"[*_#`>~]", "", text)
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

async def ask_gemini(question: str) -> str:
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")

    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=400,
        ),
    )
    return response.text or "Sorry, I could not think of an answer."

@app.get("/ping")
def ping():
    return {"status": "alive"}

# Speaks exactly the text you send (unchanged)
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

# Asks Gemini, then speaks the answer
@app.get("/ask")
async def ask_and_speak(q: str):
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = clean_for_speech(await ask_gemini(q))
        print(f"Question: {q}\nGemini answer: {answer}")

        selected_voice = get_voice_by_langdetect(answer)
        wav_bytes = await generate_true_wav_bytes(answer, selected_voice)
        return wav_response(wav_bytes)

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

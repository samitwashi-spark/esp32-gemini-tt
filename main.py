import sys

try:
    import audioop
except ImportError:
    import audioop_lts
    sys.modules['audioop'] = audioop_lts

import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import edge_tts
from pydub import AudioSegment
from langdetect import detect

app = FastAPI()

def get_voice_by_langdetect(text: str) -> str:
    try:
        lang = detect(text)
        print(f"Language identified: {lang}")
        if lang == 'bn':
            return "bn-BD-NabanitaNeural"
        elif lang == 'hi':
            return "hi-IN-SwaraNeural"
        elif lang == 'ar':
            return "ar-AE-FatimaNeural"
    except Exception as e:
        print(f"Fallback to English: {e}")
        
    return "en-US-EmmaNeural"

async def generate_true_wav_bytes(text: str, voice_model: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice_model)
    
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
            
    mp3_buffer.seek(0)
    
    sound = AudioSegment.from_mp3(mp3_buffer)
    sound = sound.set_frame_rate(24000) # 24kHz
    sound = sound.set_channels(1)       # Mono
    sound = sound.set_sample_width(2)   # 16-bit
    
    wav_buffer = io.BytesIO()
    sound.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/tts")
async def text_to_speech_wav(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
        
    try:
        selected_voice = get_voice_by_langdetect(text)
        wav_bytes = await generate_true_wav_bytes(text, selected_voice)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(wav_bytes))
            }
        )

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

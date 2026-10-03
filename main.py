import sys

try:
    import audioop
except ImportError:
    import audioop_lts
    sys.modules['audioop'] = audioop_lts

import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

async def generate_pcm_raw(text: str, voice_model: str):
    """Generates speech and returns uncompressed raw PCM data (16-bit mono 24kHz)"""
    communicate = edge_tts.Communicate(text, voice_model)
    
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
            
    mp3_buffer.seek(0)
    
    # Process using pydub
    sound = AudioSegment.from_mp3(mp3_buffer)
    sound = sound.set_frame_rate(24000) # 24kHz Sample Rate
    sound = sound.set_channels(1)       # Mono
    sound = sound.set_sample_width(2)   # 16-bit PCM (2 bytes/sample)
    
    # Export raw PCM data (no WAV headers to break streaming)
    pcm_buffer = io.BytesIO()
    sound.export(pcm_buffer, format="raw")
    return pcm_buffer.getvalue()

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/tts")
async def text_to_speech_stream(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
        
    try:
        selected_voice = get_voice_by_langdetect(text)
        pcm_data = await generate_pcm_raw(text, selected_voice)

        def iter_pcm():
            # Stream in 1024-byte chunks
            chunk_size = 1024
            for i in range(0, len(pcm_data), chunk_size):
                yield pcm_data[i:i + chunk_size]

        return StreamingResponse(
            iter_pcm(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Bits-Per-Sample": "16",
                "Content-Length": str(len(pcm_data))
            }
        )

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

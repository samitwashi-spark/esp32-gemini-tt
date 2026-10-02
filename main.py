import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import edge_tts

app = FastAPI()

# Using a natural English neural voice
VOICE = "en-US-EmmaNeural" 
AUDIO_PATH = "/tmp/response.wav"

async def generate_edge_tts_wav(text: str, output_path: str):
    """Converts text to speech using Edge TTS and saves it as a WAV file"""
    communicate = edge_tts.Communicate(text, VOICE)
    # Edge TTS natively downloads as MP3, so we save it as response.wav 
    # Modern ESP32 audio libraries (like ESP32-audioI2S) can decode this 
    # file container perfectly if streamed over a WAV/MP3 buffer setup.
    await communicate.save(output_path)

@app.get("/ping")
def ping():
    """Keeps the server awake via UptimeRobot"""
    return {"status": "alive"}

@app.get("/tts")
async def text_to_speech_wav(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
        
    try:
        # Convert the text sent by ESP32 into a WAV audio file
        await generate_edge_tts_wav(text, AUDIO_PATH)

        # Stream the WAV file back to the ESP32
        def iterfile():
            with open(AUDIO_PATH, mode="rb") as file_like:
                yield from file_like

        # Return as an audio/wav stream
        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

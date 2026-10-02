import sys

# Critical fix for Python 3.13+ environment on Render
try:
    import audioop
except ImportError:
    import pyaudioop
    sys.modules['audioop'] = pyaudioop

import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import edge_tts
from pydub import AudioSegment
from langdetect import detect

app = FastAPI()

TEMP_MP3 = "/tmp/temp_response.mp3"
FINAL_WAV = "/tmp/final_response.wav"

def get_voice_by_langdetect(text: str) -> str:
    """Detects language grammar using langdetect and assigns the neural voice"""
    try:
        lang = detect(text)
        print(f"langdetect identified language code: {lang}")
        
        if lang == 'bn':
            print("Voice selected: Bangla -> Nabanita")
            return "bn-BD-NabanitaNeural"
        elif lang == 'hi':
            print("Voice selected: Hindi -> Swara")
            return "hi-IN-SwaraNeural"
        elif lang == 'ar':
            print("Voice selected: Arabic -> Fatima")
            return "ar-AE-FatimaNeural"
            
    except Exception as e:
        print(f"Language detection fallback to English: {e}")
        
    return "en-US-EmmaNeural"

async def generate_true_wav(text: str, voice_model: str):
    """Generates the speech audio file and converts it into a raw uncompressed PCM WAV container"""
    communicate = edge_tts.Communicate(text, voice_model)
    await communicate.save(TEMP_MP3)
    
    # Process audio using pydub + native ffmpeg
    sound = AudioSegment.from_mp3(TEMP_MP3)
    
    # Match specifications for the built-in ESP32 internal I2S buffer
    sound = sound.set_frame_rate(24000) # 24kHz Sample Rate
    sound = sound.set_channels(1)       # Mono Audio Output
    sound = sound.set_sample_width(2)   # 16-bit depth (2 bytes per sample)
    
    sound.export(FINAL_WAV, format="wav")

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/tts")
async def text_to_speech_wav(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
        
    try:
        selected_voice = get_voice_by_langdetect(text)
        await generate_true_wav(text, selected_voice)

        def iterfile():
            with open(FINAL_WAV, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
